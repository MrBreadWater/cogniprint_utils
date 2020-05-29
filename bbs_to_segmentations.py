'''A simple script to convert bounding boxes to rectangular image segmentations

Supports only 3 classes (mapped 1,2,3 -> R,G,B)

Example:
     $ python bbs_to_segmentations.py \
         -c /home/michael/ai/models/research/object_detection/training_ssd_80K/export/serving/ \
         -i /media/michael/Hard Drive/Projects/Project Kronos/sorted_5k_with_synth/pass/, /media/michael/Hard Drive/Projects/Project Kronos/sorted_5k_with_synth/fail/ \
         -o /tmp/imgs/output/ \
         -m 10000 \
         -v False \
         -l /home/michael/ai/models/research/object_detection/data_kronos/1k_labels.pbtxt

Copyright © 2020 Michael Paniagua <mrbreadwater@yahoo.com>
This work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from multiprocessing import Pool
import gc
from progressbar import progressbar
from cogniprint_utils import img_utils as iu
from cogniprint_utils import model_utils as mu

MAX_NUM_IMGS = 10000
CKPT_PATH = '/home/michael/ai/models/research/object_detection/training_ssd_80K/export/serving/'
iu.SAVE_VISUALIZATIONS = False
iu.LABEL_MAP = '/home/michael/ai/models/research/object_detection/data_kronos/1k_labels.pbtxt'
iu.IMG_DIRS =  ['/media/michael/Hard Drive/Projects/Project Kronos/sorted_5k_with_synth/pass/', '/media/michael/Hard Drive/Projects/Project Kronos/sorted_5k_with_synth/fail/'] #['/tmp/imgs/pass/', '/tmp/imgs/fail/']
iu.IMG_PATHS = None
iu.OUTPUT_DIR = None # Use default

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--ckpt-path', default = CKPT_PATH, help='The path to the model checkpoints. (Default: /home/michael/ai/models/research/object_detection/training_ssd_80K/export/serving/)', required=False)
    parser.add_argument('-i','--image-folders', default = iu.IMG_DIRS, help='Image input folders. (Default: /media/michael/Hard Drive/Projects/Project Kronos/sorted_5k_with_synth/pass/, /media/michael/Hard Drive/Projects/Project Kronos/sorted_5k_with_synth/fail/'), required=False)
    parser.add_argument('-o','--output-folder', default = iu.OUTPUT_DIR, help='Image save folder (Default: /tmp/imgs/output/). NOTE: Will NOT be created automatically.', required=False)
    parser.add_argument('-m','--max-num-images', default = MAX_NUM_IMGS, help='The maximum number of images to process. (Default: 10000)', required=False)
    parser.add_argument('-v','--save-visualizations', default = iu.SAVE_VISUALIZATIONS, help='Whether or not to save visualizations. (Default: False)', required=False)
    parser.add_argument('-l','--label-map', default = iu.LABEL_MAP, help='The .pbtext for the labels. (Default: /home/michael/ai/models/research/object_detection/data_kronos/1k_labels.pbtxt). Only relevant when save-visualizations is True.', required=False)
    
    args = parser.parse_args()

    CKPT_PATH = args.ckpt_path
    iu.IMG_DIRS = args.image_folders
    iu.OUTPUT_DIR = args.output_folder
    MAX_NUM_IMGS = args.max_num_images
    iu.SAVE_VISUALIZATIONS = args.save_visualizations
    iu.LABEL_MAP = args.label_map

    sess, graph = mu.load_model(model_dir=CKPT_PATH)
    classes, scores, boxes, image_tensor = mu.get_tensors(graph)
    iu.IMG_PATHS, img_array, _ = iu.load_imgs(iu.IMG_DIRS, max_num_imgs=MAX_NUM_IMGS)
    gc.collect()
    print("img count:", len(img_array))

    pool = Pool(16)
    for i, imgs in progressbar(enumerate(zip(*[iter(img_array)]*32))): # Translation: for i, imgs in enumerated groups of 32 images
        detections = sess.run([classes, boxes, scores], {image_tensor: imgs})
        detections = list(zip(*detections)) # Combine each into individual imgs
        #detections_list += detections
        input_array = enumerate(detections)
        input_array = zip(imgs, [detection + (i,) for detection in input_array])
        pool.map(iu.gen_img_segmentation_wrapper, input_array)
