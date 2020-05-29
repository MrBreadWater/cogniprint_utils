import numpy as np
import gc
import glob
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define as func for multiprocessing
def load_img(IMG_PATH, img_dims=(800,600)):
    '''Loads an image and resizes it to the given img_dims

    Takes as input an IMG_PATH, and optionally img_dims (default (800,600))

    Returns an image scaled to img_dims
    '''
    return cv2.resize(cv2.imread(IMG_PATH), img_dims)

def load_imgs(IMG_DIRS, max_num_imgs=10000, img_dims=(800,600), one_class_name="fail", balance_classes=0.85):
    '''Loads each image using 16 threads, resizing to the given img_dims

    Takes as input an array of directories,
    optionally integer max_num_imgs (defaults to MAX_NUM_IMGS)
    and 2-tuple img_dims (default (800,600))
    and one_class_name (default 'fail') that is contained within the path or filename of the one class image examples
    and balance_classes (float >=0, default 0.85), the percentage of zero class examples to delete to balance with the (usually smaller) one class

    Returns a tuple of lists img_paths, img_array, label_array
    '''

    # Load the image array with labels
    img_paths = np.array(sum([glob.glob('%s*.jpg' % DIR) for DIR in IMG_DIRS], []))
    max_num_imgs = min(max_num_imgs, len(img_paths))
    img_paths =  np.random.choice(img_paths, size=max_num_imgs)
    label_array_all = np.array([(1 if "fail" in IMG_PATH else 0) for IMG_PATH in img_paths])
    num_examples = label_array_all.size
    pass_inds = np.where(label_array_all==0)[0]
    num_pass = pass_inds.size
    num_fail = num_examples - num_pass
    num_to_balance = num_pass - num_fail
    num_del = int(balance_classes * num_to_balance) if num_to_balance >= 0 else 0
    print("Not using %s images" % num_del)
    del_pass = np.random.choice(pass_inds, size=num_del, replace=False)
    mask = np.ones(len(label_array_all), dtype=bool)
    mask[del_pass] = False
    p = np.random.permutation(len(label_array_all[mask]))
    label_array = label_array_all[mask][p]
    pool = Pool(16)
    img_paths = img_paths[mask][p]
    img_array = np.array(pool.map(load_img, img_paths))
    del label_array_all
    pool.close()
    pool.join()
    del pool
    gc.collect()
    return img_paths, img_array, label_array

def gen_img_segmentation(image, detections, output_name, save_visualizations=False,  label_map='', output_dir='/tmp/imgs/output/', min_score_thresh=0.0):
    """Creates and saves image segmentation for an object detection.

    Takes as input:
    image (),
    detections,
    output_name

    and optionally: save_visualizations (default False)
    and ouput_dir (default '/tmp/imgs/output/')
    and label_map (default '', required if save_visualizations)
    and min_score_thresh (default 0.0)

    WARNING: Will not throw error if output_dir does not exist, and will not create it.

    Saves images and (if enabled) visualizations of the detections to the output_dir

    Returns nothing.
    """

    if save_visualizations and not label_map:
        raise Exception("label_map must be defined if save_visualizations is enabled")

    if not save_visualizations:
        del image # Remove unecessary overhead

    img_size = (75, 100)
    x = np.zeros([img_size[0],img_size[1], 3])
    for detection in zip(*detections):
        confidence = detection[2]
        if confidence > min_score_thresh:
            channel = int(detection[0] - 1) #R,G,B
            ymin, xmin, ymax, xmax = (detection[1] * np.array(img_size + img_size)).astype(int)

            x[ymin:ymax,xmin:xmax,channel] += confidence * 255

    cv2.imwrite(output_dir + 'detections_%s' % output_name, x[:,:,[2,1,0]])

    if save_visualizations:
        label_map = label_map_util.load_labelmap(label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=3, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(detections[1]),
            np.squeeze(detections[0]).astype(np.int32),
            np.squeeze(detections[2]),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=9,
            min_score_thresh=0.6
        )

        # Display output
        plt.figure(dpi=300)
        plt.imshow(image)
        plt.savefig(output_dir + 'detections_vis_%s' % output_name)
        plt.close()

def gen_img_segmentation_wrapper(input_array, **kwargs):
    '''A wrapper for the gen_img_segmentation

    Requires SAVE_VISUALIZATIONS, LABEL_MAP, OUTPUT_DIR, and IMG_PATHS to be defined.

    Takes as input a tuple: (image, (j, detections, i)) { Why is it formatted that way, you ask? Because pool.map() said so. }
    And any kwargs as options.

    Returns nothing.
    '''
    image, input_array = input_array
    j, detections, i = input_array

    kwargs["save_visualizations"] = kwargs.get("save_visualizations") or SAVE_VISUALIZATIONS
    kwargs["label_map"] = kwargs.get("label_map") or LABEL_MAP

    if OUTPUT_DIR:
        kwargs["output_dir"] = kwargs.get("output_dir") or OUTPUT_DIR

    if IMG_PATHS is not None:
        output_name = IMG_PATHS[(i * 32) + j].split('/')[-1]
    else:
        raise Exception("IMG_PATHS is not defined! Check your IMG_DIRS")
    gen_img_segmentation(image, detections, output_name, **kwargs)
