'''Tools for importing the Cogniprint Model

This script is a reusable script for loading a model

It can be imported as a module and contains the following functions:

    * load_model - returns a TensorFlow Session and its Graph, loaded from the given model_dir
    * get_tensors - returns the relevant input/outuput tensors
'''

import tensorflow as tf

def load_model(model_dir='/home/michael/ai/models/research/object_detection/training_ssd_80K/export/serving/'):
    """Loads the TensorFlow model from the given checkpoint directory
    
    Takes as input model_dir (default /home/michael/ai/models/research/object_detection/training_ssd_80K/export/serving/)
    and loads a checkpoint from that variable.

    Returns a tuple, Session and a Graph
    """
    # Start the session using checkpoint
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_dir + 'model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint(model_dir))
    graph = sess.graph

    return sess, graph

def get_tensors(graph):
    """Gets the relevant object detection tensors
    
    Takes as input a TensorFlow Graph

    Returns the tensors detection_classes (-1, 100),
    detection_scores (-1, 100), detection_boxes (-1,100,4), and image_tensor
    """
    tensors = [tf.reshape(graph.get_tensor_by_name('detection_%s:0' % name), shape) for name, shape in [('classes', (-1,100)), ('scores', (-1,100)), ('boxes', (-1,100,4))]]
    tensors += [graph.get_tensor_by_name('image_tensor:0')]
    return tensors

