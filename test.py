import config as cfg
from tools import file_option, tfrecord_option
import logging
import os
from datetime import datetime
import tensorflow as tf
from crnn_model.crnn_network import Net
import numpy as np
import cv2
from tools.inform_trans import sparse_tensor_to_str

logging.basicConfig(level=logging.INFO)

if cfg.TEST_CHECKPOINTS is not None:
    checkpoints_dir = cfg.TEST_CHECKPOINTS

pimages = file_option.get_dir_tree(root=cfg.PATH_TEST_IMAGES)


def test():
    graph = tf.Graph()
    with graph.as_default():

        input_data = tf.placeholder(dtype=tf.float32, shape=[1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3], name='input_data')

        phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')

        crnn_net = Net(phase=phase_tensor)

        with tf.variable_scope('crnn_hit', reuse=False) as scope:
            ctc_input_logist, cnn_feature_dict = crnn_net.build(inputdata=input_data)

        decoded, _ = tf.nn.ctc_beam_search_decoder(ctc_input_logist,
                                                   cfg.TRAIN_SEQUENCE_LENGTH * np.ones(1),
                                                   beam_width=10,
                                                   merge_repeated=False)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if checkpoints_dir is not None:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        for pimage in pimages:
            image = cv2.imread(pimage, cv2.IMREAD_COLOR)
            image_save = np.copy(image)

            c_h, c_s, c_v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            c_v = clahe.apply(c_v)
            image = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)

            image = np.expand_dims((cv2.resize(image, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))/255.0).astype(np.float32), axis=0)

            preds = sess.run(decoded, feed_dict={input_data: image, phase_tensor: 'test'})
            preds = sparse_tensor_to_str(preds[0])

            print(preds[0])
            cv2.imshow('1', image_save)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
