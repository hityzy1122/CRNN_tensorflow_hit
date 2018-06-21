import tensorflow as tf
import config as cfg
import os
import cv2
from tools.file_option import make_dir_tree
from tools import inform_trans
import numpy as np


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, float):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def tf_record_writer(list_path, record_path):
    assert os.path.exists(list_path)
    make_dir_tree(os.path.dirname(record_path))

    with tf.python_io.TFRecordWriter(record_path) as writer:
        with open(list_path, 'r', encoding='utf-8-sig') as train_list:
            for index, line in enumerate(train_list):
                pimage_label = line.strip().split()
                pimage = os.path.join(cfg.PATH_IMAGES, pimage_label[0])
                str_label = pimage_label[1]
                label = inform_trans.trans_str_2_label(str_label)

                image = cv2.imread(pimage, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, (100, 32))
                image = bytes(list(np.reshape(image, 9600)))  # 9600 = 100*3*32

                features = tf.train.Features(feature={
                    'label': int64_feature(label),
                    'image': bytes_feature(image)
                })

                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                if index % 100 == 0:
                    print('{} is done'.format(str(index)))

                # with tf.gfile.FastGFile(pimage, 'rb') as f:
                #     image_data = f.read()
                # img_data_jpg = tf.image.decode_jpeg(image_data, channels=3)
                # img_data_jpg = tf.reshape(img_data_jpg, shape=[31, 85, 3])
                #
                # img_decode = tf.decode_raw(image_data, tf.uint8)
                #
                # with tf.Session() as sess:
                #     img_jpg = img_data_jpg.eval()
                #     a = tf.identity(image_data)


if __name__ == '__main__':
    tf_record_writer(cfg.PATH_TRAIN_LIST, cfg.PATH_TFRECORDS_TRAIN)
