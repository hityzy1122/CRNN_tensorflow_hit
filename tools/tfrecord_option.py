import tensorflow as tf
import config as cfg
import os
import cv2
from tools import inform_trans


def tf_record_reader(record_path, batch_size):
    if not os.path.exists(record_path):
        raise FileExistsError('{} is not existed'.format(record_path))
    filename_queue = tf.train.string_input_producer([record_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.VarLenFeature(tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })
    image_buffer = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image_buffer, [32, 100, 3])

    label_buffer = features['label']
    label = tf.cast(label_buffer, tf.int32)

    image_batch, label_batch = \
        tf.train.shuffle_batch(tensors=[image, label],
                               batch_size=batch_size,
                               capacity=1024+2*batch_size,
                               min_after_dequeue=128,
                               num_threads=64, seed=8)
    return image_batch, label_batch


if __name__ == '__main__':
    # TFRecordWriter(cfg.PATH_TRAIN_LIST, cfg.PATH_TFRECORDS_TRAIN)
    tf_record_reader(cfg.PATH_TFRECORDS_TRAIN)

    images_batch, labels_batch = tf_record_reader(cfg.PATH_TFRECORDS_TRAIN)
    cv2.namedWindow('1', 0)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(200000):
            images, labels = sess.run([images_batch, labels_batch])
            labels = inform_trans.sparse_tensor_to_str(labels)
            for idx in range(1):
                print(labels[idx])
                cv2.imshow('1', images[idx, ...])
                cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads=threads)
