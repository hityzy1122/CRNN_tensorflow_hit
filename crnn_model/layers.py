import tensorflow as tf
from tensorflow.contrib import layers as tflayers


def relu(inputdata, name=None):

    return tf.nn.relu(features=inputdata, name=name)


def leaky_relu(inputdata, name=None):

    return tf.nn.leaky_relu(features=inputdata, name=name)


def pooling(inpitdata, poolsize=(2, 2), stride=(2, 2), name=None):

    return tf.layers.max_pooling2d(inputs=inpitdata,
                                   pool_size=poolsize,
                                   strides=stride, padding='same', name=name)


def layerbn(inputdata, is_training, name):

    def f1():
        # print('training')
        return tf.layers.batch_normalization(inputdata, training=True, reuse=False, name=name)

    def f2():
        # print('testing')
        return tf.layers.batch_normalization(inputdata, training=False, reuse=True, name=name)

    output = tf.cond(is_training, f1, f2, name=name)
    return output


def conv2d_bn_active(input_feature=None, filters=None,
                     kernel_size=(3, 3), strides=(1, 1), padding='same',
                     name=None, trainable=True,
                     is_training=tf.constant(True, dtype=tf.bool),
                     if_bn=True, if_active=True):

    conv = tf.layers.conv2d(inputs=input_feature, filters=filters,
                            kernel_size=kernel_size, strides=strides, padding=padding,
                            kernel_initializer=tflayers.xavier_initializer(),
                            kernel_regularizer=tflayers.l2_regularizer(scale=0.1),
                            bias_regularizer=tflayers.l2_regularizer(scale=0.1),
                            trainable=trainable, name=name)

    # bn = tf.identity(layerbn(conv, is_training=is_training, name=name+'_bn')
    #                  if if_bn else
    #                  conv, name=name+'_bn')
    #
    # act = tf.identity(leaky_relu(bn, name+'_bn_act')
    #                   if if_active else
    #                   bn, name+'_bn_act')
    if if_bn:
        bn = layerbn(conv, is_training=is_training, name=name+'_bn')
    else:
        bn = tf.identity(conv, name=name+'_bn')

    if if_active:
        act = leaky_relu(bn, name+'_bn_act')
    else:
        act = tf.identity(bn, name+'_bn_act')

    return act


if __name__ == '__main__':
    pass

    # input = tf.placeholder(dtype=tf.float32, shape=(32, 128, 128, 3), name='testinput')
    # phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')
    # accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_tensor')
    #
    # with tf.variable_scope('shadow', reuse=False):
    #     is_training = tf.equal(phase_tensor, tf.constant('Train', dtype=tf.string))
    #
    #     output = conv2d_bn_active(input_feature=input,
    #                               filters=64, kernel_size=[3,3], strides=(1, 1),
    #                               padding='same',
    #                               trainable=True,
    #                               is_training=is_training,
    #                               if_bn=True,
    #                               if_active=True,
    #                               name='testlayer')