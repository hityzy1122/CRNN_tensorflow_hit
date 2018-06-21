import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
from crnn_model import layers
import config as cfg


class Net(object):
    def __init__(self, phase, rnn_hidden_nums=256, seq_length=cfg.TRAIN_SEQUENCE_LENGTH, num_class=cfg.TRAIN_CLASS_NUMS):
        self._phase = phase
        self._rnn_hidden_nums = rnn_hidden_nums
        self._seq_length = seq_length
        self._num_class = num_class
        self.reuse = False
        self.is_training = tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    def _deep_cnn(self, input_tensor):
        with tf.variable_scope('deep_cnn', reuse=False):
            tensor_dict = dict()
            # 128*32*100*3-------------------------------------------------------------------------
            conv1 = layers.conv2d_bn_active(input_feature=input_tensor, filters=64,
                                            is_training=self.is_training, name='conv1')
            tensor_dict['conv1'] = conv1
            pool1 = layers.pooling(inpitdata=conv1, name='pool1')
            tensor_dict['pool1'] = pool1
            # 128*16*50*64------------------------------------------------------------------------------
            conv2 = layers.conv2d_bn_active(input_feature=pool1, filters=128,
                                            is_training=self.is_training, name='conv2')
            tensor_dict['conv2'] = conv2
            pool2 = layers.pooling(inpitdata=conv2, name='pool2')
            tensor_dict['pool2'] = pool2
            # 128*8*25*128------------------------------------------------------------------------------
            conv3 = layers.conv2d_bn_active(input_feature=pool2, filters=256,
                                            is_training=self.is_training, name='conv3')
            tensor_dict['conv3'] = conv3
            # 128*8*25*256------------------------------------------------------------------------------
            conv4 = layers.conv2d_bn_active(input_feature=conv3, filters=256,
                                            is_training=self.is_training, name='conv4')
            tensor_dict['conv4'] = conv4
            pool4 = layers.pooling(inpitdata=conv4, stride=[2, 1], name='pool4')
            tensor_dict['pool4'] = pool4
            # 128*4*25*512------------------------------------------------------------------------------
            conv5 = layers.conv2d_bn_active(input_feature=pool4, filters=512,
                                            is_training=self.is_training, name='conv5')
            tensor_dict['conv5'] = conv5
            # 128*4*25*512------------------------------------------------------------------------------
            conv6 = layers.conv2d_bn_active(input_feature=conv5, filters=512,
                                            is_training=self.is_training, name='conv6')
            tensor_dict['conv6'] = conv6
            pool6 = layers.pooling(inpitdata=conv6, stride=[2, 1], name='pool6')
            tensor_dict['pool6'] = pool6
            # 128*1*25*512--------------------------------------------------------------------
            conv7 = layers.conv2d_bn_active(input_feature=pool6, filters=512, kernel_size=3,
                                            strides=[2, 1], is_training=self.is_training, name='conv7')
            tensor_dict['conv7'] = conv7

        return conv7, tensor_dict

    def _BiLSTM(self, cnn_feature: tf.Tensor):
        with tf.variable_scope('BiLSTM'):
            cnn_feature_shape = cnn_feature.get_shape().as_list()
            assert cnn_feature_shape[1] == 1
            cnn_feature_squeeze = tf.squeeze(input=cnn_feature, axis=1, name='cnn_feature_squeeze')

            fw_cell_list = [rnn.BasicLSTMCell(num_units=var, forget_bias=1.0)
                            for var in [self._rnn_hidden_nums, self._rnn_hidden_nums]]
            bw_cell_list = [rnn.BasicLSTMCell(num_units=var, forget_bias=1.0)
                            for var in [self._rnn_hidden_nums, self._rnn_hidden_nums]]

            lstm_out, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=fw_cell_list, cells_bw=bw_cell_list,
                inputs=cnn_feature_squeeze, dtype=tf.float32
            )

            lstm_out = tf.cond(self.is_training,
                               lambda: tf.nn.dropout(x=lstm_out, keep_prob=0.5),
                               lambda: tf.identity(lstm_out))
            batch_size, _, _, _ = cnn_feature.get_shape().as_list()
            lstm_out_reshape = tf.reshape(lstm_out, [-1, 2*self._rnn_hidden_nums])

            logist = slim.fully_connected(inputs=lstm_out_reshape,
                                          num_outputs=self._num_class,
                                          activation_fn=None)
            logist_reshape = tf.reshape(logist, [batch_size, -1, self._num_class])

            ctc_input_logist = tf.transpose(logist_reshape, (1, 0, 2), name='ctc_input')
        return ctc_input_logist

    def build(self, inputdata):
        with tf.variable_scope('cnn_rnn', reuse=False) as scope:
            if self.reuse:
                scope.reuse_variables()
        cnn_feature, cnn_feature_dict = self._deep_cnn(input_tensor=inputdata)
        ctc_input_logist = self._BiLSTM(cnn_feature=cnn_feature)

        self.reuse = True
        return ctc_input_logist, cnn_feature_dict


if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default():
        inputdata = tf.placeholder(dtype=tf.float32, shape=[128, 32, 100, 3], name='test')
        inputdata = tf.cast(x=inputdata, dtype=tf.float32)
        phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')

        accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_tensor')
        net = Net(phase=phase_tensor, rnn_hidden_nums=256,
                  seq_length=cfg.TRAIN_SEQUENCE_LENGTH, num_class=cfg.TRAIN_CLASS_NUMS)

        with tf.variable_scope('crnn_hit', reuse=False):
            ctc_out = net.build(inputdata)
