import config as cfg
from tools import file_option, tfrecord_option
import logging
import os
from datetime import datetime
import tensorflow as tf
from crnn_model.crnn_network import Net
import numpy as np
from tools.train_option import cal_accuracy

logging.basicConfig(level=logging.INFO)

if cfg.TRAIN_LOAD_MODEL is not None:
    checkpoints_dir = os.path.join(cfg.PATH_CHECKPOINTS, str(cfg.TRAIN_LOAD_MODEL))
else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = os.path.join(cfg.PATH_CHECKPOINTS, '{}'.format(current_time))
    file_option.make_dir_tree(checkpoints_dir)


def train():
    graph = tf.Graph()
    with graph.as_default():

        train_images, train_labels = tfrecord_option.tf_record_reader(record_path=cfg.PATH_TFRECORDS_TRAIN,
                                                                      batch_size=cfg.TRAIN_BATCH_SIZE)
        val_images, val_labels = tfrecord_option.tf_record_reader(record_path=cfg.PATH_TFRECORDS_VALIDATION,
                                                                  batch_size=cfg.TRAIN_VAL_BATCH_SIZE)

        input_data_train = tf.cast(x=train_images, dtype=tf.float32)/255.0
        input_data_val = tf.cast(x=val_images, dtype=tf.float32)/255.0

        phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')
        train_accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_tensor')
        val_accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_tensor')

        crnn_net = Net(phase=phase_tensor)

        with tf.variable_scope('crnn_hit', reuse=False) as scope:
            ctc_input_logist, cnn_feature_dict = crnn_net.build(inputdata=input_data_train)
            scope.reuse_variables()
            val_logist, _ = crnn_net.build(inputdata=input_data_val)

        cost_train = tf.reduce_mean(tf.nn.ctc_loss(labels=train_labels, inputs=ctc_input_logist,
                                                   sequence_length=cfg.TRAIN_SEQUENCE_LENGTH *
                                                                   np.ones(cfg.TRAIN_BATCH_SIZE)))

        cost_val = tf.reduce_mean(tf.nn.ctc_loss(labels=val_labels, inputs=val_logist,
                                                 sequence_length=cfg.TRAIN_SEQUENCE_LENGTH *
                                                                 np.ones(cfg.TRAIN_VAL_BATCH_SIZE)))

        decoded_train, _ = tf.nn.ctc_beam_search_decoder(ctc_input_logist,
                                                         cfg.TRAIN_SEQUENCE_LENGTH * np.ones(cfg.TRAIN_BATCH_SIZE),
                                                         beam_width=10,
                                                         merge_repeated=False)

        decoded_val, _ = tf.nn.ctc_beam_search_decoder(val_logist,
                                                       cfg.TRAIN_SEQUENCE_LENGTH * np.ones(cfg.TRAIN_VAL_BATCH_SIZE),
                                                       beam_width=10,
                                                       merge_repeated=False)

        # sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))
        global_step = tf.Variable(0, name='global_step', trainable=False)

        learning_rate = tf.train.exponential_decay(learning_rate=cfg.TRAIN_LEARNING_RATE,
                                                   global_step=global_step,
                                                   decay_steps=10000,
                                                   decay_rate=0.90,
                                                   staircase=True, name='learning_rate')

        t_var = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = (tf.train.AdamOptimizer(learning_rate=learning_rate).
                         minimize(cost_train, global_step=global_step, var_list=t_var))

        lr_summary = tf.summary.scalar(name='learining_rate', tensor=learning_rate)
        tl_summary = tf.summary.scalar(name='train_loss', tensor=cost_train)
        vl_summary = tf.summary.scalar(name='val_loss', tensor=cost_val)
        ta_summary = tf.summary.scalar(name='train_accuracy', tensor=train_accuracy_tensor)
        va_summary = tf.summary.scalar(name='val_accuracy', tensor=val_accuracy_tensor)
        # td_summary = tf.summary.scalar(name='train_seq_dist', tensor=sequence_dist)

        summary_op = tf.summary.merge(inputs=[lr_summary, tl_summary, vl_summary, ta_summary, va_summary])
        summary_writer = tf.summary.FileWriter(logdir=checkpoints_dir, graph=graph)
        saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(graph=graph) as sess:
        if cfg.TRAIN_LOAD_MODEL is not None:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            logging.info('Training from scratch')
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        accuracy_train = 0
        accuracy_val = 0

        try:
            while not coord.should_stop():

                _, loss_train = sess.run([optimizer, cost_train], feed_dict={phase_tensor: "train"})
                loss_val = sess.run(cost_val, feed_dict={phase_tensor: "test"})

                if step % cfg.TRAIN_DISPLAY == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  loss_train   : {} '.format(loss_train))
                    logging.info('  loss_val     : {} '.format(loss_val))

                if step % cfg.TRAIN_VAL_SNAPSHOT == 0:
                    # ----------------------------------------------------------------------------------
                    gt_labels_train, preds_train = sess.run([train_labels, decoded_train],
                                                            feed_dict={phase_tensor: "test"})

                    accuracy_train, gt_labels_train, preds_train = cal_accuracy(gt_labels_train, preds_train)

                    # ----------------validation----------------------------------------------------------
                    gt_labels_val, preds_val = sess.run([val_labels, decoded_val],
                                                        feed_dict={phase_tensor: 'test'})

                    accuracy_val, gt_labels_val, preds_val = cal_accuracy(gt_labels_val, preds_val)
                    # ------------------------------------------------------------------------------------
                    logging.info('  accuracy train  : {} '.format(accuracy_train))
                    logging.info('  accuracy val    : {} '.format(accuracy_val))
                    logging.info('pred_val:{}'.format(preds_val[0:5]))
                    logging.info('gt_val:{}'.format(gt_labels_val[0:5]))

                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info('Model saved in files: %s' % save_path)

                train_summary = sess.run(summary_op, feed_dict={train_accuracy_tensor: accuracy_train,
                                                                val_accuracy_tensor: accuracy_val,
                                                                phase_tensor: 'train'})
                summary_writer.add_summary(summary=train_summary, global_step=step)
                summary_writer.flush()

                step += 1
                if step == cfg.TRAIN_EPHO:
                    coord.request_stop()

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
