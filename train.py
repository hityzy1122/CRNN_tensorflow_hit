import config as cfg
from tools import file_option, tfrecord_option
import logging
import os
from datetime import datetime
import tensorflow as tf
from crnn_model.crnn_network import Net
import numpy as np
from tools.inform_trans import sparse_tensor_to_str

logging.basicConfig(level=logging.INFO)

if cfg.TRAIN_LOAD_MODEL is not None:
    checkpoints_dir = os.path.join(cfg.PATH_CHECKPOINTS, str(cfg.TRAIN_LOAD_MODEL))
else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = os.path.join(cfg.PATH_CHECKPOINTS, '{}'.format(current_time))
    file_option.make_dir_tree(checkpoints_dir)


graph = tf.Graph()
with graph.as_default():

    input_images, input_labels = tfrecord_option.tf_record_reader(cfg.PATH_TFRECORDS_TRAIN)

    input_data = tf.cast(x=input_images, dtype=tf.float32)/255.0
    phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')
    train_accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_tensor')

    crnn_net = Net(phase=phase_tensor)
    with tf.variable_scope('crnn_hit', reuse=False):
        ctc_input_logist, cnn_feature_dict = crnn_net.build(inputdata=input_data)

    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=ctc_input_logist,
                                         sequence_length=cfg.TRAIN_SEQUENCE_LENGTH*np.ones(cfg.TRAIN_BATCH_SIZE)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(ctc_input_logist,
                                                      cfg.TRAIN_SEQUENCE_LENGTH * np.ones(cfg.TRAIN_BATCH_SIZE),
                                                      merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))
    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate=cfg.TRAIN_LEARNING_RATE,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.90,
                                               staircase=True, name='learning_rate')

    t_var = tf.trainable_variables()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = (tf.train.AdadeltaOptimizer(learning_rate=learning_rate).
                     minimize(loss=loss, global_step=global_step, var_list=t_var))

    lr_summary = tf.summary.scalar(name='learining_rate', tensor=learning_rate)
    tl_summary = tf.summary.scalar(name='train_loss', tensor=loss)
    ta_summary = tf.summary.scalar(name='train_accuracy', tensor=train_accuracy_tensor)
    td_summary = tf.summary.scalar(name='train_seq_dist', tensor=sequence_dist)

    summary_op = tf.summary.merge(inputs=[lr_summary, tl_summary, ta_summary, td_summary])
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
    try:
        while not coord.should_stop():
            _, loss, seq_dist, preds, gt_labels = sess.run([optimizer, loss, sequence_dist, decoded, input_labels],
                                                           feed_dict={phase_tensor: "train"})

            # -----------------------------accuracy------------------------------------------
            preds = sparse_tensor_to_str(preds[0])
            gt_labels = sparse_tensor_to_str(gt_labels)

            accuracy_train = []
            for index, gt_label in enumerate(gt_labels):
                pred = preds[index]
                total_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy_train.append(correct_count/total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy_train.append(1)
                        else:
                            accuracy_train.append(0)

            accuracy_train = np.mean(np.array(accuracy_train).astype(np.float32), axis=0)

            train_summary = sess.run(summary_op, feed_dict={train_accuracy_tensor: accuracy_train,
                                                            phase_tensor: 'train'})
            # -----------------------------accuracy--------------------------------------------------
            summary_writer.add_summary(summary=train_summary, global_step=step)
            summary_writer.flush()

            if step % 1 == 0:
                logging.info('-----------Step %d:-------------' % step)
                logging.info('  loss   : {}'.format(loss))
            if step % 1000 == 0:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info('Model saved in files: %s' % save_path)
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

        coord.request_stop()  # 停止训练
        coord.join(threads)




