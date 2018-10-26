# training one modality (RGB or depth), one stream.
# e.g. python s1_train_stream.py --dset=ntu-mini --modality=depth

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from codebase import utils
from codebase import parsers
from codebase import restorers

# tensorflow models_dir ##################################################
import sys
sys.path.insert(0, utils.tensorflow_models_dir)
import nets.resnet_v1_two_stream as resnet_v1
############################################################################


def train(exp_id, files, args):
    log_path = './log'
    ckpt_path = './checkpoint'

    # dataset ######################################################
    train_filenames, val_filenames, test_filenames = utils.get_tfrecords(
        args.eval_mode, files['data'], dataset=args.dset)
    n_classes = utils.get_n_classes(args.dset)

    with tf.device('/cpu:0'):
        dset_train = tf.contrib.data.TFRecordDataset(
            train_filenames, compression_type="GZIP")
        dset_train = dset_train.map(
            lambda x: parsers._parse_fun_one_mod(x, is_training=True, modality=args.modality))
        seed = tf.placeholder(tf.int64, shape=())  # =epoch
        dset_train = dset_train.shuffle(100, seed=seed)
        dset_train = dset_train.batch(args.batch_sz)

        if val_filenames:
            dset_val = tf.contrib.data.TFRecordDataset(
                val_filenames, compression_type="GZIP")
            dset_val = dset_val.map(
                lambda x: parsers._parse_fun_one_mod(x, is_training=False, modality=args.modality))
            dset_val = dset_val.batch(args.batch_sz)

        dset_test = tf.contrib.data.TFRecordDataset(
            test_filenames, compression_type="GZIP")
        dset_test = dset_test.map(
            lambda x: parsers._parse_fun_one_mod(x, is_training=False, modality=args.modality))
        dset_test = dset_test.batch(args.batch_sz)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                               dset_train.output_types, dset_train.output_shapes)

        train_iterator = dset_train.make_initializable_iterator()
        if val_filenames:
            val_iterator = dset_val.make_initializable_iterator()
        test_iterator = dset_test.make_initializable_iterator()

        next_element = iterator.get_next()
        images_stacked = next_element[0]  # [batch_sz, time_bottleneck, h,w,c]
        if args.dset == 'uwa3dii':  # because tfrecords labels are [1,30]
            labels = next_element[1] - 1
        elif 'ntu' in args.dset or args.dset == 'nwucla':
            labels = next_element[1]
        labels = tf.reshape(labels, [-1])
        labels = tf.one_hot(labels, n_classes)

        stack_shape = tf.shape(images_stacked)
        # reshape to [batch * pooled_frames, h,w,c]
        batch_images = tf.reshape(images_stacked, [
                                  stack_shape[0] * stack_shape[1], 224, 224, 3])

    # -----TF.CONFIGPROTO------###########################################
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    # tf Graph input ##############################################
    with tf.device(args.gpu0):
        with slim.arg_scope(resnet_v1.resnet_arg_scope(batch_norm_decay=args.bn_decay)):
            is_training = tf.placeholder(tf.bool, name="is_training")
            nr_frames = parsers.time_bottleneck
            scope = 'resnet_v1_50'

            net_out, net_endpoints = resnet_v1.resnet_one_stream_main(
                batch_images, nr_frames, num_classes=n_classes, scope=scope, gpu_id=args.gpu0, is_training=is_training)

            # predictions for each video are the avg of frames' predictions
            # TRAIN ###############################
            net_train = tf.reshape(net_out, [-1, nr_frames, n_classes])
            net_train = tf.reduce_mean(net_train, axis=1)
            # TEST ###############################
            net_test = tf.reshape(net_out, [-1, nr_frames, n_classes])
            net_test = tf.reduce_mean(net_test, axis=1)

            # loss ##########################################################
            loss = slim.losses.softmax_cross_entropy(net_train, labels)
            # optimizers ######################################################
            optimizer = tf.train.AdamOptimizer(
                learning_rate=args.learning_rate)
            minimizing = slim.learning.create_train_op(loss, optimizer)

            acc_train = utils.accuracy(net_train, labels)
            n_correct = tf.reduce_sum(
                tf.cast(utils.correct_pred(net_test, labels), tf.float32))

    summ_loss = tf.summary.scalar('loss', loss)
    summ_acc_train = tf.summary.scalar('acc_train', acc_train)
    summ_train = tf.summary.merge([summ_acc_train, summ_loss])
    accuracy_value_ = tf.placeholder(tf.float32, shape=())
    summ_acc_test = tf.summary.scalar('acc_test', accuracy_value_)
    summ_acc_val = tf.summary.scalar('acc_val', accuracy_value_)
    test_saver = tf.train.Saver(max_to_keep=3)

    with tf.Session(config=tf_config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        if val_filenames:
            val_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        summary_writer = tf.summary.FileWriter(
            os.path.join(log_path, args.dset, exp_id), sess.graph)

        f_log = open(os.path.join(
            log_path, args.dset, exp_id, 'log.txt'), 'a')
        utils.double_log(f_log, '\n###############################################\n' +
                         exp_id + '\n#####################################\n')
        f_log.write(' '.join(sys.argv[:]) + '\n')
        f_log.flush()

        sess.run(tf.global_variables_initializer())
        if args.just_eval:
            restorers.restore_weights_s1_continue(sess, args.ckpt, args.modality)
        else:
            restorers.restore_weights_s1(
                sess, files['imagenet_checkpoint_file'])

        def val_test(value_step, mode='val'):
            if mode == 'val':
                if not val_filenames:
                    return -1
                utils.double_log(f_log, "eval validation set \n")
                sess.run(val_iterator.initializer)
                step_handle = val_handle
                step_samples = len(val_filenames)
                step_summ = summ_acc_val
            elif mode == 'test':
                utils.double_log(f_log, "eval test set \n")
                sess.run(test_iterator.initializer)
                step_handle = test_handle
                step_samples = len(test_filenames)
                step_summ = summ_acc_test

            try:
                accum_correct = 0
                while True:
                    n_correct_val = sess.run(n_correct, feed_dict={
                                             handle: step_handle, is_training: False})
                    accum_correct += n_correct_val
            except tf.errors.OutOfRangeError:
                step_acc = accum_correct / step_samples
                summary_acc = sess.run(step_summ, feed_dict={
                                       accuracy_value_: step_acc})
                summary_writer.add_summary(summary_acc, value_step)
                utils.double_log(f_log, 'Accuracy = %s \n' % str(step_acc))
            return step_acc

        if args.just_eval:
            val_test(-1, mode='test')
            f_log.close()
            summary_writer.close()
            return

        val_test(-1, mode='val')
        val_test(-1, mode='test')
        n_step = 0
        best_acc = best_epoch = best_step = -1
        for epoch in range(args.n_epochs):
            utils.double_log(f_log, 'epoch %s \n' % str(epoch))
            sess.run(train_iterator.initializer, feed_dict={seed: epoch})
            try:
                while True:
                    print(n_step)
                    if n_step % 100 == 0:  # get summaries
                        _, summary = sess.run(
                            [minimizing, summ_train], feed_dict={handle: train_handle, is_training: True})
                        summary_writer.add_summary(summary, n_step)
                    else:
                        sess.run(minimizing, feed_dict={
                            handle: train_handle, is_training: True})
                    n_step += 1
            except tf.errors.OutOfRangeError:
                acc_validation = val_test(n_step, mode='val')

            if val_filenames:
                acc_epoch = acc_validation
            else:
                continue
            if acc_epoch >= best_acc:
                best_acc = acc_epoch
                best_epoch = epoch
                best_step = n_step
                test_saver.save(
                    sess, os.path.join(ckpt_path, args.dset, exp_id, 'test/model.ckpt'), global_step=n_step)

        utils.double_log(f_log, "Optimization Finished!\n")
        if val_filenames:
            utils.double_log(f_log, str(
                "Best Validation Accuracy: %f at epoch %d %d\n" % (best_acc, best_epoch, best_step)))
            variables_to_restore = slim.get_variables_to_restore()
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, os.path.join(
                ckpt_path, args.dset, exp_id, 'test/model.ckpt-' + str(best_step)))
        else:
            test_saver.save(
                sess, os.path.join(ckpt_path, args.dset, exp_id, 'test/model.ckpt'), global_step=n_step)

        val_test(n_step + 1, mode='test')
        f_log.close()
        summary_writer.close()


def main():
    args = utils.get_arguments()
    experiment_id_prefix = 's1_train_' + args.modality
    exp_id = utils.create_folders(prefix=experiment_id_prefix,
                                  dset=args.dset, eval_mode=args.eval_mode)
    files = utils.load_files_paths(args.dset)
    train(exp_id, files, args)


if __name__ == '__main__':
    main()
