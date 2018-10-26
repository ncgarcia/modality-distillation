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
    hallucination_layer_true = 'resnet_v1_50_depth/block4'
    hallucination_layer_hall = 'resnet_v1_50_hall/block4'
    scope_rgb = 'hall'

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
            lambda x: parsers._parse_fun_2stream(x, is_training=True))
        seed = tf.placeholder(tf.int64, shape=())
        dset_train = dset_train.shuffle(100, seed=seed)
        dset_train = dset_train.batch(args.batch_sz)

        if val_filenames:
            dset_val = tf.contrib.data.TFRecordDataset(
                val_filenames, compression_type="GZIP")
            dset_val = dset_val.map(
                lambda x: parsers._parse_fun_2stream(x, is_training=False))
            dset_val = dset_val.batch(args.batch_sz)

        dset_test = tf.contrib.data.TFRecordDataset(
            test_filenames, compression_type="GZIP")
        dset_test = dset_test.map(
            lambda x: parsers._parse_fun_2stream(x, is_training=False))
        dset_test = dset_test.batch(args.batch_sz)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                               dset_train.output_types, dset_train.output_shapes)

        train_iterator = dset_train.make_initializable_iterator()
        if val_filenames:
            val_iterator = dset_val.make_initializable_iterator()
        test_iterator = dset_test.make_initializable_iterator()

        next_element = iterator.get_next()
        images_depth_stacked = next_element[0]  # [batch, pooled_frames, h,w,c]
        images_rgb_stacked = next_element[1]
        if args.dset == 'uwa3dii':  # because tfrecords labels are [1,30]
            labels = next_element[2] - 1
        elif 'ntu' in args.dset or args.dset == 'nwucla':
            labels = next_element[2]
        labels = tf.reshape(labels, [-1])
        labels = tf.one_hot(labels, n_classes)

        rgb_stack_shape = tf.shape(images_rgb_stacked)
        depth_stack_shape = tf.shape(images_depth_stacked)
        # reshape to [batch * pooled_frames, h,w,c]
        images_rgb = tf.reshape(images_rgb_stacked, [
                                rgb_stack_shape[0] * rgb_stack_shape[1], 224, 224, 3])
        images_depth = tf.reshape(images_depth_stacked, [
                                  depth_stack_shape[0] * depth_stack_shape[1], 224, 224, 3])

    # -----TF.CONFIGPROTO------###########################################
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    # tf Graph input ##############################################
    with tf.device(args.gpu0):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            is_training = tf.placeholder(tf.bool, [])
            nr_frames = parsers.time_bottleneck

            net_depth_out, endpoints_depth, net_hall_out, endpoints_hall = resnet_v1.resnet_twostream_main(
                images_depth, images_rgb, nr_frames, num_classes=n_classes, scope_rgb_stream=scope_rgb, depth_training=False, is_training=is_training, interaction=args.interaction)

            # TRAIN ###############################
            net_depth_train = tf.reshape(
                net_depth_out, [-1, parsers.time_bottleneck, n_classes])
            net_depth_train = tf.reduce_mean(net_depth_train, axis=1)
            net_hall_train = tf.reshape(
                net_hall_out, [-1, parsers.time_bottleneck, n_classes])
            net_hall_train = tf.reduce_mean(net_hall_train, axis=1)

            # TEST ###############################
            net_hall_test = tf.reshape(
                net_hall_out, [-1, parsers.time_bottleneck, n_classes])
            net_hall_test = tf.reduce_mean(net_hall_test, axis=1)
            net_depth_test = tf.reshape(
                net_depth_out, [-1, parsers.time_bottleneck, n_classes])
            net_depth_test = tf.reduce_mean(net_depth_test, axis=1)

            # losses ##########################################################
            loss_hard_hall_class = slim.losses.softmax_cross_entropy(
                net_hall_train, labels)  # hard labels loss

            T = 10  # Temperature T
            teacher_soft_labels = utils.softmax_distill(net_depth_train, T)
            student_soft_labels = utils.softmax_distill(net_hall_train, T)
            loss_soft_hall_class = utils.cross_entropy(
                student_soft_labels, teacher_soft_labels)

            lambda1 = 0.5  # lambda immitation - contrib of soft_label_entropy
            loss_hall_distill = (1 - lambda1) * loss_hard_hall_class + \
                lambda1 * loss_soft_hall_class

            loss_hall_rect_static = utils.loss_hall_rect(
                endpoints_depth[hallucination_layer_true], endpoints_hall[hallucination_layer_hall])

            lambda2 = 0.5  # strength of cross entropy loss
            lambda3 = 0.01
            loss = lambda2 * loss_hall_distill + \
                (1 - lambda2) * loss_hall_rect_static * lambda3

            optimizer = tf.train.AdamOptimizer(
                learning_rate=args.learning_rate)
            # freezing depth
            to_remove = [
                x for x in list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) if 'resnet_v1_50_depth/' in x.name]
            train_vars = [x for x in tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES) if x not in to_remove]
            minimizing = slim.learning.create_train_op(
                loss, optimizer, variables_to_train=train_vars)

            ###################################################################
            acc_depth_train = utils.accuracy(net_depth_train, labels)
            acc_hall_train = utils.accuracy(net_hall_train, labels)

            n_correct_depth = tf.reduce_sum(
                tf.cast(utils.correct_pred(net_depth_test, labels), tf.float32))
            n_correct_hall = tf.reduce_sum(
                tf.cast(utils.correct_pred(net_hall_test, labels), tf.float32))

    summ_loss_combined = tf.summary.scalar(
        'loss', loss)
    summ_loss_hall_l2 = tf.summary.scalar(
        'loss_hall_l2', loss_hall_rect_static)
    summ_loss_hard_hall = tf.summary.scalar(
        'loss_hard_hall_class', loss_hard_hall_class)
    summ_loss_soft_hall = tf.summary.scalar(
        'loss_soft_hall_class', loss_soft_hall_class)
    summ_acc_train_hall = tf.summary.scalar('acc_train_hall', acc_hall_train)
    summ_acc_train_depth = tf.summary.scalar(
        'acc_train_depth', acc_depth_train)
    summ_train = tf.summary.merge([summ_acc_train_hall, summ_acc_train_depth,
                                   summ_loss_hall_l2, summ_loss_hard_hall, summ_loss_combined, summ_loss_soft_hall])

    accuracy_value_ = tf.placeholder(tf.float32, shape=())
    summ_acc_val_hall = tf.summary.scalar('acc_val_hall', accuracy_value_)
    summ_acc_val_depth = tf.summary.scalar('acc_val_depth', accuracy_value_)
    summ_acc_test_hall = tf.summary.scalar('acc_test_hall', accuracy_value_)
    summ_acc_test_depth = tf.summary.scalar('acc_test_depth', accuracy_value_)
    test_saver = tf.train.Saver(max_to_keep=3)

    with tf.Session(config=tf_config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        if val_filenames:
            val_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        summary_writer = tf.summary.FileWriter(
            os.path.join(log_path, args.dset, exp_id), sess.graph)

        f_log = open(os.path.join(log_path, args.dset, exp_id, 'log.txt'), 'a')
        utils.double_log(f_log, '\n###############################################\n' +
                         exp_id + '\n#####################################\n')
        f_log.write(' '.join(sys.argv[:]) + '\n')
        utils.update_details_file(
            f_log, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

        sess.run(tf.global_variables_initializer())
        restorers.restore_weights_s3(sess, args.ckpt)

        def val_test(value_step, mode='val'):
            if mode == 'val':
                if not val_filenames:
                    return -1
                utils.double_log(f_log, "eval validation set \n")
                sess.run(val_iterator.initializer)
                step_handle = val_handle
                step_samples = len(val_filenames)
                step_summ_hall = summ_acc_val_hall
                step_summ_depth = summ_acc_val_depth
            elif mode == 'test':
                utils.double_log(f_log, "eval test set \n")
                sess.run(test_iterator.initializer)
                step_handle = test_handle
                step_samples = len(test_filenames)
                step_summ_hall = summ_acc_test_hall
                step_summ_depth = summ_acc_test_depth
            try:
                accum_correct_depth = accum_correct_hall = 0
                while True:
                    n_correct_hall1, n_correct_depth1 = sess.run(
                        [n_correct_hall, n_correct_depth], feed_dict={handle: step_handle, is_training: False})
                    accum_correct_depth += n_correct_depth1
                    accum_correct_hall += n_correct_hall1
            except tf.errors.OutOfRangeError:
                acc_hall = accum_correct_hall / step_samples
                acc_depth = accum_correct_depth / step_samples
                summ_hall_acc = sess.run(step_summ_hall, feed_dict={
                    accuracy_value_: acc_hall})
                summary_writer.add_summary(summ_hall_acc, value_step)
                summ_depth_acc = sess.run(step_summ_depth, feed_dict={
                    accuracy_value_: acc_depth})
                summary_writer.add_summary(summ_depth_acc, value_step)
                utils.double_log(
                    f_log, 'Hall acc = %s \n' % str(acc_hall))
                utils.double_log(
                    f_log, 'Depth acc = %s \n' % str(acc_depth))
                return acc_hall

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
                    if n_step % 100 == 0:
                        _, summary = sess.run(
                            [minimizing, summ_train], feed_dict={handle: train_handle, is_training: True})
                        summary_writer.add_summary(summary, n_step)
                    else:
                        sess.run([minimizing], feed_dict={
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
        if val_filenames:  # restore best validation model
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
    experiment_id_prefix = 's3-hall-depth-distill'
    exp_id = utils.create_folders(prefix=experiment_id_prefix,
                                  dset=args.dset, eval_mode=args.eval_mode)
    files = utils.load_files_paths(args.dset)
    train(exp_id, files, args)


if __name__ == '__main__':
    main()
