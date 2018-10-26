import tensorflow as tf
import tensorflow.contrib.slim as slim
import subprocess


def restore_weights_s1(sess, checkpoint_filename):
    # used when training separately the networks - step 1
    # uses imagenet ckpt (or a previously trained network ckpt)
    to_exclude = [i.name for i in tf.global_variables()
                  if '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  or 'conv_temp' in i.name
                  ]
    to_exclude.append('resnet_v1_50/logits/*')
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, checkpoint_filename)


def restore_weights_s1_continue(sess, checkpoint_filename, modality):
    # used if testing separately the networks - step 1
    if modality == 'rgb':
        subprocess.call('python codebase/rename_ckpt.py --checkpoint_dir=' + checkpoint_filename +
                        ' --replace_from=resnet_v1_50_rgb/ --replace_to=resnet_v1_50/', shell=True)
    if modality == 'depth':
        subprocess.call('python codebase/rename_ckpt.py --checkpoint_dir=' + checkpoint_filename +
                        ' --replace_from=resnet_v1_50_depth/ --replace_to=resnet_v1_50/', shell=True)
    to_exclude = [i.name for i in tf.global_variables()
                  if '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, checkpoint_filename)


def restore_weights_s2(sess, s1_rgb_ckpt, s1_depth_ckpt):
    # used in step 2 - rgb and depth,  uses weights from step 1
    # do once the following call
    subprocess.call('python codebase/rename_ckpt.py --checkpoint_dir=' + s1_rgb_ckpt +
                    ' --replace_from=resnet_v1_50/ --replace_to=resnet_v1_50_rgb/', shell=True)
    to_exclude = [i.name for i in tf.global_variables()
                  if 'resnet_v1_50_depth' in i.name
                  or '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, s1_rgb_ckpt)

    # do once the following call
    subprocess.call('python codebase/rename_ckpt.py --checkpoint_dir=' + s1_depth_ckpt +
                    ' --replace_from=resnet_v1_50/ --replace_to=resnet_v1_50_depth/', shell=True)
    to_exclude = [i.name for i in tf.global_variables()
                  if 'resnet_v1_50_rgb' in i.name
                  or '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, s1_depth_ckpt)


def restore_weights_s2_continue(sess, s2_ckpt):
    to_exclude = [i.name for i in tf.global_variables()
                  if '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  or 'global_step' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, s2_ckpt)


def restore_weights_s3(sess, s2_ckpt):
    # depth and hall - checkpoint is the one from s2
    to_exclude = [i.name for i in tf.global_variables()
                  if 'resnet_v1_50_hall' in i.name
                  or '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, s2_ckpt)

    # hall also from s2
    subprocess.call('python codebase/rename_ckpt.py --checkpoint_dir=' +
                    s2_ckpt + ' --replace_from=resnet_v1_50_depth/ --replace_to=resnet_v1_50_hall/', shell=True)
    to_exclude = [i.name for i in tf.global_variables()
                  if 'resnet_v1_50_depth' in i.name
                  or '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, s2_ckpt)
    subprocess.call('python codebase/rename_ckpt.py --checkpoint_dir=' + s2_ckpt +
                    ' --replace_from=resnet_v1_50_hall/ --replace_to=resnet_v1_50_depth/', shell=True)


def restore_weights_hall_rgb(sess, rgb_ckpt, hall_ckpt):
    # rgb and hall
    # RGB - from s2
    to_exclude = [i.name for i in tf.global_variables()
                  if 'resnet_v1_50_hall' in i.name
                  or '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, rgb_ckpt)

    # hall - from gan hall
    to_exclude = [i.name for i in tf.global_variables()
                  if 'resnet_v1_50_rgb' in i.name
                  or '/Adam' in i.name
                  or 'beta1_power' in i.name
                  or 'beta2_power' in i.name
                  ]
    variables_to_restore = slim.get_variables_to_restore(exclude=to_exclude)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, hall_ckpt)
