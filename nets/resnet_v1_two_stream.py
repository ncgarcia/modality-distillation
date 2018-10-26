from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


@slim.add_arg_scope
def bottleneck_normal(inputs,
                      depth,
                      depth_bottleneck,
                      stride,
                      nr_frames=None,
                      rate=1,
                      outputs_collections=None,
                      scope=None,
                      use_bounded_activations=False,
                      temporal=False):
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth, [1, 1],
                stride=stride,
                activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')

        if temporal:  # temporal conv
            residual = resnet_utils.conv_temp2(
                residual, nr_frames, scope='conv_temp')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


@slim.add_arg_scope
def bottleneck_injected(inputs,
                        depth,
                        depth_bottleneck,
                        stride,
                        nr_frames=None,
                        rate=1,
                        outputs_collections=None,
                        scope=None,
                        use_bounded_activations=False,
                        temporal=False, multiplier=None, net_before_relu=None, unit_id=-1):
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth, [1, 1],
                stride=stride,
                activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                scope='shortcut')

        if temporal:
            # if unit x which coincides with temporal conv and depth / OF injection,
            # multiply feature maps
            residual_mult = tf.multiply(net_before_relu, multiplier)
            inputs = inputs + residual_mult

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')

        if temporal:  # temporal conv
            residual = resnet_utils.conv_temp2(
                residual, nr_frames, scope='conv_temp')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            if unit_id == 0:
                output_before_relu = shortcut + residual
                output = tf.nn.relu(shortcut + residual)
                return slim.utils.collect_named_outputs(outputs_collections,
                                                        sc.original_name_scope,
                                                        output), output_before_relu
            else:
                output = tf.nn.relu(shortcut + residual)
                return slim.utils.collect_named_outputs(outputs_collections,
                                                        sc.original_name_scope,
                                                        output)


############################## network #############################
def resnet_one_stream(inputs, blocks, nr_frames,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      include_root_block=True,
                      spatial_squeeze=True,
                      reuse=None,
                      scope=None,
                      gpu_id='/gpu:0'):
    bottleneck = bottleneck_normal
    with tf.device(gpu_id):
        with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, bottleneck,
                                 resnet_utils.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    if include_root_block:
                        if output_stride is not None:
                            if output_stride % 4 != 0:
                                raise ValueError(
                                    'The output_stride needs to be a multiple of 4.')
                            output_stride /= 4
                        net = resnet_utils.conv2d_same(
                            net, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(
                            net, [3, 3], stride=2, scope='pool1')
                    net = resnet_utils.stack_blocks_dense(
                        net, blocks, nr_frames, output_stride)
                    if global_pool:
                        net = tf.reduce_mean(
                            net, [1, 2], name='pool5', keep_dims=True)
                    last_pool = net
                    if num_classes is not None:
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                          normalizer_fn=None, scope='logits')
                        if spatial_squeeze:
                            net = tf.squeeze(
                                net, [1, 2], name='SpatialSqueeze')
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)
                    if num_classes is not None:
                        end_points['predictions'] = slim.softmax(
                            net, scope='predictions')
                    end_points['last_pool'] = last_pool
    return net, end_points


# S2 two stream #########################
def resnet_twostream_inter(inputs_depth, blocks_depth,
                           inputs_rgb, blocks_rgb, nr_frames,
                           num_classes=None,
                           is_training=True,
                           global_pool=True,
                           output_stride=None,
                           include_root_block=True,
                           spatial_squeeze=True,
                           reuse=None,
                           scope_depth=None, scope_rgb=None,
                           depth_training=True):
    # depth / hallucination injects signal into rgb
    # depth stream
    with tf.device('/gpu:0'):
        inputs = inputs_depth
        scope = scope_depth
        bottleneck = bottleneck_normal
        blocks = blocks_depth
        with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, bottleneck,
                                 resnet_utils.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=depth_training):
                    net = inputs
                    if include_root_block:
                        if output_stride is not None:
                            if output_stride % 4 != 0:
                                raise ValueError(
                                    'The output_stride needs to be a multiple of 4.')
                            output_stride /= 4
                        net = resnet_utils.conv2d_same(
                            net, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(
                            net, [3, 3], stride=2, scope='pool1')
                    net = resnet_utils.stack_blocks_dense(
                        net, blocks, nr_frames, output_stride)
                    if global_pool:
                        net = tf.reduce_mean(
                            net, [1, 2], name='pool5', keep_dims=True)
                    if num_classes is not None:
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                          normalizer_fn=None, scope='logits')
                        if spatial_squeeze:
                            net = tf.squeeze(
                                net, [1, 2], name='SpatialSqueeze')
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)
                    if num_classes is not None:
                        end_points['predictions'] = slim.softmax(
                            net, scope='predictions')
                    net_depth = net
                    end_points_depth = end_points

    end_points_to_pass = {}
    end_points_to_pass[scope_depth +
                       '/block1/unit_1/bottleneck_v1'] = end_points_depth[scope_depth + '/block1/unit_1/bottleneck_v1']
    end_points_to_pass[scope_depth +
                       '/block2/unit_1/bottleneck_v1'] = end_points_depth[scope_depth + '/block2/unit_1/bottleneck_v1']
    end_points_to_pass[scope_depth +
                       '/block3/unit_1/bottleneck_v1'] = end_points_depth[scope_depth + '/block3/unit_1/bottleneck_v1']
    end_points_to_pass[scope_depth +
                       '/block4/unit_1/bottleneck_v1'] = end_points_depth[scope_depth + '/block4/unit_1/bottleneck_v1']

    # rgb stream
    with tf.device('/gpu:1'):
        inputs = inputs_rgb
        scope = scope_rgb
        bottleneck = bottleneck_injected
        blocks = blocks_rgb
        with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, bottleneck,
                                 resnet_utils.stack_blocks_dense_injected],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    if include_root_block:
                        if output_stride is not None:
                            if output_stride % 4 != 0:
                                raise ValueError(
                                    'The output_stride needs to be a multiple of 4.')
                            output_stride /= 4
                        net = resnet_utils.conv2d_same(
                            net, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(
                            net, [3, 3], stride=2, scope='pool1')
                    net = resnet_utils.stack_blocks_dense_injected(
                        net, blocks, nr_frames, end_points_to_pass, 'resnet_v1_50_depth/', output_stride)
                    if global_pool:
                        net = tf.reduce_mean(
                            net, [1, 2], name='pool5', keep_dims=True)
                    if num_classes is not None:
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                          normalizer_fn=None, scope='logits')
                        if spatial_squeeze:
                            net = tf.squeeze(
                                net, [1, 2], name='SpatialSqueeze')
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)
                    if num_classes is not None:
                        end_points['predictions'] = slim.softmax(
                            net, scope='predictions')
                    net_rgb = net
                    end_points_rgb = end_points

    return net_depth, end_points_depth, net_rgb, end_points_rgb


############################# BLOCKS #################################
resnet_one_stream.default_image_size = 224
resnet_twostream_inter.default_image_size = 224


def resnet_v1_block(scope, base_depth, num_units, stride, unit_fun):
    return resnet_utils.Block(scope, unit_fun, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


def resnet_v1_50(unit_fun):
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3,
                        stride=2, unit_fun=unit_fun),
        resnet_v1_block('block2', base_depth=128, num_units=4,
                        stride=2, unit_fun=unit_fun),
        resnet_v1_block('block3', base_depth=256, num_units=6,
                        stride=2, unit_fun=unit_fun),
        resnet_v1_block('block4', base_depth=512, num_units=3,
                        stride=1, unit_fun=unit_fun),
    ]
    return blocks


resnet_v1_50.default_image_size = 224


########################################
def resnet_one_stream_main(inputs, nr_frames,
                           num_classes=None,
                           scope='resnet_v1_50', is_training=True, gpu_id='/gpu:0'):
                           # step 1
    blocks = resnet_v1_50(bottleneck_normal)
    net, endpoints = resnet_one_stream(inputs, blocks, nr_frames,
                                       num_classes=num_classes,
                                       scope=scope, is_training=is_training, gpu_id=gpu_id)
    return net, endpoints


# twostreams#################################
def resnet_twostream_main(inputs_depth, inputs_rgb, nr_frames,
                          num_classes=None, scope_rgb_stream='rgb', depth_training=True, is_training=True, interaction=False):
                          # step 2 and step 3
    scope_depth = 'resnet_v1_50_depth'
    # rgb stream may be true RGB or hallucination stream
    scope_rgb = 'resnet_v1_50_' + scope_rgb_stream

    blocks_depth = resnet_v1_50(bottleneck_normal)

    if interaction:
        blocks_rgb = resnet_v1_50(bottleneck_injected)
        net_depth, endpoints_depth, net_rgb, endpoints_rgb = resnet_twostream_inter(
            inputs_depth, blocks_depth,
            inputs_rgb, blocks_rgb, nr_frames,
            num_classes=num_classes,
            scope_depth=scope_depth, scope_rgb=scope_rgb, is_training=is_training, depth_training=depth_training)

    else:
        blocks_rgb = resnet_v1_50(bottleneck_normal)
        net_depth, endpoints_depth = resnet_one_stream(inputs_depth, blocks_depth, nr_frames,
                                                       num_classes=num_classes,
                                                       scope=scope_depth, is_training=depth_training, gpu_id='/gpu:0')
        net_rgb, endpoints_rgb = resnet_one_stream(inputs_rgb, blocks_rgb, nr_frames,
                                                   num_classes=num_classes,
                                                   scope=scope_rgb, is_training=is_training, gpu_id='/gpu:1')

    return net_depth, endpoints_depth, net_rgb, endpoints_rgb


def resnet_hall_rgb_main(inputs_rgb, nr_frames,
                         num_classes=None, is_training=True, interaction=False):
                         # step 4
    scope_depth = 'resnet_v1_50_hall'
    scope_rgb = 'resnet_v1_50_rgb'

    blocks_depth = resnet_v1_50(bottleneck_normal)

    if interaction:
        blocks_rgb = resnet_v1_50(bottleneck_injected)
        net_depth, endpoints_depth, net_rgb, endpoints_rgb = resnet_twostream_inter(
            inputs_rgb, blocks_depth,
            inputs_rgb, blocks_rgb, nr_frames,
            num_classes=num_classes,
            scope_depth=scope_depth, scope_rgb=scope_rgb, is_training=is_training)

    else:
        blocks_rgb = resnet_v1_50(bottleneck_normal)
        net_depth, endpoints_depth = resnet_one_stream(inputs_rgb, blocks_depth, nr_frames,
                                                       num_classes=num_classes,
                                                       scope=scope_depth, is_training=is_training, gpu_id='/gpu:0')
        net_rgb, endpoints_rgb = resnet_one_stream(inputs_rgb, blocks_rgb, nr_frames,
                                                   num_classes=num_classes,
                                                   scope=scope_rgb, is_training=is_training, gpu_id='/gpu:1')

    return net_depth, endpoints_depth, net_rgb, endpoints_rgb
