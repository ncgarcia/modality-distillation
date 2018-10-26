import tensorflow as tf

time_bottleneck = 5  # number of frames


def sample_mask(num_frames, sample_size, is_training):
    # randomly choose between uniform or random sampling
    # with tf.device('/cpu:0'):
    if is_training:
        mask_prob = tf.random_uniform([])
        mask = tf.cond(tf.greater(mask_prob, 0.5),
                       lambda: sample_mask_uniform(num_frames, sample_size),
                       lambda: mask_random(num_frames, sample_size))
        # mask = sample_mask_uniform(num_frames, sample_size)
    else:
        mask = sample_mask_uniform(num_frames, sample_size)
    return mask


def sample_mask_uniform(num_frames, sample_size):
    end = tf.subtract(num_frames, 1)
    indexes = tf.to_int32(tf.linspace(
        0.0, tf.to_float(end), sample_size[0]))
    updates = tf.ones(sample_size, dtype=tf.int32)
    mask = tf.scatter_nd(tf.expand_dims(indexes, 1),
                         updates, tf.expand_dims(num_frames, 0))
    compare = tf.ones([num_frames], dtype=tf.int32)
    mask = tf.equal(mask, compare)
    return mask


def mask_random(num_frames, sample_size):
    ones = tf.ones(sample_size)
    zeros = tf.zeros(num_frames - sample_size)
    mask = tf.concat([ones, zeros], axis=0)
    mask = tf.random_shuffle(mask)
    compare = tf.ones([num_frames])
    mask = tf.equal(mask, compare)
    return mask


def _parse_fun_one_mod(example_proto, is_training=True, modality=None):
    with tf.device('/cpu:0'):
        context_features = {"label": tf.FixedLenFeature(
            [], dtype=tf.int64), "video_id": tf.FixedLenFeature([], dtype=tf.string)}
        sequence_features = {"rgb": tf.FixedLenSequenceFeature(
            [], dtype=tf.string), "depth": tf.FixedLenSequenceFeature([], dtype=tf.string)}
        context, features = tf.parse_single_sequence_example(
            serialized=example_proto, context_features=context_features, sequence_features=sequence_features)

        num_frames = tf.shape(features[modality])[0]
        n_to_sample = tf.constant([time_bottleneck])
        mask = sample_mask(num_frames, n_to_sample, is_training)
        frames = tf.boolean_mask(features[modality], mask)
        frames = tf.decode_raw(frames, tf.uint8)
        # images_depth.shape = [n_to_sample, h*w*c], so need to reshape
        frames = tf.reshape(frames, [-1, 224, 224, 3])

        def flip(x): return tf.image.flip_left_right(x)
        if is_training:
            flipping_prob = tf.random_uniform([])
            frames = tf.cond(tf.greater(flipping_prob, 0.5),
                             lambda: tf.map_fn(flip, frames),
                             lambda: frames)

        frames = tf.to_float(frames)
        label = [context['label']]
        return frames, label


def _parse_fun_2stream(example_proto, is_training=True):
    # parse each video
    # mask = sample n_to_sample=time_bottleneck frames from each video, for rgb and depth
    # after dataset.batch, this will return (batch, n_to_sample, h, w, 3)

    with tf.device('/cpu:0'):
        context_features = {"label": tf.FixedLenFeature(
            [], dtype=tf.int64), "video_id": tf.FixedLenFeature([], dtype=tf.string)}
        sequence_features = {"rgb": tf.FixedLenSequenceFeature(
            [], dtype=tf.string), "depth": tf.FixedLenSequenceFeature([], dtype=tf.string)}
        context, features = tf.parse_single_sequence_example(
            serialized=example_proto, context_features=context_features, sequence_features=sequence_features)

        num_frames_rgb = tf.shape(features['rgb'])[0]
        n_to_sample = tf.constant([time_bottleneck])
        mask = sample_mask(num_frames_rgb, n_to_sample, is_training)

        images_depth = tf.boolean_mask(features['depth'], mask)
        images_rgb = tf.boolean_mask(features['rgb'], mask)
        images_depth = tf.decode_raw(images_depth, tf.uint8)
        # images_depth.shape = [n_to_sample, h*w*c], so need to reshape
        images_depth = tf.reshape(images_depth, [-1, 224, 224, 3])
        images_rgb = tf.decode_raw(images_rgb, tf.uint8)
        images_rgb = tf.reshape(images_rgb, [-1, 224, 224, 3])

        def flip(x): return tf.image.flip_left_right(x)
        if is_training:
            flipping_prob = tf.random_uniform([])
            images_depth = tf.cond(tf.greater(flipping_prob, 0.5),
                                   lambda: tf.map_fn(flip, images_depth),
                                   lambda: images_depth)
            images_rgb = tf.cond(tf.greater(flipping_prob, 0.5),
                                 lambda: tf.map_fn(flip, images_rgb),
                                 lambda: images_rgb)

        images_depth = tf.to_float(images_depth)
        images_rgb = tf.to_float(images_rgb)
        label = [context['label']]
        # label_per_frame = tf.tile(label, n_to_sample)
        # video_id = tf.convert_to_tensor(context['video_id'])
        return images_depth, images_rgb, label
