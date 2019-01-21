import os
import tensorflow as tf
# import ipdb
import concurrent.futures
import time
import gc
import psutil
# import json
# import math
# import csv
# import numpy as np


# UWA3D
rgb_dset_dir = '/media/pavis/3TB-HD2/datasets/uwa3dii/UWA3D-RGB/'
# rgb_dset_dir = '/home/ng/Datasets/UWA3D/UWA3D-RGB/'
depth_dset_dir = '/media/pavis/3TB-HD2/datasets/uwa3dii/UWA3D-Depth/WholeFrames/'
# depth_dset_dir = '/home/ng/Datasets/UWA3D/UWA3D-Depth/WholeFrames/'
output_dir = '/media/pavis/3TB-HD2/datasets/uwa3dii/tfrecords/'
# output_dir = '/home/ng/Datasets/UWA3D/tfrecords/'

videos_id = os.listdir(depth_dset_dir)
videos_id.sort()


def build_absolute_paths(video_id):
    frames_rgb = os.listdir(os.path.join(rgb_dset_dir, video_id[:-1] + '1'))
    frames_rgb.sort()
    frames_depth = os.listdir(os.path.join(depth_dset_dir, video_id))
    frames_depth.sort()

    # i = 001,.jpg     - get the intersection of frames_rgb and frames_depth
    frames = [i[:-5] for i in frames_rgb if i[:-5]+'.png' in frames_depth]

    samples_rgb = list(map(lambda x: os.path.join(
        rgb_dset_dir, video_id[:-1] + '1', x+',.jpg'), frames))

    samples_depth = list(map(lambda x: os.path.join(
        depth_dset_dir, video_id, x + '.png'), frames))

    samples_rgb.sort()
    samples_depth.sort()
    return samples_rgb, samples_depth


def main_loop(video_id):
    with tf.device('/cpu:0'):
        # no sampling, just get all frames' names
        frames_rgb, frames_depth = build_absolute_paths(video_id)
        interleaved_frames = [x for t in zip(
            frames_rgb, frames_depth) for x in t]
        filename_queue = tf.train.string_input_producer(
            interleaved_frames, shuffle=False, num_epochs=1)

        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)

        my_img_rgb = tf.image.decode_jpeg(value)

        # resize to final size
        my_img_rgb = tf.image.resize_images(my_img_rgb, [224, 224])
        my_img_rgb = tf.squeeze(my_img_rgb)

        # subtract mean
        VGG_MEAN = [123.68, 116.78, 103.94]
        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])

        my_img_rgb = tf.subtract(my_img_rgb, means)
        my_img_rgb = tf.saturate_cast(my_img_rgb, tf.uint8)

    ###########################################################

        my_img_depth = tf.image.decode_png(value)

        # resize to final size
        my_img_depth_e = tf.image.resize_images(my_img_depth, [224, 224])
        my_img_depth_rgb = tf.image.grayscale_to_rgb(my_img_depth_e)
        my_img_depth_f = tf.squeeze(my_img_depth_rgb)

        # my_img_depth_g = tf.subtract(my_img_depth_f, means)
        my_img_depth_h = tf.saturate_cast(my_img_depth_f, tf.uint8)

        # https://github.com/tensorflow/tensorflow/issues/3168
        init_op = tf.group(tf.initialize_all_variables(
        ), tf.initialize_local_variables(), tf.global_variables_initializer())
        # init_op = tf.global_variables_initializer()
        sess = tf.InteractiveSession(
            config=tf.ConfigProto(log_device_placement=True))
        with sess.as_default():
            sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_list_rgb = []
        image_list_depth = []
        # After everything is built, start the loop.
        try:
            while not coord.should_stop():
                image_list_rgb.append(my_img_rgb.eval(session=sess))
                image_list_depth.append(my_img_depth_h.eval(session=sess))
        except tf.errors.OutOfRangeError:
            try:
                # means the loop has finished, write your tfrecord
                tfrecord_name = os.path.join(
                    output_dir, video_id + '.tfrecords')
                writer = tf.python_io.TFRecordWriter(tfrecord_name, options=tf.python_io.TFRecordOptions(
                    compression_type=tf.python_io.TFRecordCompressionType.GZIP
                ))
                # video_id = 'a01_s01_v01_e00'
                label = int(video_id[1:3])
                ex = make_example(video_id, image_list_rgb,
                                  image_list_depth, label)
                writer.write(ex.SerializeToString())
                writer.close()
            except:
                pass
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            sess.run(filename_queue.close(cancel_pending_enqueues=True))
            coord.join(threads)
            sess.close()
            time.sleep(1)
            image_list_rgb = []
            image_list_depth = []
            # time.sleep(1)
            # gc.collect()
            time.sleep(1)


def make_example(video_id, rgb_frames_list, depth_frames_list, label):
    ex = tf.train.SequenceExample()
    ex.context.feature["label"].int64_list.value.append(label)
    ex.context.feature["video_id"].bytes_list.value.append(
        video_id.encode())

    rgb_frames = ex.feature_lists.feature_list["rgb"]
    depth_frames = ex.feature_lists.feature_list["depth"]

    for rgb_frame, depth_frame in zip(rgb_frames_list, depth_frames_list):
        rgb_frames.feature.add().bytes_list.value.append(rgb_frame.tostring())
        depth_frames.feature.add().bytes_list.value.append(depth_frame.tostring())

    return ex


# ipdb.set_trace()
# videos_id[0] = 'a01_s01_v01_e00'
# main_loop(videos_id[0])
# ipdb.set_trace()


with concurrent.futures.ThreadPoolExecutor() as executor:  # max_workers=45
    # future_to_url = {executor.submit(
    #     main_loop, video): video for video in videos_id}

    # https://stackoverflow.com/questions/34770169/using-concurrent-futures-without-running-out-of-ram
    jobs = {}
    MAX_JOBS_IN_QUEUE = 500
    files_left = len(videos_id)
    files_iter = iter(videos_id)

    while files_left:
        for this_file in files_iter:
            job = executor.submit(main_loop, this_file)
            jobs[job] = this_file
            if len(jobs) > MAX_JOBS_IN_QUEUE:
                break  # limit the job submission for now job

        # Get the completed jobs whenever they are done
        print("----------------DELETING JOBS-------------------------")
        for job in concurrent.futures.as_completed(jobs):
            files_left -= 1  # one down - many to go...   <---
            # delete the result from the dict as we don't need to store it.
            del jobs[job]
            # break  # give a chance to add more jobs <-----
        # https://stackoverflow.com/questions/31720674/clear-memory-in-python-loop

        mem_available = psutil.virtual_memory().available
        mem_available = mem_available >> 30
        if mem_available < 3:
            print("---------------------COLLECTING MEMORY--------------------")
            time.sleep(3)
            gc.collect()
            time.sleep(3)
