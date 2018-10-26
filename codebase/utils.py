import tensorflow as tf
import os
import time
import numpy as np
import argparse

##########################################################
tensorflow_models_dir = '../nets'
imagenet_ckpt = '/home/pavis/Documents/ng/tf-ckpt/resnet_v1_50.ckpt'
uwa3dii_dir = '/media/pavis/3TB-HD2/datasets/uwa3dii/tfrecords/'
ntu_dir = '/datasets/ntu/tfrecords_aligned_improved/'
nwucla_dir = '/datasets/nwucla/tfrecords/'
##########################################################


def get_arguments():
    parser = argparse.ArgumentParser(description='args...')
    # --dset=[nwucla/uwa3dii/ntu/ntu-mini]
    parser.add_argument('--dset', action='store',
                        dest='dset', default='ntu-mini')
    # --modality=[rgb/depth/depth_bott]
    parser.add_argument('--modality', action='store',
                        dest='modality', default='depth_bott')
    #  --eval=[cross_subj/cross_view]
    # only ntu and ntu-mini have cross_subj protocol
    parser.add_argument('--eval', action='store',
                        dest='eval_mode', default='cross_view')
    # doesn't train, just evaluates on validation and test set, given a ckpt
    # only ntu and ntu-mini have validation sets
    parser.add_argument('--just_eval', action='store_true',
                        dest='just_eval', default=False)
    # batch size = number of videos. We sample 5 frames from each video.
    parser.add_argument('--batch_sz', action='store',
                        dest='batch_sz', default='5', type=int)
    parser.add_argument('--n_epochs', action='store',
                        dest='n_epochs', default='40', type=int)
    parser.add_argument('--lr', action='store',
                        dest='learning_rate', default='0.001', type=float)
    parser.add_argument('--ckpt', action='store',
                        dest='ckpt', default='')
    parser.add_argument('--interaction', action='store_true',
                        dest='interaction', default=False)
    # standard deviation of noise, to test twostream_depth_rgb model
    parser.add_argument('--noise', action='store',
                        dest='noise', default='0', type=float)
    parser.add_argument('--bn_decay', action='store',
                        dest='bn_decay', default='0.997', type=float)
    parser.add_argument('--gpu0', action='store',
                        dest='gpu0', default='/gpu:0')
    args = parser.parse_args()
    return args


def create_folders(prefix, **kwargs):
    datetime = time.strftime("%d%m%Y_%H%M%S")
    hyperparameters = ''.join('_' + str(key) + '_' + str(value)
                              for key, value in sorted(kwargs.items()))
    experiment_id = prefix + '_' + datetime + '_' + hyperparameters

    ckpt_path = './checkpoint'
    log_path = './log'
    os.makedirs(os.path.join(ckpt_path, kwargs['dset'], experiment_id))
    os.makedirs(os.path.join(log_path, kwargs['dset'], experiment_id))
    return experiment_id


def load_files_paths(dataset):
    files = {}
    files['imagenet_checkpoint_file'] = imagenet_ckpt
    if dataset == 'uwa3dii':
        files['data'] = uwa3dii_dir
    elif dataset in ['ntu', 'ntu-mini']:
        files['data'] = ntu_dir
    elif dataset == 'nwucla':
        files['data'] = nwucla_dir
    return files


def get_tfrecords(eval_mode, data_dir, mini=False, dataset=''):
    if dataset == 'uwa3dii':
        return get_tfrecords_uwa3dii_noval(eval_mode, data_dir)
    if dataset == 'ntu':
        return get_tfrecords_ntu(eval_mode, data_dir, mini=False)
    if dataset == 'ntu-mini':
        return get_tfrecords_ntu(eval_mode, data_dir, mini=True)
    if dataset == 'nwucla':
        return get_tfrecords_nwucla_noval(eval_mode, data_dir)


def get_tfrecords_nwucla_noval(eval_mode, dset_dir):
    training_view_ids = ['1', '2']
    test_view_id = '3'

    train_list = []
    test_list = []

    for v in training_view_ids:
        view = 'view_' + v
        videos = os.listdir(os.path.join(dset_dir, view))
        videos = list(map(lambda x: view + '/' + x, videos))
        train_list.append(videos)

    train_list = [y for x in train_list for y in x]
    train_list = list(map(lambda x: os.path.join(dset_dir, x), train_list))

    test_list = os.listdir(os.path.join(dset_dir, 'view_' + test_view_id))
    test_list = list(map(lambda x: os.path.join(
        dset_dir, 'view_' + test_view_id, x), test_list))

    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)
    return train_list, [], test_list


def get_tfrecords_uwa3dii_noval(eval_mode, dset_dir):
    training_view_ids = ['01', '02', '03']
    test_view_ids = ['04']

    train_list = []
    test_list = []

    recs = os.listdir(dset_dir)  # 'a01_s01_v01_e00.tfrecords'
    recs = [x for x in recs if '.tfrecords' in x]

    if eval_mode == 'cross_view':
        train_list = [x for x in recs if x[9:11] in training_view_ids]
        test_list = [x for x in recs if x[9:11] in test_view_ids]
        train_list = list(map(lambda x: os.path.join(dset_dir, x), train_list))
        test_list = list(map(lambda x: os.path.join(dset_dir, x), test_list))

    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)
    return train_list, [], test_list


def get_tfrecords_ntu(eval_mode, dset_dir, mini=False):
    # eval_mode is cross_subject or cross_view
    training_subj_ids = [2, 4, 5, 8, 9, 13, 14, 15,
                         16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    training_view_ids = [2, 3]
    # subj=1 was in training_ids. it has 2016 videos, bit >5% of training
    # samples
    validation_subj_id = [1]
    validation_view_size = 1882  # 5% of training
    # SsssCcccPpppRrrrAaaa (e.g. S001C002P003R002A013), for which sss is the
    # setup number, ccc is the camera ID, ppp is the performer ID, rrr is the
    # replication number (1 or 2), and aaa is the action class label.

    train_list = []
    test_list = []
    val_list = []
    recs = os.listdir(dset_dir)

    if eval_mode == 'cross_subj':
        train_list = [x for x in recs if int(
            x[9:12]) in training_subj_ids]
        val_list = [x for x in recs if int(
            x[9:12]) in validation_subj_id]
        test_list = [x for x in recs if int(
            x[9:12]) not in training_subj_ids and int(x[9:12]) not in validation_subj_id]

    if eval_mode == 'cross_view':
        train_list = [x for x in recs if int(
            x[5:8]) in training_view_ids]
        test_list = [x for x in recs if int(
            x[5:8]) not in training_view_ids]

        # choosing samples from training list for the validation set
        val_samples_per_class = int(validation_view_size / 60)
        dict_class = {}  # {class, [name of file,...]}
        for i in range(60):
            dict_class[i + 1] = []
        for i, sample in enumerate(train_list):
            dict_class[int(sample[17:20])].append((i, sample))
        for i in range(60):
            files = dict_class[i + 1]
            files.sort(key=lambda tup: tup[1])
            np.random.seed(0)
            indexes = np.random.choice(
                len(files), val_samples_per_class, replace=False)
            val_list.extend([train_list[files[j][0]] for j in indexes])
        train_list = [x for x in train_list if x not in val_list]

    train_list.sort()
    np.random.seed(0)
    np.random.shuffle(train_list)

    train_list = list(map(lambda x: os.path.join(dset_dir, x), train_list))
    val_list = list(map(lambda x: os.path.join(dset_dir, x), val_list))
    test_list = list(map(lambda x: os.path.join(dset_dir, x), test_list))

    if mini:
        len_list = len(train_list)
        train_list = train_list[:int(len_list / 3)]
    return train_list, val_list, test_list


def get_n_classes(dataset):
    n_classes_dict = {'ntu': 60,
                      'ntu-mini': 60,
                      'uwa3dii': 30,
                      'nwucla': 10
                      }
    return n_classes_dict[dataset]


def loss_hall_rect(x1, x2): return tf.reduce_sum(
    tf.square(tf.subtract(tf.sigmoid(x1), tf.sigmoid(x2))))


def correct_pred(net, y): return tf.equal(
    tf.argmax(net, 1), tf.argmax(y, 1))


def accuracy(net, y): return tf.reduce_mean(
    tf.cast(correct_pred(net, y), tf.float32))


def double_log(f_log, string):
    print(string.rstrip())
    f_log.write(string)
    f_log.flush()


def update_details_file(f_log, **kwargs):
    f_log.write('\n'.join('_' + str(key) + '_' + str(value)
                          for key, value in sorted(kwargs.items())))
    f_log.write('\n')
    f_log.flush()


def softmax_distill(logits, T):
    # TODO check implementation
    logits = logits - tf.reduce_max(logits, 1, keep_dims=True)
    e = tf.exp(logits / T)
    return e / tf.reduce_sum(e, 1, keep_dims=True)


def cross_entropy(student, teacher):
    # https://github.com/tensorflow/tensorflow/issues/2462
    # http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
    # https://stackoverflow.com/questions/42521400/calculating-cross-entropy-in-tensorflow
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(teacher *
                                                  tf.log(student), 1))
    return cross_entropy
