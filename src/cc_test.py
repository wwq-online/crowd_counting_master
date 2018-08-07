# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import os
import random
import time
from cc_utils import *
from cc_nets import crowd_counting_cnn
from cc_configs import *

np.set_printoptions(threshold=np.inf)


def set_GPU(gpu='0'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


set_GPU()
img_root_dir = '/home/wwq/Data/ShangHaiTech_Datasets/part_A_final/test_data/images/'
gt_root_dir = '/home/wwq/Data/ShangHaiTech_Datasets/part_A_final/test_data/ground_truth/'

img_file_list = os.listdir(img_root_dir)
gt_file_list = os.listdir(gt_root_dir)

d_weight = 1.0
h_weight = 1.0

configs = ConfigFactory()


def train():
    # place holder
    input_img_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    head_count_placeholder = tf.placeholder(tf.float32, shape=(None, 1))

    # 网络输出
    [inference_density_map, inference_head_countd] = crowd_counting_cnn(input_img_placeholder)

    # 密度图损失
    density_map_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(density_map_placeholder, inference_density_map))))
    # 人群计数损失
    head_count_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(head_count_placeholder, inference_head_countd))))
    # 联合训练
    joint_loss = d_weight * density_map_loss + h_weight * head_count_loss

    init = tf.global_variables_initializer()
    # 启动会话
    sess = tf.InteractiveSession()
    sess.run(init)

    file_path = configs.log_router
    if (not os.path.exists(file_path)):
        os.makedirs(file_path)
    log = open(configs.log_router + configs.model_name + '_test_' + r'.logs', mode='a+', encoding='utf-8')

    saver = tf.train.Saver(max_to_keep=configs.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(configs.ckpt_router)

    if ckpt and ckpt.model_checkpoint_path:
        print('load model')
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        mae = 0.0
        mse = 0.0
        # test
        for file_index in range(len(img_file_list)):
            img_path = img_root_dir + img_file_list[file_index]
            gt_path = gt_root_dir + 'GT_' + img_file_list[file_index].split(r'.')[0]
            img, density_map, head_number = read_test_data(img_path, gt_path)

            feed_dict = {input_img_placeholder: img, density_map_placeholder: density_map,
                         head_count_placeholder: head_number}

            h_count, j_loss, d_loss, h_loss = sess.run(
                [inference_head_countd, joint_loss, density_map_loss, head_count_loss], feed_dict=feed_dict)
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            format_str = 'step %d, joint loss = %.5f, density map loss = %.5f, head count loss = %5f ' \
                         'inference =  %5f, gt = %5f'
            log_line = format_time, img_file_list[file_index], format_str % (file_index, j_loss, d_loss, h_loss, h_count, head_number)
            print(log_line)
            log.writelines(str(log_line) + '\n')
            mae += abs(abs(h_count) - abs(head_number))
            mse += pow(abs(h_count) - abs(head_number), 2)
        print('mae', mae / len(img_file_list))
        print('mse', pow(mse / len(img_file_list), 0.5))

if __name__ == '__main__':
    train()
