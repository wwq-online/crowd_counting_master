# -*-coding:utf-8-*-

import numpy as np
import cv2 as cv
import math
import csv
import os
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

from numpy.core.multiarray import ndarray

np.set_printoptions(threshold=np.inf)


def gaussian_2d(krow, kcol, sigma=0.3):
    '''
    生成二维高斯分布矩阵
    :param krow: 矩阵行数
    :param kcol: 矩阵列数
    :param sigma: 方差
    :return:
    '''
    x, y = np.mgrid[-krow / 2 + 0.5:krow // 2 + 0.5, -kcol / 2 + 0.5:kcol // 2 + 0.5]
    gaussian_distri = np.exp(-(np.square(x) + np.square(y)) / (2 * np.power(sigma, 2))) / (2 * np.power(sigma, 2))
    result = gaussian_distri / np.sum(np.sum(gaussian_distri))
    return result


def get_knn_diatance(pointx_x, point_y, points, k=2):
    '''
    计算人头标签KNN
    :param pointx_x: 行坐标
    :param point_y: 列坐标
    :param points: 人头标签集合
    :param k: k值
    :return:
    '''
    num_points = len(points)
    if k >= num_points:
        return 1.0
    else:
        distance = np.zeros((num_points, 1))
        for i in range(num_points):
            x1 = points[i, 1]
            y1 = points[i, 0]
            # 欧式距离
            distance[i, 0] = math.sqrt(math.pow(pointx_x - x1, 2) + math.pow(point_y - y1, 2))
        distance = np.sort(distance)
        sum = 0.0
        for j in range(k + 1):
            sum = sum + distance[j, 0]
        return sum / k


def get_density_map(img_gray, points, knn_phase=False):
    '''
    生成密度图
    :param img_gray: 灰度图
    :param points: 人头标签集合
    :param knn_phase: 是否使用几何自适应
    :return:
    '''
    h, w = img_gray.shape[0], img_gray.shape[1]
    img_density = np.zeros((h, w))
    num = len(points)
    if num == 0:
        return img_density
    for i in range(num):
        krow = 15
        kcol = 15
        sigma = 4.0
        # 标签坐标 先y后x
        x = min(h, max(0, abs(int(math.floor(points[i, 1])))))
        y = min(w, max(0, abs(int(math.floor(points[i, 0])))))

        x1 = x - int(math.floor(kcol / 2))
        y1 = y - int(math.floor(kcol / 2))
        x2 = x + int(math.ceil(krow / 2))
        y2 = y + int(math.ceil(krow / 2))

        # 边界处理
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(h, x2)
        y2 = min(w, y2)

        # 重设高斯核大小
        krow = x2 - x1
        kcol = y2 - y1
        if knn_phase:
            m_distance = get_knn_diatance(x, y, points)
            sigma = sigma * m_distance
        H = gaussian_2d(krow, kcol, sigma)
        img_density[x1:x2, y1:y2] = img_density[x1:x2, y1:y2] + H
        img_density = np.asarray(img_density)
    return img_density

def read_train_data(img_path, gt_path, size=128, scale=4, knn_phase=True):
    '''
    训练数据
    :param img_path: 输入图像路径
    :param gt_path: 标签路径
    :return:
    '''
    img_color = cv.imread(img_path)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    data = loadmat(gt_path)  # 读取mat文件
    points = data['image_info'][0][0]['location'][0][0]
    # number = data['image_info'][0][0]['number'][0][0]
    img_color_crop, img_gray_crop, points_crop, number = random_crop(img_color, img_gray, points, size=size, scale=scale)
    img_density = get_density_map(img_gray_crop, points_crop, knn_phase=knn_phase)
    img_color_crop = img_color_crop.reshape((1, img_color_crop.shape[0], img_color_crop.shape[1], img_color_crop.shape[2]))
    img_density = img_density.reshape((1, img_density.shape[0], img_density.shape[1], 1))
    number = np.asarray(number)
    number = number.reshape((1, 1))
    return img_color_crop / 255., img_density, number


def read_test_data(img_path, gt_path, scale=4, knn_phase=True):
    img_color = cv.imread(img_path)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    data = loadmat(gt_path)  # 读取mat文件
    points = data['image_info'][0][0]['location'][0][0]
    number = data['image_info'][0][0]['number'][0][0]

    img_gray = cv.resize(img_gray, (img_gray.shape[1] // (scale), img_gray.shape[0] // (scale)))

    for i in range(len(points)):
        points[i] = points[i] / (scale)

    img_density = get_density_map(img_gray, points, knn_phase=knn_phase)
    img_color = img_color.reshape((1, img_color.shape[0], img_color.shape[1], img_color.shape[2]))
    img_density = img_density.reshape((1, img_density.shape[0], img_density.shape[1], 1))
    number = np.asarray(number)
    number = number.reshape((1, 1))
    return img_color, img_density, number



def random_crop(img_color, img_gray, points, size=128, scale=4):
    '''
    crop
    :param img_color:
    :param img_gray:
    :param points:
    :param size:
    :param scale:
    :return:
    '''
    h, w = img_color.shape[0], img_color.shape[1]
    x1 = random.randint(0, h - size)
    y1 = random.randint(0, w - size)

    x2 = x1 + size
    y2 = y1 + size
    img_color_crop = img_color[x1:x2, y1:y2, ...]
    img_gray_crop = img_gray[x1:x2, y1:y2]
    img_gray_crop = cv.resize(img_gray_crop, (img_gray_crop.shape[1] // (scale), img_gray_crop.shape[0] // (scale)))
    points_crop = []
    for i in range(len(points)):
        if points[i, 1] >= x1 and points[i, 1] <= x2 and points[i, 0] >= y1 and points[i, 0] <= y2:
            points[i, 0] = points[i, 0] - y1
            points[i, 1] = points[i, 1] - x1
            points_crop.append(points[i] / (scale))
    points_crop = np.asarray(points_crop)
    number = len(points_crop)
    return img_color_crop, img_gray_crop, points_crop, number


def show_density_map_as_heatmap(density_map):
    '''
    密度热力图展示
    :param density_map:
    :return:
    '''
    plt.imshow(density_map, cmap='jet')
    plt.show()
    return


def read_datasets(img_path, density_map_path, scale=4):
    head_count = img_path.split(r'/')[-1].split(r'.')[0].split(r'_')[-1]
    head_count = np.asarray(int(head_count), dtype=np.float32)
    im = cv.imread(img_path, 0)

    im = np.asarray(im, dtype=np.float32)
    density_map = np.loadtxt(open(density_map_path, 'rb'), dtype=np.float32, delimiter=",", skiprows=0)

    # ht = im.shape[0]
    # wd = im.shape[1]
    # ht_1 = (ht / scale) * scale
    # wd_1 = (wd / scale) * scale

    # wd_1 = int(wd_1 / scale)
    # ht_1 = int(ht_1 / scale)
    # density_map = cv.resize(density_map, (wd_1, ht_1))
    # density_map = density_map * ((wd * ht) / (wd_1 * ht_1))

    # density_map = cv.resize(density_map, (0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv.INTER_CUBIC)
    im = im.reshape((1, im.shape[0], im.shape[1], 1))
    density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))
    head_count = head_count.reshape((1, 1))
    return [im, density_map, head_count]


if __name__ == '__main__':
    cc = 0
    # print(gaussian_2d(8, 8, 4.0))
    # img_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/train/'
    # den_root_dir = './datasets/formatted_trainval/shanghaitech_part_A_patches_9/train_den/'
    #
    # # 列出文件夹下所有的目录与文件
    # list = os.listdir(img_root_dir)
    # img_path = img_root_dir + list[0]
    # density_map_path = den_root_dir + str(list[0]).split(r'.')[0] + r'.csv'
    #
    # [img, density_map, head_count] = read_datasets(img_path, density_map_path)
    # print(img.shape)
    # print(density_map.shape)
    # data = loadmat('dp.mat')  # 读取mat文件
    # data = np.mat(data['im_density'])
    #
    # show_density_map_as_heatmap(data)
    # img, density_map, number = read_train_data('./IMG_2.jpg', './GT_IMG_2.mat')
    # show_density_map_as_heatmap(res)
    # sum = np.sum(np.sum(res))
    # print(sum)
    #
