# -*- coding: UTF-8 -*-
import cv2
import glob
import numpy as np
from memory_profiler import profile
from dataGenerate.utils import write_Hdf5, read_Hdf5


# 程序功能： 生成RGB的数据集
# 根据网格点计算label和input

#########################################
# 输入x和y起始和终点坐标,划分间隔,# 输出网格坐标
def meshpic(_startx, _endx, _intervalx, _starty, _endy, _intervaly):
    x = np.arange(_startx, _endx, _intervalx)  # 水平方向上的划分，对应图像是列数
    y = np.arange(_starty, _endy, _intervaly)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    _coor = np.vstack((X, Y))
    return _coor


@profile  # 内存检测
#########################################
# 输入路径，patch尺寸，采样间隔，返回一批patch
def image_patch(_datapath, _patchsize, _interval):
    files = glob.glob(_datapath)
    print(files.__len__())
    # files = files[0:200]  # 只用一半的数据集
    LabelImgs = []  # original
    InputImgs = []  # 2倍降采样
    for file in files:
        image = cv2.imread(file)  # originale image
        input = cv2.resize(image, dsize=None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)  # 降采样
        input = cv2.resize(input, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # 升采样
        label = image.astype(np.float64) / 255.
        input = input.astype(np.float64) / 255.
        # 生成网格坐标,这个坐标是patch的中心坐标 ###############
        coor = meshpic(_startx=_patchsize // 2, _endx=label.shape[1] - _patchsize // 2, _intervalx=_interval,
                       _starty=_patchsize // 2, _endy=label.shape[0] - _patchsize // 2, _intervaly=_interval)
        # 根据网格坐标取patch,patch中心是网格坐标########
        # label
        for i in range(coor.shape[1]):
            x_patch, y_patch = coor[0][i], coor[1][i]
            labelPatch = label[y_patch - _patchsize // 2:y_patch + _patchsize // 2,
                         x_patch - _patchsize // 2:x_patch + _patchsize // 2]  # 取patch
            # print(labelPatch.shape)
            LabelImgs.append(labelPatch)
        ########
        # input
        for i in range(coor.shape[1]):
            x_patch, y_patch = coor[0][i], coor[1][i]
            input_patch = input[y_patch - _patchsize // 2:y_patch + _patchsize // 2,
                          x_patch - _patchsize // 2:x_patch + _patchsize // 2]  # 取patch
            # print(image_x2_patch.shape)
            InputImgs.append(input_patch)

    print(InputImgs.__len__())  # 下采样图像序列的大小
    InputImgs = np.array(InputImgs, dtype=np.float32)  # list转化为numpy,需要list维度一致
    print(LabelImgs.__len__())
    LabelImgs = np.array(LabelImgs, dtype=np.float32)
    return InputImgs, LabelImgs


if __name__ == "__main__":
    train_interval = 12  # 取图像间隔
    test_interval = 18
    scale = 4
    label_patchsize = 72
    train_H5 = "../data/H5/Yang91/{}_91Train_x{}.h5".format('RGB', scale)
    test_H5 = "../data/H5/Yang91/{}_91Test_x{}.h5".format('RGB', scale)
    train_Path = '../data/SetData/Train/*.bmp'
    test_Path = '../data/SetData/Test/Set*/*.bmp'
    input, label = image_patch(_datapath=train_Path, _patchsize=label_patchsize,
                               _interval=train_interval)
    write_Hdf5(input, label, train_H5)
    input, label = read_Hdf5(train_H5)
    print("train", input.shape, label.shape)
    # 测试集##################################
    input, label = image_patch(_datapath=test_Path, _patchsize=label_patchsize, _interval=test_interval)
    write_Hdf5(input, label, test_H5)
    input, label = read_Hdf5(test_H5)
    print(input.shape, label.shape, label.shape)
