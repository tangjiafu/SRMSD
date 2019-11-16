import glob
import cv2
import numpy as np
import random
import h5py
import os


# 创建写入h5文件
def write_Hdf5(_input, _label, output_filename, _inputKey='input', _labelKey='label'):
    _input = _input.astype(np.float32)
    _label = _label.astype(np.float32)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset(_inputKey, data=_input, shape=_input.shape)
        h.create_dataset(_labelKey, data=_label, shape=_label.shape)
    h.close()


# 读取h5文件数据
def read_Hdf5(file, _inputKey='input', _labelKey='label'):
    with h5py.File(file, 'r') as hf:
        input = np.array(hf.get(_inputKey))
        label = np.array(hf.get(_labelKey))
        return input, label


# 计算两个文件对应图像的PSNR
def psnr_dir(file_1, file_2, _formate, result, scale):
    psnr_mean = []
    files = glob.glob(file_1 + "/" + "*" + _formate)
    print(files.__len__())
    print(file_1 + "/" + "*" + _formate)

    result_txt = open(result, "w+")
    result_txt.truncate()  # 清空文件
    for file in files:
        pic_name = file.split("/")[-1]
        src1 = cv2.imread(file_1 + "/" + pic_name)
        src2 = cv2.imread(file_2 + "/" + pic_name)
        h, w = src1.shape[0], src1.shape[1]
        src1 = src1[0:h - h % scale, 0:w - w % scale, :]
        print(src1.shape)
        print(src2.shape)
        if src1.shape == src2.shape:
            src1=cv2.cvtColor(src1,cv2.COLOR_BGR2YCrCb)[:,:,0]
            src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            psnr = cv2.PSNR(src1, src2)
            psnr_mean.append(psnr)
            result_txt.write("{}".format(psnr) + "  " + file_1 + "/" + pic_name + "  " + file_2 + "/" + pic_name + "\n")
    psnr_mean = np.mean(psnr_mean)
    result_txt.write("{}".format(psnr_mean) + "\n")
    result_txt.write("-----------------------------------------")
    result_txt.close()


def psnr_set():
    scale = 4
    psnr_dir(file_1="../data/SetData/Test/Set5", file_2="../Result/RGBX2", _formate=".bmp",
             result="../Result/result5_{}.txt".format(scale), scale=scale)
    psnr_dir(file_1="../data/SetData/Test/Set14", file_2="../Result/RGBX2", _formate=".bmp",
             result="../Result/result14_{}.txt".format(scale), scale=scale)


if __name__ == "__main__":
    psnr_set()
    # print(os.getcwd())
