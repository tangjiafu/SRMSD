from PIL import Image
import glob
import numpy as np
from skimage.measure import compare_ssim
import ssim
import psnr
import os


def psnr_ssim(_data_path, _result):
    file_list = glob.glob(_data_path)
    scale_list = [2, 3, 4]
    A_result = {"2": [], "3": [], "4": []}  # 结果字典
    A_psnr_sum = {"2": 0, "3": 0, "4": 0}  # psnr总和
    A_ssim_sum = {"2": 0, "3": 0, "4": 0}  # ssim总和
    for file in file_list:
        img = Image.open(file).convert("YCbCr")  # type:Image.Image
        y, _, _ = img.split()  # type:Image.Image
        w, h = y.size[0], y.size[1]
        # y.resize()
        for scale in scale_list:
            ###################################
            result_path = os.path.join(_result, "X{}".format(scale),
                                       file.split("/")[-1].split(".")[0] + ".bmp")  # 存储图像目录
            w_lr, h_lr = w // scale, h // scale  # 下采样的图像
            w_hr, h_hr = w_lr * scale, h_lr * scale
            y_hr = y.crop((0, 0, w_hr, h_hr))  # type:Image.Image # 保证整除,hr图像
            y_lr = y_hr.resize(size=(w_lr, h_lr), resample=Image.BICUBIC)  # lr图像
            y_hr_p = y_lr.resize(size=(w_hr, h_hr), resample=Image.BICUBIC)  # 重构hr图像
            # y_hr_p.save(result_path)  ###保存重构图像
            y_hr_np = np.array(y_hr)
            y_hr_p_np = np.array(y_hr_p)
            y_ssim = ssim.ssim_tf(y_hr_np, y_hr_p_np)
            y_psnr = psnr.psnr_py(y_hr_np, y_hr_p_np)
            y_result = "{}__{}__{}".format(file.split("/")[-1], y_psnr, y_ssim)
            A_result["{}".format(scale)].append(y_result)
            ###psnr,ssim的总和
            A_psnr_sum["{}".format(scale)] = A_psnr_sum["{}".format(scale)] + y_psnr
            A_ssim_sum["{}".format(scale)] = A_ssim_sum["{}".format(scale)] + y_ssim
    # 计算均值
    for scale in scale_list:
        mean_psnr = A_psnr_sum["{}".format(scale)] / len(file_list)
        mean_ssim = A_ssim_sum["{}".format(scale)] / len(file_list)
        mean_result = "scale{}_{}_{}".format(scale, mean_psnr, mean_ssim)
        A_result["{}".format(scale)].append(mean_result)
    ################################################################
    f = open(os.path.join(_result, "result.txt"), "w+")
    f.truncate()
    for scale in scale_list:
        for str in A_result["{}".format(scale)]:
            f.writelines(str + "\n")
        f.writelines("---------------------------------\n")
    f.close()


Set5_path = "/home/laglangyue/AApython/Data/SetData/Test/Set5/*.bmp"
Set14_path = "/home/laglangyue/AApython/Data/SetData/Test/Set14/*.bmp"
B100_path = "/home/laglangyue/AApython/Data/BSDS300/valid/*.jpg"
U100_path = "/home/laglangyue/AApython/Data/Urban100/*.jpg"

Set5 = os.path.join("./", "result", "Set5")
Set14 = os.path.join("./", "result", "Set14")
B100 = os.path.join("./", "result", "B100")
U100 = os.path.join("./", "result", "U100")

psnr_ssim(Set5_path, Set5)
psnr_ssim(Set14_path, Set14)
psnr_ssim(B100_path, B100)
psnr_ssim(U100_path, U100)
