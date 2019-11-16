from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
import ssim

scale = 4
img = Image.open('/home/laglangyue/AApython/Data/SetData/Test/Set5/baby_GT.bmp')
y, _, _ = img.split()  # type:Image.Image
w, h = y.size[0], y.size[1]
###########################

w_lr, h_lr = w // scale, h // scale  # 下采样的图像
w_hr, h_hr = w_lr * scale, h_lr * scale
y_hr = y.crop((0, 0, w_hr, h_hr))  # type:Image.Image # 保证整除,hr图像
y_lr = y_hr.resize(size=(w_lr, h_lr), resample=Image.BICUBIC)  # lr图像
y_hr_p = y_lr.resize(size=(w_hr, h_hr), resample=Image.BICUBIC)  # 重构hr图像
y_hr_np = np.array(y_hr)
y_hr_p_np = np.array(y_hr_p)
y_ssim = ssim.ssim_tf(y_hr_np, y_hr_p_np)
print(compare_ssim(y_hr_np, y_hr_p_np, data_range=255))
print(y_ssim)