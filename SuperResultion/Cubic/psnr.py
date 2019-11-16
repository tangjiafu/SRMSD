import numpy as np
import math


def psnr_py(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:  # 防止分母为0
        return 1000
    return 10 * math.log10(255.0 ** 2 / mse)
