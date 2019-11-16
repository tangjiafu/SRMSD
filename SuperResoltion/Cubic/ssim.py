import tensorflow as tf
import numpy as np
import os




def ssim_tf(img1, img2):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True  # allocate dynamically
    # tensorflow要求是H,W,C三维
    if img1.ndim == 2:
        img1 = img1[:, :, np.newaxis]
    if img2.ndim == 2:
        img2 = img2[:, :, np.newaxis]
    im1 = tf.convert_to_tensor(img1)  # np -->>tensor
    im2 = tf.convert_to_tensor(img2)
    # Compute SSIM over tf.uint8 Tensors.
    ssim1 = tf.image.ssim(im1, im2, max_val=255)
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(ssim1)
        return ssim1.eval()
