from keras.models import Model
from keras.layers import Conv2D, Input, Activation, add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from dataGenerate.utils import read_Hdf5, write_Hdf5
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.backend.tensorflow_backend import set_session
import glob
import scipy.io as sio

################################################   keras使用GPU
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU的第二种方法

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 定量
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))


################################################
def model():
    input = Input([None, None, 1])
    ####  1
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(input)
    model = Activation("relu")(model)
    ####  2
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  3
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  4
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  5
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  6
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  7
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  8
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  9
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  10
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  11
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  12
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  13
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  14
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  15
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  16
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  17
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  18
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  19
    model = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####  20
    model = Conv2D(nb_filter=1, nb_row=3, nb_col=3, init='glorot_uniform',
                   padding='same', bias=True, input_shape=(None, None, 1))(model)
    model = Activation("relu")(model)
    ####
    model = add(inputs=[input, model])
    # optimizers
    _VDSR = Model(inputs=[input], outputs=[model])
    adam = Adam(lr=0.0003)
    _VDSR.compile(optimizer=adam, loss='mean_squared_error', metrics=[psnr])
    return _VDSR


def psnr(y_true, y_pred):
    psnr_cal = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr_cal


def train():
    vdsr = model()
    print(vdsr.summary())
    input, label = read_Hdf5("/home/laglangyue/AApython/Data/H5/BSD_Y91_train.h5")
    val_input, val_label = read_Hdf5("/home/laglangyue/AApython/Data/H5/BSD_Y91_test.h5")

    checkpoint = ModelCheckpoint("./data/H5/VDSR.h5", monitor='val_psnr', verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False, mode='max')
    callbacks_list = [checkpoint]

    history = vdsr.fit(input, label, batch_size=128, validation_data=(val_input, val_label),
                       callbacks=callbacks_list, shuffle=True, epochs=1000, verbose=1)
    # 绘制训练 & 验证的损失值
    #################################################################
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    sio.savemat("./vdsr.mat",
                {'vdsr_loss': history.history['loss'], 'vdsr_val_loss': history.history['val_loss']})
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("./vdsr.png")
    #################################################################


def predict(path, result_path, scale):
    '''
    :param path:输入图像的文件夹，进行测试
    :param result_path: 经过超分模型输出图像的文件夹
    :param scale: 超分倍率
    :return: 0
    '''
    color = 'Y'
    # color = 'RGB'  # 基于RGB重建
    srcnn_model = model()
    srcnn_model.load_weights("./data/H5/VDSR_{}_{}.h5".format(color, scale))
    files = glob.glob(path)
    for file in files:
        imgrec = cv2.imread(file, cv2.IMREAD_COLOR)
        ################################################# 保证scale整除
        h, w = imgrec.shape[0], imgrec.shape[1]
        imgrec = imgrec[0:h - h % scale, 0:w - w % scale, :]
        #################################################
        img_name = result_path + file.split("/")[-1]
        if color == 'Y':
            imgrec = cv2.cvtColor(imgrec, cv2.COLOR_BGR2YCrCb)
            imgrec = cv2.resize(imgrec, dsize=None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
            img = cv2.resize(imgrec, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            Y_img = img[:, :, 0]  # 提取亮度分量
            Y_img = Y_img / 255.
            Y_img = Y_img[np.newaxis, :, :, np.newaxis]
            pre = srcnn_model.predict(Y_img, batch_size=1) * 255.
            pre = np.clip(pre, 0, 255)
            pre = pre.astype(np.uint8)
            pre = np.squeeze(pre, 0)  # 去除第0维
            img[:, :, 0] = pre[:, :, 0]
            img_result = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(img_name, img_result)
        elif color == 'RGB':
            img = cv2.resize(imgrec, dsize=None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = img / 255.  # 归一化到0-1
            img = img[np.newaxis, :, :, :]
            pre = srcnn_model.predict(img, batch_size=1) * 255.
            pre = np.clip(pre, 0, 255)
            pre = pre.astype(np.uint8)
            img_result = np.squeeze(pre, 0)  # 去除第0维
            img_result = cv2.cvtColor(img_result, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(img_name, img_result)


if __name__ == "__main__":
    scale = 4
    train()
    # path = "./data/SetData/Test/Set*/*.bmp"

    # result_path = "./Result/{}/".format("RGBX{}".format(scale))
    # predict(path, result_path, scale)
