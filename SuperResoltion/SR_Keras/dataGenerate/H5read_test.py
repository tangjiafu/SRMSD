from dataGenerate.utils import read_Hdf5
import numpy as np
import cv2

input, label = read_Hdf5("../data/H5/Yang91/RGB_91Test_x2.h5")
print(input.shape, label.shape)
print(np.dtype(input[4, 0, 0, 0]))
test_input = input[100, :, :, :] * 255
test_label = label[100, :, :, :] * 255

test_input = np.clip(test_input, 0, 255)
test_input = test_input.astype(np.uint8)

test_label = np.clip(test_label, 0, 255)
test_label = test_label.astype(np.uint8)
cv2.imshow("test1", test_input)
cv2.imshow("test2", test_label)
cv2.waitKey(0)
