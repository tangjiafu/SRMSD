import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data = sio.loadmat('./srcnn_10000.mat')
test = data["srcnn_loss"]
plt.plot(test[0, :])
plt.show()
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
