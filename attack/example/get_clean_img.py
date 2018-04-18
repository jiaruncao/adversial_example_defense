

import numpy as np
import matplotlib
matplotlib.use('Agg')           # noqa: E402
import tensorflow as tf
from PIL import Image

img_size = 32
img_chan = 3
n_classes = 10


print('\nLoading CIFAR10')

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

for i in range(10000):

    temp_img = X_test[i] * 255
    #temp_img = temp_img.reshape(28, 28)
    im = Image.fromarray(temp_img.astype('uint8'))
    im.save("cifar10_img/" + str(i) + ".png")


