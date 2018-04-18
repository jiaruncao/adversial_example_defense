import os

import numpy as np

import matplotlib

matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from attacks import fgmt
# import fgsm_mnist
from attacks import deepfool
from attacks import fgm
from attacks import jsma
from PIL import Image
from attacks import fgmt
from scipy import misc

img_size = 32
img_chan = 3
n_classes = 10
batch_size = 128

print('ok!!')
print('\nLoading CIFAR10')

cifar10 = tf.keras.datasets.cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255
print(X_test.shape)
to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1 - VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]


def model(x, logits=False, training=False):
    y = x
    with tf.name_scope('conv0'):
        y = tf.layers.conv2d(y, filters=64, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    with tf.name_scope('conv2'):
        y = tf.layers.conv2d(y, filters=64, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='valid')
    with tf.name_scope('conv3'):
        y = tf.layers.conv2d(y, filters=128, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    with tf.name_scope('conv4'):
        y = tf.layers.conv2d(y, filters=128, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='valid')
    with tf.name_scope('conv5'):
        y = tf.layers.conv2d(y, filters=256, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    with tf.name_scope('con6'):
        y = tf.layers.conv2d(y, filters=256, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='valid')
    with tf.variable_scope('flatten'):
        shape = y.get_shape().as_list()
        y = tf.reshape(y, [-1, np.prod(shape[1:])])
    with tf.variable_scope('mlp'):
        y = tf.layers.dense(y, units=128, activation=tf.nn.relu)
        y = tf.layers.dropout(y, rate=0.5, training=training)
    logits_ = tf.layers.dense(y, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()

with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()
with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgmt = fgmt(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def make_fgmt(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgmt, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


#
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.import_meta_graph('my_model/cifar10-85.meta')
    saver.restore(sess, tf.train.latest_checkpoint('my_model/'))
    # evaluate(sess,env,X_test,y_test)
    X_adv = make_fgmt(sess, env, X_test, eps=0.05, epochs=12)
    # evaluate(sess,env,X_adv,y_test)

    print('-----convert start-----')

    for i in range(10000):
        # print(X_adv[i].shape)
        # temp_img = np.transpose(X_adv[i], [1, 2, 0])
        temp_img = X_adv[i] * 255
        #temp_img = temp_img.reshape(32, 32, 3)
        # misc.imsave("ad_img/"+str(i)+".png",temp_img)
        im = Image.fromarray(temp_img.astype('uint8'))
        im.save("ad_cifar10_fgmt_eps_0.05/" + str(i) + ".png")

# if __name__ == '__main__':
#    main()
