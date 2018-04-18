"""
Use fast gradient sign method to craft adversarial on MNIST.

Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
import time
from attacks import fgmt
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization

img_size = 32
img_chan = 3
n_classes = 10


print('\nLoading CIFAR10')

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')



def model(x, logits=False, training=False):
    y = x
    with tf.name_scope('conv0'):
        y = tf.layers.conv2d(y, filters=64, kernel_size=3,strides=1,
                             padding='same',activation=tf.nn.relu)
    with tf.name_scope('conv2'):
        y = tf.layers.conv2d(y, filters=64, kernel_size=3,strides=1,
                             padding='same', activation=tf.nn.relu)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2,padding = 'valid')
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

with tf.variable_scope('model', reuse=True):
    env.cifar10_eps = tf.placeholder(tf.float32, (), name='cifar10_eps')
    env.cifar10_epochs = tf.placeholder(tf.int32, (), name='cifar10_epochs')
    #env.x_fgmt = fgmt(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

print('\nInitializing graph')



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=256):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=256, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'my_model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    step = 0
    os.makedirs('my_model', exist_ok=True)
    for epoch in range(epochs):
        step += 1
        start = time.time()
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)
        print('Time Spent: %.2f seconds' % (time.time() - start))
        if step % 5 == 0:
            print('\n Saving model')
            #saver.save(sess, checkpoint_path, global_step=step)
            env.saver.save(sess,'./my_model/cifar10', global_step=step)
    '''        
    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('my_model', exist_ok=True)
        env.saver.save(sess, 'my_model/{}'.format(name))
    '''

print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=100,
      name='cifar10')
