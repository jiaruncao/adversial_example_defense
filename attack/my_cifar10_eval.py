import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402


import tensorflow as tf
from attacks import fgmt
#import fgsm_mnist

from attacks import fgm
from attacks import cw
from PIL import Image
from scipy import misc

img_size = 32
img_chan = 3
n_classes = 10
batch_size = 128
print ('cifar10_eval')
print('\nLoading CIFAR10')

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

to_categorical = tf.keras.utils.to_categorical
y_test = to_categorical(y_test)

X_adv = np.empty([10000,32,32,3])
for i in range(10000):
    load_img = Image.open('new_image/'+str(i)+'.png')
    load_img = np.array(load_img)
    #print (type(load_img))
    load_img = load_img.astype(np.float32)
    load_img = load_img.reshape(32,32,1)
    X_adv[i] = load_img





def model(x, logits=False, training=False):
    y = x
    with tf.name_scope('conv0'):
        y = tf.layers.conv2d(y, filters=64, kernel_size=3,strides=1,
                             padding='same',activation=tf.nn.relu)
        #y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
        #                    kernel_initializer='he_normal')(y)
    with tf.name_scope('conv2'):
        y = tf.layers.conv2d(y, filters=64, kernel_size=3,strides=1,
                             padding='same', activation=tf.nn.relu)
    #y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
     #                 kernel_initializer='he_normal')(y)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2,padding = 'valid')
        #y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)
    with tf.name_scope('conv3'):
        y = tf.layers.conv2d(y, filters=128, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
        #y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
        #                kernel_initializer='he_normal')(y)
    with tf.name_scope('conv4'):
        y = tf.layers.conv2d(y, filters=128, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
        #y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
        #                 kernel_initializer='he_normal')(y)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='valid')
        #y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)
    with tf.name_scope('conv5'):
        y = tf.layers.conv2d(y, filters=256, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    #y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
     #                 kernel_initializer='he_normal')(y)
    with tf.name_scope('con6'):
        y = tf.layers.conv2d(y, filters=256, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    #y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
     #                 kernel_initializer='he_normal')(y)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='valid')
    #y = MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)
    with tf.variable_scope('flatten'):
        shape = y.get_shape().as_list()
        y = tf.reshape(y, [-1, np.prod(shape[1:])])
    #y = Flatten()(y)
    with tf.variable_scope('mlp'):
        y = tf.layers.dense(y, units=128, activation=tf.nn.relu)
        y = tf.layers.dropout(y, rate=0.5, training=training)
    #y = Dense(units=128, activation='relu', kernel_initializer='he_normal')(y)
    #y = Dropout(0.5)(y)
    #logits_ = y
    #y = Dense(units=n_classes, activation='softmax', kernel_initializer='he_normal')(y)
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
#    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
                #                               y=env.adv_y, eps=env.adv_eps,
             #                                  optimizer=optimizer)


def evaluate(sess, env, X_data, y_data, batch_size=128):
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

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_model/cifar10-85.meta')
    saver.restore(sess, tf.train.latest_checkpoint('my_model/'))
    evaluate(sess,env,X_adv,y_test)

    #evaluate(sess,env,X_adv,y_test)
