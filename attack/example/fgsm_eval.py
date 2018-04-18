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

img_size = 28
img_chan = 1
n_classes = 10
batch_size = 32
print ('fgsm_eval')
print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

to_categorical = tf.keras.utils.to_categorical
y_test = to_categorical(y_test)

X_adv = np.empty([10000,28,28,1])
for i in range(10000):
    load_img = Image.open('my_img/dct_jsma/'+str(i)+'.png')
    load_img = np.array(load_img)
    load_img =  load_img.astype(np.float32)
    load_img = load_img.reshape(28,28,1)
    X_adv[i] = load_img





def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    return y, logits_


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
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
        vs = tf.global_variables()
        env.train_op = optimizer.minimize(env.loss, var_list=vs)

    env.saver = tf.train.Saver()

    # Note here that the shape has to be fixed during the graph construction
    # since the internal variable depends upon the shape.
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
                                               y=env.adv_y, eps=env.adv_eps,
                                               optimizer=optimizer)



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
    saver = tf.train.import_meta_graph('model/mnist.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))
    evaluate(sess,env,X_adv,y_test)

    #evaluate(sess,env,X_adv,y_test)
