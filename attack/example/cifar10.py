# -*- coding:utf-8 -*-
# @time: 2017.7.14
# @athor: xzy
#
# --------------------------------------------

"""构建CIFAR-10网络.

 # 计算训练输入图片和标签. 如果想要运行验证使用 inputs()函数代替.
 inputs, labels = distorted_inputs()
 
 # inference
 predictions = inference(inputs)
 
 # loss
 loss = loss(predictions, labels)

 # training
 train_op = train(loss, global_step)
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# 参数设置
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")

# 描述CIFAR-10数据集的全局常量,来自cifar_input.py文件
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

""" 对于一个数据集来讲，运行一个epoch就是将这个数据集中的图片全部计算一遍。
如一个数据集中有三张图片A.jpg、B.jpg、C.jpg，那么跑一个epoch就是指对A、B、C
三张图片都计算了一遍。两个epoch就是指先对A、B、C各计算一遍，然后再全部计算一遍
"""

# 描述训练过程的常数
MOVING_AVERAGE_DECAY = 0.9999  # 衰变率
NUM_EPOCHS_PER_DECAY = 350.0  # 每批次轮训350，进行衰变
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习率衰减因子.
INITIAL_LEARNING_RATE = 0.1  # 学习率初始值

# 如果要使用多GPUs训练, 所有的操作都带上tower_name前缀
TOWER_NAME = 'tower'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """创建summaries收集信息,便于可视化.

  收集 histogram.
  收集scalar

  Args:
    x: 输入张量
  Returns:
    无
  """
    # 如果是多GPU训练, 将x.op.name里的'tower_[0-9]/'替换成空串
    # re.sub是个正则表达式方面的函数，用来实现比普通字符串的replace更加强大的替换功能
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))  # 记录0在x中的小数比例


def _variable_on_cpu(name, shape, initializer):
    """在CPU内存进行持久化Variable操作.

  Args:
    name: variable名字
    shape: 形状
    initializer: initializer for Variable

  Returns:
    持久化tensor
  """
    with tf.device('/cpu:0'):  # 指定操作在0号CPU上进行
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """使用带有weight衰变来初始化Variable.

  使用截断正态分布(限制x的取值范围的一种分布)初始化Variable,特别指定时才使用权重衰变

  Args:
    name: variable名字
    shape: 张量的形状
    stddev: 标准差
    wd: 添加 L2Loss weight衰变,再乘wd. wd值为空,不添加衰变

  Returns:
    Variable Tensor
  """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)  # 将value以name的名称存储在收集器(collection)中
    return var


def distorted_inputs():
    """调用cifar10_input.py中的distorted_input()对CIFAR数据集进行变形处理

  Returns:
    images: Images. 4D 张量 [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] .
    labels: Labels. 1D 张量 [batch_size].

  Raises:
    ValueError: 没有data_dir将报错
  """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """调用cifar10_input中的input()函数处理输入

  Args:
    eval_data: 表明是验证数据还是训练数据

  Returns:
    images: Images. 4D  [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] .
    labels: Labels. 1D  [batch_size].

  Raises:
    ValueError: 没有data_dir报错
  """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    print("=======cifar10====image", images)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images, batch_size=FLAGS.batch_size):
    """前向预测

  Args:
    images: 由 distorted_inputs() 或者 inputs()提供的输入图片
    batch_size: 批次大小， 默认值是128
  Returns:
    Logits.
  """
    # 使用多GPU训练的话,将tf.Variable()用tf.get_variable()取代
    #
    # 卷积层1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # 池化层1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # 标准化，LRN详解见http://blog.csdn.net/banana1006034246/article/details/75204013
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # 卷积层2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # 此处标准化为什么放在池化前面？？
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # 池化2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # batch_size = 128
        reshape = tf.reshape(pool2, [batch_size, -1])  # -1代表这项为缺省项
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # 线性层完成的也是(WX + b),
    # 在这我们不使用softmax 因为tf.nn.sparse_softmax_cross_entropy_with_logits接收一个未缩放logits
    # 并且为了高效在内部实现softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """为可训练的参数添加loss损失

  收集"Loss" 和 "Loss/avg"信息
  Args:
    logits: 来自inference()的 Logits
    labels: 来自distorted_inputs 或者 inputs()的Labels. 1-D [batch_size]张量

  Returns:
    类型为float的Loss tensor
  """
    # 计算批次上的平均交叉熵损失.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)  # 以key='losses'存储均值交叉熵

    # 权重衰变产生的交叉熵损失累加作为总的loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """对 CIFAR-10 模型loss添加汇总.

  针对总的loss产生滑动均值和网络性能可视化相关信息汇总.

  Args:
    total_loss: 从 loss()得到总的loss.
  Returns:
    loss_averages_op: 产生losses移动均值op操作.
  """
    # ExponentialMovingAverage在采用随机梯度下降算法训练神经网络时，使用 tf.train.ExponentialMovingAverage
    # 滑动平均操作的意义在于提高模型在测试数据上的健壮性（robustness），需要提供一个衰
    # 减率（decay）。该衰减率用于控制模型更新的速度， 它对每一个（待更新训练学习的）变
    # 量（variable）都会维护一个影子变量（shadow variable）
    # 计算所有单个 losses 和 总 loss的滑动均值.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # 从字典集合中返回关键字'losses'对应的所有变量，包括交叉熵损失和正则项损失
    losses = tf.get_collection('losses')
    # apply() 方法会添加 trained variables 的 shadow copies，并添加操作来维护变量的滑动均值到 shadow copies
    # shadow variables 的更新公式 shadow_variable = decay * shadow_variable + (1 - decay) * variable
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # 对单个loss和总loss添加一个acalar汇总; 对平均loss也做同样的操作.
    for loss_iterator in losses + [total_loss]:
        # 将每个损失命名为'(raw)',并将损失的移动平均版本命名为原始损失名称。
        tf.summary.scalar(loss_iterator.op.name + ' (raw)', loss_iterator)
        # average() 方法可以访问 shadow variables
        tf.summary.scalar(loss_iterator.op.name, loss_averages.average(loss_iterator))

    return loss_averages_op


def train(total_loss, global_step):
    """ CIFAR-10 训练模型

  构建一个 optimizer 并且运用到可训练 variables

  Args:
    total_loss: 从 loss()得到总loss.
    global_step: 训练步长
  Returns:
    train_op: training op操作
  """
    # 批次数 = 训练数/批次大小
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)  # 衰变步长

    # 指数下降调整步长
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # 产生所有loss滑动均值并做汇总
    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度gradients.
    with tf.control_dependencies([loss_averages_op]):  # 定义控制依赖，后面的操作必须在loss_averages_op执行之后
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)  # 计算梯度，它是minimize()函数的第一个部分

    # 应用梯度，它是minimize()函数的第二个部分.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 为每个变量做 histograms 汇总
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 每个梯度做 histograms 汇总.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # 追踪可训练参数的滑动均值. MOVING_AVERAGE_DECAY = 0.99----衰变率
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())  # 更新滑动均值

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')  # 定义训练op

    return train_op


def maybe_download_and_extract():
    """从给定的网址上下载CIFAR_10数据集"""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):  # 下载的数据块数目，每个数据块的大小，服务器端的总的数据大小
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # 下载数据,_progress在终端显示下载文件和大小就被销毁,打印进度显示下载进度
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()  # 输出空行
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)  # 将所有数据抽取放在cifar-10-batches-bin目录下
