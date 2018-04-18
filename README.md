# 对抗性样本攻击与防御（中文版）

该代码探究了一种对于图像的攻击与防御方法。使用多种对抗性样本生成方法，在标准数据集上批量生成对抗性样本，从而达到欺骗神经网络的效果。

同时，对于已经生成的对抗性样本做某些图像变换，去除对抗性样本噪声(noise)，从而消除对抗性样本的影响，达到防御的效果。

**【注】：实验数据采用了Mnist/Cifar-10**
## 攻击

###攻击方法
* [fgsm](https://arxiv.org/abs/1412.6572)(fast gradient sign method)
* [fgmt](https://arxiv.org/pdf/1607.02533.pdf)(fast gradient method with target)
* [deeo fool](https://arxiv.org/abs/1511.04599)
* [jsma](https://arxiv.org/abs/1511.07528)(Jacobian-based saliency map approach)

**点击以查看相关论文**

###代码
####`attack/`文件夹包含了攻击方法的代码，具体如下：
* `fgsm_mnist.py`：基于Mnist数据集训练模型
* `fgsm_eval.py`：模型预测
* `fgsm_make_ad.py`：对Mnist数据集生成对抗性样本（FGSM）
* `my_cifar10_train.py`：基于Cifar-10数据集训练模型
* `my_cifar10_eval.py`：模型预测
* `my_cifar10_maka_ad.py`：对Mnist数据集生成对抗性样本（FGMT）
* `get_clean_img.py`：提取Cifar-10数据集，以.png格式保存到本地

####`attack/attacks`文件夹是对攻击方法的代码封装
####`attack/example`文件夹是上述所有攻击方法在Mnist数据集的实现（尝试了不同的迭代次数、扰动值等）

## 防御

###防御方法
对图像做DCT变换，然后逆变换回图像，只保留其基本的直流分量，以实现对抗性样本的去噪处理
**[DCT原理](https://en.wikipedia.org/wiki/Discrete_cosine_transform)**

###代码
####`defense/`文件夹包含了防御方法的代码，具体如下：
* `create_gray_scale_image.py`：生成灰度图
* `dct&idct.py`：对图像做DCT&IDCT变换，然后将新图像保存到本地

## 原始版本
####`initial_version/`文件夹是最原始的版本，实现了对单张图像的对抗性攻击和DCT去噪处理，
数据集采用ImageNet中的图像，模型采用了Inception V3（Keras中的预训练模型），攻击方法采用FGMT，具体如下：
* `dection.py`：模型预测
* `dct.py`：将输入的图片做dct变换，再做逆变换，并保存新图  
* `create_ad.py`：生成对抗性图片 
* `create_gray_scale_image`：生成灰度图
* `imagenet_classes.py`：对抗性攻击目标
**[代码来源](https://mp.weixin.qq.com/s/jgeCqz1VwY92BfTLFahu1Q)**

## 目前的问题
* 对于cifar-10的模型识别准确率低于baseline（模型采用TensorFlow官方网站的tutorial）
* 基于cifar-10生成的对抗性样本效果很差，完全看不出图像原始的样子，基本无法使用

## 作者相关
  Author Name：Jiarun Cao
  Address：Beijing Insitute of Technology
  E-mail:2211241432@qq.com && jiaruncao.china@Gmail.com

