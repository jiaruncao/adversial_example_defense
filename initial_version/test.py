import keras
import numpy as np
import foolbox
#import matplotlib.pyplot as plt
from imagenet_classes import class_names
#import cv2
import  tensorflow as tf
keras.backend.set_learning_phase(0)
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('cifar10_train/')
    if ckpt and ckpt.model_checkpoint_path:
     # Restores from checkpoint
     kmodel =  saver.restore(sess, ckpt.model_checkpoint_path)

#kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
    print (kmodel)

preprocessing = (np.array([104, 116, 123]), 1)
model = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

image, _ = foolbox.utils.imagenet_example()
predict = model.predictions(image[:,:,::-1])
predict_label = np.argmax(predict)
predict_conf = predict[predict_label]

attack = foolbox.attacks.FGSM(model)
adversarial = attack(image[:,:,::-1], predict_label)
att = model.predictions(adversarial.astype(np.uint8))
att_label = np.argmax(att)
att_value = att[att_label]


print('Predicted:', class_names[att_label], ',', att_value)

#plt.show(adversarial)





