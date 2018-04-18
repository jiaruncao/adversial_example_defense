import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
y = cv2.imread('/Users/app/documents/dl/bigdata_article/article1/cat_downloads.png',0)

print (y.shape)


'''
def dctTrans(img):
    y = cv2.imread(img,0)
    rows,cols = y.shape[:2]
    imf = np.float32(y)/255.0
    #trans = cv2.dct(imf)
   # init = cv2.dct(trans,flags = 1)

    rgb = imf
    numrgb = countnum(rgb)
    rgb1 = rgb.reshape(numrgb,1)
    trans = cv2.dct(rgb1)
    init = cv2.dct(trans,flags = 1)
    rgb1 = init
    d1,d2,d3 = rgb.shape
    rgb = rgb1.reshape(d1,d2,d3)
    arrayRGB2Image = Image.fromarray(rgb)
    arrayRGB2Image.save('dct_trans.png')
'''