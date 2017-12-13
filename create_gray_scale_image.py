import numpy as np
#from keras.preprocessing import image
#from keras.applications import inception_v3
import numpy as np
import argparse
import cv2
#from matplotlib import pyplot as plt
#from matplotlib.colors import Normalize
from PIL import Image

mypath = '/Users/app/documents/dl/bigdata_article/article1/'
y = cv2.imread(mypath+'cat_downloads.png',0)

cv2.imwrite(mypath+'gray_cat.png',y)





