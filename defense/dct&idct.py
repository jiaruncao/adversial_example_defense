from scipy import fftpack
import numpy as np
from PIL import Image
'''
# dct & idct convert
for i in range(10000):
    img0 = Image.open('ad_cifar10_fgmt_12/'+str(i)+'.png')
    
    img0 = np.array(img0)
    
    dct_img0 = fftpack.dct(img0)
    
    idct_img0 = fftpack.idct(dct_img0)
    idct_img0 = idct_img0
    idct_img0 = Image.fromarray(idct_img0.astype('uint8'))
    idct_img0.save('dct_cifar10_fgmt_12/'+str(i)+'.png')
'''
''''
# for i in range(10000):
img0 = Image.open('ad_img_deepfool/0.png')

img0 = np.array(img0)
img_show = Image.fromarray(img0)
#img_show.show()
dct_img0 = fftpack.dct(img0)

idct_img0 = fftpack.idct(dct_img0)
idct_img0 = idct_img0/255.0
idct_img0 = Image.fromarray(idct_img0.astype('uint8'), 'L')
#idct_img0.save('clean_dct_img/' + str(i) + '.png')

#dct_img0 = Image.fromarray(dct_img0.astype('uint8'),'L')
#dct_img0.show()

idct_img0.show()
'''

# RGB to grayscale
for i in range(10000):
    img0 = Image.open('ad_cifar10_fgmt_12/'+str(i)+'.png')
    img = np.array(img0.convert('L'))
    img = Image.fromarray(img.astype('uint8'),'L')
    img.save('gray_cifar10_fgmt_12/' + str(i) + '.png')

#img0 = np.array(img0)
#dct_img0 = fftpack.dct(img0)

#idct_img0 = fftpack.idct(dct_img0)
#idct_img0 = idct_img0
#idct_img0 = Image.fromarray(idct_img0.astype('uint8'))
#idct_img0.save('dct_cifar10_fgmt_12/' + str(i) + '.png')