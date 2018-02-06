from scipy import fftpack
import numpy as np
from PIL import Image

for i in range(10000):

    img0 = Image.open('clean_img/'+str(i)+'.png')
    
    img0 = np.array(img0)
    img0 = img0/255.0
    
    
    dct_img0 = fftpack.dct(img0)
    
    idct_img0 = fftpack.idct(dct_img0)
    
    idct_img0 = Image.fromarray(idct_img0.astype('uint8'), 'L')
    idct_img0.save('clean_dct_img/'+str(i)+'.png')

