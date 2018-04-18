from  PIL import Image
import  numpy as np
X_test = np.empty([10000,28,28,1])
for i  in range(10000):
    load_img = Image.open('ad_img_res/'+str(i)+'.png')
    load_img = np.array(load_img)
    load_img =  load_img.astype(np.float32) / 255
    load_img = load_img.reshape(28,28,1)
    X_test[i] = load_img


print (X_test)