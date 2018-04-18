import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
def my_detection(filename):
    # Load pre-trained image recognition model
    model = inception_v3.InceptionV3()

    # Load the image file and convert it to a numpy array
    img = image.load_img(filename, target_size=(299, 299))
    input_image = image.img_to_array(img)
    #print (input_image.shape)

    # Scale the image so all pixel intensities are between [-1, 1] as the model expects
    input_image /= 255.
    input_image -= 0.5
    input_image *= 2.

    # Add a 4th dimension for batch size (as Keras expects)
    input_image = np.expand_dims(input_image, axis=0)

    # Run the image through the neural network
    #print (input_image.shape)
    predictions = model.predict(input_image)

    # Convert the predictions into text and print them
    predicted_classes = inception_v3.decode_predictions(predictions, top=1)
    imagenet_id, name, confidence = predicted_classes[0][0]
    print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))


#my_detection('/Users/app/documents/dl/bigdata_article/article1/hacked_cat.png')
for i in range(3,4):
    print(str(i)+':')
    my_detection('generate_img2/'+str(i)+'.png')
