import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense 
from keras.optimizers import Adam 
from keras.metrics import catagorical_crossentropy 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model 
from keras.applications import imagenet_utils
from sklearn.meterics import confusion_matrix 
import itertools 
from IPython.display import Image
import matplotlib.pyplot as plt


mobile = keras.applications.mobilenet.MobileNet()

def prepare_image(file):
	img_path ='deep_learning_face_detection/'
	img= image.load_img(img_path +file, target_size+(244,244))
	img_array = image.img_to_array(img)
	img_array_expanded_dims = np.expanded_dims(img_array, axis=0)
	return keras.applicationsmobilenet.preprocess_input(img_array_expanded_dims)

image(filename= 'deep_learning_face_detection/rooster.jpg', width=200, height= 200)
preprocessed_image = prepare_image('rooster.jpg')
predictions= mobile.predict(preprocessed_image)
results= imagenet_utils.decode_predictions(predictions)

bestResult= result[0]
name= bestResult[1]
nameLetters= name.split()

firstResponse= input("Would you like to know what is in the image?")



if firstResponse == "yes" or "Yes":
	if nameLetters[0]== 'a','e','i', 'o', 'u':
		print("In the image, there is an " + name)
	else:
		print("In the image, there is a "+ name)
else:
	print("OK. Bye")
