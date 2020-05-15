import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers


folder= 'skin-cancer-maligniant-vs-benign'

train_imgs= os.listdir(f"{folder}/data/train")
test_imgs= os.listdir(f"{folder}/data/train")

train_labels=['benign', 'cancerous']

test_labels= ['benign', 'cancerous']


train_imgs, test_imgs= train_imgs/255.0, test_imgs/255.0 

model= Sequential()

model.add(layers.Conv2D( 32, (3,3), activation='relu'), (input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D( 64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D( 64, (3,3), activation='relu')))
model.add(layers.MaxPooling2D(2,2))

model.compile(optimizer='adam', 
	metrics=['accuracy'], 
	loss = sparse_catagorical_crossentropy)

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
