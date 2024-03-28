import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('C:\\Users\\huthi\\Desktop\\face shape detector'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

train = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('C:\\Users\\huthi\\Desktop\\face shape detector',target_size=(200,200),batch_size=3, class_mode = 'binary')

train_dataset.class_indices

classess = ['diamond','heart','oblong','oval','round','square','triangle']

'''

ann = models.Sequential([
        layers.Flatten(input_shape=(200,200,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(7, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(train_dataset ,epochs=5)

'''

'''

import tensorflow as tf

# Convert SavedModel to TensorFlow Lite
saved_model_dir = 'C:\\Users\\huthi\\Desktop\\model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
model_lite = converter.convert()
'''

'''
img = cv2.imread("C:\\Users\\huthi\\Desktop\\face shape detector\\diamond\\download (10).jpg")
plt.imshow(img)
plt.show()
img = cv2.resize(img,(200,200))
img = np.reshape(img,[1,200,200,3])
classes = ann.predict(img)
print(np.argmax(classes))
z = (np.argmax(classes))
print(classess[z])
'''

 
cnn=tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64,padding='same',strides=2,kernel_size=3,activation='relu',input_shape=(200,200,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',strides=2,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',strides=2,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(7,activation='softmax'))

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(train_dataset,epochs=10)

 
img = cv2.imread("C:\\Users\\huthi\\Desktop\\face shape detector\\diamond\\download (10).jpg")
plt.imshow(img)
plt.show()
img = cv2.resize(img,(200,200))
img = np.reshape(img,[1,200,200,3])
classes = cnn.predict(img)
print(np.argmax(classes))

img = cv2.imread("C:\\Users\\huthi\\Desktop\\face shape detector\\oval\\download (1).jpg")
plt.imshow(img)
plt.show()
img = cv2.resize(img,(200,200))
img = np.reshape(img,[1,200,200,3]) 
classes = cnn.predict(img)
z = (np.argmax(classes))
print(z)


cnn.save('saved_model')

