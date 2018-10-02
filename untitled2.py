# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 02:26:21 2018

@author: aayush
"""

#Impoting Libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import MaxPooling2D

#Creating A CNN
classifier2 = Sequential()
classifier2.add(Conv2D(64,3,3, input_shape = (128,128,3), activation = 'relu'))
classifier2.add(Conv2D(64,3,3, activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2,2)))
classifier2.add(Conv2D(64,3,3, activation = 'relu'))
classifier2.add(Conv2D(64,3,3, activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2,2)))
classifier2.add(Conv2D(64,3,3, activation = 'relu'))
classifier2.add(Conv2D(64,3,3, activation = 'relu'))
classifier2.add(MaxPooling2D(pool_size = (2,2)))
classifier2.add(Flatten())
classifier2.add(Dense(output_dim = 100, activation = 'relu'))
classifier2.add(Dropout(0.5))
classifier2.add(Dense(output_dim = 100, activation = 'relu'))
classifier2.add(Dropout(0.5))
classifier2.add(Dense(output_dim = 1, activation = 'softmax'))
classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =  ['accuracy'])

    
#Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('rotten/train', target_size = (128,128), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('rotten/test', target_size = (128,128), batch_size = 32, class_mode = 'categorical')

#fitting the model onto images
classifier2.fit_generator(training_set, samples_per_epoch = 136, nb_epoch = 11, validation_data = test_set, nb_val_samples = 64)



