# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:40:28 2018

@author: aayush
"""

import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(96, 96, 3))

classifier = Sequential()

classifier.add(conv_base)
classifier.add(Flatten())
classifier.add(Dense(output_dim = 150, activation = 'relu', input_dim = 7*7*512))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 150, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =  ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('New folder/train', target_size = (128,128), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('New folder/test', target_size = (128,128), batch_size = 32, class_mode = 'categorical')


classifier.fit_generator(training_set, samples_per_epoch = 1555, nb_epoch = 30, validation_data = test_set, nb_val_samples = 490)



