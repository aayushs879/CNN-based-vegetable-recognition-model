# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:27:51 2018

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
classifier1 = Sequential()
classifier1.add(Conv2D(64,3,3, input_shape = (128,128,3), activation = 'relu'))
classifier1.add(Conv2D(64,3,3, activation = 'relu'))
classifier1.add(Conv2D(32,3,3, activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2,2)))
classifier1.add(Conv2D(32,3,3, activation = 'relu'))
classifier1.add(Conv2D(32,3,3, activation = 'relu'))
classifier1.add(Conv2D(16,3,3, activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2,2)))
classifier1.add(Conv2D(16,3,3, activation = 'relu'))
classifier1.add(Conv2D(16,3,3, activation = 'relu'))
classifier1.add(Conv2D(16,3,3, activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2,2)))
classifier1.add(Flatten())
classifier1.add(Dense(output_dim = 100, activation = 'relu'))
classifier1.add(Dropout(0.5))
classifier1.add(Dense(output_dim = 100, activation = 'relu'))
classifier1.add(Dropout(0.5))
classifier1.add(Dense(output_dim = 3, activation = 'softmax'))
classifier1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =  ['accuracy'])


#Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('New folder/train', target_size = (128,128), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('New folder/test', target_size = (128,128), batch_size = 32, class_mode = 'categorical')

#fitting the model onto images
classifier1.fit_generator(training_set, samples_per_epoch = 166, nb_epoch = 25, validation_data = test_set, nb_val_samples = 56)

classifier1.save('fruit_classifier.h5')

label2index = tests.class_indices

classifier1.predict_generator()

tests = ImageDataGenerator(rescale = 1./255)
test = tests.flow_from_directory('test', target_size = (64,64), batch_size = 1, class_mode = 'categorical')

groun_truth = test.classes

predictions = classifier1.predict_generator(test, steps = 42)

import numpy as np
predicted_classes = np.argmax(predictions,axis=1)



