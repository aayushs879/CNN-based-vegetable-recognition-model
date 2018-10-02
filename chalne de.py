# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:29:50 2018

@author: aayush
"""

from keras.models import load_model
classifier = load_model('vegetable_classifier2.h5')

import cv2


test = cv2.imread('test1.jpg') 


import matplotlib.pyplot as plt


plt.imshow(test)



import numpy as np
test_potato = np.reshape(test,[1, 128,128,3])

classifier.predict_classes(test)

def tell_me_the_veggy(img):
    
    img = cv2.resize(img, (128,128))
    flag = classifier.predict_classes(np.reshape(img,[1,128,128,3]))
    if(flag == 0):
        print('The given Vegetable was an onion')
    elif(flag == 1):
        print('The given Vegetable was a potato')
    elif(flag == 2):
        print('The given Vegetable was a tomato')
        

