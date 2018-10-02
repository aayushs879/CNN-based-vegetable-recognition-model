# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 03:50:10 2018

@author: aayush
"""

def getSizedFrame(width, height):
  
    s, img = self.cam.read()

    # Only process valid image frames
    if s:
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return s, img