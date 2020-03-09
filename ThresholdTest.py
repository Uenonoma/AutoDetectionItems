# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:05:50 2019

@author: daniel
"""

import cv2
import numpy as np



img_stat = 0

def img_pretreatment(dst, blue_min, blue_max):
    dst_hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(dst_hsv, blue_min, blue_max) #Item is Black, backgraund is white
    mask_inverse = cv2.bitwise_not(mask) #Item is White, backgraund is Black
    return mask_inverse, dst_hsv



try:
    capture = cv2.VideoCapture(1)
    
    while(True):
        ret, frame = capture.read()
        if ret == False:
            print('Cannot get the frame')
            break

        
        dst = frame[100:420, 10:630]

        blue_min = np.array([0, 0, 100], np.uint8) 
        blue_max = np.array([180, 45, 240], np.uint8)        
        mask_inverse, check_hsv = img_pretreatment(dst, blue_min, blue_max)
        mask_inverse = cv2.medianBlur(mask_inverse, 5)
        contours, _ = cv2.findContours(mask_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(dst, contours, -1, (255, 0, 0), 2)
        
        
        KeyCheck = cv2.waitKey(1)&0xFF
        

        
        if KeyCheck == ord('d'):
            img_stat = 1
            
            
        elif KeyCheck == ord('q'):
            break
        
        
        if img_stat == 1:
            cv2.imshow('mask_inverse', mask_inverse)
            
        cv2.imshow('dst', img)
            
    
    capture.release()
    cv2.destroyAllWindows()
    
except:
    import sys
    print("Error:", sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))