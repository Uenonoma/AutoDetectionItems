# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:58:16 2020

@author: daniel
""" 



 
#--------------------------------------------------------------------------
import cv2
import numpy as np


CircleSize = 10

def img_pretreatment(dst):
    blue_min = np.array([0, 0, 100], np.uint8) 
    blue_max = np.array([180, 60, 240], np.uint8)   
    dst_hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(dst_hsv, blue_min, blue_max) #Item is Black, backgraund is white
    mask_inverse = cv2.bitwise_not(mask) #Item is White, backgraund is Black
    mask_inverse = cv2.medianBlur(mask_inverse, 5)
    return mask_inverse, dst_hsv

def CircleOnTeeth(mask_inverse, dst, CircleSize):
    contours, _  = cv2.findContours(mask_inverse, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=True)
    #img = cv2.drawContours(upstate, [hull], -1, (0,255,0), 10)
    hull_dim2 = hull.squeeze()
    for x,y in hull_dim2:
        img = cv2.circle(dst, (x,y), CircleSize, (0,255,0), 2)
        
    return img



try:
    capture = cv2.VideoCapture(1)
    
    while(True):
        ret, frame = capture.read()
        if ret == False:
            print('Cannot get the frame')
            break

        
        dst = frame[100:420, 10:630]

     
        mask_inverse, check_hsv = img_pretreatment(dst)
        img = CircleOnTeeth(mask_inverse,dst,CircleSize)
        
        
        
        
        KeyCheck = cv2.waitKey(1)&0xFF
            
        if KeyCheck == ord('d'):
            CircleSize = int(input())
            
            
            
        elif KeyCheck == ord('q'):
            break
        
            
        cv2.imshow('dst', img)
            
    
    capture.release()
    cv2.destroyAllWindows()
    
except:
    import sys
    print("Error:", sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))