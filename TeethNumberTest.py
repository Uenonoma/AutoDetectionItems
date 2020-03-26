# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:01:50 2020

@author: daniel
"""

import cv2
import numpy as np


def img_pretreatment(dst, blue_min, blue_max):
    dst_hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(dst_hsv, blue_min, blue_max) #Item is Black, backgraund is white
    mask_inverse = cv2.bitwise_not(mask) #Item is White, backgraund is Black
    return mask_inverse, dst_hsv


def Detec_CTSB(mask_inverse):
    contours, _  = cv2.findContours(mask_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints = True)
    
    hull_dim2 = hull.squeeze()
    ep =np.array([len(hull)])-1
    cal_base = np.insert(hull_dim2, 0, hull_dim2[0], axis=0)
    cal_ded = np.append(hull_dim2, hull_dim2[ep], axis=0)
    
    differ = cal_base - cal_ded
    power = np.sqrt(np.sum(differ**2, axis=1))
    differ_parameter = 3
    final_point = np.array([(np.where(power > differ_parameter))])
    return final_point, hull_dim2


def Pointing_edge(mask_inverse, final_point, hull_dim2):
    for i in final_point:
        for x in hull_dim2[i]:
            cv2.circle(mask_inverse, (x[0,0], x[1,0]), 10, (255, 255, 0), 10)
    return mask_inverse



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
        mask_inverse, _, = img_pretreatment(dst, blue_min, blue_max)
        mask_inverse = cv2.medianBlur(mask_inverse, 5)
        
        final_point, hull_dim2 = Detec_CTSB(mask_inverse)
        mask_inverse = Pointing_edge(mask_inverse, final_point, hull_dim2 )
        
        
        cv2.imshow('dst', mask_inverse)
              
        
        KeyCheck = cv2.waitKey(1)&0xFF
            
        if KeyCheck == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
    
except:
    import sys
    print("Error:", sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))