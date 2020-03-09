# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:05:50 2019

@author: daniel
"""


#φ85 -- 619


import cv2
import  numpy as np



def img_pretreatment(dst):
    dst_hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    blue_min = np.array([100, 0, 0], np.uint8)
    blue_max = np.array([140, 255, 255], np.uint8)
    mask = cv2.inRange(dst_hsv, blue_min, blue_max)
    mask_inverse = cv2.bitwise_not(mask)
    return mask_inverse
    

def SizeCheck(mask_inverse): 
    contours, _  = cv2.findContours(mask_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    
    leftmost = cnt[cnt[:,:,0].argmin()][0][0]
    rightmost = cnt[cnt[:,:,0].argmax()][0][0]
    
    a = rightmost - leftmost
    
    print(str(a))
    
    
def Draw_contours(mask_inverse, dst):
    contours, _ = cv2.findContours(mask_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    dst = cv2.drawContours(dst, cnt, -1, (255, 0, 0), 10)
    return dst




try:
    capture = cv2.VideoCapture(1)
    
    while(True):
        ret, frame = capture.read()
        if ret == False:
            print('Cannot get the frame')
            break
        
        
        dst = frame[100:420, 10:630]
        
        
        KeyCheck = cv2.waitKey(1) & 0xFF
      
        if KeyCheck == ord('a'):
            mask_inverse = img_pretreatment(dst)
            SizeCheck(mask_inverse)

                  
        elif KeyCheck == ord('q'):
            break
        
        mask_inverse = img_pretreatment(dst)
        dst = Draw_contours(mask_inverse, dst)
        
        cv2.imshow('f', dst)
        
    

    capture.release()
    cv2.destroyAllWindows()
    
except:
    import sys
    print("Error:", sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))