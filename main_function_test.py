# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:33:56 2020

@author: daniel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



def img_pretreatment(upstate):
    upstate_hsv = cv2.cvtColor(upstate, cv2.COLOR_RGB2HSV)
    blue_min = np.array([100, 0, 0], np.uint8)
    blue_max = np.array([140, 255, 255], np.uint8)
    mask = cv2.inRange(upstate_hsv, blue_min, blue_max) #Item is Black, backgraund is white
    mask = cv2.medianBlur(mask, 21)
    mask_inverse = cv2.bitwise_not(mask) #Item is White, backgraund is Black
    return mask_inverse, mask


def Detec_CTSB(mask_inverse, mask):
    contours, _  = cv2.findContours(mask_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints = True)
    
    hull_dim2 = hull.squeeze()
    ep =np.array([len(hull)])-1
    cal_base = np.insert(hull_dim2, 0, hull_dim2[0], axis=0)
    cal_ded = np.append(hull_dim2, hull_dim2[ep], axis=0)
    
    differ = cal_base - cal_ded
    power = np.sqrt(np.sum(differ**2, axis=1))
    differ_parameter = 60
    final_point = np.array([(np.where(power > differ_parameter))])
    #print(final_point.size)
    

# =============================================================================
#     p = final_point.flatten() 
#     for i in p:
#         mask = cv2.circle(mask, (hull_dim2[i,0],hull_dim2[i,1]) , differ_parameter, (0,0,255), 20)
# =============================================================================

    
    #plt.imshow(cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2RGB))

    leftmost = cnt[cnt[:,:,0].argmin()][0][0]
    rightmost = cnt[cnt[:,:,0].argmax()][0][0]
    #topmost = cnt[cnt[:,:,1].argmin()][0][1]
    a = rightmost - leftmost
    
    if 2300 < a <2500 :
        size = 345
  
     
       
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    x,y,w,h = cv2.boundingRect(cnt)
    mask = cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),20)     
    cv2.putText(mask, str(size) + " x " + str(final_point.size), (x ,y-100 ), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 25, cv2.LINE_4,bottomLeftOrigin = False)
    plt.imshow(mask)
          
  


upstate = cv2.imread("c:\\Users\\merro\\Desktop\\345-50.jpg")
img , mask = img_pretreatment(upstate)
Detec_CTSB(img, mask)
    
    

