import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt



house_min = np.array([100, 0, 0], np.uint8)
house_max = np.array([140, 255, 255], np.uint8)
    
upstate = cv2.imread("C:\\Users\\merro\\Desktop\\work\\Programming\\python\\OpenCV\\Picture\\Picture for testing\\main_test1.jpg")
upstate_hsv = cv2.cvtColor(upstate, cv2.COLOR_RGB2HSV)
blue_min = house_min
blue_max = house_max
mask = cv2.inRange(upstate_hsv, blue_min, blue_max)
mask = cv2.GaussianBlur(mask, (15,15), 0)
contours, _  = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
cnt_dim2 = cnt.squeeze()

leftmost = cnt[cnt[:,:,0].argmin(),0,:]
rightmost = cnt[cnt[:,:,0].argmax(),0,:]
center = np.abs(rightmost - leftmost)/2+leftmost

pre = cnt_dim2-center
power = np.sqrt(np.sum(pre**2, axis=1))

x = []
total = 0
for i in range(power.size):
    x.append(i)
    
    
threshold = (power[power.argmax()] - power[power.argmin()])/3+power[power.argmin()]
threshold = int(threshold)
    
y = [] 
check = 0
teeth = 0

for j in power:
    if j >= threshold:
        y.append(1)
        h = 1
        check = h

    else:
        y.append(0)
        h = 0
        if check - h == 1:
            teeth +=1
            check = h
        
        
print(teeth)
print(center)
print(threshold)

plt.plot(x,y)
