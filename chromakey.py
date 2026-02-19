
import torch
import cv2
import time
import numpy as np

from datetime import datetime


def nothing(x):
    pass

capture = cv2.VideoCapture(0)




cv2.namedWindow("Results")
cv2.createTrackbar("min H", "Results", 50, 255, nothing)
cv2.createTrackbar("max H", "Results", 100, 255, nothing)

cv2.createTrackbar("min S", "Results", 50, 255, nothing)
cv2.createTrackbar("max S", "Results", 255, 255, nothing)

cv2.createTrackbar("min V", "Results", 50, 255, nothing)
cv2.createTrackbar("max V", "Results", 255, 255, nothing)




imageWidth = 160
imageHeight = 160

background = cv2.imread("D://hvs//Hyvsion_Projects//Actions_project//Bg_images//KakaoTalk_20250205_094441982_11.jpg")
background = cv2.resize(background, dsize=(560,400))

start = datetime.now()
while True:
    ret, original = capture.read()
    original = original[40: 440, 40: 600]
    result = original.copy()

    minH = cv2.getTrackbarPos('min H','Results')
    maxH = cv2.getTrackbarPos('max H','Results')

    minS = cv2.getTrackbarPos('min S','Results')
    maxS = cv2.getTrackbarPos('max S','Results')

    minV = cv2.getTrackbarPos('min V','Results')
    maxV = cv2.getTrackbarPos('max V','Results')




    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (minV, minS, minH), (maxV, maxS, maxH))     

    cv2.copyTo(background, mask, result)  
    cv2.imshow("mask", mask)
    cv2.imshow("VideoFrame", original)
    cv2.imshow("Results", result)
    cv2.waitKey(1)




    
        
