# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:58:46 2020

@author: Subinoy Mukherjee
"""

import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0)
while True:
    ret , img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y) , (x+w,y+h), (255,0,0) , 3)
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = img[y:y+h , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey) , (ex+ew,ey+eh), (0,255,0) , 2)
          
    cv2.imshow('img' , img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyALLWindows()