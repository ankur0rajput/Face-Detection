# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 5:24:41 2019

@author: hp
"""
import math
import cv2
import os
import numpy as np
    
cap=cv2.VideoCapture('Tere_Bin_(Simmba)_Full_HD(bossmobi).mp4')

face_cas=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    faces=face_cas.detectMultiScale(gray, 1.5, 3)
    
    #drawing circle around the faces
    #cv2.shape(line, rectangle etc)(image, Pt1, radius, color, thickness)
    for (x,y,w,h) in faces:
        
        cv2.circle(frame, ( int((x + x + w )/2), int((y + y + h)/2 )), int (h / 2), (0, 0, 255), 3)
        
    
    #display the resulting frames
    cv2.imshow('Frame', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#when evrything done ,releases the videocapture object
cap.release()

#close all the frames
cv2.destroyAllWindows()