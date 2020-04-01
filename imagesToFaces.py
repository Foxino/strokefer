#script to convert dataset into just images of faces

import cv2
import os

face_front_csc = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
dir = "raw"
target = "trainingdata"
path = os.path.join(os.getcwd(),dir)
count = 0

for filename in os.listdir(path):
    count += 1
    if filename.endswith(".png"):
        t = os.path.join(path, filename)
        cap = cv2.VideoCapture(t)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_front_csc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            img_item = target + "/"+str(count)+".png"
            cv2.imwrite(img_item, roi_gray)
