import numpy as np
import cv2
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

face_front_csc = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

emotion_classifier = load_model('SmartAlarm.h5')
#emotion_classifier = model_from_json(open("fer.json", "r").read())
#emotion_classifier.load_weights("fer.h5")

labels = ["Angry","FP","Happy","Neutral","Sad", "Surprise", "Neutral"]
#               Angry,  FacialPalsy,  Happy,   Neutral,   Sad,      Shock
labels_col = [(255,0,0),(0,255,0),(255,0,255),(255,255,0),(5,0,255),(5,0,255),(5,0,255),(5,0,255)]

sample_vid = "sample-video/happyspidey.mp4"

cap = cv2.VideoCapture(sample_vid) ## sample_vid for testing against sample video, 0 for webcam

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_front_csc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]

        roi_flip = cv2.flip(roi_gray,1)

        roi = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        #roi /= 255

        flipped = cv2.resize(roi_flip, (48,48), interpolation=cv2.INTER_AREA)
        flipped = flipped.astype('float')/255.0
        flipped = img_to_array(flipped)
        flipped = np.expand_dims(flipped, axis=0)
        #flipped /= 255

        predFlip = emotion_classifier.predict(flipped)[0]

        pred = emotion_classifier.predict(roi)[0]
        label = labels[pred.argmax()]
        col = labels_col[pred.argmax()]
        stroke = 3
        cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, col, stroke, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
