import os
import sys
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

webcam = False
fp = False
samplevid = "sample-video/fast.mp4"

#handling arguments
if len(sys.argv) > 1:
    if sys.argv[1] == "webcam":
        webcam = True


if len(sys.argv) > 2:
    if sys.argv[2] == "fp":
        fp = True

if fp == True:
    print("Using Modified DataSet... ")
    modelSource = "alarm"
else:
    print("Using FER Dataset... ")
    modelSource = "fer2013"

if webcam == True:
    print("Capturing From Webcam")
    captureSource = 0
else:
    print("Using Demo Clip")
    captureSource = samplevid
#load model
model = model_from_json(open(f"{modelSource}.json", "r").read())
#load weights
model.load_weights(f'{modelSource}.h5')
#facial classifer
face_haar_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')


if webcam == True:
    captureSource = 0
else:
    captureSource = samplevid

cap=cv2.VideoCapture(captureSource)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,255),thickness=7)

        #selecting face for prediction
        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        roi_flip = cv2.flip(roi_gray,1)
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        #second prediction (flipped)
        img_pixels_f = image.img_to_array(roi_flip)
        img_pixels_f = np.expand_dims(img_pixels, axis = 0)
        img_pixels_f /= 255

        #flipped prediction
        flipped_pred = model.predict(img_pixels)

        #predicting
        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        #extra check on stroke by flipping the image
        if np.argmax(flipped_pred[0]) == 7:
            max_index = 7

        #get the emotion
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral','!!')
        color_emotions = ((0,0,255), (0,255,100), (255,0,0),(0,255,0), (255,0,100), (255,255,255), (255,0,0))
        predicted_emotion =  emotions[max_index]

        #print to screen
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_emotions[max_index], 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)


    # Q will Quit
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
