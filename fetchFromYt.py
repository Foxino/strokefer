import os
import cv2
import numpy as np

cd = os.getcwd()
trainingdata = "trainingdata/"
frameRate = 15 ## all videos used likely to be 30FPS, frameRate = 30 = 1 img/s, frameRate = 15 = 2 img/s and so on.
face_csc = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# turn video into frames

def splitVideoIntoFrames(path):
    head, tail = os.path.split(path)
    tail = tail.replace(".mp4", "")
    cap = cv2.VideoCapture(path)
    clen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 1
    if clen == 0:
        return
    else:
        ret, frame = cap.read()
        while ret:
            ret, frame = cap.read()
            if ret and count % frameRate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_csc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                for (x, y, w, h) in faces:
                    fpos = gray[y:y+h, x:x+w]
                    fpos = cv2.resize(fpos,(48,48))
                    cv2.imwrite(os.path.join(cd,trainingdata+"/"+"image_"+tail+"_"+str(count))+".jpg", fpos)

            #if ret and count % frameRate == 0:
            #    cv2.imwrite(os.path.join(cd,trainingdata+"/"+"image_"+tail+"_"+str(count))+".jpg",frame)
            count += 1

# create a dataset using youtube as a source.

from pytube import YouTube as yt

links = ["https://www.youtube.com/watch?v=PdCgmL1EocU"]

for link in links:
    v = yt(link)
    print("Downloading from ", link)
    try:
        #attempt to download in 720p, 1080p too large of a download
        v.streams.filter(mime_type="video/mp4", res="720p").first().download()
    except Exception as e:
        #720p not available, download any version
        v.streams.filter(mime_type="video/mp4").first().download()
    print("Downloaded ", link)


# get all videos from folder

for file in os.listdir(cd):
    if file.endswith("mp4"):
        print("Classifying faces on ", file)
        path=os.path.join(cd, file)
        #splitVideoIntoFrames(path)
        #print("Removing ", file)
        #os.remove(path)
