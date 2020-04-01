import cv2
import os

p = os.path.join(os.getcwd(),"raw")
co = 0
for filename in os.listdir(p):
    img = os.path.join(p,filename)
    i = cv2.imread(img, 0)
    print(filename)
    try:
        cv2.imwrite(str(co)+".png", i)
    except Exception as e:
        raise
    co += 1
