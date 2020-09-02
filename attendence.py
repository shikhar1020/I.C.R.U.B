import cv2
import numpy as np
import face_recognition
import os

path = 'Images'

images = []
classNames = []
imagesList = os.listdir(path)
print(imagesList)

for img in imagesList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    classNames.append(os.path.splitext(img)[0])
print(classNames)