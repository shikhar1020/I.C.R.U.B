import cv2
import numpy as np
import face_recognition

imgsangam = face_recognition.load_image_file('Shikhar.jpg')
imgsangam = cv2.cvtColor(imgsangam,cv2.COLOR_BGR2RGB)

cv2.imshow('Sangam Image', imgsangam)
cv2.waitKey(0)