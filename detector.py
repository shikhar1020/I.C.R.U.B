import cv2
import numpy as np
import face_recognition

#test image
imgsangam = face_recognition.load_image_file('Shikhar.jpg')
imgsangam = cv2.cvtColor(imgsangam,cv2.COLOR_BGR2RGB)

imgsangam1 = face_recognition.load_image_file('Shikhar1.jpeg')
imgsangam1 = cv2.cvtColor(imgsangam1,cv2.COLOR_BGR2RGB)


#faceLocation
LocationFace = face_recognition.face_locations(imgsangam)[0]
encodeSangam =face_recognition.face_encodings(imgsangam)[0]
cv2.rectangle(imgsangam,(LocationFace[3],LocationFace[0]),(LocationFace[1],LocationFace[2]),(255,0,255),2)
cv2.imshow('Sangam Image', imgsangam)
#print(LocationFace)


LocationFace1 = face_recognition.face_locations(imgsangam1)[0]
encodeSangam1 =face_recognition.face_encodings(imgsangam1)[0]
cv2.rectangle(imgsangam1,(LocationFace1[3],LocationFace1[0]),(LocationFace1[1],LocationFace1[2]),(255,0,255),2)
cv2.imshow('Sangam Image 1', imgsangam1)
#print(LocationFace)

cv2.waitKey(0)