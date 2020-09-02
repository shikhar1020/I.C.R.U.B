import cv2
import numpy as np
import face_recognition

#test image
Image1 = face_recognition.load_image_file('Sangam1.jpg')
Image1 = cv2.cvtColor(Image1,cv2.COLOR_BGR2RGB)

Image2 = face_recognition.load_image_file('Sangam2.jpg')
Image2 = cv2.cvtColor(Image2,cv2.COLOR_BGR2RGB)


#------------Face Location and Encoding----------------------#
#------IMAGE 1-----------#
LocationFace1 = face_recognition.face_locations(Image1)[0]
encodeImage1 =face_recognition.face_encodings(Image1)[0]
cv2.rectangle(Image1,(LocationFace1[3],LocationFace1[0]),(LocationFace1[1],LocationFace1[2]),(255,0,255),2)
#createwindowtoshowimage
cv2.namedWindow('Image 1 ',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Image 1 ', 600,600)

#print(LocationFace1)

#------IMAGE 2-----------#
LocationFace2 = face_recognition.face_locations(Image2)[0]
encodeImage2 =face_recognition.face_encodings(Image2)[0]
cv2.rectangle(Image2,(LocationFace2[3],LocationFace2[0]),(LocationFace2[1],LocationFace2[2]),(255,0,255),2)
#createwindowtoshowimage
cv2.namedWindow('Image 2',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Image 2', 600,600)

#print(LocationFace2)

#----------------faceEncodingCheck------------------#
encodingresult = face_recognition.compare_faces([encodeImage1],encodeImage2)
faceDistance = face_recognition.face_distance([encodeImage1],encodeImage2)
print(encodingresult, faceDistance)
cv2.putText(Image2, f'{encodingresult}{round(faceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,146,0),)

#---------------showImage---------------#
cv2.imshow('Image 2', Image2)
cv2.imshow('Image 1 ', Image1)

cv2.waitKey(0)