import cv2
import numpy as np
import face_recognition
import os

path = 'Images'

images = []
classNames = []
imagesList = os.listdir(path)
print(imagesList)

#----------list of images from given directory-----------#
for img in imagesList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    classNames.append(os.path.splitext(img)[0])
print(classNames)


#--------------function for finding encodings---------#
def findEncodings(images):
    encodelist = []
    for pic in images:
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        encodepic =face_recognition.face_encodings(pic)[0]
        encodelist.append(encodepic)
    return encodelist


encodeknownlist = findEncodings(images)
print(len(encodeknownlist))
print("Yay! Some known faces found!!")


#------------------------Initialise WebCam---------------#
capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    capturedImage = cv2.resize(img,(0,0),None,0.25,0.25)#reduce the size of captured image
    capturedImage = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2RGB)

    frameface = face_recognition.face_locations(capturedImage)
    frameEncoded = face_recognition.face_encodings(capturedImage,frameface)


    for encodeFace,faceloc in zip(frameEncoded, frameface):
        matches = face_recognition.compare_faces(encodeknownlist, encodeFace)
        faceDist = face_recognition.face_distance(encodeknownlist, encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        #matching list
        if  matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)


    #show webcam
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)


#capture.release(0, cv2.CAP_DSHOW)
#cv2.destroyAllWindows()

'''
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

'''