import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

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

#-------------function to list/mark the attendence sheet with time-------#
def markAttendence(name):
    with open("attendencelist.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeknownlist = findEncodings(images)
print(len(encodeknownlist))
print("Yay! Some known faces found!!")


#------------------------WebCam Functionalities---------------#
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

        #matching from list and showing name in webcam image
        if  matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 #since we resized the images to .25 or 1/4 intially
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,225,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
            markAttendence(name)

    #show webcam
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)


#capture.release(0, cv2.CAP_DSHOW)
#cv2.destroyAllWindows()

