import cv2
from pickle import load
from datetime import datetime
from numpy import argmin
from face_recognition import face_locations,face_encodings,face_distance


encodin_list = load( open( "encodin list.p", "rb" ) )
class_name = load( open( "class name.p", "rb" ) )

def attendence(name):
    with open('Today attendance.csv','r+') as f:
        myData = f.readlines()
        namelist = []
        for line in myData:
            entry = line.split(',')[0]
            namelist.append(entry)
        if name not in namelist:
            f.writelines(f"\n{name},{datetime.now()}")

def detact_human(img_s):
    img_s = cv2.resize(img, (0,0), None, 0.33, 0.33)

    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    faceLoc_,name_ = [],[]

    facesCurFrame = face_locations(img_s)
    encodeCurFrame = face_encodings(img_s, facesCurFrame)

    for i in range(5):
        for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):

            faceDis = face_distance(encodin_list , encodeFace)

            find = argmin(faceDis)
            name = class_name[find].upper()

            if str(name in name_) == 'False' and faceDis[find]*100 < 48.00:
                faceLoc_.append(faceLoc)
                name_.append(name)
    return faceLoc_,name_

cap = cv2.VideoCapture(0)

while 1:

    _,img = cap.read()

    faceLoc_,name_ = detact_human(img)

    for faceLoc,name in zip(faceLoc_,name_):

        y1,x2,y2,x1 = faceLoc

        y1,x2,y2,x1 = y1*3, x2*3, y2*3, x1*3

        img = cv2.rectangle(img, (x1,y1), (x2,y2), (36,255,12), 1)

        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 166, 255), 1)

        attendence(name)

    cv2.imshow("Cam_1", img)
    if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()()