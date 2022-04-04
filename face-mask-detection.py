# -*- coding: utf-8 -*-
"""

"""
import cv2

face_mask = cv2.CascadeClassifier(r'E:\AYU\KULIAH\Semester 8\Kecerdasan Buatan\Tubes AI\classifier\cascade.xml')
img1 = cv2.imread(r'E:\AYU\KULIAH\Semester 8\Kecerdasan Buatan\Tubes AI\n\irfan-n.jpeg',1)
img = cv2.resize(img1,(240,300))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = face_mask.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'Using Mask', (55,280), font,0.5,(255,0,0))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Without Mask',(20,200),font,0.5,(255,255,255))
cv2.imshow('test',img)
cv2.waitKey(0)    
cv2.destroyAllWindows()