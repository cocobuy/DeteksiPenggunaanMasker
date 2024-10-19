import numpy as np
import cv2
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cam = cv2.VideoCapture(0)
data_num = 1
getData = 0
while True:
	ret,img = cam.read()
	img = imutils.resize(img, width = 400)
	img = cv2.flip(img,1)
	img = cv2.GaussianBlur(img,(5,5),0)
	resized = img.copy()
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.1,4)
	for x, y, w, h in faces:
		roi = img[y:y+h, x:x+w]
		if getData == 1:
			cv2.imwrite('data'+str(data_num)+'.png',roi)
			print("data{}.png tersimpan".format(data_num))
			data_num += 1
		cv2.imshow('roi', roi)
		cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
	
	cv2.imshow('img', img)
	
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	
	elif k == ord('s'):
		getData = 1

	elif k == ord('u'):
		getData = 0

cam.release()
cv2.destroyAllWindows()
print('Program dihentikan')