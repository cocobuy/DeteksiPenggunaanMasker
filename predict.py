from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cam = cv2.VideoCapture(0)

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default = "model.model", help="file model")
args = vars(ap.parse_args())

print("[INFO] loading network . . .")
model = load_model(args["model"])

maxVal = 0
maxIndex = 0
status = [0,0,0]

font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
font_scale = 0.5

weared_mask = "Terimakasih telah memakai masker"
not_weared_mask = "Anda tidak memakai masker!!!"
label = ""
reset = 15
flag = 0
countdown = reset
countdown2 = reset

green = (0,255,0)
red = (0,0,255)
color_rect = green
color_text = green

prevTime = 0

while True:
	currentTime = time.time()
	ret,img = cam.read()
	img = imutils.resize(img, width = 400)
	img = cv2.flip(img,1)
	img = cv2.GaussianBlur(img,(5,5),0)
	resized = img.copy()
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.1,4)
	if len(faces) > 0:
		for x, y, w, h in faces:
			roi = img[y:y+h, x:x+w]
			prep = roi.copy()
			prep = cv2.resize(prep, (30, 30))
			prep = prep.astype("float") / 255.0
			prep = img_to_array(prep)
			prep = np.expand_dims(prep, axis=0)
			status[0],status[1]= model.predict(prep)[0]
			maxVal = max(status)
			maxIndex = status.index(maxVal)
			if maxIndex == 1:
				color_rect = green
				countdown = reset
				if(countdown2 > 0):
					countdown2 -= 1
				else:
					if flag != 1:
						label = weared_mask
						color_text = green
						flag = 1
					elif flag == 1:
						label = weared_mask
						color_text = green
			else:
				color_rect = red
				countdown2 = reset
				if countdown > 0:
					countdown -= 1
				else:
					label = not_weared_mask
					color_text = red
					if flag != 2:
						flag = 2
			cv2.rectangle(img, (x, y), ((x + w), (y + h)), color_rect, 2)
	
	else:
		if flag == 1:
			if countdown > 0:
				countdown -= 1
			else:
				label = ""
				countdown = reset
				countdown2 = reset
				flag = 0
		elif flag == 2:
			if countdown2 > 0:
				countdown2 -= 1
			else:
				label = ""
				countdown2 = reset
				countdown = reset
				flag = 0
	
	fps = 1 / (currentTime - prevTime)
	prevTime = currentTime
	cv2.rectangle(img,(0,0),(400,50),(255,255,255),-1)
	cv2.putText(img, label, (10,40), font, font_scale, color_text, thickness, cv2.LINE_AA)	
	cv2.putText(img, "FPS = {:.2F}".format(fps), (10,20), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)	
	cv2.imshow('img', img)
		
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

cam.release()
cv2.destroyAllWindows()
print('Program dihentikan')