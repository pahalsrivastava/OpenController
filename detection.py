import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

model = load_model("trained_model/model.h5")

classes = ['call me', 'hi', 'no hand', 'okay', 'peace', 'rock', 'straight', 'thumb']

rimage = cv2.imread("nimage/peace.png")
rimage = cv2.resize(rimage, (400,400))

gaussianBlur = cv2.GaussianBlur(rimage,(5,5),0)
cv2.imshow("blur image", gaussianBlur)

hsvImage = cv2.cvtColor(gaussianBlur,cv2.COLOR_BGR2HSV)
cv2.imshow("hsv Image", hsvImage)

lowerSkinColor = np.array([0,20,20],dtype=np.uint8)
upperSkinColor = np.array([50,255,255],dtype=np.uint8)

mask = cv2.inRange(hsvImage, lowerSkinColor, upperSkinColor)
cv2.imshow("Mask", mask)

resizedMask = cv2.resize(mask, (128, 128))
resizedMask = resizedMask / 255.0

prediction = model.predict(np.array([resizedMask.reshape((128, 128, 1))]))[0]
predictedClassIndex = np.argmax(prediction)
predictedGesture = classes[predictedClassIndex]
print("Confidence Score:", prediction[predictedClassIndex])
print("Predicted Gesture:", predictedGesture)

cv2.imshow("original Image", rimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
