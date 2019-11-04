import numpy as np
import cv2
import sys
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
from functools import partial

from mainPCA import *
from plotGraph import *
from OlivettiDataset.mypackage.download_olivetty_faces import getAccuracy
global canvas

class Application:

	def __init__(self):
		window = tk.Tk()
		window.title("Face Detection & Recognition")
		window.geometry('700x400')
		canvas_width = 700
		canvas_height = 400
		self.canvas = Canvas(window, width = canvas_width, height = canvas_height, relief = 'raised')
		self.canvas.pack()


		head = Label(window, text= "Face Detection & Recognition", width=50, fg= "blue", font=" Georgia 16 bold")
		head.place(x=40, y=20)
		path = 'C:\\Users\\swoichha\\Documents\\GitHub\\FaceRecognition\\yalefaces\\yalefaces\\default.jpg'
		self.photo = ImageTk.PhotoImage(Image.open(path))
		self.canvas.create_image(300,70, anchor=NW, image=self.photo)
		self.imgs = []


		button1= tk.Button(window, text = "Browse", width = 20, command = self.browse_image)
		button1.place(x=60 , y=70)

		button2= tk.Button(window, text = "Face Detection ", width=20, command = self.detect_face)
		button2.place(x=60 , y=120)

		self.button3= tk.Button(window, text = "Face Recognition", width=20,command = lambda : displayIndex(self.imgs))
		self.button3["state"]="disabled"
		self.button3.place(x=60 , y=170)

		button4= tk.Button(window, text = "Variance Graph ",width = 20,command =plotGraph)
		button4.place(x=60 , y=220)

		button5= tk.Button(window, text="Real Time Detection",width=20, command = realTimeDetection)
		button5.place(x=60 , y=270)

		window.mainloop()

	def browse_image(self):
		global imgs
		path_to_image = filedialog.askopenfilename(initialdir = "C:\\Users\\swoichha\\Documents\\GitHub\\FaceRecognition\\yalefaces\\yalefaces\\testemotion",title = "Select file",filetypes = (("jpeg files","*.jpg"),("jpeg files","*.jpeg*")))
		if path_to_image:
			self.imgs = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
			self.imgs = cv2.resize(self.imgs, (220,233))
			self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imgs))
			self.canvas.create_image(300,70, anchor=NW, image=self.photo)
			self.button3["state"]="disabled"

	def detect_face(self):
		face_cascade = cv2.CascadeClassifier("C:\\Users\\swoichha\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
		faces = face_cascade.detectMultiScale(self.imgs, scaleFactor = 1.16, minNeighbors = 5)

		if (len(faces) == 0):
			messagebox.showinfo("Detection Error", "Could not find face!")
		else:
			self.button3["state"]="active"	
		print(faces)

		for x,y,w,h in faces:
			self.canvas.create_rectangle(x + 300, y + 70, x + w + 300, y + h + 70, outline = '#39ff14' , width = 2)
			cropped =  self.imgs[y: y + h, x: x + w]
			print(cropped.shape)

def realTimeDetection():

	face_cascade = cv2.CascadeClassifier("C:\\Users\\swoichha\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
	imageCounter = 0
	face_cascade = cv2.CascadeClassifier("C:\\Users\\swoichha\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
	video = cv2.VideoCapture(0)

	while True:

		check, frame = video.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#cropped = grayImg[0:70, 410:640]

		faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.16, minNeighbors = 5)

		for (x, y, w, h) in faces:
			cv2.rectangle(gray, (x,y), (x + w, y + h),(0, 255, 0), 2)
			cropped = grayImg[y:y+h, x:x+w]		

		cv2.imshow('Capturing', gray)
		key = cv2.waitKey(1)

		if key == 27:
			break
		elif cv2.getWindowProperty('Capturing',1) == -1 :
			break
		elif key == 32:
			sub_face = cv2.resize(cropped, (220,233))
			#cv2.imwrite(os.path.join(save, 'data'+str(imageCounter)+'.jpg'), sub_face)
			#imageCounter+=1
			displayIndex(sub_face)
			break

	video.release()
	cv2.destroyAllWindows()
	#print(frame.sum())

		
if __name__ == '__main__':
	Application()