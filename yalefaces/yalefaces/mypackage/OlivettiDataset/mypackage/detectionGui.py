import numpy as np
import cv2
import sys
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
from functools import partial

from download_olivetty_faces import *
global canvas


def motion_detector():

	face_cascade = cv2.CascadeClassifier("C:\\Users\\swoichha\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
	video = cv2.VideoCapture(0)

	while True:

		check, frame = video.read()

		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.16, minNeighbors = 5)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x,y), (x + w, y + h),(0, 255, 0), 2)

		cv2.imshow('Capturing', frame)

		key = cv2.waitKey(1)

		if key == ord('q'):
			break

	video.release()
	cv2.destroyAllWindows()



class Application:

	def __init__(self):
		window = tk.Tk()
		window.title("Face Recognition")
		window.geometry('600x400')
		canvas_width = 600
		canvas_height = 400
		self.canvas = Canvas(window, width = canvas_width, height = canvas_height, relief = 'raised')
		self.canvas.pack()


		head = Label(window, text= "Face Detection & Recognition", width=30, fg= "blue", font=" Georgia 14 bold")
		head.place(x=20, y=20)
		path = 'C:\\Users\\swoichha\\Documents\\GitHub\\FaceRecognition\\yalefaces\\yalefaces\\default.jpg'
		self.photo = ImageTk.PhotoImage(Image.open(path))
		self.canvas.create_image(250,70, anchor=NW, image=self.photo)
		self.imgs = []


		button1= tk.Button(window, text = "Browse", width = 20, command = self.browse_image)
		button1.place(x=60 , y=70)

		button2= tk.Button(window, text = "Add Image to Dataset",width = 20)
		button2.place(x=60 , y=115)

		button3= tk.Button(window, text = "Face Detection ", width=20, command = self.detect_face)
		button3.place(x=60 , y=160)


		self.button4= tk.Button(window, text = "Face Recognition", width=20,command = lambda : displayIndex(self.imgs))
		self.button4["state"]="disabled"
		self.button4.place(x=60 , y=205)

		button5= tk.Button(window, text="Motion detector",width=20, command = motion_detector)
		button5.place(x=60 , y=250)

		button6= tk.Button(window, text="Accuracy",width=20, command = self.displayAccuracy)
		button6.place(x=60 , y=295)

		window.mainloop()

	def browse_image(self):
		global imgs
		path_to_image = filedialog.askopenfilename(initialdir = "C:\\Users\\swoichha\\Documents\\GitHub\\FaceRecognition\\yalefaces\\yalefaces\\testset",title = "Select file",filetypes = (("jpeg files","*.jpg"),("jpeg files","*.jpeg*")))
		if path_to_image:
			self.imgs = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
			self.imgs = cv2.resize(self.imgs, (220,233))
			self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.imgs))
			self.canvas.create_image(250,70, anchor=NW, image=self.photo)
			self.button4["state"]="disabled"

	def detect_face(self):
		face_cascade = cv2.CascadeClassifier("C:\\Users\\swoichha\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
		faces = face_cascade.detectMultiScale(self.imgs, scaleFactor = 1.16, minNeighbors = 5)

		if (len(faces) == 0):
			messagebox.showinfo("Detection Error", "Could not find face!")
		else:
			self.button4["state"]="active"	
		print(faces)

		for x,y,w,h in faces:
			self.canvas.create_rectangle(x + 250, y + 70, x + w + 250, y + h + 70, outline = '#39ff14' , width = 2)
			cropped =  self.imgs[y: y + h, x: x + w]
			print(cropped.shape)

		
if __name__ == '__main__':
	Application()