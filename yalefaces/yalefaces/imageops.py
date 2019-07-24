#this is inside the dataset code
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.image as mpimg



DATADIR = 'C:/Users/swoichha/Documents/GitHub/FaceRecognition/yalefaces/yalefaces/dataset'
CATEGORIES = ['subject01','subject02','subject03','subject04','subject05','subject06','subject07','subject08','subject09','subject10','subject11','subject12','subject13','subject14','subject15']

img_count = 0
normalize_face_vector = np.zeros((150,51260),dtype=np.uint8, order='C')
# resized_normalize_face_vector = np.zeros((150,51260),dtype=np.uint8, order='C')
imageArray = np.zeros((150,51260),dtype=np.uint8, order='C')	#Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
print(imageArray)
# imageArray = imageArray.reshape(150,51260,1)			#150 elements with 51260 rows ans 1 column
print('\n',imageArray[9].size)								#only from 0-9
print('\n',imageArray.shape)

dir = ""

for category in CATEGORIES:	
	path = os.path.join(DATADIR,category) 				#-------path to all directory of images of diff subject
	for img in os.listdir(path):						#-------iterate through all those images
		dir = (os.path.join(path,img))
		img_array = mpimg.imread(dir)
		img_array.resize(1,51260)
		# print(imageArray.shape)
		# print('\n This is actual array \n',imageArray)
				
		imageArray[img_count] = np.array(img_array)
		# i = i + 1
		img_count += 1
		

# print(img_array[512]) 	#------0 to 9 only------#
print(imageArray[5])		#------its size is 51260--------#	


def averageOfArray(x,total):
	x = np.sum(x, axis=0)
	x = (x/total)
	return x

avg_faceVector = averageOfArray(imageArray,img_count)
print(avg_faceVector.shape)

resized_avg_faceVector = np.reshape(avg_faceVector,(233,220))

plt.imshow(resized_avg_faceVector,cmap='gray')
plt.show()

# print(avg_faceVector[512])		#----from 0 to 51259-----#


#------Normalize face vector
i = 0
while (i < img_count): 
	normalize_face_vector[i] = np.array(imageArray[i] - avg_faceVector)
	# print('\n normalize_face_vector of image',i,'is', normalize_face_vector[i])
	resized_normalize_face_vector = np.reshape(normalize_face_vector[i],(233,220))
	i += 1
	# plt.imshow(resized_normalize_face_vector,cmap='gray')
	# plt.show()

print('\n shape  \n',normalize_face_vector.shape)

# #------covariance of matrix	is done to find eigen vector
covariance_Matrix = np.cov(normalize_face_vector)	
print('\n covariance matrix \n',covariance_Matrix)
print('\n shape  \n',covariance_Matrix.shape)		#-----150*150-----#

	
