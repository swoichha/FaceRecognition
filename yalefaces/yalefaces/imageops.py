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
imageArray = np.zeros((150,51260),dtype=int,order='C')	#Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
print(imageArray)
# img_array = img_array.reshape(150,51260,1)			#150 elements with 51260 rows ans 1 column
print(imageArray[9].size)								#only from 0-9
print(imageArray.shape)

img_count = 0
dir = ""
for category in CATEGORIES:	
	path = os.path.join(DATADIR,category) 				#-------path to all directory of images of diff subject
	for img in os.listdir(path):						#-------iterate through all those images
		dir = (os.path.join(path,img))
		img_array = mpimg.imread(dir)
		img_array.resize(1,233*220)
		# print(imageArray.shape)
		# print('\n This is actual array \n',imageArray)
				
		imageArray[img_count] = np.array(img_array)
		# i = i + 1
		img_count += 1
		# a = np.copyto(img_array,imageArray,casting='same_kind', where=True)
		# # print(img_array.shape)
		# # print('\n This is array after place \n',img_array)

		# a = np.place(img_array,img_array == 0,imageArray)
		# print(img_array.shape)
		# print('\n This is array after place \n',img_array)

		

print(img_count)
# print(img_array[512]) #------0 to 9 only
print(imageArray[5])		#its size is 51260	
print(imageArray[9])	


def averageOfArray(x,total):
	x = np.sum(x, axis=0)
	x = (x/total)

	return x

avg_faceVector = averageOfArray(imageArray,img_count)
print(avg_faceVector)
# # print(avg_faceVector.shape)
# # print(avg_faceVector[512])
resized_avg_faceVector = np.reshape(avg_faceVector,(233,220))
plt.imshow(resized_avg_faceVector,cmap='gray')
plt.show()


#------Normalize face vector
i = 0
while (i < img_count): 
	normalize_face_vector = np.array(imageArray[i] - avg_faceVector)
	normalize_face_vector = np.reshape(normalize_face_vector,(233,220))
	i += 1
	# plt.imshow(normalize_face_vector,cmap='gray')
	# plt.show()

#------covariance of matrix	is done to find eigen vector
covariance_Matrix = np.cov(normalize_face_vector)	
print('\n covariance matrix \n',covariance_Matrix)

	
