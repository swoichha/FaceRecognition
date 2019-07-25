#this is inside the dataset code

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import linalg as LA


global img_count
global directory
global imageArray
global normalize_face_vector

img_count = 0
directory = ""
imageArray = np.zeros((150,51260),dtype=np.uint8, order='C')	#Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
normalize_face_vector = np.zeros((150,51260),dtype=np.int8, order='C')
# resized_normalize_face_vector = np.zeros((150,51260),dtype=np.uint8, order='C')
# imageArray = imageArray.reshape(150,51260,1)			#150 elements with 51260 rows ans 1 column

print(imageArray)
print('\n',imageArray[9].size)								#only from 0-149

DATADIR = 'C:/Users/swoichha/Documents/GitHub/FaceRecognition/yalefaces/yalefaces/dataset'
CATEGORIES = ['subject01','subject02','subject03','subject04','subject05','subject06','subject07','subject08','subject09','subject10','subject11','subject12','subject13','subject14','subject15']

for category in CATEGORIES:	
	path = os.path.join(DATADIR,category) 				#-------path to all directory of images of diff subject
	for img in os.listdir(path):						#-------iterate through all those images
		directory = (os.path.join(path,img))
		img_array = mpimg.imread(directory)
		img_array.resize(1,51260)				
		imageArray[img_count] = np.array(img_array)
		img_count += 1
		
print(np.array_equal(imageArray[4],imageArray[50]))		#checking if array of images are correctly stored


def averageOfArray(x,total):
	x = np.sum(x, axis=0)
	x = (x/total)
	return x

avg_img = averageOfArray(imageArray,img_count)
print('\n average face vector \n',avg_img.shape)
print(avg_img)

resized_avg_img = np.reshape(avg_img,(233,220))
# plt.imshow(resized_avg_img,cmap='gray')
# plt.show()

print(avg_img[512])		#----from 0 to 51259-----#


#---------------Normalize face vector-------------#
i = 0
while (i < img_count): 
	normalize_face_vector[i] = np.array(imageArray[i] - avg_img)
	# print('\n normalize_face_vector of image',i,'is', normalize_face_vector[i])
	resized_normalize_face_vector = np.reshape(normalize_face_vector[i],(233,220))
	i += 1
	plt.imshow(resized_normalize_face_vector,cmap='gray')
	plt.show()




#-------covariance of matrix is done to find eigen vector. total 150 eigen vectors each of dimension 150*1
covariance_Matrix = np.cov(normalize_face_vector)
print('\n covariance matrix of 148th image:\n',covariance_Matrix[148].shape)		#-----gives dimension 150*1 covariance value of 148th image
print('\n shape of covariance matrix  as a whole of shape',covariance_Matrix.shape, '\n',covariance_Matrix)				#-----150*150-----#

#--------eigen values and faces-----------
print('\n minimum value:',np.min(covariance_Matrix))
eigenvalue,eigenvector = LA.eig(covariance_Matrix)
print('\n Eigen values are: \n',eigenvalue)
print('\n Eigen vector are: \n',eigenvector)

#------ Make a list of (eigenvalue, eigenvector) tuples--------
eig_pairs = [(np.abs(eigenvalue[i]), eigenvector[:,i]) for i in range(len(eigenvalue))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\n \n Eigenvalues in descending order:\n')
for i in eig_pairs:
    print(i[0])

	
