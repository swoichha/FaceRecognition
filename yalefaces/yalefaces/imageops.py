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
normalize_face_vector = np.zeros((150,51260), order='C')
k_eigenvector = np.zeros((11,51260), order='C')



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
	# plt.imshow(resized_normalize_face_vector,cmap='gray')
	# plt.show()


print('\n minimum value:',np.min(normalize_face_vector[2]))
print('\n shape of 148th image:  \n',len(normalize_face_vector[148]))
# print('\n shape of 148th image:  \n',normalize_face_vector[148].shape)
print('\n shape as a whole :\n',normalize_face_vector.shape)

#-------covariance of matrix is done to find eigen vector. total 150 eigen vectors each of dimension 150*1
covariance_Matrix = np.cov(normalize_face_vector)
print('\n covariance matrix of 148th image:\n',covariance_Matrix[148].shape)		#-----gives dimension 150*1 covariance value of 148th image
print('\n shape of covariance matrix  as a whole of shape',covariance_Matrix.shape, 
'\n \n The covariance matrix is: \n \n',covariance_Matrix)				#-----150*150-----#

#--------eigen values and faces-----------
print('\n minimum value:',np.min(covariance_Matrix))
eigenvalue,eigenvector = LA.eig(covariance_Matrix)
print(len(eigenvalue))

print('\n Eigen values are: \n',eigenvalue)
print('\n Eigen vector are: \n',eigenvector)

#------ Make a list of (eigenvalue, eigenvector) tuples--------
eig_pairs = [(np.abs(eigenvalue[i]), eigenvector[:,i]) for i in range(len(eigenvalue))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True) 

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\n \n Eigenvalues in descending order:\n')
for i in eig_pairs:
    print(i[0])		#prints eigen value as it is in first column and eigen vector is in 2nd column



#-----select k eigen faces such that k<m i.e k<150 and can represent the whole training set
tot = sum(eigenvalue)
var_exp = [(i / tot)*100 for i in sorted(eigenvalue, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# print ("Variance captured by each component is \n",var_exp)
print(40 * '-')
# print ("Cumulative variance captured as we travel each component \n",cum_var_exp)

	
print ("All Eigen Values along with Eigen Vectors")
# print(print(eig_pairs))
print(40 * '-' )
matrix_w = np.hstack((eig_pairs[0][1].reshape(150,1),
                      eig_pairs[1][1].reshape(150,1),
                      eig_pairs[2][1].reshape(150,1),
                      eig_pairs[3][1].reshape(150,1),
                      eig_pairs[4][1].reshape(150,1),
                      eig_pairs[5][1].reshape(150,1),
                      eig_pairs[6][1].reshape(150,1),
                      eig_pairs[7][1].reshape(150,1),
                      eig_pairs[8][1].reshape(150,1),
                      eig_pairs[9][1].reshape(150,1),
                      eig_pairs[10][1].reshape(150,1),
                      eig_pairs[11][1].reshape(150,1),
                      eig_pairs[12][1].reshape(150,1),
                      eig_pairs[13][1].reshape(150,1),
                      eig_pairs[14][1].reshape(150,1),
                      eig_pairs[15][1].reshape(150,1),
                      eig_pairs[16][1].reshape(150,1),
                      eig_pairs[17][1].reshape(150,1),
                      eig_pairs[18][1].reshape(150,1),
                      eig_pairs[19][1].reshape(150,1),
                      eig_pairs[20][1].reshape(150,1),
                      eig_pairs[21][1].reshape(150,1),
                      eig_pairs[22][1].reshape(150,1),
                      eig_pairs[23][1].reshape(150,1)))
# print(eigenvector[10].shape)
# print(np.array_equal((eig_pairs[0][1]),matrix_w[:,0]))	
# print ('Matrix W:\n', matrix_w)

imageArrayTranspose = np.transpose(imageArray)
print(imageArrayTranspose.shape)

Y = imageArrayTranspose.dot(matrix_w)
print(Y.shape)

k_eigenvector = np.transpose(Y)
print(k_eigenvector.shape) 
print(np.array_equal((Y[:,0]),k_eigenvector[0,:]))
print(normalize_face_vector.shape)


# for i in range(len(k_eigenvector)):

# 	resized_eigenvector = np.array(np.reshape(k_eigenvector[i],(233,220)), dtype=float)
# 	i += 1
# 	print(i)
# 	# plt.imshow(resized_eigenvector,cmap='gray')
# 	# plt.show()

# wcoeff = pca(matrix_w)
# print(wcoeff)

normalize_face_vector_Transpose = np.transpose(normalize_face_vector)
print(normalize_face_vector_Transpose.shape)

weigth_coeff = np.zeros((150,24),dtype=np.complex128, order='C')
# wcoeff = np.zeros((150,11),dtype=np.uint8, order='C')
for i in range(img_count):
	weigth_coeff[i] = np.array(k_eigenvector.dot(normalize_face_vector_Transpose[:,i]))
print(weigth_coeff[2])


# print(matrix_w.shape)

print(weigth_coeff.shape)
testImg = np.array((weigth_coeff[0].dot(k_eigenvector)),dtype = float)
plotImg = np.reshape(testImg,(233,220))
plt.imshow(plotImg,cmap='gray')
plt.show()




