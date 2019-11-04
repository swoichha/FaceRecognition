import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import linalg as LA
from scipy.spatial import distance
from detectionGui import imgs as testImg



global img_count
global directory
global imageArray
global normalize_face_vector

img_count = 0
directory = ""
imageArray = np.zeros((150,51260),dtype=np.uint8, order='C')  #Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
normalize_face_vector = np.zeros((150,51260), order='C')
# k_eigenvector = np.zeros((11,51260), order='C')



# resized_normalize_face_vector = np.zeros((150,51260),dtype=np.uint8, order='C')
# imageArray = imageArray.reshape(150,51260,1)      #150 elements with 51260 rows ans 1 column

print(imageArray)
print('\n',imageArray[9].size)                #only from 0-149

DATADIR = 'C:\\Users\\swoichha\\Documents\\GitHub\\FaceRecognition\\yalefaces\\yalefaces\\dataset'
CATEGORIES = ['subject01','subject02','subject03','subject04','subject05','subject06','subject07','subject08','subject09','subject10','subject11','subject12','subject13','subject14','subject15']
# CATEGORIES = ['subject01']

for category in CATEGORIES: 
  path = os.path.join(DATADIR,category)         #-------path to all directory of images of diff subject
  for img in os.listdir(path):            #-------iterate through all those images
    directory = (os.path.join(path,img))
    img_array = mpimg.imread(directory)
    img_array.resize(1,51260)       
    imageArray[img_count] = np.array(img_array)
    img_count += 1
    
print(np.array_equal(imageArray[4],imageArray[50]))   #checking if array of images are correctly stored


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

print(avg_img[512])   #----from 0 to 51259-----#


#---------------Normalize face vector-------------#
i = 0
while (i < img_count): 
  normalize_face_vector[i] = np.array(imageArray[i] - avg_img)
  # print('\n normalize_face_vector of image',i,'is', normalize_face_vector[i])
  resized_normalize_face_vector = np.reshape(normalize_face_vector[i],(233,220))
  i += 1
  # plt.imshow(resized_normalize_face_vector,cmap='gray')
  # plt.show()

normalize_face_vector = normalize_face_vector.transpose() #(51260,150)
imageArray = imageArray.transpose()
# resized_img = np.reshape(imageArray[:,1],(233,220))
# plt.imshow(resized_img,cmap='gray')
# plt.show()


#-------covariance of matrix is done to find eigen vector. total 150 eigen vectors each of dimension 150*1
covariance_Matrix = np.cov(normalize_face_vector,rowvar=False)  #rowvar = Flase : each column represents a variable, while the rows contain observations.
print('\n covariance matrix of 148th image:\n',covariance_Matrix[148].shape)    #-----gives dimension 150*1 covariance value of 148th image
print('\n shape of covariance matrix  as a whole of shape',covariance_Matrix.shape, 
'\n \n The covariance matrix is: \n \n',covariance_Matrix)        #-----150*150-----#
print(covariance_Matrix.shape)
print(normalize_face_vector.shape)

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
    print(i[0])   #prints eigen value as it is in first column and eigen vector is in 2nd column



#-----select k eigen faces such that k<m i.e k<150 and can represent the whole training set
tot = sum(eigenvalue)
var_exp = [(i / tot)*100 for i in sorted(eigenvalue, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# print ("Variance captured by each component is \n",var_exp)
print(80 * '-')
# print ("Cumulative variance captured as we travel each component \n",cum_var_exp)

k=0
for i in range(len(cum_var_exp)):
  if cum_var_exp[i] < 92:
    k = k+1
    
print(k)
matrix_w = np.zeros((k,150))

for i in range(k):
  matrix_w[i] = np.array(eig_pairs[i][1].reshape(1,150))
 

print(matrix_w.shape)
print(np.array_equal((eig_pairs[0][1]),matrix_w[0,:]))  
print ('Matrix W:\n', matrix_w)

# imageArrayTranspose = np.transpose(imageArray)
print(normalize_face_vector.shape)
print(matrix_w.shape)

matrix_wT = matrix_w.transpose()
print(np.array_equal(matrix_wT[:,26],matrix_w[26,:])) 
print(matrix_wT.shape)
Y = normalize_face_vector.dot(matrix_wT)
print(Y.shape)

k_eigenvector = np.transpose(Y)
print(k_eigenvector.shape) 
# print(np.array_equal((Y[:,0]),k_eigenvector[0,:]))
# print(normalize_face_vector.shape)


for i in range(len(k_eigenvector)):

  resized_eigenvector = np.array(np.reshape(k_eigenvector[i],(233,220)), dtype=float)
  i += 1
  print(i)
  # plt.imshow(resized_eigenvector,cmap='gray')
  # plt.show()

# wcoeff = pca(matrix_w)
# print(wcoeff)

normalize_face_vector_Transpose = np.transpose(normalize_face_vector)
print(normalize_face_vector.shape)

weight_coeff = np.zeros((150,k))
# wcoeff = np.zeros((150,11),dtype=np.uint8, order='C')
for i in range(img_count):
  weight_coeff[i] = np.array(k_eigenvector.dot(normalize_face_vector[:,i]))
print(weight_coeff[2])  #weight of 36 eigen faces for 2nd image in traning


# print(matrix_wT.shape)
print(weight_coeff.shape)
print(k_eigenvector.shape)
print(normalize_face_vector.shape)
print(avg_img.shape)


# testImg = mpimg.imread(inputImg)
print(testImg)
testImg.resize(1,51260)


normalizeTestImg = testImg - avg_img
normalizeTestImg = normalizeTestImg.transpose()
testImgWeight = k_eigenvector.dot(normalizeTestImg)
testImgWeightT = testImgWeight.transpose() 
print(testImgWeightT.shape)
print(weight_coeff.shape)

# Array to store difference in weights between test image and training images to find best match

# Function for euclidean distance
def euclidean_distance(v, u):
   return np.sqrt(np.sum((v - u) ** 2))

dist = np.zeros((150,1))
for i in range(img_count):
  # dist[i] = euclidean_distance(testImgWeightT, weight_coeff[i,:])
  # dist = LA.norm(testImgWeightT - weight_coeff[i,:])
  dist[i] = distance.euclidean(testImgWeightT, weight_coeff[i,:])

print(dist[2].shape)
print(dist.shape)




minval = np.amin(dist)
print(minval)
index = np.argmin(dist)
print(index)

# result = np.where(dist == minval)

if index < 10:
  print('subject 01')
elif index < 20:
  print('subject 02')
elif index < 30:
  print('subject 03')  
elif index < 40:
  print('subject 04')
elif index < 50:
  print('subject 05')
elif index < 60:
  print('subject 06')
elif index < 70:
  print('subject 07')
elif index < 80:
  print('subject 08')
elif index < 90:
  print('subject 09')
elif index < 100:
  print('subject 10')
elif index < 110:
  print('subject 11')
elif index < 120:
  print('subject 12')
elif index < 130:
  print('subject 13')  
elif index < 140:
  print('subject 14')
elif index < 150:
  print('subject 15')
else:
  print('No match found')    

 
 