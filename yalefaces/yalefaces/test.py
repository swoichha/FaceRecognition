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
CATEGORIES = ['subject01']#,'subject02','subject03','subject04','subject05','subject06','subject07','subject08','subject09','subject10','subject11','subject12','subject13','subject14','subject15']
img_count = 0
array_sum = 0
print(type(img_count))

for category in CATEGORIES:
	path = os.path.join(DATADIR,category) #path to all directory of images of diff subject
	for img in os.listdir(path):		#iterate through all those images
		img_array = mpimg.imread(os.path.join(path,img))
		img_array.resize(1,233*220)
		array_sum += img_array

		# print('\nthis is array \n',img_array)
		#print(array_sum)
		img_count += 1



# np.save('imageArrayfile', img_array)
print(type(img_array))
print(type(array_sum))
print(img_count)

# array_sum = np.ndarray.sum(img_array)
print('\n sum of array of all image \n',array_sum)


print('\n  total number of train image \n ',img_count)
avg_faceVector = array_sum/img_count 

print(avg_faceVector)

avg_faceVector = avg_faceVector.astype(int)
print('\n The average face vector \n',avg_faceVector)

resized_avg_faceVector = np.reshape(avg_faceVector,(233,220))
print('\n',resized_avg_faceVector)
print(resized_avg_faceVector.shape)

plt.imshow(resized_avg_faceVector,cmap='gray')
plt.show()
plt.pause(10)

plt.close()


def normalizeVector(imgArray, averageImage):
	return (imgArray-averageImage)
normalize_face_vector = 0
# to normalize face vector of image
normalize_face_vector = 0
for category in CATEGORIES:
	path = os.path.join(DATADIR,category) #path to all directory of images of diff subject
	for img in os.listdir(path):		#iterate through all those images
		img_array = cv2.imread(os.path.join(path,img),0)
		normalize_face_vector = normalizeVector(img_array,resized_avg_faceVector)
		plt.imshow(normalize_face_vector, cmap='gray')
		plt.show()
		print('\n this is normald vector \n',normalize_face_vector)
		#print(array_sum)
		img_count += 1
print(type(normalize_face_vector))
print('\n test print which should be same as above \n \n',normalize_face_vector)



# --------------------------------------
# import cv2 as cv

# img = cv.imread("img_name_here.jpg", 0) # 0 means read the image as gray
# print(type(img))
# resized_image = cv.resize(img, (-----)) # substitute --- with the image size in tuples

#----------------------------------------

# plt.imshow(resized_image, cmap='gray')
# plt.show()
# b = np.reshape(avg_faceVector,(233,220))
# print(b)

# cv2.imshow('average image',b)
# cv2.waitKey(0)

#to show images 
# plt.imshow(img_array)
# plt.show()
# print('\n Face vector of ',img +'\n',img_array)
# ----------------------------------	

# img = Image.fromarray(avg_faceVector)



# print(type(img_array))
# the array is loaded into b 
# b = np.load('imageArrayfile.npy') 		

# array_length =len(img_array)
# print(array_length)

# x = img_array.data
# n_features = x.shape[1]
