# Imports

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import linalg as LA
from showFaces import show_40_distinct_people, show_10_faces_of_n_subject
from sklearn.metrics import classification_report
#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from time import time

from faceRecognition import faceRecog
from weights import weight 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
imageArray = ''


def getAccuracy():
	data=np.load("../olivetti_faces.npy")
	target=np.load("../olivetti_faces_target.npy")

	# print("There are {} images in the dataset".format(len(data)))
	# print("There are {} unique targets in the dataset".format(len(np.unique(target))))
	# print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))
	# print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4]))

	# print("unique target number:",np.unique(target))



	# show_40_distinct_people(data, np.unique(target))    
		
	#You can playaround subject_ids to see other people faces
	# show_10_faces_of_n_subject(images=data, subject_ids=[0,5, 21, 24, 36])    
	# plt.show()
	print("X shape:",data.shape)
	#We reshape images for machine learnig  model
	X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
	print("X shape:",X.shape)

	X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
	print("X_train shape:",X_train.shape)
	print("y_train shape:{}".format(y_train.shape))

	dim = np.shape(X_train)

	imageCount, total_pixels = dim


	global normalizeFaceVector
	normalizeFaceVector = np.zeros((imageCount, total_pixels), order='C')

	def averageOfArray(x,total):
		x = np.sum(x, axis = 0)
		x = (x/total)
		return x

	  

	averageImage = averageOfArray(X_train,imageCount)
	# ------------------Ploting image from imageArray------------------------------
	resizedAverageImage = np.reshape(averageImage,(64,64))
	# plt.imshow(resizedAverageImage,cmap='gray')
	# plt.show()

	cum_var_exp, eig_pairs,normalizeFaceVector = faceRecog(X_train,imageCount,averageImage,normalizeFaceVector)

	#k is equals to the n_componet of pca
	k=0
	for i in range(len(cum_var_exp)):
		if cum_var_exp[i] < 95:
		  k = k+1

	matrix_w = np.zeros((k,280))
	  #------storing lower dimensionality k eigenvectors into matrix_w-----------
	for i in range(k):
		matrix_w[i] = np.array(eig_pairs[i][1].reshape(1,280))

	matrix_wT = matrix_w.transpose()

	#-----------converting lower dimension k eigenvectors to original face dimensionality---------------------
	Y = normalizeFaceVector.dot(matrix_wT)
	k_eigenVector = np.transpose(Y)
	 

	# for i in range(len(k_eigenVector)):
	#   resized_eigenvector = np.array(np.reshape(k_eigenVector[i],(64,64)), dtype=float)
	  
	#   plt.imshow(resized_eigenvector,cmap='gray')
	#   plt.show()

	weight_coeff = weight(k_eigenVector,normalizeFaceVector,0)
	X_train_pca = weight_coeff
	# Initialize Classifer and fit training data
	clf = SVC(kernel='rbf',C=10000,gamma=0.000001)
	clf = clf.fit(X_train_pca, y_train)

	print(k)

	normalizeTestImg = X_test - averageImage
	normalizeTestImg = normalizeTestImg.transpose()
	testImgWeight = weight(k_eigenVector,normalizeTestImg,1)
	print(testImgWeight.shape)
	testImgWeightT = testImgWeight.transpose() 
	X_test_pca = testImgWeightT

	y_pred = clf.predict(X_test_pca)
	print("accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
	# print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))

	models=[]
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(("SVM",SVC(kernel='rbf',C=10000,gamma=0.000001)))
	models.append(("LR",LogisticRegression()))
	models.append(("NB",GaussianNB()))
	models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))
	models.append(("DT",DecisionTreeClassifier()))


	for name, model in models:
	    
	    clf = model

	    clf.fit(X_train_pca, y_train)
	    y_pred = clf.predict(X_test_pca)
	    print(10*"=","{} Result".format(name).upper(),10*"=")
	    print("Accuracy score:{:0.4f}".format(metrics.accuracy_score(y_test, y_pred)))
	    print()
	    
	return    

getAccuracy()