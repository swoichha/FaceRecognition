import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import linalg as LA

def faceRecog(imageArray,imageCount,averageImage,normalizeFaceVector):
	i = 0
	while (i < imageCount): 
	  normalizeFaceVector[i] = np.array(imageArray[i] - averageImage)
	  resizedNormalizeFaceVector = np.reshape(normalizeFaceVector[i],(233,220))
	  i += 1

	normalizeFaceVector = normalizeFaceVector.transpose() #(51260,150)
	imageArray = imageArray.transpose()  

	#-------covariance of matrix is done to find eigen vector. total 150 eigen vectors each of dimension 150*1
	covarianceMatrix = np.cov(normalizeFaceVector,rowvar=False)  #rowvar = Flase : each column represents a variable, while the rows contain observations.

	# or can do this way also
	# covarianceMatrix = np.cov(normalizeFaceVector.transpose())

	# mean_vec = np.mean(normalizeFaceVector, axis=0)
	# cov_mat = (normalizeFaceVector - mean_vec).T.dot((normalizeFaceVector - mean_vec)) / (normalizeFaceVector.shape[0]-1)


	#--------eigen values and faces-----------
	eigenValue,eigenVector = LA.eig(covarianceMatrix)

	#------ Make a list of (eigenValue, eigenVector) tuples--------
	eig_pairs = [(np.abs(eigenValue[i]), eigenVector[:,i]) for i in range(len(eigenValue))]

	# Sort the (eigenValue, eigenVector) tuples from high to low
	eig_pairs.sort(key=lambda x: x[0], reverse=True) 


	#-----select k eigen faces such that k<m i.e k<150 and can represent the whole training set
	total = sum(eigenValue)
	# The explained variance tells us how much information (variance) can be attributed to each of the principal components.
	var_exp = [(i / total)*100 for i in sorted(eigenValue, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	

	return cum_var_exp, eig_pairs,normalizeFaceVector