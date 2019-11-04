import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import linalg as LA

from scipy.spatial import distance

from readImages import readFiles
from faceRecognition import faceRecog
from weights import weight 
from displayName import imgName

normalizeFaceVector = np.zeros((150,51260), order='C')

imageArray, imageCount  = readFiles()
def averageOfArray(x,total):
  x = np.sum(x, axis = 0)
  x = (x/total)
  return x

averageImage = averageOfArray(imageArray,imageCount)

cum_var_exp, eig_pairs,normalizeFaceVector = faceRecog(imageArray,imageCount,averageImage,normalizeFaceVector)

def getVariance():
  return cum_var_exp

def displayIndex(testImg):
  print('testImg',testImg)
    
  k=0
  for i in range(len(cum_var_exp)):
    if cum_var_exp[i] < 96:
      k = k+1

  matrix_w = np.zeros((k,150))
  #------storing lower dimensionality k eigenvectors into matrix_w-----------
  for i in range(k):
    matrix_w[i] = np.array(eig_pairs[i][1].reshape(1,150))

  matrix_wT = matrix_w.transpose()
  #-----------converting lower dimension k eigenvectors to original face dimensionality---------------------
  Y = normalizeFaceVector.dot(matrix_wT)
  k_eigenVector = np.transpose(Y)
  print(k_eigenVector.shape) 
  print(len(k_eigenVector))  

  weight_coeff = weight(k_eigenVector,normalizeFaceVector,0)
  # testImg = img_select()
  testImg.resize(1,51260)

  normalizeTestImg = testImg - averageImage
  normalizeTestImg = normalizeTestImg.transpose()
  testImgWeight = weight(k_eigenVector,normalizeTestImg,1)
  testImgWeightT = testImgWeight.transpose() 

  # Function for euclidean distance
  def euclidean_distance(v, u):
     return np.sqrt(np.sum((v - u) ** 2))
  def manhattan_distance(u,v):
    return distance.cityblock(u,v)
       
  dist = np.zeros((150,1))
  for i in range(imageCount):
    dist[i] = euclidean_distance(testImgWeightT, weight_coeff[i,:])
    # dist = LA.norm(testImgWeightT - weight_coeff[i,:])
    # dist[i] = distance.euclidean(testImgWeightT, weight_coeff[i,:])

  print(dist[2].shape)
  print(dist.shape)
  minval = np.amin(dist)
  print(minval)
  index = np.argmin(dist)
  print(index)
  
 
 
  result = np.array(np.reshape(imageArray[index],(233,220)))
  testImg_resized = np.reshape(testImg,(233,220))
  # cv2.imshow("Recognized Image", result)
  f = plt.figure()
  f.add_subplot(1,2, 1)
  matchedImg = imgName(index)
  plt.suptitle('Matched with ' + matchedImg)
  plt.imshow(result,cmap='gray')
  f.add_subplot(1,2, 2)
  plt.imshow(testImg_resized,cmap='gray')
  plt.show(block=True)
  
  return result
# print(result)
# plt.imshow(result,cmap='gray')  
# plt.show()
