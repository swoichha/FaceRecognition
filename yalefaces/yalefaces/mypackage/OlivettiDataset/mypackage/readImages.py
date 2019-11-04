# import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


global directory



directory = ""
imageArray = np.zeros((150,51260),dtype=np.uint8, order='C')  #Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.
    

def readFiles():
  imageCount = 0
  DATADIR = 'C:\\Users\\swoichha\\Documents\\GitHub\\FaceRecognition\\yalefaces\\yalefaces\\dataset'
  CATEGORIES = ['subject01','subject02','subject03','subject04','subject05','subject06','subject07','subject08','subject09','subject10','subject11','subject12','subject13','subject14','subject15']
  
  global imageArray
  
  for category in CATEGORIES: 
    path = os.path.join(DATADIR,category)         #-------path to all directory of images of diff subject
    for img in os.listdir(path):            #-------iterate through all those images
      directory = (os.path.join(path,img))
      currentImageArray = mpimg.imread(directory)
      currentImageArray.resize(1,51260)       
      imageArray[imageCount] = np.array(currentImageArray)
      imageCount += 1
  return imageArray, imageCount 

# imageArray, imageCount  = readFiles()
# print(imageArray.shape)
# print(imageCount)