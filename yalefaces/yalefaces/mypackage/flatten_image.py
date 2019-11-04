import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def flatten_image(imageName):
	img = mpimg.imread(imageName)
	gray = rgb2gray(img)
	fimage = imresize(double(gray),[220,233])
	[testm,testn] = size(fimage)
    fimage = reshape(fimage,[1,testm*testn])
	fimage = imageName.resize(1,51260)
	