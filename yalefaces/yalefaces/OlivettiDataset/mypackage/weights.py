import numpy as np

def weight(k_eigenVector,normalizeFaceVector,test):
	img_count = 280
	if test == 0:
		
		k = len(k_eigenVector)
		wcoeff = np.zeros((280,k))
		# wcoeff = np.zeros((280,11),dtype=np.uint8, order='C')
		for i in range(img_count):
		  wcoeff[i] = np.array(k_eigenVector.dot(normalizeFaceVector[:,i]))
		
		return wcoeff	
	else:
		wcoeff = k_eigenVector.dot(normalizeFaceVector)
		return wcoeff



	
	
