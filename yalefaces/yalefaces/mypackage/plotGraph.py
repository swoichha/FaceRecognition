# importing the required module 
import matplotlib.pyplot as plt 
import numpy as np

from mainPCA import getVariance
def plotGraph():
	plt.subplot2grid((2,1),(0,0))
	
	plt.title('Accuracy')
	y = [6.666,13.33,33.33,60,73.33] 
	plt.plot(y) 
	plt.xlabel('DataSet Taken') 
	plt.ylabel('Error Rate') 
	plt.xticks(np.arange(0, 1, step=0.5))
	plt.xticks(np.arange(5), ('Normal face','Emotion','With glasses', 'Central ligthing','Right ligthing'))  

	plt.subplot2grid((2,1),(1,0))
	cum_var_exp = getVariance()		
	plt.figure(1, figsize=(12,8))
	plt.plot(cum_var_exp,linewidth=2)
	plt.xlabel('Components')
	plt.ylabel('Explained Variances')
	plt.show()
	

# def dataSetAccuraacy():	  
	
# 	# function to show the plot 
# 	plt.show() 
# def componentGraph():
# 	cum_var_exp = getVariance()		
# 	plt.figure(1, figsize=(12,8))
# 	plt.plot(cum_var_exp,linewidth=2)
# 	plt.xlabel('COMPONENTS')
# 	plt.ylabel('EXPLAINED VARIANCES')
# 	plt.show()
