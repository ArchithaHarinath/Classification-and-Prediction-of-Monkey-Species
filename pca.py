import os
import glob
import math
import numpy as np
import cv2
import operator

#Read image data from the directories
def dataread(path,data):
	for file in glob.glob(path):
		a=cv2.imread(file,0)	
		re=cv2.resize(a,(70,70))
		data.append(re)
	return data	

#Calculate cartesian distance for each data  	
def cartesiandistance(instance1, instance2, length):
	distance=0
	for x in range(length):
		distance+=pow((instance1[x]-instance2[x]), 2)
	return math.sqrt(distance)	

#The prediction 	
def classmajority(neighbours):
	classcount={}
	for x in range(len(neighbours)):
		temp=neighbours[x][0]
		if temp in classcount:
			classcount[temp]+= 1
		else:
			classcount[temp]= 1
	temp=classcount.items()
	temp.sort(key=operator.itemgetter(1),reverse=True)
	return temp

def main():

	#Data path for reading training data
	data=[]
	path = "D:/ms sem 2/ml/final_project/mushrooms/train/agaricus/*jpg"
	data=dataread(path,data)
	path = "D:/ms sem 2/ml/final_project/mushrooms/train/amantia/*jpg"
	data=dataread(path,data)
	path = "D:/ms sem 2/ml/final_project/mushrooms/train/entoloma/*jpg"
	data=dataread(path,data)
	path = "D:/ms sem 2/ml/final_project/mushrooms/train/suillus/*jpg"
	data=dataread(path,data)
	
	#flattening the data read from the directories
	train=[]
	for i in range(len(data)):
		train.append(data[i].flatten())
	
	#List of Class names for the training data
	cl_names=['agaricus','amantia','entoloma','suillus']
	
	#PCA Algorithm
	arr=np.array(train)
	#mean of the image data
	mean_v=np.mean(arr.T,axis=1)
	#mean corrected matrix data
	arr=arr-mean_v
	#Covariance of the mean corrected array
	Cov=np.dot(arr.T,arr)/len(data)
	Cov=np.array(Cov)
	#Eigen values and eigen vectors of the covariance matrix
	e,u=np.linalg.eigh(Cov)
	#Eigen value pairs
	eig_pairs = [(np.abs(e[i]), u[:,i]) for i in range(len(u))]
	#Choosing the eigen vectors with high values for a better fit
	eig_pairs.sort()
	eig_pairs.reverse()
	
	#choosing the particular number of feature vectors
	feature=[]
	for i in range(20):
		feature.append(eig_pairs[i][1])
	feature=np.array(feature)
	#projecting the data onto a reduced dimentions
	project=np.dot(arr,feature.T)

	#Data path for reading the test data
	test=[]
	path = "D:/ms sem 2/ml/final_project/mushrooms/test/agaricus/*jpg"
	test=dataread(path,test)
	path = "D:/ms sem 2/ml/final_project/mushrooms/test/amantia/*jpg"
	test=dataread(path,test)
	path = "D:/ms sem 2/ml/final_project/mushrooms/test/entoloma/*jpg"
	test=dataread(path,test)
	path = "D:/ms sem 2/ml/final_project/mushrooms/test/suillus/*jpg"
	test=dataread(path,test)

	#prepare the test set data
	test_v=[]
	#flatten the test data
	for i in range(len(test)):
		test_v.append(test[i].flatten())
	arr_v=np.array(test_v)
	#mean of the test data
	mean_v=np.mean(arr_v.T,axis=1)
	#Mean corrected test data set
	arr_v=arr_v-mean_v
	arr_v=np.array(arr_v)
	#The final test data for passing it to K-Nearest neighbors for calculating the nearest neighbours
	test_data=np.dot(arr_v,feature.T)
	
	#KNN Algorithm
	distances=[]
	length=len(test_data[0])-1
	temp=[]
	result=[]
	for i in range(len(test_data)):
		dista=[]
		neighbours=[]
		for x in range(len(data)):
			#Calculate the cartesian distances
			dist=cartesiandistance(test_data[i],project[x],length)
			#Append the class labels with the corresponding test set cartesian distance
			if(x>=0 and x<=99):
				temp.append(cl_names[0])
			if(x>=100 and x<=199):
				temp.append(cl_names[1])
			if(x>=200 and x<=299):
				temp.append(cl_names[2])
			if(x>=300 and x<=399):
				temp.append(cl_names[3])
			temp.append(dist)
			dista.append(temp)
			temp=[]
			#sort the cartesain distance in ascending order
			dista.sort(key=operator.itemgetter(1))
		
		#storing the number of wanted sorted distances
		for i in range(5):
			neighbours.append(dista[i])
		#Predicting the result for the test data
		temp=classmajority(neighbours)
		result.append(temp[0][0])
	
	#original Result Set
	original_res=[]
	for x in range(len(test_data)):
		if(x>=0 and x<=24):
			original_res.append(cl_names[0])
		if(x>=25 and x<=49):
			original_res.append(cl_names[1])
		if(x>=50 and x<=74):
			original_res.append(cl_names[2])
		if(x>=75 and x<=99):
			original_res.append(cl_names[3])
	
	#Accuracy of the predictions
	count=0
	for i in range(len(original_res)):
		if(original_res[i]==result[i]):
			count+=1
	acc=float(count)/float(len(result))
	
	print ("The accuracy of the predictions "+str(acc*100)+ "%")
	
main()	