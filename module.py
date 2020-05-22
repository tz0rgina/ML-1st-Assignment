
import numpy as np
import pandas as pd
from csv import reader
import matplotlib.pyplot as plt

def Diff(list1, list2):
    return [x1 - x2 for (x1, x2) in zip(list1, list2)]

#create noise of the training set
def createNoise(mean, variance, N):
    noiseMean=mean
    noiseVariance=variance
    noise= np.random.normal(noiseMean, noiseVariance, N)
    return noise

def leastSquares(X,y):
    Xt = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]  
    from numpy.linalg import inv
    a=np.linalg.inv(np.matmul(Xt, X))
    b=np.matmul(a, Xt)
    tE=np.matmul(b, y)
    return tE   

def calculateMSE(yT,yF,N):
    diff= Diff(yF,yT)
    error=[]
    for i in range (0,len(diff)):
        error.append(diff[i]** 2)
    mse=sum(error)/N
    return mse 

def column(matrix, i):
    return [row[i] for row in matrix]

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def find_classes(d):
    #print(np.unique(d).tolist())
    return np.unique(d).tolist()
           
def separate_data(data,classes):
    matrix=[]
    for i in classes:
        matrix.append([])
    for j in range(0, len(data)):
        last_col=(column(data,-1))
        class_Of_Row=classes.index(last_col[j])
        matrix[class_Of_Row].append(data[j][0 : len(data[0]) - 1])
    #print(matrix)
    return matrix

def load_data(name):  
    data=load_csv(name)
    for i in range(len(data[0])-1):
        str_column_to_float(data, i) 
    return data

#accuracy = (correctly predicted class / total testing class) × 100%
def calculate_accuracy(data,function, k, step, isNaive, x_axis, lab , f):
    accuracy=[]
    k_counter=[]
    for i in range (1,k,step):
        r, w=function(k,data,isNaive)
        accuracy.append(r)
        k_counter.append(i)
    #print(accuracy)
    m = max(accuracy)
    index=[i for i, j in enumerate(accuracy) if j == m]
    _=plt.figure(f)
    _=plt.axis([k_counter[0], k_counter[-1], min(accuracy)-10,100]) 
    _=plt.xlabel(x_axis)
    _=plt.ylabel('Accuracy (%)')
    _=plt.plot(k_counter, accuracy, label=lab)
    #print(accuracy[index[0]])
 
    return k_counter[index[0]], accuracy[index[0]]


