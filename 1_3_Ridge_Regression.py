# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:19:50 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import Polynomial as pl
import module as mod
from numpy.linalg import inv

def RidgeRegression(X,y,l, thetaZero, N, noiseVar): 
    a=np.linalg.inv(np.matmul(np.transpose(X), X)+l*np.identity(len(X[0])))
    b=np.matmul(a, np.transpose(X))
    theta=np.matmul(b, y)
    return theta             
  
def experimentWithRidgeRegression(lamda,plotCounter,thetaZero,N,noiseVar):
    trueModel=pl.Polynomial(len(thetaZero)-1)
    
    #create the N true points   
    x_true=np.arange(0, 2, 2/float(N))
    y_true=trueModel.produce_set(x_true,thetaZero) 
    
    #create training set of N points
    noise=mod.createNoise(0,noiseVar,N)
    y_training=y_true+noise

    theta=RidgeRegression(trueModel.X, y_training,lamda,thetaZero,N,noiseVar)

    #create test set
    xTest=(np.random.uniform(0.0, 2.0, 1000)).tolist()
    #print(xTest)
    testModel=pl.Polynomial(len(thetaZero)-1)
    ytest=testModel.produce_set(xTest,thetaZero)
    noiseT=mod.createNoise(0, 0.1,1000)
    ytest=ytest+noiseT 
    
    #create y from regression model for test set
    predictedModel=pl.Polynomial(len(thetaZero)-1)
    ypre=predictedModel.produce_set(xTest,theta)
    
    #create y from regression model for training set
    predictedModelT=pl.Polynomial(len(thetaZero)-1)
    ypreT=predictedModelT.produce_set(x_true,theta)
        
    MSEtraining=mod.calculateMSE(y_training,ypreT,20)
    MSEtest=mod.calculateMSE(ytest,ypre,1000)

    #print( MSEtraining)
    #print( MSEtest)
    
    #Plot the training set of 20 points and the curves
    _=plt.figure(plotCounter)
    _=plt.plot(0,label='MSEtraining='+str("%.5f" % MSEtraining), color='white')
    _=plt.plot(0,label='MSEtest='+str("%.5f" % MSEtest), color='white')
    _=plt.xlabel('x')
    _=plt.ylabel('y')
    _=plt.plot(x_true, y_training, 'o', label='Training Set')
    _=trueModel.graph(thetaZero,"True Model",0,2)
    _=pl.Polynomial(len(thetaZero)-1).graph(theta,"Fitted Model",0,2)
    _=plt.title('Ridge Regression over 20 points training set with λ = ' + str(lamda))
    _= plt.legend(fontsize='small')
    _=plt.show()
    
L=[0.000001, 0, 0.001, 0.003, 0.0035, 0.0025, 0.0055, 0.1, 0.3 , 1.0, 10.0, 100.0, 1000]
plotCounter=0
thetaZero=[0.2, -1, 0.9, 0.7, 0, -0.2]
N=20
noiseVar=0.1
for lamda in L:
    experimentWithRidgeRegression(lamda,plotCounter,thetaZero,N, noiseVar)
    plotCounter=plotCounter+1