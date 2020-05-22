# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:14:10 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import Polynomial as pl
import module as mod
from numpy.linalg import inv

class experiment:
    def __init__(exp,mean, variance, N, xtrue, ytrue, gradeOfReg):
        exp.noise=mod.createNoise(mean,variance,N)
        exp.y_training=ytrue+exp.noise
        exp.regressionModel=pl.Polynomial(gradeOfReg)
        exp.regressionModel.create_X_matrix(xtrue)
        exp.theta=mod.leastSquares(exp.regressionModel.X, exp.y_training)

thetaZero=[0.2, -1, 0.9, 0.7, 0, -0.2]
N=20
    
trueModel=pl.Polynomial(5)
    
#create the N true points   
x_true=np.arange(0, 2, 2/float(N))
y_true=trueModel.produce_set(x_true,thetaZero) 
    
Y2=[]
for i in range(100):
    experiment_2nd_degree=experiment(0,0.1,20,x_true,y_true,2)
    y_training2=experiment_2nd_degree.y_training
    theta2=experiment_2nd_degree.theta
    yFitted=experiment_2nd_degree.regressionModel.produce_set(x_true,theta2)
    Y2.append(yFitted)
means2= np.mean(Y2, axis=0)
variance2= np.std(Y2, axis=0)
#print(variance2)
        
#Note
_=plt.figure(1)
_=plt.style.use('seaborn-whitegrid')
_=plt.errorbar(x_true, means2, yerr=variance2,  ecolor='grey',fmt='.k', color='black',label='Fitted data');
_=trueModel.graph(thetaZero,"True Model",0,2)
_=plt.title('Exersice 1.2- Fitting with 2nd degree polynomial')
_= plt.legend(fontsize='small')

#Note
Y10=[]
for i in range(100):
    experiment_10th_degree=experiment(0,0.1,20,x_true,y_true,10)
    y_training10=experiment_10th_degree.y_training
    theta10=experiment_10th_degree.theta
    yFitted10=experiment_10th_degree.regressionModel.produce_set(x_true,theta10)
    Y10.append(yFitted10)
means10= np.mean(Y10, axis=0)
variance10= np.std(Y10, axis=0)
print(variance10)  
    
#Note
_=plt.figure(2)
_=plt.style.use('seaborn-whitegrid')
_=plt.errorbar(x_true, means10, yerr=variance10, fmt='.k', color='black', ecolor='grey',label='Fitted data');
_=trueModel.graph(thetaZero,"True Model",0,2)
_=plt.title('Exersice 1.2 - Fitting with 10th degree polynomial')
_= plt.legend(fontsize='small')
_=plt.show()
    
  
    