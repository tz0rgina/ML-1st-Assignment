# -*- coding: utf-8 -*-
"""
Last Modified on Mon Dec  2 23:10:08 2019

@author: Georgia Paraskevopoulou
"""
import numpy as np
import random
import Polynomial as pl
import matplotlib.pyplot as plt

#Returns a list in which elements are the difference of the elements in list1 and list2 respectively.
def Diff(list1, list2):
    return [x1 - x2 for (x1, x2) in zip(list1, list2)]

#create noise of the training set
def createNoise(mean, variance, N):
    noiseMean=mean
    noiseVariance=variance
    noise= np.random.normal(noiseMean, noiseVariance, N)
    return noise

#basic calculation (aI+bΦ'Φ)^-1
def calculation(a, b, X):
    Xt_X = np.matmul(np.transpose(X), X)
    return np.linalg.inv(a*np.identity(len(Xt_X[0]))+b*Xt_X)

#Calculate  parameters of posterior distribution p(θ|y)
#Returns only the mean which we need for calculation of distribution of pridictive distribution
def  posterior_distribution(thetaZero,thetaVar,noiseVar, X, y):
    covariance_matrix=calculation(1/thetaVar,1/noiseVar, X)
    posterior_mean=thetaZero+(1/noiseVar)*np.matmul(np.matmul(covariance_matrix, np.transpose(X)),Diff(y,np.matmul(X, thetaZero)))
    return posterior_mean

#Calculate parameters of predictive distribution
def  predictive_distribution(posterior_mean, thetaVar, noiseVar, X, Xtraining ):
    predictive_mean=[]
    predictive_variance=[]
    for i in range(0, len(X)):
        predictive_mean.append(np.matmul(np.transpose(X[i]),posterior_mean))
        product=np.matmul(np.transpose(X[i]),np.matmul(calculation(noiseVar,thetaVar,Xtraining), X[i]))
        predictive_variance.append(noiseVar+noiseVar*thetaVar*product)
    return predictive_mean,predictive_variance

"""
Runs an experiment with Bayesian Inference.
"""
def experiment_bayesian_inference(N, N_test, thetaVar, noiseVar, thetaTrue, thetaZero, trainingModel, y_training , experimentCounter, exercise):
 
    x_vector=xTest=(np.random.uniform(0.0, 2.0, N_test)).tolist()
    #print(x_vector)
    
    predictiveModel=pl.Polynomial(len(thetaZero)-1)
    predictiveModel.create_X_matrix(x_vector) 
    X_pred= predictiveModel.X
   # print(X_pred)
    posterior_mean= posterior_distribution(thetaZero,  thetaVar, noiseVar, trainingModel.X, y_training) 
    #print( "posterior_mean=" + str(posterior_mean))
    
    predMean, predVar=predictive_distribution(posterior_mean, thetaVar, noiseVar, X_pred, trainingModel.X)
    #predMean, predVar=predictive_distribution(posterior_mean, thetaVar, noiseVar, X_pred, X_pred)
    #print(predMean)
    
    #calculate errors
    errors=[]
    for i in predVar:
        errors.append(np.sqrt(i)/np.sqrt(N_test-1))
        #errors.append(np.sqrt(i))
    #print("errors = " + str(errors))

    _=plt.figure(experimentCounter)
    _=plt.title('Exercise 1.' + str(exercise) +' - N = ' + str (N) + '\n Noise Variance ='+str("%.3f" % noiseVar) + ' ,Theta Variance ='+str("%.3f" % thetaVar) )
    _=plt.xlabel('x')
    _=plt.ylabel('y')
    _=trainingModel.graph(thetaTrue,"True Model",0,2)
    _=plt.errorbar(x_vector, predMean, yerr=errors, ecolor='grey',fmt='.k', color='black',label='Predicted Points')
    _=plt.legend(fontsize='small')
    _=plt.show()
 
def exercise1_4(N,Ntest):
    thetaTrue=[0.2, -1, 0.9, 0.7, 0, -0.2]
    noiseVar_vector=[0.05,0.15]
    experimentCounter=0
    thetaZero=[0.2, -1, 0.9, 0.7, 0, -0.2]
    
    
    for i in noiseVar_vector:
        #Traininig Set
        x_training=np.arange(0, 2, 2/float(N))
        trainingModel=pl.Polynomial(len(thetaTrue)-1)
        y_true=trainingModel.produce_set(x_training,thetaTrue) 
        noise=createNoise(0,i, N)
        y_training=y_true+noise
        experimentCounter=experimentCounter+1
        experiment_bayesian_inference(N, Ntest, 0.1, i, thetaTrue, thetaZero,trainingModel, y_training, experimentCounter, 4)

def exercise1_5():
    thetaTrue=[0.2, -1, 0.9, 0.7, 0, -0.2]
    noiseVar=0.05
    thetaZero=[-10.54, -0.465, 0.0087, -0.093, 0,  -0.004]
    N_vector=[20, 500]
    thetaVar_vector=[0.1, 2]
    experimentCounter=0
    
    for i in N_vector:
        #Traininig Set
        x_training=np.arange(0, 2, 2/float(i))
        trainingModel=pl.Polynomial(len(thetaTrue)-1)
        y_true=trainingModel.produce_set(x_training,thetaTrue) 
        noise=createNoise(0, noiseVar, i)
        y_training=y_true+noise
        for j in thetaVar_vector:
            experimentCounter=experimentCounter+1
            experiment_bayesian_inference(i, 20, j, noiseVar, thetaTrue, thetaZero, trainingModel, y_training, experimentCounter, 5)

exercise1_5()
#exercise1_4(20,20)