# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:31:45 2019

@author: user
"""

import numpy as np
import random
import Polynomial as pl
import matplotlib.pyplot as plt
import Bayesian_Inference_1_4_5 as BI

"""
Runs Expectation-Maximization Algorithm with arbitrary positive initial values a=b=1.
When the algorithm converges returns :
    ->the resulting values of a and b
    ->the estimated mean and covariance matrix of the posterior p(θ|y) 
"""
def EM_algorithm(X, y, dimensionsOfTheta, criterion): 
    
    a =[1]
    b =[1]    
    
    while True:
        
        #E - step 
        #Parameters of posterior distribution p(θ|y)
        covarianceTheta=BI.calculation(a[-1],b[-1],X)
        meanTheta = b[-1]*np.matmul(np.matmul(covarianceTheta, np.transpose(X)), y)
        
        A=(np.linalg.norm(meanTheta)**2 + np.trace(covarianceTheta))
        B=(np.linalg.norm(BI.Diff(y,np.matmul(X,meanTheta)))**2 + np.trace(np.matmul(np.matmul(X,covarianceTheta), np.transpose(X))))
        
        #M-step
        a.append(dimensionsOfTheta/A)
        N=len(y)
        b.append(N/B)
        if (abs(a[-1]-a[-2]) < criterion or abs(b[-1] - b[-2]) < criterion):
            break
    
    return covarianceTheta, meanTheta, a, b

"""
//Creates a training set.
//Runs the EM_Algorithm. 
//Calculates the Bayesian Inference predictions using the values 
of ση^2 and σθ^2 obtained via the M_Algorithm.
//Plots the origimal graph from which the training points were sampled, the respective 
predictions y^ and the associated error bars from 20 randomly choosen points.
The values for the error bars can be computed with 2 different ways. 
//Plots the convergence curve for ση^2 as a function of the iterations of the EM-Algorithm
"""
def LinearRegression_EM_Algorithm(thetaZero, noiseVar, N, N_test, criterion, exp_counter):
    
    x_training=np.arange(0, 2, 2/float(N))
    trainingModel=pl.Polynomial(len(thetaZero)-1)
    y_true=trainingModel.produce_set(x_training,thetaZero) 
    
    #create training set of N points
    noise=BI.createNoise(0, noiseVar, N)
    y_training=y_true+noise
    
    covarianceTheta, meanTheta, a, b=EM_algorithm(trainingModel.X, y_training, len(thetaZero), criterion)
    
    print()
    print("μ(θ|y) = " + str(meanTheta))
    print()
    
    x_vector=xTest=(np.random.uniform(0.0, 2.0, N_test)).tolist()
    predictiveModel=pl.Polynomial(len(thetaZero)-1)
    predictiveModel.create_X_matrix(x_vector) 
    X_pred= predictiveModel.X
    
    noiseVar_v=[]
    iteration=[]
    for i in range(0, len(b)):
        iteration.append(i)
        noiseVar_v.append(1/b[i])
    #print(a[-1])
    #print(b[-1])
    
    thetaVar=1/a[-1]
    predMean, predVar=BI.predictive_distribution(meanTheta, thetaVar, noiseVar_v[-1], predictiveModel.X, trainingModel.X )
    
    
    print("a = " + str(a))
    print()
    print("b = " + str(b))
    #thetaVar=1/a[-1]
    errors=[]
    for i in predVar:
        #errors.append(np.sqrt(i)/np.sqrt(N_test-1))
        errors.append(np.sqrt(i))
    #print("errors = " + str(errors))

    _=plt.figure(exp_counter)
    _=plt.title('Exercise 1.6 - N = ' + str (N) + '\n Noise Variance ='+str("%.3f" % noiseVar_v[-1]) + ' ,Theta Variance ='+str("%.3f" % thetaVar) )
    _=plt.plot(0,label='ε ='+str(criterion), color='white')
    _=plt.xlabel('x')
    _=plt.ylabel('y')
    _=trainingModel.graph(thetaZero,"True Model",0,2)
    _=plt.errorbar(x_vector, predMean, yerr=errors, ecolor='grey',fmt='.k', color='black',label='Predicted Points')
    _=plt.legend(fontsize='small')#xperiment_bayesian_inference(N, N_test, thetaVar, noiseVar_v[-1], thetaZero, meanTheta, trainingModel, y_training , exp_counter, 6)
    
    _=plt.figure(exp_counter+1)
    _=plt.title('Exercise 1.6 - Convergence curve')
    _=plt.plot(label='ε ='+str(criterion), color='white')
    _=plt.xlabel('Iterations')
    _=plt.ylabel('Noise Variance')
    _=plt.plot(iteration,noiseVar_v ) 
    _=plt.show()
    
 

#Example with the generalized Linear Regression Model of our exersice. 
def exercise1_6():
    thetaZero=[0.2, -1, 0.9, 0.7, 0, -0.2]
    N=500
    N_test=20
    noiseVar=0.05
    criterion=[0.00000000001]
    exp_counter=0
    for i in criterion:
        exp_counter=exp_counter+1
        LinearRegression_EM_Algorithm(thetaZero, noiseVar, N, N_test, i, exp_counter)
        exp_counter=exp_counter+1

exercise1_6()

    
    
    

