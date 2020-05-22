
import numpy as np
import matplotlib.pyplot as plt
import Polynomial as pl
import module as mod
from numpy.linalg import inv
import pickle
 
def leastSquares(X,y):
    Xt = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]  
    from numpy.linalg import inv
    a=np.linalg.inv(np.matmul(Xt, X))
    b=np.matmul(a, Xt)
    tE=np.matmul(b, y)
    return tE  

def Diff(list1, list2):
    return [x1 - x2 for (x1, x2) in zip(list1, list2)]
#need generalization
    
thetaZero=[0.2, -1, 0.9, 0.7, 0, -0.2]
N=20
    
trueModel=pl.Polynomial(5)
    
#create the N true points   
x_true=np.arange(0, 2, 2/float(N))
y_true=trueModel.produce_set(x_true,thetaZero) 
    
#create training set of N points
noise=mod.createNoise(0,0.1,20)
y_training=y_true+noise

theta=leastSquares(trueModel.X, y_training)
print(theta)
    
#create test set
xTest=(np.random.uniform(0.0, 2.0, 1000)).tolist()
testModel=pl.Polynomial(5)
ytest=testModel.produce_set(xTest,thetaZero)
noiseT=mod.createNoise(0, 0.1,1000)
ytest=ytest+noiseT 
#ytest=ytest
    
#create y from regression model for test set
predictedModel=pl.Polynomial(5)
ypre=predictedModel.produce_set(xTest,theta)
    
#create y from regression model for training set
predictedModelT=pl.Polynomial(5)
ypreT=predictedModelT.produce_set(x_true,theta)
        
MSEtraining=mod.calculateMSE(y_training,ypreT,20)
MSEtest=mod.calculateMSE(ytest,ypre,1000)

#print( MSEtraining)
#print( MSEtest)
    
    
#Plot the training set of 20 points and the curves
_=plt.figure(0)
_=plt.plot(0,label='MSEtraining='+str("%.3f" % MSEtraining), color='white')
_=plt.plot(0,label='MSEtest='+str("%.3f" % MSEtest), color='white')
_=plt.plot(x_true, y_training, 'o', label='Training Set')
_=trueModel.graph(thetaZero,"True Model",0,2)
_=pl.Polynomial(len(theta)-1).graph(theta,"Fitted Model",0,2)
_=plt.title('Least Square Method over 20 points training set')
_= plt.legend(fontsize='small')
_=plt.show()




               
               
               
               