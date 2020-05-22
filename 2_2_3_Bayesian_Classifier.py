# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:25:06 2019
[
"""


import pandas as pd
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from random import randrange
import math
from math import sqrt
from math import pi
from math import exp

def load_data(name):  
   data=load_csv(name)
   for i in range(len(data[0])-1):
      str_column_to_float(data, i) 
   return data

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
   
# Calculates v1' * M * v2
def multiplication(v1, M, v2):     
    return np.matmul(np.matmul(np.transpose(v1), np.linalg.inv(M)), v2)

#Returns the column i of a matrix as a list
def column(matrix, i):
    return [row[i] for row in matrix]

def multiplyList(myList) : 
    result = 1
    for x in myList: 
         result = result * x  
    return result

def Diff(list1, list2):
    return [x1 - x2 for (x1, x2) in zip(list1, list2)]
def find_classes(d):
    #print(np.unique(d).tolist())
    return np.unique(d).tolist()

def class_parameters(matrix, N, isNaive):
    #Calculate prior probability
    prior=len(matrix)/N
        
    # Gaussian distribution parameters     
    # Calculate the mean
    m = []
    for i in range(0, len(matrix[0])):        
        m.append(np.mean(column(matrix, i)))
    #print(m)
    # Calculate the covariance matrix 
    S=[]
    if (isNaive):
        for i in range(0, len(matrix[0])): 
            S.append(np.std(column(matrix, i), axis = 0, ddof=1))
            #covariance= variance*np.identity(len(variance))
    else:
        S= np.cov(matrix, rowvar=False)
            
    return  prior, m, S

def separate_data(data,classes):
    matrix=[]
    for i in classes:
        matrix.append([])
    for j in range(0, len(data)):
        last_col=(column(data,-1))
        class_Of_Row=classes.index(last_col[j])
        matrix[class_Of_Row].append(data[j][0 : len(data[0]) - 1])
    return matrix

def preparing_data(data, isNaive): 
    classes=find_classes(column(data,-1))
    groupByClass=separate_data(data,classes)
   # print(groupByClass)
    
    #define list of the priof probabilities of each class 
    prior=[]
    #define list of means and covariance matrices of each class 
    m=[]
    S=[]
    for i in range(0,len(classes)):
        prior_temp, m_temp, S_temp=class_parameters(groupByClass[i], len(data), isNaive)
        prior.append(prior_temp)
        m.append(m_temp)
        S.append(S_temp)
    return classes, prior, m, S

# Calculates the Gaussian probability distribution function for x
def  GaussianProbabilityDensity(x, m, s, isNaive):
    #UniVariate Gaussian
    if (isNaive):
        p=(1 / (sqrt(2 * pi) * s)) * exp(-((x-m)**2 / (2 * s**2 )))
    else:
        #Multivariate Gaussian
        p=(1 / (sqrt((2 * pi)**len(m) * np.linalg.det(s)))) * exp(-((multiplication(Diff(x,m), s, Diff(x,m)) / 2)))
    return p

#Calculates the Likelihood using the product rule if Naive Classification is required
def probability(x,m, s, feature_matrix, isNaive):
    p=[]
    if(isNaive):
        p_i=[]
        for i in range(0, len(x)):
            p_i.append(GaussianProbabilityDensity(x[i],m[i],s[i],isNaive))
            p=multiplyList(p_i)
    else:
        p=GaussianProbabilityDensity(x, m, s, isNaive)
    return p

"""
//Calculates the posterior probabilities of each class given the priors.
//Returns the class of an unknown vector x according to the rule
Assign x to wi : argMax P(wi|x), j=1,2,..,M 
"""
def discrimination(classes, prior, m, s, x, isNaive):
    marginal=[]
    posterior=[]
    for i in range(0, len(classes)):
        marginal.append(probability(x, m[i], s[i], classes[i], isNaive))
        posterior.append(marginal[i]*prior[i])
    m = max(posterior)
    index=[i for i, j in enumerate(posterior) if j == m]
    #print(classes[index[0]])  
    return classes[index[0]]

"""
//Returns the training set splited into k-folds
//Each folds contains the required number of training points randomly selected 
from the initial training set.
"""
def K_fold(k, data):
    fold_size = int(len(data) / k)
    k_folds=[]
    training=data.copy()
    for i in range (0 , fold_size):
        fold=[]
        while(len(fold)<k):
            index=randrange(len(training))
            fold.append(training.pop(index))
        k_folds.append(fold)
    return k_folds

#Merges the (k-1)folds as the training set
def trainingSet(k_1folds):
    training=[]
    for i in range (0, len(k_1folds)):
        fold=k_1folds[i]
        for j in range(0, len(k_1folds[i])):
            training.append(fold[j])
    return training

"""
//Makes Bayes Classification (Naive/Simple) via k-fold Cross Validation.
//Returns the percentage of correctly and incorrectly predicted class.
--------------------------------------------
|Classification           | Simple | Naive |
--------------------------------------------
|Value of Argument isNaive| False  |  True |
--------------------------------------------
"""
def BayesClassifier(k,data,isNaive):
    right_f=[]
    wrong_f=[]
    k_folds=K_fold(k, data)
    #print(k_folds)
    for fold in k_folds:
        #print(fold)
        pre_training=list(k_folds)
        #print(pre_training)
        pre_training.remove(fold)
        training=trainingSet(pre_training)
        #print(training)
        classes, prior, m, S=preparing_data(training, isNaive)
        #print(classes)
        right=0
        wrong=0
        counter=0
        for j in range (0, len(fold)):
            #print(fold[j][0 : len(fold[j])-1])
            if (isNaive):
                result=discrimination(classes, prior, m, S, fold[j][0 : len(fold[j])-1], isNaive)
            else:
                result=discrimination(classes, prior, m, S, fold[j][0 : len(fold[j])-1], isNaive)
                #result=discrimination_function(classes, prior, m, S, fold[j][0 : len(fold[j])-1])
            counter+=1
            if (result==fold[j][-1]):
                right+=1
            else: 
                wrong+=1
        right_f.append(right/counter)
        wrong_f.append(wrong/counter)
    percentage_right=(sum(right_f)/len(right_f))*100
    percentage_wrong=(sum(wrong_f)/len(wrong_f))*100
    
    return percentage_right, percentage_wrong

#Plots results of different method in the same graph 
def plot_results(title, labels, groupLabels, group1, group2):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x - width/2, group1, width, label= groupLabels[0])
    rects2 = ax.bar(x + width/2, group2, width, label= groupLabels[1])
    
    ax.set_ylabel('Frequency (%)')
    ax.set_title(title , fontstyle='italic')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    
    def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    
    plt.show()
    
"""
//Bayes Classification with k-Fold Cross Validation
with Simple and Naive Bayes Classifier.
//Compares the resuls plot them in the same graph.
"""
def BayesClassification(k,data, title):
    r, w=BayesClassifier(k,data, False)
    r2, w2=BayesClassifier(k,data, True)
    labels = ['Simple Bayes','Naive']
    groupLabels= ['Right', 'Wrong']
    right_guesses=[float("%.2f" % r),float("%.2f" % r2)]
    wrong_guesses=[float("%.2f" % w),float("%.2f" % w2)]
    title="Classification for " + title + "\n via "+str(k)+"-Fold Cross Validation"
    plot_results(title, labels,  groupLabels, right_guesses, wrong_guesses)
   
#Calculates mean absolute deviation
def calculate_mean_absolute_deviation(mean, x):
    sum=0
    for i in range (0, len(x)):
        sum+=abs(x[i]-mean)
    return sum/len(x)

"""
//Returns the first element of both lists of k-values with the best accuracy and accuracy.
//accuracy = (correctly predicted class / total testing class) × 100%
"""
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
 
    return k_counter[index[0]], accuracy[index[0]], accuracy

"""
--------------------------------------------------------------------------------------
Runs iterated Cross Validation experiments using different values of k via Bayesian 
Classification (Simple and Naive)
--------------------------------------------------------------------------------------.
//Computes the accuracy of each experiment.
//Plots the acuuracy as a function of k values.
//Prints the mean of the accuracy list for each type
of Bayesian Classification.
//Prints standard and absolute deviation of the accuracy list for each type
of Bayesian Classification as presicion indicators.
//Plots the percentages at the same graph. 
"""
def plot_accuracy(data, max_k, step,label):
    index, b, accuraces=calculate_accuracy(data ,BayesClassifier, max_k ,step, False, "k-folds", 'Simple Bayes', 0)
    mean=np.mean(accuraces)
    std=np.std(accuraces)
    abs_std=calculate_mean_absolute_deviation(mean,accuraces)
    print ([mean,std,abs_std])
    print()
    index2, b2, accuraces2=calculate_accuracy(pima,BayesClassifier, max_k ,step,  True, "k-folds", "Naive Bayes" , 0)
    mean2=np.mean(accuraces2)
    std2=np.std(accuraces2)
    abs_std2=calculate_mean_absolute_deviation(mean,accuraces2)
    print ([mean2,std2,abs_std2])
    print()
    labels = ['Simple Bayes','Naive']
    groupLabels= ['Right', 'Wrong']
    right_guesses=[float("%.2f" % b),float("%.2f" % b2)]
    wrong_guesses=[float("%.2f" % (100-b)),float("%.2f" % (100-b2))]
    title="Classification for "+label+ "\n via "+str(index)+"/"+str(index2)+"-Fold Cross Validation"
    plot_results(title, labels,  groupLabels, right_guesses, wrong_guesses)

    
pima=load_data("pima-indians-diabetes.csv")
iris=load_data("iris.csv") 
data=[pima,iris]
max_k=[300,40]
step=[10,5]
label=["Pima", "Iris"]
for i in range (0,2):
    plot_accuracy(data[i], max_k[i], step[i],label[i] )




