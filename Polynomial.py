# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:10:19 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

class Polynomial:
    def __init__(pol,degree):
         pol.degree=degree
         pol.X=[]
         pol.plotX=[]
         pol.y_plot=[]
         
    def produce_x_vector(pol,x):
        switcher = {
            1: [1,x],
            2: [1, x, x**2],
            3: [1, x, x**2, x**3],
            4: [1, x, x**2, x**3, x**4],
            5: [1, x, x**2, x**3, x**4, x**5],
            6: [1, x, x**2, x**3, x**4, x**5, x**6],
            7: [1, x, x**2, x**3, x**4, x**5, x**6, x**7],
            8: [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8],
            9: [1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9],
            10:[1, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10],
            }
        return switcher.get(pol.degree)
    
    def produce_my_y(pol,x_vector,coef):
        y=np.inner(x_vector,coef)
        return y
    
    def create_X_matrix(pol,x_data):
        for i in range(0, len(x_data)):
            x_vector=pol.produce_x_vector(x_data[i])
            pol.X.append(x_vector)
    
    def produce_set(pol,x_data,coef):
        y_vector=[]
        for i in range(0, len(x_data)):
            x_vector=pol.produce_x_vector(x_data[i])
            pol.X.append(x_vector)
            y_vector.append(pol.produce_my_y(x_vector,coef))
        return y_vector
    
    def graph(pol, coef,lb, i, j ):
        x = np.linspace(i, j, 1000)
        pol.y_plot=[]
        for i in range(0, 1000):
            x_plot=pol.produce_x_vector(x[i])
            pol.plotX.append(x_plot)
            pol.y_plot.append(pol.produce_my_y(x_plot,coef))
        return plt.plot(x, pol.y_plot, label=lb, color='black')