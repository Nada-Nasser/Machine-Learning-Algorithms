# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 01:40:15 2020

@author: Toshiba
"""

import numpy as np


def compute_cost(x_vectors, y_vector, theta):
    diff = (np.power(((x_vectors * theta.T) - y_vector) , 2))
    cost = (np.sum(diff) /(len(x_vectors)))
    return cost

def gradientDescent(X_matrix, Y_matrix, theta, alpha, n_iterations):
    
    

    m = len(X_matrix) # m = size of the training data
    new_theta = np.matrix(np.zeros(theta.shape)) # store new value of theta 
                                                 #for each iteration, to be able 
                                                 #to change all theta values with
                                                 # the old theta values
    n_theta = int(theta.ravel().shape[1])
    cost = np.zeros(n_iterations) # list to store the cost in every iteration, 
                                  #initialize it with zeros
  
    for iteration in range (n_iterations): # start the algorithm
        
        inner_diff = ((X_matrix * theta.T) - Y_matrix) # calculate the inner difference using 
                                         #the current theta values 
        
        
        
        for j in range(n_theta):# for each theta j, update its value
            inner_sum =np.sum(np.multiply(inner_diff , X_matrix[:,j]))
            new_theta[0,j] = theta[0,j] - ((alpha / m) * (inner_sum))
        
        theta = new_theta
        cost[iteration] = compute_cost(X_matrix,Y_matrix,theta) #
        
        if(iteration-1 >= 0):
            if(cost[iteration] > cost[iteration-1]):
                print(cost[iteration])
    test = (theta * X_matrix.T)
    print(test)
#    test = (X_matrix * theta.T)
#    print(test)
#    
    print(cost[n_iterations-1])        
    return theta,cost



def hypothesis(x , theta):
    return (theta * x.T)


def new_compute_cost(x, y, theta):
    n = len(x)
    diff = 0

    for i in range(n):
        hxi = hypothesis(x[i] , theta)
        diff += np.power((hxi - y[i]),2)    
    return diff/n




def new_gradientDescent(X_matrix, Y_matrix, theta, alpha, n_iterations):

    m = len(X_matrix) # m = size of the training data

    new_theta = np.matrix(np.zeros(theta.shape)) # store new value of theta 
                                                 #for each iteration, to be able 
                                                 #to change all theta values with
                                                 # the old theta values
    n_theta = int(theta.ravel().shape[1])
    cost = np.zeros(n_iterations) # list to store the cost in every iteration, 
                                  #initialize it with zeros
    
    for iteration in range(n_iterations):
        hxi =  hypothesis(X_matrix , theta)
        diff = (hxi - Y_matrix)
        for j in range(n_theta):    
            inner_term = diff * X_matrix[:,j]
#            for i in range(m):
#                hxi =  hypothesis(X_matrix[i] , theta)
#                inner_term = (hxi - Y_matrix[i]) * X_matrix[i,j]
#            
            new_theta[0,j] = theta[0,j] - ((alpha/m) * sum(inner_term))
            
        theta = new_theta
        cost[iteration] = new_compute_cost(X_matrix ,Y_matrix,theta)
        print(cost[iteration])
        
    return theta,cost

