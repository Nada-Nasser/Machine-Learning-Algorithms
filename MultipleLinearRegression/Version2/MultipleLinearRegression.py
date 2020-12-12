# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:48:30 2020

@author: Toshiba
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hypothesis(x , theta):
    return (theta * x)

def compute_cost(x, y, theta):
    y_hat = hypothesis(x , theta)
    y_hat = np.sum(y_hat, axis=1)
    diff = (np.power((y_hat - y) , 2))
    cost = (1/len(x) * sum(diff))
    return cost

def gradientDescent(x, y, theta, alpha, n_iterations):
    
    m = len(x) # m = size of the training data
    n_theta = 6
    cost = np.zeros(n_iterations) # list to store the cost in every iteration, 
                                  #initialize it with zeros # list to store the cost in every iteration, 
                                  #initialize it with zeros
    
    for iteration in range (n_iterations): # start the algorithm
        
        hx = (hypothesis(x , theta))
        hx = np.sum(hx, axis=1)

        for j in range(n_theta):# for each theta j, update its value            
            theta[j] = theta[j] - (alpha/m) * sum((hx - y)*x.iloc[:,j])
            
        cost[iteration] = (compute_cost(x,y,theta))
    
    return theta,cost



input_fields = ['grade', 'bathrooms', 'lat', 'sqft_living', 'view']
output_fields = ['price']

path = "house_data.csv"

file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields]
output_data = file_data[output_fields]

# rescaling data
input_data = (input_data - input_data.mean()) / input_data.std()

#input_data = (input_data )/ (input_data.max())

# add ones column
input_data.insert(0, 'Ones', 1)


# separate X (training data) from y (target variable)
training_data = input_data.iloc[:,:] # all rows x all columns expect the last one
target_variable = output_data.iloc[:,0] # all rows x last column

thetas = np.array([0]*len(training_data.columns))


#print(training_data.shape)
#print(target_variable.shape)
#print(thetas.shape)
#
print(training_data.head(10))
#print(target_variable.head(10))
#print(thetas)



alpha = 0.1
n_iterations = 100

optimized_thetas,cost = gradientDescent(training_data,target_variable
                                        ,thetas,alpha,n_iterations)

print("Optimized THETA : ")
print(optimized_thetas)

#print(cost[n_iterations-1])


y_hat = hypothesis(optimized_thetas, training_data)
y_hat = np.sum(y_hat, axis=1)


## draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(n_iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('total cost graph')


#=========================================================================
# get best fit line
fig, ax2 = plt.subplots(figsize=(5,5))


ax2.scatter(x = list(range(0, 21613)), y = target_variable, label='Traning Data' , color='green')
ax2.scatter(x = list(range(0, 21613)), y =  y_hat, label='predicted Data' , color='blue')


ax2.legend(loc=2)

ax2.set_xlabel('house')
ax2.set_ylabel('price')
ax2.set_title('house ID vs. house price') 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def predict_price(inputs):
    
    file_data = pd.read_csv(inputs, skipinitialspace=True)
    in_data = file_data[input_fields]
    out_data = file_data[output_fields]
    
    # rescaling data
    in_data = (in_data - in_data.mean()) / in_data.std()
    
    # add ones column
    in_data.insert(0, 'Ones', 1)
    
    
    # separate X (training data) from y (target variable)
    x = in_data.iloc[:,:] # all rows x all columns expect the last one
    y = out_data.iloc[:,0] # all rows x last column
        
    y_hat = hypothesis(optimized_thetas, x)
    y_hat = np.sum(y_hat, axis=1)
    
    # get best fit line
    fig, ax2 = plt.subplots(figsize=(5,5))
    
#    print(x.shape[0])
    
    ax2.scatter(x = list(range(0, x.shape[0])), y = y, label='testing Data' , color='green')
    ax2.scatter(x = list(range(0, x.shape[0])), y =  y_hat, label='predicted Data' , color='blue')
        
    ax2.legend(loc=2)
    
    ax2.set_xlabel('house')
    ax2.set_ylabel('price')
    ax2.set_title('house ID vs. house price')

    


