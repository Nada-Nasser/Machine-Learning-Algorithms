# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:48:30 2020

@author: Toshiba
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions as MLR


input_fields = ['grade', 'bathrooms', 'lat', 'sqft_living', 'view']
output_fields = ['price']

path = "house_data.csv"

file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields]
output_data = file_data[output_fields]

print("\ninput data (Xs) : ")
print(input_data.head(10))
print("\ntarget data (Ys) : ")
print(output_data.head(10))


# rescaling data
input_data = (input_data - input_data.mean()) / input_data.std()

print(input_data.head(10))


# add ones column
input_data.insert(0, 'Ones', 1)
#print(input_data.head(10))

# separate X (training data) from y (target variable)
training_data = input_data.iloc[:,:] # all rows x all columns expect the last one
target_variable = output_data.iloc[:,:] # all rows x last column

#print(target_variable.head(10))
#
#print(training_data.head(10))

# Vectorization
# x and y are a list of vectors (Matrices) 
x = np.matrix(training_data.values) # vectors of the training data
y = np.matrix(target_variable.values) # vector of all target values

#initialize thetas
thetas = np.matrix(np.array([0,0,0,0,0,0]))


#print(x)
#print(y)
#print(thetas)

alpha = 0.01
n_iterations = 1000
#
#print(thetas.T.shape)
#
#print(x[0].shape)

optimized_thetas,cost = MLR.gradientDescent(x,y,thetas,alpha,n_iterations)

print("\noptimized thetas = ")
print(optimized_thetas)

#print(cost)

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(n_iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')



def predict_price(inputs):
    file_data = pd.read_csv(inputs, skipinitialspace=True)
    in_data = file_data[input_fields]
    
    # rescaling data
    in_data = (in_data - in_data.mean()) / in_data.std()
    real_out_data = file_data[output_fields]
        
    # add ones column
    in_data.insert(0, 'Ones', 1)
    #print(input_data.head(10))
    
    training_in_data = in_data.iloc[:,:]
    target_out_variable = real_out_data.iloc[:,:] # all rows x last column

    
    x_in_data = np.matrix(training_in_data.values) # vectors of the training data
    y_in_data = np.matrix(target_out_variable.values) # vector of all target values


    outputs = x_in_data *  optimized_thetas.T
    c = MLR.compute_cost(x_in_data,y_in_data,optimized_thetas)
    
    return c,outputs


#48992470760.25859
#92349754598.17032

