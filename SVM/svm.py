import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SVM.gradient_descent import *

input_fields = ['age', 'trestbps']
output_fields = ['target']

path = "heart.csv"

file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields].head(100)
output_data = file_data[output_fields].head(100)

color = file_data[output_fields[0]].apply(lambda x: 'red' if x == 0 else 'green')
file_data.plot(kind='scatter', x=input_fields[0], y=input_fields[1], c=color)
# plt.show()

# add ones column
input_data.insert(2, 'Ones', 1)

# print(input_data)

x = input_data.iloc[:, :]  # all rows x all columns expect the last one
y = output_data.iloc[:, 0]  # all rows x last column

# print(x)
# print(y)

w = np.array([0] * (len(x.columns)))
# print(w)
alpha = 0.1
n_iterations = 100

# print(y * np.sum(hypothesis(x, w), axis=1))
#
# print("COST = ")
# print(compute_cost(x, y, w))

optimized_w, cost = gradient_descent(x, y, w, alpha, n_iterations)


print("Optimized THETA : ")
print(optimized_w)
#
