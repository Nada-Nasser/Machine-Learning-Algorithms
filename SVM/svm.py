import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SVM.gradient_descent import *

input_fields = ['age', 'cp']
output_fields = ['target']

path = "heart.csv"

file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields].head(300)
output_data = file_data[output_fields].head(300)

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
alpha = 0.111
n_iterations = 10
lmda = 0.111

optimized_w, cost = gradient_descent(x, y, w, alpha, lmda, n_iterations)

print("Optimized THETA : ")
print(optimized_w)
print(cost)


print(hypothesis(x, optimized_w))
