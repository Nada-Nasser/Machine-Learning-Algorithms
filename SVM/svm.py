import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SVM.gradient_descent import *

# input_fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

input_fields = ['trestbps', 'chol','thalach', 'oldpeak']
output_fields = ['target']
path = "training_data.csv"

file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields].head(100)
output_data = file_data[output_fields].head(100)


# rescaling data
input_data = (input_data - input_data.mean()) / input_data.std()

# add ones column
input_data.insert(2, 'Ones', 1)

x = input_data.iloc[:, :]  # all rows x all columns expect the last one
y = output_data.iloc[:, 0]  # all rows x last column
y = y.apply(lambda v: -1 if v < 1 else 1)
w = np.array([0] * (len(x.columns))) # [ 0 0 0]


alpha = 0.0001
n_iterations = 500
lmda = 0

print(w)

w, cost = gradient_descent(x, y, w, alpha, lmda, n_iterations)

print("Optimized THETA : ")
print(w)
print(cost)

## draw error graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(n_iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('total cost graph')


hx, fx = hypothesis(x, w)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
xx = input_data.iloc[:, 1]
yy = input_data.iloc[:, 2]
zz = input_data.iloc[:, 3]
c = output_data.iloc[:, -1]
img = ax.scatter(xx, yy, zz, c = c)
fig.colorbar(img)
plt.title("Original Output")


fig2 = plt.figure(figsize=(5, 5))
ax = fig2.add_subplot(111, projection='3d')
xx = input_data.iloc[:, 1]
yy = input_data.iloc[:, 2]
zz = input_data.iloc[:, 3]
c = hx.apply(lambda v: 0 if v < 1 else 1)
img2 = ax.scatter(xx,yy,zz,c = c)
fig2.colorbar(img2)
plt.title("predicted Output")


print(hx)

plt.show()