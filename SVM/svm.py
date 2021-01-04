import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SVM.gradient_descent import *


def gaussian_kernel_gram_matrix_full(X1, X2, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum(np.power((x1 - x2), 2)) / float(2 * (sigma ** 2)))
    return gram_matrix


# f1=exp(−||x−l(1)||22σ2)
# input_fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
# input_fields = ['age','sex','cp','trestbps','chol','fbs']
input_fields = ['trestbps', 'chol', 'thalach', 'oldpeak']

output_fields = ['target']
path = "train.csv"


file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields]
output_data = file_data[output_fields]

# rescaling data
input_data = (input_data - input_data.mean()) / input_data.std()

# a = np.tanh(input_data)
# input_data = pd.DataFrame(a)

# a = gaussian_kernel_gram_matrix_full(np.array(input_data), np.array(input_data))
# input_data = pd.DataFrame(a)

# add ones column
input_data.insert(0, 'Ones', 1)
# input_data.insert(len(input_fields), 'Ones', 1)

x = input_data.iloc[:, :]  # all rows x all columns expect the last one
y = output_data.iloc[:, 0]  # all rows x last column
y = y.apply(lambda v: -1 if v < 1 else 1)
w = np.array([0] * len(x.columns))  # [ 0 0 0]

# alpha = 0.0001
alpha = 0.00005
n_iterations = 500
lmda = 0

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

# hx, fx = hypothesis(x, w)
#
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection='3d')
# xx = input_data.iloc[:, 0]
# yy = input_data.iloc[:, 1]
# zz = input_data.iloc[:, 2]
# c = output_data.iloc[:, -1]
# img = ax.scatter(xx, yy, zz, c=c)
# fig.colorbar(img)
# plt.title("Original Output")
#
# fig2 = plt.figure(figsize=(5, 5))
# ax = fig2.add_subplot(111, projection='3d')
# xx = input_data.iloc[:, 0]
# yy = input_data.iloc[:, 1]
# zz = input_data.iloc[:, 2]
# c = hx.apply(lambda v: 0 if v < 1 else 1)
# img2 = ax.scatter(xx, yy, zz, c=c)
# fig2.colorbar(img2)
# plt.title("predicted Output")
#
# print(hx)
#
# plt.show()

def predict_price(inputs):
    file_data = pd.read_csv(inputs, skipinitialspace=True)
    in_data = file_data[input_fields]
    out_data = file_data[output_fields]
    original_in = in_data

    # rescaling data
    in_data = (in_data - in_data.mean()) / in_data.std()

    # a = gaussian_kernel_gram_matrix_full(np.array(in_data), np.array(in_data))
    # in_data = pd.DataFrame(a)

    # a = np.tanh(in_data)
    # in_data = pd.DataFrame(a)

    # add ones column
    in_data.insert(0, 'Ones', 1)

    # separate X (training data) from y (target variable)
    x_testing = in_data.iloc[:, :]  # all rows x all columns expect the last one
    hx_testing, fx_testing = hypothesis(x_testing, w)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    xx = original_in.iloc[:, 0]
    yy = original_in.iloc[:, 1]
    zz = original_in.iloc[:, 2]
    cc = out_data.iloc[:, -1]
    img_t = ax.scatter(xx, yy, zz, c=cc)
    fig.colorbar(img_t)
    plt.title("Original testing Output")

    fig2 = plt.figure(figsize=(5, 5))
    ax = fig2.add_subplot(111, projection='3d')
    xx = original_in.iloc[:, 0]
    yy = original_in.iloc[:, 1]
    zz = original_in.iloc[:, 2]
    # c = hx_testing.apply(lambda v: 0 if v < 1 else 1)
    ct= fx_testing.apply(lambda v: 1 if v >= 1 else 0)
    img2 = ax.scatter(xx, yy, zz, c=ct)
    fig2.colorbar(img2)
    plt.title("predicted Output")

    error = 0
    o = 0
    for i in range(len(ct)):
        if ct[i] == 1:
            o+=1
        if ct[i] != cc[i]:
            error+=1

    print("Accuracy =  ", (len(ct) - error)/len(ct))
    print(o)

    plt.show()


predict_price('test.csv')


