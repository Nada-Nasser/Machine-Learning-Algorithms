import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SVM.gradient_descent import *


# input_fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
input_fields = ['trestbps', 'chol', 'thalach', 'oldpeak']

output_fields = ['target']
path = "train.csv"


file_data = pd.read_csv(path, skipinitialspace=True)
input_data = file_data[input_fields]
output_data = file_data[output_fields]

# rescaling data
input_data = (input_data - input_data.mean()) / input_data.std()

# add ones column
input_data.insert(0, 'Ones', 1)

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

# draw error graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(n_iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('total cost graph')


def predict_price(inputs):
    file_data = pd.read_csv(inputs, skipinitialspace=True)
    in_data = file_data[input_fields]
    out_data = file_data[output_fields]
    original_in = in_data

    # rescaling data
    in_data = (in_data - in_data.mean()) / in_data.std()

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
    for i in range(len(ct)):
        if ct[i] != cc[i]:
            error+=1

    print("Accuracy =  ", (len(ct) - error)/len(ct))

    plt.show()


predict_price('test.csv')


