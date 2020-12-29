import numpy as np


def hypothesis(x, theta):
    return theta * x


def compute_cost(x, y, theta):
    fx = np.sum(hypothesis(x, theta), axis=1)
    temp = y * fx
    cost = temp.apply(lambda v: 0 if v >= 1 else max(0, 1-v))
    return cost


def gradient_descent(x, y, theta, alpha, n_iterations):
    m = len(x)  # m = size of the training data
    n_theta = 3
    cost = np.zeros(n_iterations)  # list to store the cost in every iteration,
    # initialize it with zeros # list to store the cost in every iteration,
    # initialize it with zeros

    for iteration in range(n_iterations):  # start the algorithm

        hx = np.sum(hypothesis(x, theta), axis=1)
        temp = y * hx

        # for j in range(n_theta):  # for each theta j, update its value
        #     theta[j] = theta[j] - (alpha / m) * sum((hx - y) * x.iloc[:, j])
        #
        # cost[iteration] = (compute_cost(x, y, theta))

    return theta, cost
