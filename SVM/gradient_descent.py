import numpy as np


def hypothesis(x, theta):
    fx = np.sum(theta.T * x, axis=1)
    hx = fx.apply(lambda v: 1 if v >= 1 else -1)
    return hx, fx


def compute_cost(x, y, theta):
    hx, fx = hypothesis(x, theta)
    cost = (y*fx).apply(lambda v: 0 if v >= 1 else max(0, 1 - v))
    return cost


def gradient_descent(x, y, theta, alpha, lmda, n_iterations):
    m = len(x)  # m = size of the training data
    cost = np.zeros(n_iterations)  # list to store the cost in every iteration,

    for iteration in range(n_iterations):  # start the algorithm
        hx, fx = hypothesis(x, theta)
        for i in range(m):
            if y[i] * fx[i] >= 1:
                theta = np.array(theta - (alpha * ((2 * lmda) * theta)))
            else:
                theta = np.array(theta + (alpha * ((y[i] * x.iloc[i, :]) - ((2 * lmda) * theta))))

        c = np.sum(compute_cost(x, y, theta))
        lmda = 1.0/c
        cost[iteration] = c
    return theta, cost
