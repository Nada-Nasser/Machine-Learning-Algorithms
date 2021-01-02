import numpy as np


def hypothesis(x, theta):
    f = np.sum(theta.T * x, axis=1)
    hx = f.apply(lambda v: 1 if v >= 0 else 0)
    return hx


def compute_cost(x, y, theta):
    fx = y * hypothesis(x, theta)
    cost = np.sum(fx.apply(lambda v: 0 if v >= 1 else max(0, 1 - v)))
    return cost


def gradient_descent(x, y, theta, alpha, lmda, n_iterations):
    m = len(x)  # m = size of the training data
    cost = np.zeros(n_iterations)  # list to store the cost in every iteration,

    for iteration in range(n_iterations):  # start the algorithm
        o=0
        u=0
        hx = hypothesis(x, theta)

        for i in range(m):
            if y[i] == hx[i]:
                theta = np.array(theta - (alpha * ((2 * lmda) * theta)))
                o = o+1
            else:
                u = u+1
                theta = np.array(theta + (alpha * ((y[i] * x.iloc[i, :]) - ((2 * lmda) * theta))))

        cost[iteration] = np.sum(compute_cost(x, y, theta))
        print(theta)
    print(o)
    print(u)
    return theta, cost
