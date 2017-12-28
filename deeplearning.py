import h5py
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    with h5py.File('datasets/train.h5', 'r') as train:
        sample_in = train['train_set_x'][:]
        sample_out = train['train_set_y'][:]

    m = len(sample_in)
    transformed_in = sample_in.reshape((m, -1)).T / 255
    weights, bias = np.zeros((len(transformed_in), 1)), 0

    learning_rate = 0.005
    cost = 2 ** np.MAXDIMS - 1
    while cost >= learning_rate:
        print(cost)
        a = sigmoid(np.dot(weights.T, transformed_in) + bias)
        cost = np.squeeze(np.sum(-(sample_out * np.log(a) + (1 - sample_out) * np.log(1 - a))) / m)

        weights = weights - learning_rate * (np.dot(transformed_in, (a - sample_out).T) / m)
        bias = bias - learning_rate * (np.sum(a - sample_out) / m)

    print(sigmoid(np.dot(weights.T, transformed_in) + bias))