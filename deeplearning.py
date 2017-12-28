import h5py
import numpy as np
import time


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_linear_reg(training_input, training_output, max_iterations=10000, learning_rate=0.005):
    m = len(training_input)
    transformed_in = normalize_data(training_input, m)
    weights, bias = np.zeros((len(transformed_in), 1)), 0

    iteration = 0
    cost = 2 ** np.MAXDIMS - 1
    while cost >= learning_rate and iteration < max_iterations:
        # print(cost)
        a = sigmoid(np.dot(weights.T, transformed_in) + bias)
        cost = np.squeeze(np.sum(-(training_output * np.log(a) + (1 - training_output) * np.log(1 - a))) / m)

        weights = weights - learning_rate * (np.dot(transformed_in, (a - training_output).T) / m)
        bias = bias - learning_rate * (np.sum(a - training_output) / m)
        iteration = iteration + 1

    return weights, bias, iteration


def normalize_data(data, number_of_samples):
    return data.reshape((number_of_samples, -1)).T / 255


def get_accuracy(expecteds, actuals):
    return sum([1 for expected, actual in zip(expecteds, actuals) if expected == actual]) / len(actuals)


def predict(trained_weights, trained_bias, the_input):
    predictions = sigmoid(np.dot(trained_weights.T, the_input) + trained_bias)
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    return np.squeeze(predictions)


if __name__ == '__main__':
    with h5py.File('datasets/train.h5', 'r') as train:
        train_in = train['train_set_x'][:]
        train_out = train['train_set_y'][:]

    start = time.time()
    trained_weights, trained_bias, iterations = train_linear_reg(train_in, train_out, max_iterations=70000)
    end = time.time()
    print('Learned after {} iterations, taking {} seconds'.format(iterations, end - start))

    with h5py.File('datasets/test.h5', 'r') as test:
        test_in = test['test_set_x'][:]
        test_out = test['test_set_y'][:]

    expecteds = predict(trained_weights, trained_bias, normalize_data(test_in, len(test_in)))
    print('Accuracy of {}%'.format(get_accuracy(expecteds, test_out) * 100))
