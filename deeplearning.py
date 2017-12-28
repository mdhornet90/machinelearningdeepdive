import h5py
import numpy as np
import time


class Classifier(object):
    def __init__(self):
        self.trained_weights = None
        self.trained_bias = None

    def train(self, training_input, training_output, max_iterations=10000, learning_rate=0.005):
        m = len(training_input)
        transformed_in = normalize_data(training_input, m)
        weights = np.zeros((len(transformed_in), 1))
        bias = 0

        _iterations = 0
        cost = 2 ** np.MAXDIMS - 1
        while cost >= learning_rate and _iterations < max_iterations:
            cost, differences = self._calculate_intermediate_results(weights, bias, transformed_in, training_output, m)

            # this causes all weights to step toward the global minimum in n-dimensional space
            weights = weights - learning_rate * (np.dot(transformed_in, differences.T) / m)
            bias = bias - learning_rate * (np.sum(differences) / m)
            _iterations = _iterations + 1

        self.trained_weights = weights
        self.trained_bias = bias

        return _iterations

    def _calculate_intermediate_results(self, in_weights, in_bias, in_train, out_train, m):
        # This step is identical to actually getting a prediction
        a = sigmoid(np.dot(in_weights.T, in_train) + in_bias)
        # Quantify how far off the predictions were from the results, averaged over each sample
        cost = np.squeeze(-np.sum(out_train * np.log(a) + (1 - out_train) * np.log(1 - a)) / m)
        # TODO re-learn why the above cost is calculated this way
        return cost, a - out_train

    def predict_results(self, test_input):
        _predictions = sigmoid(np.dot(self.trained_weights.T, test_input) + self.trained_bias)
        _predictions[_predictions > 0.5] = 1
        _predictions[_predictions <= 0.5] = 0

        return np.squeeze(_predictions)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def normalize_data(data, number_of_samples):
    return data.reshape((number_of_samples, -1)).T / 255


def get_accuracy(expecteds, actuals):
    return sum([1 for expected, actual in zip(expecteds, actuals) if expected == actual]) / len(actuals)


if __name__ == '__main__':
    with h5py.File('datasets/train.h5', 'r') as train:
        train_in = train['train_set_x'][:]
        train_out = train['train_set_y'][:]

    classifier = Classifier()
    start = time.time()
    iterations = classifier.train(train_in, train_out)
    end = time.time()
    print('Learned after {} iterations, taking {} seconds'.format(iterations, end - start))

    with h5py.File('datasets/test.h5', 'r') as test:
        test_in = test['test_set_x'][:]
        test_out = test['test_set_y'][:]

    predictions = classifier.predict_results(normalize_data(test_in, len(test_in)))
    print('Accuracy of {}%'.format(get_accuracy(predictions, test_out) * 100))
