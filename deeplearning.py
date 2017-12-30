import h5py
import numpy as np
import time


class ShallowLearningClassifier(object):
    def __init__(self, hidden_unit_count):
        self.trained_weights = None
        self.trained_bias = None
        self.hidden_unit_count = hidden_unit_count

    def train(self, training_input, training_output, max_iterations=10000, learning_rate=0.005):
        m = len(training_input)
        transformed_in = normalize_data(training_input, m)
        hidden_weights = np.random.randn(len(transformed_in), self.hidden_unit_count) * 0.01
        hidden_bias = np.zeros((self.hidden_unit_count, 1))
        output_weights = np.random.randn(self.hidden_unit_count, 1) * 0.01
        output_bias = np.zeros((1, 1))

        _iterations = 0
        cost = 2 ** np.MAXDIMS - 1
        while cost >= learning_rate and _iterations < max_iterations:
            # forward propagation
            z_hidden = calculate_z(transformed_in, hidden_weights, hidden_bias)
            a_hidden = relu(z_hidden)
            z_output = calculate_z(a_hidden, output_weights, output_bias)
            a_output = sigmoid(z_output)

            # calculate cost
            cost = calculate_cost(a_output, training_output, m)

            # backward propagation
            d_z_output = a_output - training_output
            d_output_weights = np.dot(a_hidden, d_z_output.T) / m
            d_output_bias = np.sum(d_z_output, axis=1, keepdims=True) / m
            d_z_hidden = np.dot(d_output_weights, d_z_output) * relu_prime(z_hidden)
            d_hidden_weights = np.dot(transformed_in, d_z_hidden.T) / m
            d_hidden_bias = np.sum(d_z_hidden, axis=1, keepdims=True) / m

            # update weights
            output_weights = output_weights - learning_rate * d_output_weights
            output_bias = output_bias - learning_rate * d_output_bias
            hidden_weights = hidden_weights - learning_rate * d_hidden_weights
            hidden_bias = hidden_bias - learning_rate * d_hidden_bias

            _iterations = _iterations + 1

        self.trained_weights = output_weights
        self.trained_bias = output_bias

        return _iterations

    def forward_propagation(self):

    def predict_results(self, test_input):
        _predictions = sigmoid(calculate_z(test_input, self.trained_weights, self.trained_bias))
        _predictions[_predictions > 0.5] = 1
        _predictions[_predictions <= 0.5] = 0

        return np.squeeze(_predictions)


class LinearClassifier(object):
    def __init__(self):
        self.trained_weights = None
        self.trained_bias = None

    def train(self, training_input, training_output, max_iterations=10000, learning_rate=0.005):
        m = len(training_input)
        transformed_in = normalize_data(training_input, m)
        weights = np.zeros((len(transformed_in), 1))
        bias = np.zeros((1, 1))

        _iterations = 0
        cost = 2 ** np.MAXDIMS - 1
        while cost >= learning_rate and _iterations < max_iterations:
            # forward propagation
            a = sigmoid(calculate_z(transformed_in, weights, bias))

            # calculate cost
            cost = calculate_cost(a, training_output, m)

            # backward propagation
            differences = a - training_output
            d_weights = np.dot(transformed_in, differences.T) / m
            d_bias = np.sum(differences) / m

            # this causes all weights to step toward the global minimum in n-dimensional space
            weights = weights - learning_rate * d_weights
            bias = bias - learning_rate * d_bias

            _iterations = _iterations + 1

        self.trained_weights = weights
        self.trained_bias = bias

        return _iterations

    def predict_results(self, test_input):
        _predictions = sigmoid(calculate_z(test_input, self.trained_weights, self.trained_bias))
        _predictions[_predictions > 0.5] = 1
        _predictions[_predictions <= 0.5] = 0

        return np.squeeze(_predictions)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    result = np.array(z, copy=True)
    result[z <= 0] = 0
    # result[z > 0] = 1
    return result


def normalize_data(data, number_of_samples):
    return data.reshape((number_of_samples, -1)).T / 255


def calculate_z(input_data, weights, bias):
    return np.dot(weights.T, input_data) + bias


def calculate_cost(a, out_train, m):
    return np.squeeze(-np.sum(np.multiply(out_train, np.log(a)) + np.multiply(1 - out_train, np.log(1 - a))) / m)


def get_accuracy(expecteds, actuals):
    return sum([1 for expected, actual in zip(expecteds, actuals) if expected == actual]) / len(actuals)


def run_classification(classifier, train_input, train_output, test_input, test_output):
    start = time.time()
    iterations = classifier.train(train_input, train_output)
    end = time.time()
    print('Learned after {} iterations, taking {} seconds'.format(iterations, end - start))

    predictions = classifier.predict_results(normalize_data(test_input, len(test_input)))
    print('Accuracy of {}%'.format(get_accuracy(predictions, test_output) * 100))


if __name__ == '__main__':
    with h5py.File('datasets/train.h5', 'r') as train:
        train_in = train['train_set_x'][:]
        train_out = train['train_set_y'][:]
        train_out.reshape((1, train_out.shape[0]))
    with h5py.File('datasets/test.h5', 'r') as test:
        test_in = test['test_set_x'][:]
        test_out = test['test_set_y'][:]
        test_out.reshape((1, test_out.shape[0]))

    # print('Running linear classification...')
    # run_classification(LinearClassifier(), train_in, train_out, test_in, test_out)

    print('Running shallow learning classification...')
    run_classification(ShallowLearningClassifier(5), train_in, train_out, test_in, test_out)
