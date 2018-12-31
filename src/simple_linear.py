import numpy as np
import sys

def weight_update(y, y_hat, x):
    return (-2 * x * (y - y_hat)) / len(y)

def bias_update(y, y_hat, x):
    return (-2 * (y - y_hat)) / len(y)

def cost_function(y_hat, y):
    return np.mean((y_hat - y) ** 2)

def minimize_cost(X, Y, learning_rate=.00000000001, tolerance=.0000001):
    weight = np.random.random()
    bias = np.random.random()
    cost = 1 / tolerance
    iteration = 0

    while cost > tolerance and iteration < 10000:
        y_hat = weight * X + bias
        differences = y_hat - Y
        cost = np.mean(differences ** 2)
        weight_updates = 2 * np.mean(np.dot(differences, X.T))
        bias_updates = 2 * np.mean(differences)
        weight -= learning_rate * weight_updates
        bias -= learning_rate * bias_updates
        iteration += 1

    return (weight, bias)

def predict(X, weights, biases):
    return X * weights + biases

def r_score(predictedY, Y):
    total_sum_squares = np.sum((np.mean(Y) - Y) ** 2)
    residual_sum_squares = np.sum((predictedY - Y) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)

def get_linear_dataset(dataset_name):
    with open('datasets/{}'.format(dataset_name)) as trainingData:
        pairs = [line.split() for line in trainingData.readlines()]
        trainX = [int(pair[0]) for pair in pairs]
        trainY = [int(pair[1]) for pair in pairs]
    return (trainX, trainY)
    
def train_and_test(train_data_file_name, test_data_file_name):
    (trainX, trainY) = get_linear_dataset(train_data_file_name)
    print('Training...')
    weights, biases = minimize_cost(np.array(trainX).reshape(1, -1), np.array(trainY).reshape(1, -1))
    print('Weight: {}, Bias: {}'.format(weights, biases))

    (testX, testY) = get_linear_dataset(test_data_file_name)
    print('Predicting...')
    predictedY = predict(np.array(testX), weights, biases)
    print('RSS: {}'.format(np.sum((predictedY - testY) ** 2)))
    print('R Score: {}'.format(r_score(predictedY, np.array(testY))))

if __name__ == "__main__":
    train_and_test(sys.argv[1], sys.argv[2])
