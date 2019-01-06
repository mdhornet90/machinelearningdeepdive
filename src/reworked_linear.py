import csv
import random
import numpy as np
    
class SimpleLinearModel(object):
    def __init__(self, headers):
        self.headers = headers

    def train(self, train_data):
        pass

    def predict(self, test_data):
        pass

if __name__ == "__main__":
    with open('datasets/{}'.format('winequality-red.csv')) as file:
        headers, *data = csv.reader(file, delimiter=',')
    shuffled_data = [line for line in data]
    random.shuffle(shuffled_data)
    train_test_split_idx = int(len(shuffled_data) / 3)
    train_data = np.array(shuffled_data[0:train_test_split_idx]).T
    test_data = np.array(shuffled_data[train_test_split_idx:]).T
    model = SimpleLinearModel(headers)
    model.train(train_data[::,0])