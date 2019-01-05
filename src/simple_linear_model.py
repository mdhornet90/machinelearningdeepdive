import random

class SimpleLinearModel(object):
    def __init__(self, headers, data):
        self.headers = headers
        shuffled_data = [line for line in data]
        random.shuffle(shuffled_data)
        train_test_split_idx = int(len(shuffled_data) / 3)
        self.train_data = shuffled_data[0:train_test_split_idx]
        self.test_data = shuffled_data[train_test_split_idx:]

    def train(self):
        pass

    def predict(self):
        pass