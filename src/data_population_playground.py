import csv
from simple_linear_model import SimpleLinearModel

def getModel(dataset_name):
    with open('datasets/{}'.format(dataset_name)) as file:
        headers, *data = csv.reader(file, delimiter=',')
    return SimpleLinearModel(headers, data)

if __name__ == "__main__":
    getModel('winequality-red.csv')