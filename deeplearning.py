import h5py
import definitions
from os import path


def load_train_data(location=path.join(definitions.DATASETS, 'train.h5')):
    with h5py.File(location, mode='r') as train:
        temp_in = train['train_set_x'][:]
        temp_out = train['train_set_y'][:]
        temp_classes = train['list_classes'][:]

    return temp_in, temp_out, temp_classes


def transform_matrices_to_vectors(matrices):
    return matrices.reshape(matrices.shape[0], -1).T


if __name__ == '__main__':
    train_in, train_out, classification_options = load_train_data()
    trans_in = transform_matrices_to_vectors(train_in)
