import h5py
import definitions
from os import path


class DataLoad(object):
    def __init__(self, location=definitions.DATASETS):
        with h5py.File(path.join(location, 'train.h5')) as train:
            self.train_inputs = train['train_set_x'][:]
            self.train_outputs = train['train_set_y'][:]
        with h5py.File(path.join(location, 'test.h5')) as test:
            self.test_inputs = test['test_set_x'][:]
            self.test_outputs = test['test_set_y'][:]
