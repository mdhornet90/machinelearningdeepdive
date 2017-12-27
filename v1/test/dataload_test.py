import unittest
import dataload


class DataLoadTest(unittest.TestCase):
    def setUp(self):
        self.dataload = dataload.DataLoad()

    def test_training_data_exists(self):
        self.assertTrue(len(self.dataload.train_inputs) > 0)
        self.assertTrue(len(self.dataload.train_outputs) > 0)

    def test_testing_data_exists(self):
        self.assertTrue(len(self.dataload.test_inputs) > 0)
        self.assertTrue(len(self.dataload.test_outputs) > 0)
