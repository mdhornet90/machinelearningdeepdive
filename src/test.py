import unittest

class TestingTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual('hello world'.upper(), 'HELLO WORLD')
