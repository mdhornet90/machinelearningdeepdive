import random


def get_random_image_data(loaded):
    return loaded.train_inputs[random.randint(0, len(loaded.train_inputs) - 1)]
