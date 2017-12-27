import random
import itertools


def get_random_image_data(loaded):
    return loaded.train_inputs[random.randint(0, len(loaded.train_inputs) - 1)]


def unroll_image_data(image):
    assert len(image) == 3, 'expected image data with 3 channels (R, G, B)'
    vector = []
    [vector.extend(itertools.chain.from_iterable(component)) for component in image]

    return vector
