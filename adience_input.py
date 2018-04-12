import tensorflow as tf

IMAGE_SIZE = 227
NUM_CLASSES = 8
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

def read_adience(filename_queue):
    result = [1, 2, 3]
    return result

def distored_inputs(data_dir, batch_size):
    image, label = [1, 2, 3]
    return image, label

def inputs(eval_data, data_dir, batch_size):
    image, label = [1, 2, 3]
    return image, label