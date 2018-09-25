import glob
import os


def find_tfrecords_in_dir(data_dir, file_pattern):
    expr = os.path.join(data_dir, file_pattern)
    return glob.glob(expr)

def create_dir_if_not_exist(path):
    # os.makedirs does recursive dir creation for path if not exist.
    # This does not take a list of paths as input.
    os.makedirs(path, exist_ok=True)
