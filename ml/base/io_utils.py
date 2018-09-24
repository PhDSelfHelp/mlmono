import glob
import os


def find_tfrecords_in_dir(data_dir, file_pattern):
    expr = os.path.join(data_dir, file_pattern)
    return glob.glob(expr)
