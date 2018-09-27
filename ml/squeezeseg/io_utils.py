import tensorflow as tf 

import numpy as np

import glob

from PIL import Image

# Converting the values into features
# _int64 is used for numeric values

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = 'something.tfrecords'

# Initiating the writer and creating the tfrecords file.

writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.

images = glob.glob('data/*.jpg')
for image in images[:1]:
    img = Image.open(image)
    img = np.array(img.resize((32,32)))
label = 0 if 'apple' in image else 1
feature = { 'label': _int64_feature(label),
              'image': _bytes_feature(img.tostring()) }

# Create an example protocol buffer

example = tf.train.Example(features=tf.train.Features(feature=feature))

# Writing the serialized example.

writer.write(example.SerializeToString())

writer.close()