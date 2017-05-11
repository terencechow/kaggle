# %matplotlib inline
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

image_dir = "data/dogsandcats/sample/train"

os.chdir(image_dir)

samples = []
for f in os.listdir():
    im = Image.open(f)
    data = np.asarray(im)
    flattened = data.flatten()
    # transposed = np.transpose(data, (1, 0, 2))
    samples.append(flattened)

# note the samples are not the same size
samples = np.asarray(samples)
