import fnmatch
import os
import re
import threading

import fitsio
import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*.fits', sortby='shuffle'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if sortby == 'shuffle':
        np.random.shuffle(files)
    return files

def load_batch(batch_size, files, index):
	batch = []
	index = index % len(files)
	for i in range(index, index+batch_size):
		batch.append(fitsio.read(files[i]))
	batch = np.stack(batch, axis=0)
	batch = batch[...,np.newaxis]
	return batch


