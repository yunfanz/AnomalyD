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

def find_pairs(directory, pattern='*OFF.fits', sortby='shuffle'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            on_name = '_'.join(filename.split('_')[:-1])+'.fits'
            files.append((os.path.join(root, on_name), os.path.join(root, filename)))
    if sortby == 'shuffle':
        np.random.shuffle(files)
    return files

def load_batch(batch_size, files, index, with_y=False):
    batch = []
    if index % 30000 == 0:
        np.random.shuffle(files)
    index = index % (len(files)-batch_size)
    for i in range(index, index+batch_size):
        if not with_y:
            img = fitsio.read(files[i])
        else:
            img = np.vstack([fitsio.read(f) for f in files[i]])
        batch.append(img)
    batch = np.stack(batch, axis=0)
    batch = batch[...,np.newaxis]/np.amax(batch)#*20.
    return batch