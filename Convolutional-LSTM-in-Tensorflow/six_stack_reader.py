import fnmatch
import os
import re
import threading
import glob
import numpy as np
import tensorflow as tf
from itertools import chain


def train_test_split(directory, train, test, split=.8, sortby='shuffle'):
    files = glob.glob(directory + "/*.npy")
    if sortby == 'shuffle':
        np.random.shuffle(files)

    if not files:
        print('No data files found!\n')
        import sys
        sys.exit(1)

    train_len = int(len(files) * split)
    with open(train, 'w+') as train_file:
        mixed = files[:train_len]
        for file in mixed:
            train_file.write(file + '\n')

    with open(test, 'w+') as test_file:
        mixed = files[train_len:]
        for file in mixed:
            test_file.write(file + '\n')


def find_files(directory, sortby='shuffle'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    with open(directory) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for filename in content:
            files.append(filename)
    if sortby == 'shuffle':
        np.random.shuffle(files)
    return files

def load_batch(batch_size, files, index, normalize='max'):
    batch = []
    if index % 20000 == 0:
        np.random.shuffle(files)
    index = index % (len(files)-batch_size)
    for i in range(index, index+batch_size):
        img = np.load(files[i])
        batch.append(img)
    batch = np.stack(batch, axis=0)
    if normalize == 'batch':
        batch = batch/np.amax(batch)#*20.
    elif normalize == 'max':
        batch = batch/np.amax(batch, axis=(2,3),keepdims=True)#*20.
    elif normalize == 'mean':
        batch = batch/np.mean(batch, axis=(2,3),keepdims=True)#*20.
    elif normalize == 'std':
        batch = batch - np.amin(batch, axis=(2,3),keepdims=True)#*20.
        batch = batch/np.std(batch, axis=(2,3),keepdims=True)#*20.
    elif normalize == 'shift_std':
        mean = np.mean(batch, axis=(2,3),keepdims=True)
        batch = (batch - mean) / np.std(batch, axis=(2,3),keepdims=True)#*20.
        batch += mean - np.amin(batch, axis=(2,3),keepdims=True)
    elif normalize == 'noise':
        mask = batch < np.percentile(batch, q=95, axis=(2,3), keepdims=True)
        batch /= np.mean(batch*mask, axis=(2,3), keepdims=True)*5
    return batch

def random_flip(batch):
    n = len(batch)
    for i in range(n):
        # flip about 1/3 of time
        if np.random.random() <= 0.33:
            if np.random.random() > 0.5:
                batch[i] = np.fliplr(batch[i])
            else:
                batch[i] = np.flipud(batch[i])
        else:
            if np.random.random() > 0.5:
                batch[i] = np.flipud(np.fliplr(batch[i]))
    return batch
