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
    for i in range(batch.shape[0]):
        if normalize == 'batch':
            batch[i] = batch[i]/np.amax(batch[i])#*20.
        elif normalize == 'max':
            batch[i] = batch[i]/np.amax(batch[i], axis=(1,2),keepdims=True)#*20.
        elif normalize == 'noise':
            mask = batch[i] < np.percentile(batch[i], q=95, axis=(1,2), keepdims=True)
            batch[i] /= np.mean(batch[i]*mask, axis=(1,2), keepdims=True)*5
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
