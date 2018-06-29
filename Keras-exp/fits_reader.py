import fnmatch
import os
import re
import threading
import fitsio
import numpy as np
import tensorflow as tf
from itertools import chain


def train_test_split(directory, train, test, split=.8, pattern='*OFF.fits', sortby='shuffle'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            on_name = '_'.join(filename.split('_')[:-1])+'.fits'
            files.append((os.path.join(root, on_name), os.path.join(root, filename)))
    if sortby == 'shuffle':
        np.random.shuffle(files)

    train_len = int(len(files) * split)
    with open(train, 'w+') as train_file:
        mixed = list(chain(*files[:train_len]))
        for file in mixed:
            train_file.write(file + '\n')

    with open(test, 'w+') as test_file:
        mixed = list(chain(*files[train_len:]))
        for file in mixed:
            test_file.write(file + '\n')


def find_files(directory, pattern='*.fits', sortby='shuffle', pair=True):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if pair:
                on_name = '_'.join(filename.split('_')[:-1])+'.fits'
                files.append((os.path.join(root, on_name), os.path.join(root, filename)))
            else:
                files.append(os.path.join(root, filename))
    if sortby == 'shuffle':
        np.random.shuffle(files)
    return files


def find_pairs(directory, sortby='shuffle'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    with open(directory) as f:
        content = f.readlines()
        content = [x.strip() for x in content if 'OFF' in x]
        for filename in content:
            on_name = '_'.join(filename.split('_')[:-1])+'.fits'
            files.append((on_name, filename))
    if sortby == 'shuffle':
        np.random.shuffle(files)
    return files


def load_batch(batch_size, files, index, with_y=False, normalize='max'):
    batch = []
    if index % 20000 == 0:
        np.random.shuffle(files)
    index = index % (len(files)-batch_size)
    for i in range(index, index+batch_size):
        if not with_y:
            img = fitsio.read(files[i])
        else:
            img = np.vstack([fitsio.read(f) for f in files[i]])
        batch.append(img)
    batch = np.stack(batch, axis=0)
    batch = batch[...,np.newaxis]#/np.amax(batch)#*20.
    if normalize == 'batch':
        batch = batch/np.amax(batch)#*20.
    elif normalize == 'max':
        batch = batch/np.amax(batch, axis=(1,2),keepdims=True)#*20.
    elif normalize == 'noise':
        mask = batch < np.percentile(batch, q=95, axis=(1,2), keepdims=True)
        batch /= np.mean(batch*mask, axis=(1,2), keepdims=True)*5
    return batch


def load_batch_pair(batch_size, files, index, normalize='max'):
    batch = []
    if index % 15000 == 0:
        np.random.shuffle(files)
    index = index % (len(files)-batch_size)
    for i in range(index, index+batch_size):
        f_on = files[i]
        f_off = '.'.join(f_on.split('.')[:-1])+'_OFF.fits'
        a_on  = fitsio.read(f_on).squeeze()
        a_off = fitsio.read(f_off).squeeze()
        batch.append(np.concatenate([a_on, a_off], axis=0))
    batch = np.stack(batch, axis=0)[...,np.newaxis]
    if normalize == 'batch':
        batch = batch/np.amax(batch)#*20.
    elif normalize == 'max':
        batch = batch/np.amax(batch, axis=(1,2),keepdims=True)#*20.
    elif normalize == 'noise':
        mask = batch < np.percentile(batch, q=95, axis=(1,2), keepdims=True)
        batch /= np.mean(batch[mask], axis=(1,2))*5
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