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
            if not 'OFF' in filename:
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

def load_batch(batch_size, files, index, with_y=False, normalize=True):
    batch = []
    if index % 30000 == 0:
        np.random.shuffle(files)
    index = index % (len(files)-batch_size)
    for i in range(index, index+batch_size):
        if not with_y:
            img = fitsio.read(files[i])
        else:
            img = np.vstack([fitsio.read(f) for f in files[i]])
        if normalize:
            mask = img < np.percentile(img, q=95)
            img /= np.mean(img[mask])*5
        #print(np.mean(img), np.amax(img))
        batch.append(img)
    batch = np.stack(batch, axis=0)
    batch = batch[...,np.newaxis]/np.amax(batch)#*20.
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
