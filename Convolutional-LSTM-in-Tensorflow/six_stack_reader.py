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
    '''finds all files matching the pattern from a directory list file.'''
    files = []
    with open(directory) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for filename in content:
            files.append(filename)
    if sortby == 'shuffle':
        np.random.shuffle(files)
    return files

def _find_channel(filename):
    ''' find coarse channel from filename '''
    tmp = filename[:filename.rindex('_')]
    return int(tmp[tmp.rindex('_') + 1:])

def bin_files_by_channel(files):
    ''' bin files according to coarse channel, as inferred from file name: ..._chnl_id.npy '''
    files_bins = {}
    for i, f in enumerate(files):
        chnl = _find_channel(f)
        if chnl not in files_bins:
            files_bins[chnl] = []
        files_bins[chnl].append(i)
    for chnl in files_bins.keys():
        files_bins[chnl].sort()
    return files_bins

def _normalize(batch, method='max'):
    ''' normalize a batch of data using the given method '''
    if method == 'batch':
        batch = batch/np.amax(batch)#*20.
    elif method == 'max':
        batch = batch/np.amax(batch, axis=(2,3),keepdims=True)#*20.
    elif method == 'mean':
        batch = batch/np.mean(batch, axis=(2,3),keepdims=True)#*20.
    elif method == 'std':
        batch -= np.mean(batch, axis=(2,3),keepdims=True)
        batch /= np.std(batch, axis=(2,3),keepdims=True)#*20.
    elif method == 'shift_std':
        batch -= np.mean(batch, axis=(2,3),keepdims=True)
        batch /= np.std(batch, axis=(2,3),keepdims=True)#*20.
        batch -= np.amin(batch, axis=(2,3),keepdims=True)
    elif method == 'noise':
        mask = batch < np.percentile(batch, q=95, axis=(2,3), keepdims=True)
        batch /= np.mean(batch*mask, axis=(2,3), keepdims=True)*5
    #elif method == 'none':
    #    pass
    return batch

def load_batch_with_fakes(batch_size, files, files_by_channel, index, normalize='max', real_prob=0.5):
    '''
    load batch, with mixed fake/real data.
    Requires files_by_channel(files) (from bin_files_by_channel) in addition to files (from find_files)
    returns: (batch, is_real) where is_real[i] is 1 iff i-th sequence is real
    '''
    batch = []
    np.random.shuffle(files)
    if index == -1:
        index = np.random.randint(0, len(files))
    else:
        index = index % len(files)
    is_real = (np.random.random(batch_size) < np.ones(batch_size) * real_prob)*1.
    for i in range(index, index+batch_size):
        i_mod = i % len(files)
        img = np.load(files[i_mod])
        if is_real[i-index]:
            batch.append(img)
        else:
            # add fake data
            chnl = _find_channel(files[i_mod])
            chnl_files = files_by_channel[chnl]
            # choose random number not in channel
            r = np.random.randint(0, len(files) - len(chnl_files))
            for idx in chnl_files:
                if r >= idx:
                    r += 1
            img_fake = np.load(files[r])
            img_fake[::2] = img[::2]
            batch.append(img_fake)
    batch = _normalize(np.stack(batch, axis=0), normalize)
    return batch, is_real 

def load_batch_fr_paired(batch_size, files, files_by_channel, index, normalize='max'):
    '''
    load batch, with paired fake/real data.
    Requires files_by_channel(files) (from bin_files_by_channel) in addition to files (from find_files)
    returns: (batch_fake, batch_real) where each index represents the same sequence, but where batch_fake
             has 'off' images swapped with a sequence from another channel
    '''
    batch_fake, batch_real = [], []
    np.random.shuffle(files)
    if index == -1:
        index = np.random.randint(0, len(files))
    else:
        index = index % len(files)

    for i in range(index, index+batch_size):
        i_mod = i % len(files)
        img = np.load(files[i_mod])
        batch_real.append(img)
        # add fake data
        chnl = _find_channel(files[i_mod])
        chnl_files = files_by_channel[chnl]
        # choose random number not in channel
        r = np.random.randint(0, len(files) - len(chnl_files))
        for idx in chnl_files:
            if r >= idx:
                r += 1
        img_fake = np.load(files[r])
        img_fake[::2] = img[::2]
        batch_fake.append(img_fake)
    batch_fake = _normalize(np.stack(batch_fake, axis=0), normalize)
    batch_real = _normalize(np.stack(batch_real, axis=0), normalize)
    return batch_fake, batch_real 

def load_batch(batch_size, files, index, normalize='max'):
    ''' load batch of data (real only) '''
    batch = []
    np.random.shuffle(files)
    if index == -1:
        index = np.random.randint(0, len(files))
    else:
        index = index % len(files)
    for i in range(index, index+batch_size):
        img = np.load(files[i])
        batch.append(img)
    batch = _normalize(np.stack(batch, axis=0), normalize)
    return batch

"""
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
"""
