import glob, os
import numpy as np
import fnmatch
DATA_PATH = '/home/yunfanz/Data/6-stacked/'
DATA_FMT = '*.npy'
OUT_PATH = '/home/yunfanz/Data/6-stacked-max200'
MAX_SAMP = 200

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
def find_files(directory, pattern='*.png', sortby="shuffle"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    if sortby == 'auto':
        files = np.sort(files)
    elif sortby == 'shuffle':
        np.random.shuffle(files)
    return files

#files = glob.glob(os.path.join(DATA_PATH, DATA_FMT))
files = find_files(DATA_PATH, pattern=DATA_FMT)
print("{} files".format(len(files)))
for file in files:
    dat = np.load(file)
    spl = file[len(DATA_PATH)+1:-4].split('/')
    base_name = os.path.join(OUT_PATH, '_'.join(spl))
    
    nimgs, _, h, w = dat.shape
    if nimgs >= MAX_SAMP:
        choices = np.random.choice(np.arange(nimgs), MAX_SAMP)
        dat = dat[choices]
    else:
        nsamps = min(3*nimgs, MAX_SAMP)
        choices = np.random.choice(np.arange(nimgs), nsamps, replace=True)
        dat = dat[choices]
    for i in range(dat.shape[0]):
        out_name = base_name + '_' + str(i)
        if not os.path.exists(out_name):
            np.save(out_name, dat[i].reshape(*dat.shape[1:], 1))

