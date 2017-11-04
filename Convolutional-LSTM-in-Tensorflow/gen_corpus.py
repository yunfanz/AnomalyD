import numpy as np
import pylab as plt
from blimpy import Waterfall
import os, time, fnmatch
from scipy import ndimage
import scipy.misc, fitsio
from skimage import measure
from joblib import Parallel, delayed


def get_pulse(outdir, ind, batch_size=None, ret=False, noise=True):
    if ind % 5000 == 0:
        print("{}".format(ind))
    if batch_size is None:
        batch_size = np.random.randint(1,5, size=1)
    seed = (int(time.clock()*1000)*ind)%(4294967295)  #2^32-1
    np.random.seed(seed=seed)
    f_0 = np.random.uniform(0., 512., size=batch_size)[...,np.newaxis, np.newaxis] #in ms 
    amp = np.random.uniform(0.25, 2.5, size=batch_size)[...,np.newaxis, np.newaxis]
    width = np.random.uniform(1., 20., size=batch_size)[...,np.newaxis, np.newaxis]
    slope = np.random.uniform(-15, 15, size=batch_size)[...,np.newaxis, np.newaxis]
    t = np.arange(16)[np.newaxis, ..., np.newaxis]
    f = np.arange(512)[np.newaxis, np.newaxis, ...]
    f0_all = f_0 + slope * t
    pulse = np.sum(amp*np.exp(-0.5 * (f - f0_all) ** 2 / width ** 2.), axis=0)
    if noise:
        noise_level = 0.2
        pulse += noise_level * np.random.random(pulse.shape)
    fitsio.write(outdir+"img_"+str(ind)+'.fits', pulse)
    scipy.misc.imsave(outdir+"pulse_"+str(ind)+'.png', pulse)
    if ret:
        return pulse
    #np.save(outdir+"pulse_"+str(ind), pulse)
    #scipy.misc.imsave(outdir+"pulse_"+str(ind)+'.png', measure.block_reduce(pulse, (32,1), np.mean))

def gen_train(N, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in xrange(N):
        pulse = get_pulse(outdir, i)

def gen_train_parallel(N, outdir):
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    _ = Parallel(n_jobs=8)(delayed(get_pulse)(outdir, i) for i in xrange(N))

def gen_noise_corpus(N, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in xrange(N):
        pulse = get_pulse(outdir, i, batch_size=0)

if __name__ == "__main__":
    OUTDIR = "./Data/simu2/"
    gen_train_parallel(32000, OUTDIR)