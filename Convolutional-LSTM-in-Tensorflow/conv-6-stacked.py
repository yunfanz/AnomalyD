import glob, os
import numpy as np
DATA_PATH = '6-stacked'
DATA_FMT = '*/*.npy'
OUT_PATH = '/datax/6-stacked'

files = glob.glob(os.path.join(DATA_PATH, DATA_FMT))
for file in files:
    dat = np.load(file)
    print(file, dat.shape)
    spl = file[len(DATA_PATH):-4].split('/')
    base_name = os.path.join(OUT_PATH, '_'.join(spl))
    
    for i in range(dat.shape[0]):
        out_name = base_name + '_' + str(i)
        if not os.path.exists(out_name):
            np.save(out_name, dat[i].reshape(*dat.shape[1:], 1))

