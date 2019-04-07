#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys
im = np.load(sys.argv[1])
plt.imshow(im)
plt.title(sys.argv[1])
plt.show()
