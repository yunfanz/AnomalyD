#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
im = plt.imread(sys.argv[1])
plt.imshow(im)
plt.title(sys.argv[1])
plt.show()
