import numpy as np
import time

def mult(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)

    start = time.time()

    np.dot(a, b)

    diff = time.time() - start
    print "size {}, time = {:.2f}".format(size, diff)

mult(1000)
mult(2000)
mult(4000)

########## Laptop ############
# original:
# size 1000, time = 0.19
# size 2000, time = 1.48
# size 4000, time = 11.88

# with openblas:
# size 1000, time = 0.12
# size 2000, time = 1.01
# size 4000, time = 8.03
