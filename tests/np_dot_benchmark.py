'''
Small benchmark to test the speed of matrix multiplication. 

Using OpenBlas will make dot multithreaded. When you run this benchmark,
use top or htop commands to make sure multiple threads are being used.

Stanley Bak
April 2017


See numpy configuration (maybe openblas is already being used):
> python
> import numpy
> numpy.show_config()

To set the number of threads used by openblas, use the environment variable OMP_NUM_THREADS:
>export OMP_NUM_THREADS=1

Some resources:

http://stackoverflow.com/questions/36483054/install-openblas-via-apt-get-sudo-apt-get-install-openblas-dev
> apt-cache search  openblas (see what packages are available
> dpkg -L libopenblas-base (see where it's installed)

http://stackoverflow.com/questions/21671040/link-atlas-mkl-to-an-installed-numpy
> ldd /<path_to_site-packages>/numpy/core/_dotblas.so (see which library numpy.dot uses, if numpy.__version__ < 1.10)
-- or --
> ldd /usr/lib/python2.7/dist-packages/numpy/core/multiarray.x86_64-linux-gnu.so    (if numpy.__version__ >= 1.10)
usually the answer will be /usr/lib/libblas.so.3

Commands to install openblas and link openblas to existing numpy:
> sudo apt-get install libopenblas-base
> sudo update-alternatives --install /usr/lib/libblas.so.3 libblas.so.3 /usr/lib/libopenblas.so.0 50

(optional, check the one being used, should be set based on priority 50 in the above command)
> sudo update-alternatives --config libblas.so.3


Here are some sample results:
########## Intel Core i5-5300U @ 2.30GHz (4 cores) ############
Without OpenBLAS:
size 1000, time = 0.19
size 2000, time = 1.48
size 4000, time = 11.88

With OpenBLAS:
size 1000, time = 0.12
size 2000, time = 1.01
size 4000, time = 8.03

########## Intel Core i7-5930K @ 3.50GHz (12 cores) ############
Without OpenBLAS:
size 1000, time = 0.57
size 2000, time = 5.54
size 4000, time = 45.93

With OpenBLAS:
size 1000, time = 0.04
size 2000, time = 0.14
size 4000, time = 0.60
'''

import numpy as np
import time
from hylaa.openblas import has_openblas

def mult(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)

    start = time.time()

    np.dot(a, b)

    diff = time.time() - start
    print "size {}, time = {:.2f}".format(size, diff)

print "OpenBLAS detected: {}".format(has_openblas())
mult(1000)
mult(2000)
mult(4000)


