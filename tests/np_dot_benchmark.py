'''
Small benchmark to test the speed of matrix multiplication. 

Using OpenBlas will make dot multithreaded. When you run this benchmark,
use top or htop commands to make sure multiple threads are being used.

Stanley Bak
April 2017


See configuration:
> import numpy
> numpy.show_config()

Instructions to install openblas and link openblas to existing numpy:

http://stackoverflow.com/questions/36483054/install-openblas-via-apt-get-sudo-apt-get-install-openblas-dev
> apt-cache search  openblas (see what packages are available
> dpkg -L libopenblas-base (see where it's installed)

http://stackoverflow.com/questions/21671040/link-atlas-mkl-to-an-installed-numpy
> ldd /<path_to_site-packages>/numpy/core/_dotblas.so (see which library numpy.dot uses)

Commands:
> sudo apt-get install libopenblas-base
> sudo update-alternatives --install /usr/lib/libblas.so.3 libblas.so.3 /usr/lib/libopenblas.so.0 50

(optional, check the one being used, should be set based on priority 50 in the above command)
> sudo update-alternatives --config libblas.so.3
'''

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
