'''
Stanley Bak
April 2017

OpenBLAS is a fast parallel library for matrix operations, which is 
typically multithreaded for large matrices. This file provides some
functions which let you control certain aspects of OpenBLAS.

has_openblas()     - checks if OpenBLAS was detected on the system
get_num_threads()  - gets the number of threads used by OpenBLAS, or -1
set_num_threads(n) - sets the number of threads if OpenBLAS is installed
OpenBlasThreads(n) - a context object which saves and sets the number of 
                     threads OpenBLAS uses. This works even if OpenBLAS 
                     is not installed (in which case, it does nothing).


Example code:

print "OpenBLAS detected: {}".format(has_openblas())

with OpenBlasThreads(4):
    print "Number of threads: ", get_num_threads()


Reference link: http://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas#
'''

import ctypes
from ctypes.util import find_library

# prioritize hand-compiled OpenBLAS library over default in /usr/lib
paths = ['/opt/OpenBLAS/lib/libopenblas.so',
             '/usr/lib/openblas-base/libblas.so.3',
             '/lib/libopenblas.so',
             '/usr/lib/libopenblas.so.0',
             find_library('openblas')]
openblas_lib = None

for lib_path in paths:
    try:
        openblas_lib = ctypes.cdll.LoadLibrary(lib_path)
        break
    except OSError:
        continue
        
def has_openblas():
    'was openblas sucessfully detected?'
    return openblas_lib is not None


class OpenBlasThreads(object):

    def __init__(self, num_threads):
        self._old_num_threads = get_num_threads()
        self.num_threads = num_threads

    def __enter__(self):
        set_num_threads(self.num_threads)

    def __exit__(self, *args):
        set_num_threads(self._old_num_threads)

def set_num_threads(n):
    'Set the current number of threads used by OpenBLAS'

    if has_openblas():
        openblas_lib.openblas_set_num_threads(int(n))

def get_num_threads():
    'Get the current number of threads used by OpenBLAS'
    rv = -1

    if has_openblas():
        rv = openblas_lib.openblas_get_num_threads()

    return rv
