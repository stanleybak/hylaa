'''
General python utilities, which aren't necessary specific to Hylaa's objects.

Methods / Classes in this one shouldn't require non-standard imports.
'''

import os
import numpy as np

class Freezable(object):
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

def get_script_path(filename):
    '''get the path this script, pass in __file__ for the filename'''
    return os.path.dirname(os.path.realpath(filename))

def matrix_to_string(m):
    'get a matrix as a string'

    return "\n".join([", ".join([str(val) for val in row]) for row in m])

def safe_zeros(label, shape, dtype=float, use_empty=False, alloc=True):
    'check if enough memory is available and if so, allocate it using np.zeros'

    assert isinstance(label, str)

    ## deprecated memory check (os.popen fails unncessarily sometimes)
    #gb_free = get_free_memory_mb(label) / 1024.
    #
    ## try to keep 1 GB free for other things
    #if gb_free > 1:
    #    gb_free -= 1
    #else:
    #    gb_free = 0
    #
    #bytes_needed = 8 # assume everything is float or int64... maybe a bit pessimistic
    #
    #for dim in shape:
    #    bytes_needed *= dim
    #
    #gb_needed = bytes_needed / 1024. / 1024. / 1024.
    #
    #if gb_needed > gb_free:
    #    raise RuntimeError('Memory allocation for {} failed. Free: {:.2f} GB, Needed: {:.2f} GB'.format(
    #        label, gb_free, gb_needed))
    
    rv = None

    if alloc:
        rv = np.zeros(shape, dtype=dtype) if not use_empty else np.empty(shape, dtype=dtype)

    return rv

def get_free_memory_mb_deprecated(label):
    'get the amount of free memory available'

    # This function fails under high memory usages

    try:
        # one-liner to get free memory from:
        # https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
        _, _, available_mb = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    except OSError:
        raise RuntimeError('OS.popen failed (likely not enough memory): {}'.format(label))

    return available_mb
