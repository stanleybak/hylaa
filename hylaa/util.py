'''
General python utilities, which aren't necessary specific to Hylaa's objects.

Methods / Classes in this one shouldn't require non-standard imports.
'''

import os

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

def get_free_memory_mb_deprecated():
    'get the amount of free memory available'

    # one-liner to get free memory from:
    # https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
    _, _, available_mb = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

    return available_mb
