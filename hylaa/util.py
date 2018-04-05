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
