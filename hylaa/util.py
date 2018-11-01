'''
General python utilities, which aren't necessary specific to Hylaa's objects.

Methods / Classes in this one shouldn't require non-standard imports.
'''

import os
import sys

from collections import deque

class Freezable():
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

DID_PYTHON3_CHECK = False

if not DID_PYTHON3_CHECK:
    # check that we're using python 3

    if sys.version_info < (3, 0):
        sys.stdout.write("Hylaa requires Python 3, but was run with Python {}.{}.\n".format(
            sys.version_info[0], sys.version_info[1]))
        sys.exit(1)

def execute_delayed_action(actions):
    '''execute the first delayed action in the given deque, where each element is (func, args)

    where func returns a 2-tuple: list_of_new_actions, should_pause_plot

    this modifies the actions list in place, and returns True/False if we should pauase after the given action
    '''

    assert isinstance(actions, deque)
    should_pause = False

    while not should_pause and actions:
        func, param = actions.popleft()

        more_actions, should_pause = func(*param)

        # if there were more actions, prepend them
        if more_actions:
            actions.extendleft(reversed(more_actions))

    return should_pause
