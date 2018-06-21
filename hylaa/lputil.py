'''
LP Utilities

This assumes a common LP structure, where the
first N columns correspond to the current-time variables, and
the first N rows are the current-time constraints (equality constraints equal to zero)
'''

from hylaa.glpk.python_sparse_glpk import LpInstance

def from_box(box_list):
    'make a new lp instance from a passed-in box'

    here

