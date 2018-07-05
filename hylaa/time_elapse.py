'''
Time Elapse Computation. This module is primarily responsive for computing
l * e^{At} where l is some direction of interest, and t is a multiple of some time step
'''

import numpy as np

from hylaa.util import Freezable
from hylaa.timerutil import Timers
from hylaa.time_elapse_expm import TimeElapseExpmMult
from hylaa.hybrid_automaton import Mode

class TimeElapser(Freezable):
    'Object which computes the time-elapse function for a single mode at multiples of the time step'

    def __init__(self, mode, step_size):
        assert isinstance(mode, Mode)

        self.mode = mode
        self.a_matrix = mode.a_matrix
        self.b_matrix = mode.b_matrix
        self.step_size = step_size
        self.dims = self.a_matrix.shape[0]
        self.inputs = 0 if self.b_matrix is None else self.b_matrix.shape[1]

        self.time_elapse_obj = None

        self.freeze_attrs()

    def get_basis_matrix(self, step_num):
        '''perform the computation for the the basis matrix and input effects matrix at the passed-in step

        returns a tuple (basis_matrix, input_effects matrix)
        '''

        if self.time_elapse_obj is None:
            Timers.tic('init time_elapse_obj')
            self.time_elapse_obj = TimeElapseExpmMult(self)
            Timers.toc('init time_elapse_obj')

        Timers.tic('step')
        self.time_elapse_obj.assign_basis_matrix(step_num)
        Timers.toc('step')

        basis_mat = self.time_elapse_obj.cur_basis_matrix
        input_effects_mat = self.time_elapse_obj.cur_input_effects_matrix

        # post-conditions check
        assert isinstance(basis_mat, np.ndarray), "cur_basis_mat should be an np.array, " + \
            "but it was {}".format(type(basis_mat))

        assert basis_mat.shape == (self.dims, self.dims), \
            "cur_basis mat shape({}) should be {}".format(basis_mat.shape, (self.dims, self.dims))

        if self.inputs == 0 or step_num == 0: # 0-th step input should be null
            assert input_effects_mat is None
        else:
            assert isinstance(input_effects_mat, np.ndarray)
            assert input_effects_mat.shape == (self.dims, self.inputs)

        return basis_mat, input_effects_mat

