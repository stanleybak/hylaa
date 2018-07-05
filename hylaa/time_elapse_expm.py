'''
Time-elapse object for matrix exponential and expm-mul methods

Stanley Bak
April 2018
'''

from scipy.sparse import csc_matrix
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

import numpy as np

from hylaa.util import Freezable
from hylaa.timerutil import Timers

class TimeElapseExpmMult(Freezable):
    'container object for expm + matrix-vec mult method'

    def __init__(self, time_elapser):
        self.settings = time_elapser.settings

        self.time_elapser = time_elapser
        self.a_matrix = time_elapser.a_matrix
        self.b_matrix = None if time_elapser.b_matrix is None else time_elapser.b_matrix
        self.dims = time_elapser.dims

        self.cur_step = 0
        self.cur_basis_matrix = None
        self.cur_input_effects_matrix = None

        self.one_step_matrix_exp = None # one step matrix exponential
        self.one_step_input_effects_matrix = None # one step input effects matrix, if inputs exist

        self.freeze_attrs()

    def init_matrices(self):
        'initialize the one-step basis and input effects matrices'

        dims = self.dims
        Timers.tic('expm')
        self.one_step_matrix_exp = expm(self.a_matrix * self.time_elapser.step_size)
        Timers.toc('expm')

        if self.b_matrix is not None:
            self.one_step_input_effects_matrix = np.zeros(self.b_matrix.shape, dtype=float)

            for c in xrange(self.time_elapser.inputs):
                # create the a_matrix augmented with a column of the b_matrix as an affine term
                a = csc_matrix(self.a_matrix)
                b = self.time_elapser.b_matrix

                assert isinstance(a, csc_matrix)
                assert isinstance(b, csc_matrix)

                indptr = b.indptr

                data = np.concatenate((a.data, b.data[indptr[c]:indptr[c+1]]))
                indices = np.concatenate((a.indices, b.indices[indptr[c]:indptr[c+1]]))
                indptr = np.concatenate((a.indptr, [len(data)]))

                aug_a_matrix = csc_matrix((data, indices, indptr), shape=(dims + 1, dims + 1))

                mat = aug_a_matrix * self.settings.step

                # the last column of matrix_exp is the same as multiplying it by the initial state [0, 0, ..., 1]
                init_state = np.zeros(dims + 1, dtype=float)
                init_state[dims] = 1.0
                col = expm_multiply(mat, init_state)

                self.one_step_input_effects_matrix[:, c] = col[:dims]

    def assign_basis_matrix(self, step_num):
        'first step matrix exp, other steps matrix multiplication'

        Timers.tic('init_matrices')
        if self.one_step_matrix_exp is None:
            self.init_matrices()
        Timers.toc('init_matrices')

        if step_num == 0: # step zero, basis matrix is identity matrix
            self.time_elapser.cur_basis_mat = np.identity(self.dims, dtype=float)
            self.time_elapser.cur_input_effects_matrix = None
        elif step_num == 1:
            self.cur_basis_matrix = self.one_step_matrix_exp
            self.cur_input_effects_matrix = self.one_step_input_effects_matrix
        elif step_num == self.cur_step + 1:
            Timers.tic('quick_step')
            self.cur_basis_matrix = np.dot(self.cur_basis_matrix, self.one_step_matrix_exp)

            # inputs
            if self.b_matrix is not None:
                self.cur_input_effects_matrix = np.dot(self.one_step_matrix_exp, self.cur_input_effects_matrix)

            Timers.toc('quick_step')
        else:
            Timers.tic('slow_step')

            Timers.tic('expm')
            mat_exp = expm(self.a_matrix * step_num)
            Timers.toc('expm')

            self.cur_basis_matrix = mat_exp

            # inputs
            if self.b_matrix is not None:
                self.cur_input_effects_matrix = np.dot(mat_exp, self.one_step_input_effects_matrix)

            Timers.toc('slow_step')

        self.cur_step = step_num
