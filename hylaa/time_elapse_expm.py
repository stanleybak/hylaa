'''
Time-elapse object for matrix exponential and expm-mul methods

Stanley Bak
April 2018
'''

import sys

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply

import numpy as np

from hylaa.util import Freezable
from hylaa.timerutil import Timers

class TimeElapseMatrixExp(Freezable):
    'container object for expm-at-each step method'

    def __init__(self, time_elapser):
        self.time_elapser = time_elapser
        self.a_matrix_csc = csc_matrix(time_elapser.a_matrix)
        self.b_matrix_csc = None if time_elapser.b_matrix is None else csc_matrix(time_elapser.b_matrix)

        self.freeze_attrs()

    def step(self):
        'matrix exp every step'

        cur_time = self.time_elapser.settings.step * self.time_elapser.next_step
        time_mat = self.a_matrix_csc * cur_time
        exp = expm(time_mat)

        init_space = self.time_elapser.init_space_csc
        output_space = self.time_elapser.output_space_csr

        self.time_elapser.cur_basis_mat = (output_space * exp * init_space).toarray()

        # compute input effects
        if self.time_elapser.inputs != 0 and self.time_elapser.next_step > 0:
            input_effects_matrix = np.zeros(self.time_elapser.b_matrix.shape, dtype=float)

            for c in xrange(self.time_elapser.inputs):
                # create the a_matrix augmented with a column of the b_matrix as an affine term
                a = self.a_matrix_csc
                b = self.b_matrix_csc

                indptr = b.indptr

                data = np.concatenate((a.data, b.data[indptr[c]:indptr[c+1]]))
                indices = np.concatenate((a.indices, b.indices[indptr[c]:indptr[c+1]]))
                indptr = np.concatenate((a.indptr, [len(data)]))

                dims = self.time_elapser.dims
                aug_a_matrix = csc_matrix((data, indices, indptr), shape=(dims + 1, dims + 1))

                matrix_exp = expm(aug_a_matrix * self.time_elapser.settings.step).toarray()

                # the last column of matrix_exp is the same as multiplying it by the initial state [0, 0, ..., 1]
                col = matrix_exp[:, -1]

                input_effects_matrix[:, c] = col[:dims]

            prev_exp = expm(self.a_matrix_csc * (self.time_elapser.settings.step * (self.time_elapser.next_step - 1)))
            full_input_effects = (prev_exp * input_effects_matrix)
            self.time_elapser.cur_input_effects_matrix = output_space * full_input_effects

class TimeElapseExpmMult(Freezable):
    'container object for expm + matrix-vec mult method'

    def __init__(self, time_elapser):
        self.settings = time_elapser.settings

        self.time_elapser = time_elapser
        self.a_matrix_csc = csc_matrix(time_elapser.a_matrix)
        self.b_matrix_csc = None if time_elapser.mode.b_matrix is None else csc_matrix(time_elapser.b_matrix)

        self.cur_input_projection_matrix = None

        self.stored_vec = None
        self.one_step_matrix_exp = None # one step matrix exponential
        self.one_step_input_effects_matrix = None # one step input effects matrix, if inputs exist

        self.freeze_attrs()

    def step(self):
        'first step matrix exp, other steps matrix multiplication'

        init_space = self.time_elapser.init_space_csc
        output_space = self.time_elapser.output_space_csr
        dims = self.time_elapser.dims

        if self.time_elapser.next_step == 0: # step zero
            self.time_elapser.cur_basis_mat = (output_space * init_space).toarray()

            # store from either output_vec or input_vec
            if self.time_elapser.use_init_space:
                self.stored_vec = output_space.toarray()
            else:
                self.stored_vec = init_space.toarray()

        elif self.one_step_matrix_exp is None: # step one
            assert self.time_elapser.next_step == 1
            Timers.tic('time_elapse.exp_mult first step')

            print_status = dims > 100 and self.settings.print_output

            if print_status:
                print "Computing the one-step matrix exponential for the {}-dimensional system...".format(dims),
                sys.stdout.flush()

            a_step_mat = self.a_matrix_csc * self.settings.step

            self.one_step_matrix_exp = expm(a_step_mat).toarray()

            if print_status:
                print "done"

            if self.time_elapser.use_init_space:
                self.stored_vec = np.dot(self.stored_vec, self.one_step_matrix_exp)
                self.time_elapser.cur_basis_mat = self.stored_vec * init_space
            else:
                self.stored_vec = np.dot(self.one_step_matrix_exp, self.stored_vec)
                self.time_elapser.cur_basis_mat = output_space * self.stored_vec

            # make it c-contiguous in memory (instead of fortran-contiguous)
            self.time_elapser.cur_basis_mat = self.time_elapser.cur_basis_mat.copy()

            if self.time_elapser.inputs > 0:
                self.one_step_input_effects_matrix = np.zeros(self.time_elapser.b_matrix.shape, dtype=float)

                for c in xrange(self.time_elapser.inputs):
                    # create the a_matrix augmented with a column of the b_matrix as an affine term
                    a = self.time_elapser.a_matrix
                    b = self.time_elapser.b_matrix

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

                self.cur_input_projection_matrix = np.array(output_space.toarray(), dtype=float)
                self.time_elapser.cur_input_effects_matrix = np.dot(self.cur_input_projection_matrix,
                                                                    self.one_step_input_effects_matrix)

            Timers.toc('time_elapse.exp_mult first step')
        else:
            Timers.tic('time_elapse.exp_mult other steps')

            if self.time_elapser.use_init_space:
                self.stored_vec = np.dot(self.stored_vec, self.one_step_matrix_exp)
                self.time_elapser.cur_basis_mat = self.stored_vec * init_space
            else:
                self.stored_vec = np.dot(self.one_step_matrix_exp, self.stored_vec)
                self.time_elapser.cur_basis_mat = output_space * self.stored_vec

            # make it c-contiguous (instead of fortran-contiguous)
            self.time_elapser.cur_basis_mat = self.time_elapser.cur_basis_mat.copy()

            # inputs
            if self.time_elapser.inputs > 0:
                self.cur_input_projection_matrix = np.dot(self.cur_input_projection_matrix, self.one_step_matrix_exp)

                self.time_elapser.cur_input_effects_matrix = np.dot(self.cur_input_projection_matrix,
                                                                    self.one_step_input_effects_matrix)

            Timers.toc('time_elapse.exp_mult other steps')
