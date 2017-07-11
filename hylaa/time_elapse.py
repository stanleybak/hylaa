'''
Time Elapse Computation. This module is primarily responsive for computing
l * e^{At} where l is some direction of interest, and t is a multiple of some time step
'''

import sys
import numpy as np

from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import expm, expm_multiply

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.containers import HylaaSettings, PlotSettings, SimulationSettings
from hylaa.timerutil import Timers

class TimeElapser(Freezable):
    'Object which computes the time-elapse function for a single mode at multiples of the time step'

    def __init__(self, mode, hylaa_settings):
        assert isinstance(mode, LinearAutomatonMode)
        assert isinstance(hylaa_settings, HylaaSettings)

        self.settings = hylaa_settings
        self.a_matrix = mode.a_matrix
        self.b_matrix = mode.b_matrix
        self.dims = self.a_matrix.shape[0]
        self.inputs = 0 if mode.b_matrix is None else mode.b_matrix.shape[1]

        self.next_step = 0
        self.key_dir_mat = None # csr_matrix
        self.cur_time_elapse_mat = None # assigned on step()
        self.cur_input_effects_matrix = None # assigned on step() if inputs exist
        self.cur_input_projection_matrix = None

        # used for certain simulation modes
        self.one_step_matrix_exp = None # one step matrix exponential
        self.one_step_input_effects_matrix = None # one step input effects matrix, if inputs exist
        self._extract_key_directions(mode)

        self.freeze_attrs()

    def _extract_key_directions(self, mode):
        'extract the key directions for lp solving'

        num_directions = 0 if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE else 2

        for t in mode.transitions:
            num_directions += t.guard_matrix.shape[0]

        lil_dir_mat = lil_matrix((num_directions, self.dims), dtype=float)

        # fill the matrix
        dir_index = 0

        if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            if isinstance(self.settings.plot.xdim_dir, int):
                lil_dir_mat[0, self.settings.plot.xdim_dir] = 1.0
            else:
                lil_dir_mat[0, :] = self.settings.plot.xdim_dir

            if isinstance(self.settings.plot.ydim_dir, int):
                lil_dir_mat[1, self.settings.plot.ydim_dir] = 1.0
            else:
                lil_dir_mat[1, :] = self.settings.plot.ydim_dir

            dir_index += 2

        for t in mode.transitions:
            for row in t.guard_matrix:
                lil_dir_mat[dir_index, :] = row
                dir_index += 1

        # done constructing, convert to csc_matrix
        self.key_dir_mat = csr_matrix(lil_dir_mat)

    def step_exp_mult(self):
        'first step matrix exp, other steps matrix multiplication'

        if self.next_step == 0:
            self.cur_time_elapse_mat = np.array(self.key_dir_mat.todense(), dtype=float)
        elif self.one_step_matrix_exp is None:
            assert self.next_step == 1
            assert isinstance(self.key_dir_mat, csr_matrix)
            Timers.tic('time_elapse.step first step')

            print_status = self.a_matrix.shape[0] > 100

            if print_status:
                print "Computing the one-step matrix exponential for the {}-dimensional system...".format(
                    self.a_matrix.shape[0]),
                sys.stdout.flush()

            self.one_step_matrix_exp = np.array(expm(self.a_matrix * self.settings.step).todense(), dtype=float)

            if print_status:
                print "done"


            self.cur_time_elapse_mat = self.key_dir_mat * self.one_step_matrix_exp

            if self.inputs > 0:
                self.one_step_input_effects_matrix = np.zeros(self.b_matrix.shape, dtype=float)

                for c in xrange(self.inputs):
                    # create the a_matrix augmented with a column of the b_matrix as an affine term
                    a = self.a_matrix
                    b = self.b_matrix

                    indptr = b.indptr

                    data = np.concatenate((a.data, b.data[indptr[c]:indptr[c+1]]))
                    indices = np.concatenate((a.indices, b.indices[indptr[c]:indptr[c+1]]))
                    indptr = np.concatenate((a.indptr, [len(data)]))

                    aug_a_matrix = csc_matrix((data, indices, indptr), shape=(self.dims + 1, self.dims + 1))

                    mat = aug_a_matrix * self.settings.step

                    # the last column of matrix_exp is the same as multiplying it by the initial state [0, 0, ..., 1]
                    init_state = np.zeros(self.dims + 1, dtype=float)
                    init_state[self.dims] = 1.0
                    col = expm_multiply(mat, init_state)

                    #matrix_exp = np.array(expm(mat).todense(), dtype=float)
                    #col = matrix_exp[:, -1]

                    self.one_step_input_effects_matrix[:, c] = col[:self.dims]

                self.cur_input_projection_matrix = np.array(self.key_dir_mat.toarray(), dtype=float)
                self.cur_input_effects_matrix = np.dot(self.cur_input_projection_matrix,
                                                       self.one_step_input_effects_matrix)

            Timers.toc('time_elapse.step first step')
        else:
            Timers.tic('time_elapse.step other steps')

            self.cur_time_elapse_mat = np.dot(self.cur_time_elapse_mat, self.one_step_matrix_exp)

            # inputs
            if self.inputs > 0:
                self.cur_input_projection_matrix = np.dot(self.cur_input_projection_matrix, self.one_step_matrix_exp)

                self.cur_input_effects_matrix = np.dot(self.cur_input_projection_matrix,
                                                       self.one_step_input_effects_matrix)

            Timers.toc('time_elapse.step other steps')

    def step_matrix_exp(self):
        'matrix exp every step'

        cur_time = self.settings.step * self.next_step
        time_mat = self.a_matrix * cur_time
        exp = expm(time_mat)

        self.cur_time_elapse_mat = np.array((self.key_dir_mat * exp).todense(), dtype=float)

        if self.inputs != 0 and self.next_step > 0:
            input_effects_matrix = np.zeros(self.b_matrix.shape, dtype=float)

            for c in xrange(self.inputs):
                # create the a_matrix augmented with a column of the b_matrix as an affine term
                a = self.a_matrix
                b = self.b_matrix

                indptr = b.indptr

                data = np.concatenate((a.data, b.data[indptr[c]:indptr[c+1]]))
                indices = np.concatenate((a.indices, b.indices[indptr[c]:indptr[c+1]]))
                indptr = np.concatenate((a.indptr, [len(data)]))

                aug_a_matrix = csc_matrix((data, indices, indptr), shape=(self.dims + 1, self.dims + 1))

                matrix_exp = np.array(expm(aug_a_matrix * self.settings.step).todense(), dtype=float)

                # the last column of matrix_exp is the same as multiplying it by the initial state [0, 0, ..., 1]
                col = matrix_exp[:, -1]

                input_effects_matrix[:, c] = col[:self.dims]

            prev_exp = expm(self.a_matrix * (self.settings.step * (self.next_step - 1)))
            full_input_effects = (prev_exp * input_effects_matrix)
            self.cur_input_effects_matrix = self.key_dir_mat * full_input_effects

    def step(self):
        'perform the computation to obtain the values of the key directions the current time'

        Timers.tic('time_elapse.step Total')

        if self.settings.simulation.sim_mode == SimulationSettings.MATRIX_EXP:
            self.step_matrix_exp()
        elif self.settings.simulation.sim_mode == SimulationSettings.EXP_MULT:
            self.step_exp_mult()
        else:
            raise RuntimeError('Unimplemented sim_mode {}'.format(self.settings.simulation.sim_mode))

        self.next_step += 1

        Timers.toc('time_elapse.step Total')

        # post-conditions check
        assert isinstance(self.cur_time_elapse_mat, np.ndarray), "cur_time_elapse_mat should be an np.array"
        assert self.cur_time_elapse_mat.shape == self.key_dir_mat.shape, \
            "cur_time_elapse mat shape({}) should be {}".format(self.cur_time_elapse_mat.shape, self.key_dir_mat.shape)

        if self.inputs == 0 or self.next_step == 1: # 0-th step input should be null
            assert self.cur_input_effects_matrix is None
        else:
            assert isinstance(self.cur_input_effects_matrix, np.ndarray)
            assert self.cur_input_effects_matrix.shape == (self.key_dir_mat.shape[0], self.inputs)
