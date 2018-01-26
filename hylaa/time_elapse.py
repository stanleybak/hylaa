'''
Time Elapse Computation. This module is primarily responsive for computing
l * e^{At} where l is some direction of interest, and t is a multiple of some time step
'''

import sys

import numpy as np

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm, expm_multiply

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.settings import HylaaSettings, PlotSettings, SimulationSettings
from hylaa.timerutil import Timers
from hylaa.time_elapse_krylov import make_cur_basis_mat_list
from hylaa.krylov_python import KrylovIterator

class TimeElapser(Freezable):
    'Object which computes the time-elapse function for a single mode at multiples of the time step'

    def __init__(self, mode, hylaa_settings, init_space_csc):
        assert isinstance(mode, LinearAutomatonMode)
        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(init_space_csc, csc_matrix)

        self.settings = hylaa_settings
        self.init_space_csc = init_space_csc

        if self.settings.simulation.sim_mode == SimulationSettings.MATRIX_EXP or \
           self.settings.simulation.sim_mode == SimulationSettings.EXP_MULT or \
            self.settings.simulation.check_answer:
            Timers.tic("convert a_matrix and b_matrix to csc matrix")
            self.a_matrix_csc = csc_matrix(mode.a_matrix)
            self.b_matrix_csc = None if mode.b_matrix is None else csc_matrix(mode.b_matrix)
            Timers.toc("convert a_matrix and b_matrix to csc matrix")

        self.a_matrix = mode.a_matrix
        self.b_matrix = mode.b_matrix

        self.dims = self.a_matrix.shape[0]
        self.inputs = 0 if mode.b_matrix is None else mode.b_matrix.shape[1]

        self.next_step = 0
        self.key_dir_mat = None # csr_matrix
        self.cur_basis_mat = None # assigned on step()
        self.cur_input_effects_matrix = None # assigned on step() if inputs exist
        self.cur_input_projection_matrix = None

        # used for certain simulation modes
        Timers.tic("extract key directions")
        self._extract_key_directions(mode)
        Timers.toc("extract key directions")

        # method-specific settings
        if self.settings.simulation.sim_mode == SimulationSettings.KRYLOV:
            self.cur_basis_mat_list = None

            if self.settings.simulation.krylov_transpose:
                output_mat = csr_matrix(self.init_space_csc.transpose())
            else:
                output_mat = self.key_dir_mat

            self.krylov_iterator = KrylovIterator(hylaa_settings, self.a_matrix, output_mat)

        elif self.settings.simulation.sim_mode == SimulationSettings.EXP_MULT:
            self.stored_vec = None
            self.one_step_matrix_exp = None # one step matrix exponential
            self.one_step_input_effects_matrix = None # one step input effects matrix, if inputs exist

        # -- performance statistics --
        # arnoldi_iter -> list, with 0 = fixed-effect, others are the tuned arnoldi iterations
        # arnoldi_mem_start -> mb, total amount of memory on device
        #
        self.stats = {} # performance statistics, map name -> value

        self.freeze_attrs()

    def _extract_key_directions(self, mode):
        'extract the key directions for lp solving'

        num_directions = 0

        data = []
        cols = []
        indptr = [0]

        if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            dirs = [self.settings.plot.xdim_dir, self.settings.plot.ydim_dir]

            for plot_dir in dirs:
                if isinstance(plot_dir, int):
                    data.append(1.0)
                    cols.append(plot_dir)
                    indptr.append(len(data))
                    num_directions += 1
                elif plot_dir is not None:
                    xdir = csr_matrix(plot_dir)
                    data += [n for n in xdir.data]
                    cols += [n for n in xdir.cols]
                    indptr.append(len(data))
                    num_directions += 1

        for t in mode.transitions:
            num_directions += t.output_space_csr.shape[0]
            offset = len(data)

            data += [n for n in t.output_space_csr.data]
            cols += [n for n in t.output_space_csr.indices]
            indptr += [i + offset for i in t.output_space_csr.indptr[1:]]

            # actually, we just need to add one of the transitions since they must share the same output space
            break

        self.key_dir_mat = csr_matrix((data, cols, indptr), shape=(num_directions, self.dims), dtype=float)

    def step_exp_mult(self):
        'first step matrix exp, other steps matrix multiplication'

        if self.next_step == 0:

            self.cur_basis_mat = (self.key_dir_mat * self.init_space_csc).toarray()

            # store either output_vec or input_vec
            if self.settings.simulation.exp_mult_output_vec:
                self.stored_vec = self.key_dir_mat.toarray()
            else:
                self.stored_vec = self.init_space_csc.toarray()
        elif self.one_step_matrix_exp is None:
            assert self.next_step == 1
            assert isinstance(self.key_dir_mat, csr_matrix)
            Timers.tic('time_elapse.step first step')

            print_status = self.a_matrix.shape[0] > 100 and self.settings.print_output

            if print_status:
                print "Computing the one-step matrix exponential for the {}-dimensional system...".format(
                    self.a_matrix.shape[0]),
                sys.stdout.flush()

            a_step_mat = self.a_matrix_csc * self.settings.step

            self.one_step_matrix_exp = np.array(expm(a_step_mat).todense(), dtype=float)

            if print_status:
                print "done"

            if self.settings.simulation.exp_mult_output_vec:
                self.stored_vec = np.dot(self.stored_vec, self.one_step_matrix_exp)
                self.cur_basis_mat = self.stored_vec * self.init_space_csc
            else:
                self.stored_vec = np.dot(self.one_step_matrix_exp, self.stored_vec)
                self.cur_basis_mat = self.key_dir_mat * self.stored_vec

            self.cur_basis_mat = self.cur_basis_mat.copy() # make it c-contiguous (instead of fortran-contiguous)

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


            if self.settings.simulation.exp_mult_output_vec:
                self.stored_vec = np.dot(self.stored_vec, self.one_step_matrix_exp)
                self.cur_basis_mat = self.stored_vec * self.init_space_csc
            else:
                self.stored_vec = np.dot(self.one_step_matrix_exp, self.stored_vec)
                self.cur_basis_mat = self.key_dir_mat * self.stored_vec

            self.cur_basis_mat = self.cur_basis_mat.copy() # make it c-contiguous (instead of fortran-contiguous)

            # inputs
            if self.inputs > 0:
                self.cur_input_projection_matrix = np.dot(self.cur_input_projection_matrix, self.one_step_matrix_exp)

                self.cur_input_effects_matrix = np.dot(self.cur_input_projection_matrix,
                                                       self.one_step_input_effects_matrix)

            Timers.toc('time_elapse.step other steps')

    def step_matrix_exp(self):
        'matrix exp every step'

        cur_time = self.settings.step * self.next_step
        time_mat = self.a_matrix_csc * cur_time
        exp = expm(time_mat)

        self.cur_basis_mat = np.array((self.key_dir_mat * exp * self.init_space_csc).todense(), dtype=float)

        if self.inputs != 0 and self.next_step > 0:
            input_effects_matrix = np.zeros(self.b_matrix.shape, dtype=float)

            for c in xrange(self.inputs):
                # create the a_matrix augmented with a column of the b_matrix as an affine term
                a = self.a_matrix_csc
                b = self.b_matrix_csc

                indptr = b.indptr

                data = np.concatenate((a.data, b.data[indptr[c]:indptr[c+1]]))
                indices = np.concatenate((a.indices, b.indices[indptr[c]:indptr[c+1]]))
                indptr = np.concatenate((a.indptr, [len(data)]))

                aug_a_matrix = csc_matrix((data, indices, indptr), shape=(self.dims + 1, self.dims + 1))

                matrix_exp = np.array(expm(aug_a_matrix * self.settings.step).todense(), dtype=float)

                # the last column of matrix_exp is the same as multiplying it by the initial state [0, 0, ..., 1]
                col = matrix_exp[:, -1]

                input_effects_matrix[:, c] = col[:self.dims]

            prev_exp = expm(self.a_matrix_csc * (self.settings.step * (self.next_step - 1)))
            full_input_effects = (prev_exp * input_effects_matrix)
            self.cur_input_effects_matrix = self.key_dir_mat * full_input_effects

    def step_krylov(self):
        'krylov-based step function'

        if self.cur_basis_mat_list is None:
            if self.settings.print_output:
                print "Using Krylov method to make basis matrices"
                
            self.cur_basis_mat_list = make_cur_basis_mat_list(self)

        self.cur_basis_mat = self.cur_basis_mat_list[self.next_step].copy()

    def step(self):
        'perform the computation to obtain the values of the key directions the current time'

        Timers.tic('time_elapse.step Total')

        if self.settings.simulation.sim_mode == SimulationSettings.MATRIX_EXP:
            self.step_matrix_exp()
        elif self.settings.simulation.sim_mode == SimulationSettings.EXP_MULT:
            self.step_exp_mult()
        elif self.settings.simulation.sim_mode == SimulationSettings.KRYLOV:
            self.step_krylov()
        else:
            raise RuntimeError('Unimplemented sim_mode {}'.format(self.settings.simulation.sim_mode))

        self.next_step += 1

        Timers.toc('time_elapse.step Total')

        # post-conditions check
        assert isinstance(self.cur_basis_mat, np.ndarray), "cur_basis_mat should be an np.array, " + \
            "but it was {}".format(type(self.cur_basis_mat))

        expected_basis_shape = (self.key_dir_mat.shape[0], self.init_space_csc.shape[1])

        assert self.cur_basis_mat.shape == expected_basis_shape, \
            "cur_basis mat shape({}) should be {}".format(self.cur_basis_mat.shape, expected_basis_shape)

        if self.inputs == 0 or self.next_step == 1: # 0-th step input should be null
            assert self.cur_input_effects_matrix is None
        else:
            assert isinstance(self.cur_input_effects_matrix, np.ndarray)
            assert self.cur_input_effects_matrix.shape == (self.key_dir_mat.shape[0], self.inputs)

        # answer accuracy check (optional)
        if self.settings.simulation.check_answer:
            self.check_answer()

    def check_answer(self):
        'check the correctness of the answer versus expm'

        Timers.tic('expm check answer')
        assert self.a_matrix.shape[0] <= 1000, "settings.simulation.check_answer == True with large matrix"
        tol = self.settings.simulation.check_answer_abs_tol

        t = self.settings.step * (self.next_step - 1)
        exp = expm(self.a_matrix_csc * t)
        expected = np.array((self.key_dir_mat * exp * self.init_space_csc).todense(), dtype=float)

        assert self.cur_basis_mat.shape == expected.shape, \
            "wrong shape in check_answer(), got {}, expected {}".format(self.cur_basis_mat.shape, expected.shape)

        #print "expected:\n{}".format(expected)
        #print "got:\n{}".format(self.cur_basis_mat)

        for dim in xrange(expected.shape[1]):
            col_expected = expected[:, dim]
            col_got = self.cur_basis_mat[:, dim]

            same = True

            for a, b in zip(col_expected, col_got):
                if abs(a - b) > tol:
                    same = False
                    break

            if not same:
                print "Answer was incorrect in column {}".format(dim)
                print "Expected {}".format(col_expected)
                print "Got {}".format(col_got)

                assert False, "answer was incorrect"

        Timers.toc('expm check answer')
