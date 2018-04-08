'''
Time Elapse Computation. This module is primarily responsive for computing
l * e^{At} where l is some direction of interest, and t is a multiple of some time step
'''

import numpy as np

from scipy.sparse import csr_matrix, csc_matrix

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.settings import HylaaSettings, PlotSettings, TimeElapseSettings
from hylaa.timerutil import Timers
from hylaa.time_elapse_expm import TimeElapseMatrixExp, TimeElapseExpmMult
from hylaa.time_elapse_krylov import TimeElapseKrylov
from hylaa.time_elapse_scipy_sim import TimeElapseScipySim

class TimeElapser(Freezable):
    'Object which computes the time-elapse function for a single mode at multiples of the time step'

    def __init__(self, mode, hylaa_settings, init_space_csc):
        assert isinstance(mode, LinearAutomatonMode)
        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(init_space_csc, csc_matrix)

        self.mode = mode
        self.settings = hylaa_settings
        self.a_matrix = mode.a_matrix_csr
        self.b_matrix = mode.b_matrix_csc
        self.dims = self.a_matrix.shape[0]
        self.inputs = 0 if self.b_matrix is None else self.b_matrix.shape[1]

        self.output_space_csr = create_output_space_csr(hylaa_settings.plot, mode)
        self.init_space_csc = init_space_csc

        self.next_step = 0
        self.cur_basis_mat = None # assigned on step()
        self.cur_input_effects_matrix = None # assigned on step() if inputs exist
        self.cur_max_one_norm = None # assigned on step() for certain methods

        self.use_init_space = self.settings.time_elapse.force_init_space

        if self.use_init_space is None:
            # auto detect strategy: use the lower dimension space
            self.use_init_space = self.init_space_csc.shape[1] <= self.output_space_csr.shape[0]

        # initialize method-specific container objects
        if self.settings.time_elapse.check_answer:
            self.checker_obj = TimeElapseMatrixExp(self)
        else:
            self.checker_obj = None

        method = self.settings.time_elapse.method

        if method == TimeElapseSettings.MATRIX_EXP:
            self.time_elapse_obj = TimeElapseMatrixExp(self)
        elif method == TimeElapseSettings.EXP_MULT:
            self.time_elapse_obj = TimeElapseExpmMult(self)
        elif method == TimeElapseSettings.KRYLOV:
            self.time_elapse_obj = TimeElapseKrylov(self)
        elif method == TimeElapseSettings.SCIPY_SIM:
            self.time_elapse_obj = TimeElapseScipySim(self)
        else:
            raise RuntimeError("Unsupported Time Elapse Method: {}".format(method))

        self.freeze_attrs()

    def step(self):
        'perform the computation to obtain the values of the key directions the current time'

        Timers.tic('time_elapse.step Total')

        self.time_elapse_obj.step()
        self.next_step += 1

        Timers.toc('time_elapse.step Total')

        # post-conditions check
        assert isinstance(self.cur_basis_mat, np.ndarray), "cur_basis_mat should be an np.array, " + \
            "but it was {}".format(type(self.cur_basis_mat))

        expected_basis_shape = (self.output_space_csr.shape[0], self.init_space_csc.shape[1])

        assert self.cur_basis_mat.shape == expected_basis_shape, \
            "cur_basis mat shape({}) should be {}".format(self.cur_basis_mat.shape, expected_basis_shape)

        if self.inputs == 0 or self.next_step == 1: # 0-th step input should be null
            assert self.cur_input_effects_matrix is None
        else:
            assert isinstance(self.cur_input_effects_matrix, np.ndarray)
            assert self.cur_input_effects_matrix.shape == (self.output_space_csr.shape[0], self.inputs)

        # answer accuracy check (optional)
        if self.checker_obj is not None:
            self.check_answer()

    def check_answer(self):
        'check the correctness of the answer at the current step'

        # save current basis matrix and current input effects matrix, as these will get overriden by check_obj.step()
        saved_basis_mat = self.cur_basis_mat
        saved_input_effects_matrix = self.cur_input_effects_matrix

        Timers.tic('expm check answer')

        assert self.a_matrix.shape[0] <= 1000, "check_answer = True with large matrix (dims > 1000)"
        tol = self.settings.time_elapse.check_answer_abs_tol

        # the step number was already advanced, so decrease it by one before calling step()
        self.next_step -= 1
        self.checker_obj.step()
        self.next_step += 1

        expected = self.cur_basis_mat
        expected_input = self.cur_input_effects_matrix

        # compare basis matrix
        for dim in xrange(expected.shape[1]):
            col_expected = expected[:, dim]
            col_got = saved_basis_mat[:, dim]

            same = True

            for a, b in zip(col_expected, col_got):
                if abs(a - b) > tol:
                    same = False
                    break

            if not same:
                print "Answer was incorrect in basis matrix column {}".format(dim)
                print "Expected {}".format(col_expected)
                print "Got {}".format(col_got)

                raise RuntimeError("answer was incorrect")

        # compare input effects matrix
        if expected_input is not None:
            assert saved_input_effects_matrix is not None, "incorrect answer: expected input effects matrix"

            for dim in xrange(expected_input.shape[1]):
                col_expected = expected_input[:, dim]
                col_got = saved_input_effects_matrix[:, dim]

                same = True

                for a, b in zip(col_expected, col_got):
                    if abs(a - b) > tol:
                        same = False
                        break

                if not same:
                    print "Answer was incorrect in input effects matrix column {}".format(dim)
                    print "Expected {}".format(col_expected)
                    print "Got {}".format(col_got)

                    raise RuntimeError("answer was incorrect")


        Timers.toc('expm check answer')

        # restore the saved basis matrix and input effects matrix
        self.cur_basis_mat = saved_basis_mat
        self.cur_input_effects_matrix = saved_input_effects_matrix

def create_output_space_csr(plot_settings, ha_mode):
    'create the output space matrix'

    num_directions = 0

    data = []
    cols = []
    indptr = [0]

    if plot_settings.plot_mode != PlotSettings.PLOT_NONE:
        dirs = [plot_settings.xdim_dir, plot_settings.ydim_dir]

        for plot_dir in dirs:
            if isinstance(plot_dir, int):
                data.append(1.0)
                cols.append(plot_dir)
                indptr.append(len(data))
                num_directions += 1
            elif plot_dir is not None:
                xdir = csr_matrix(plot_dir)
                assert len(xdir.shape) == 1 or xdir.shape[0] == 1, \
                    "expected row vector for plot direction, got shape: {}".format(plot_dir.shape)

                data += [n for n in xdir.data]
                cols += [n for n in xdir.indices]
                indptr.append(len(data))
                num_directions += 1

    if ha_mode.output_space_csr is not None:
        num_directions += ha_mode.output_space_csr.shape[0]
        offset = len(data)

        data += [n for n in ha_mode.output_space_csr.data]
        cols += [n for n in ha_mode.output_space_csr.indices]
        indptr += [i + offset for i in ha_mode.output_space_csr.indptr[1:]]

    dims = ha_mode.a_matrix_csr.shape[0]

    return csr_matrix((data, cols, indptr), shape=(num_directions, dims), dtype=float)
