'''
Time Elapse Computation. This module is primarily responsive for computing
l * e^{At} where l is some direction of interest, and t is a multiple of some time step
'''

import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import expm

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
        self.dims = self.a_matrix.shape[0]

        self.next_step = 0
        self.key_dir_mat = None # csr_matrix
        self.cur_time_elapse_mat = None # assigned on step()
        self.one_step_matrix_exp = None # one step matrix exponential, used for sim_mode == EXP_MULT
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
            self.cur_time_elapse_mat = self.key_dir_mat * np.identity(self.dims)
        elif self.one_step_matrix_exp is None:
            assert self.next_step == 1
            assert isinstance(self.key_dir_mat, csr_matrix)
            Timers.tic('time_elapse.step first')

            self.one_step_matrix_exp = np.array(expm(self.a_matrix * self.settings.step).todense(), dtype=float)
            self.cur_time_elapse_mat = np.array((self.key_dir_mat * self.one_step_matrix_exp).todense(), dtype=float)

            Timers.toc('time_elapse.step first')
        else:
            Timers.tic('time_elapse.step others')
            self.cur_time_elapse_mat = np.dot(self.cur_time_elapse_mat, self.one_step_matrix_exp)
            Timers.toc('time_elapse.step others')

    def step_matrix_exp(self):
        'matrix exp every step'

        cur_time = self.settings.step * self.next_step
        time_mat = self.a_matrix * cur_time
        exp = expm(time_mat)

        self.cur_time_elapse_mat = np.array((self.key_dir_mat * exp).todense(), dtype=float)

    def step(self):
        'perform the computation to obtain the values of the key directions the current time'

        Timers.tic('time_elapse.step')

        if self.settings.simulation.sim_mode == SimulationSettings.MATRIX_EXP:
            self.step_matrix_exp()
        elif self.settings.simulation.sim_mode == SimulationSettings.EXP_MULT:
            self.step_exp_mult()
        else:
            raise RuntimeError('Unimplemented sim_mode {}'.format(self.settings.simulation.sim_mode))

        self.next_step += 1

        Timers.toc('time_elapse.step')
        assert isinstance(self.cur_time_elapse_mat, np.ndarray), "cur_time_elapse_mat should be an np.array"
        assert self.cur_time_elapse_mat.shape == self.key_dir_mat.shape, \
            "cur_time_elapse mat shape({}) should be {}".format(self.cur_time_elapse_mat.shape, self.key_dir_mat.shape)
