'''
Time Elapse Computation. This module is primarily responsive for computing
l * e^{At} where l is some direction of interest, and t is a multiple of some time step
'''

import numpy as np

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import expm

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.containers import HylaaSettings, PlotSettings
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
        self.key_dir_mat = None # csc_matrix
        self.cur_time_elapse_mat = None # assigned on update()
        self._extract_directions(mode)

        self.freeze_attrs()

    def _extract_directions(self, mode):
        'extract the directions which are of interest'

        num_directions = 0 if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE else 2

        for t in mode.transitions:
            num_directions += len(t.condition_list)

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
            for lc in t.condition_list:
                lil_dir_mat[dir_index, :] = lc.vector
                dir_index += 1

        # done constructing, convert to csc_matrix
        self.key_dir_mat = csc_matrix(lil_dir_mat)

    def step(self):
        'perform the computation to obtain the values of the key directions the current time'

        Timers.tic('time_elapse.step')

        cur_time = self.settings.step * self.next_step

        time_mat = self.a_matrix * cur_time
        exp = expm(time_mat)

        self.cur_time_elapse_mat = np.array((self.key_dir_mat * exp).todense(), dtype=float)

        self.next_step += 1

        Timers.toc('time_elapse.step')
