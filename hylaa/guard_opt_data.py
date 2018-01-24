'''
Stanley Bak
Hylaa-Continuous Guard Optimization Logic
July 2017
'''

import numpy as np

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.settings import PlotSettings
from hylaa.glpk_interface import LpInstance

class GuardOptData(Freezable):
    'Guard optimization data'

    def __init__(self, star, mode, transition_index):
        assert isinstance(mode, LinearAutomatonMode)

        self.settings = star.settings
        self.mode = mode
        self.inputs = star.inputs

        self.star = star
        self.transition = mode.transitions[transition_index]
        self.num_output_vars = self.transition.guard_matrix_csr.shape[1]

        self.key_dir_offset = 0

        if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            if self.settings.plot.xdim_dir is not None:
                self.key_dir_offset += 1

            if self.settings.plot.ydim_dir is not None:
                self.key_dir_offset += 1

        self.lpi = LpInstance(self.num_output_vars, star.num_init_vars, self.inputs)

        self.lpi.set_init_constraints(star.init_mat, star.init_rhs)

        self.lpi.set_output_constraints(self.transition.guard_matrix_csr, self.transition.guard_rhs)

        if star.inputs > 0:
            self.lpi.set_input_constraints_csr(star.input_mat_csr, star.input_rhs)

        self.freeze_attrs()

    def update_full_lp(self):
        '''update the LP solution and, if it's feasible, get its solution, for GUARD_FULL_LP'''

        cur_basis_mat = self.star.time_elapse.cur_basis_mat

        # start and end rows of key-dir matrices
        start = self.key_dir_offset
        end = self.key_dir_offset + self.num_output_vars

        self.lpi.update_basis_matrix(cur_basis_mat[start:end])

        # add input effects for the current step (if it exists)
        if self.star.time_elapse.cur_input_effects_matrix is not None:
            input_effects_mat = self.star.time_elapse.cur_input_effects_matrix
            self.lpi.add_input_effects_matrix(input_effects_mat[start:end])

        result_len = self.num_output_vars + self.star.num_init_vars

        if self.inputs > 0:
            num_steps = self.star.time_elapse.next_step - 1
            result_len += self.inputs * num_steps

        result = np.zeros((result_len), dtype=float)
        direction = np.zeros((self.num_output_vars,), dtype=float)

        # self.lpi.reset_lp()

        is_feasible = self.lpi.minimize(direction, result, error_if_infeasible=False)

        return result if is_feasible else None

    def get_guard_lpi(self):
        '''get the current full lp instance for this guard'''

        return self.lpi

    def get_updated_lp_solution(self):
        '''update the LP solution and, if it's feasible, get its solution'''

        return self.update_full_lp()
