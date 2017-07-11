'''
Stanley Bak
Hylaa-Continuous Guard Optimization Logic
July 2017
'''

import numpy as np

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.containers import SimulationSettings, PlotSettings
from hylaa.glpk_interface import LpInstance

class GuardOptData(Freezable):
    'Guard optimization data'

    def __init__(self, star, mode, transition_index):
        assert isinstance(mode, LinearAutomatonMode)

        self.settings = star.settings
        self.mode = mode
        self.dims = star.dims
        self.inputs = star.inputs

        self.star = star
        self.transition = mode.transitions[transition_index]
        self.num_constraints = self.transition.guard_matrix.shape[0]

        self.key_dir_offset = 0 if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE else 2

        for t_index in xrange(transition_index):
            self.key_dir_offset += self.mode.transitions[t_index].guard_matrix.shape[0]

        if self.settings.simulation.guard_mode == SimulationSettings.GUARD_FULL_LP:
            self.lpi = LpInstance(self.num_constraints, self.dims, self.inputs)
            self.lpi.set_init_constraints_csr(star.init_mat_csr, star.init_rhs)
            self.lpi.set_cur_time_constraint_bounds(self.transition.guard_rhs)

            if star.inputs > 0:
                self.lpi.set_input_constraints_csr(star.input_mat_csr, star.input_rhs)
        elif self.settings.simulation.guard_mode == SimulationSettings.GUARD_DECOMPOSED:
            assert self.num_constraints == 1, \
                "SimulationSettings.guard_mode == GUARD_DECOMPOSED requires num transition constraints to be 1"

            self.noinput_lpi = LpInstance(self.num_constraints, self.dims, 0)
            self.noinput_lpi.set_init_constraints_csr(star.init_mat_csr, star.init_rhs)

            self.input_lpi = LpInstance(self.num_constraints, self.inputs, 0)
            self.input_lpi.set_init_constraints_csr(star.input_mat_csr, star.input_rhs)

            self.input_history = [] # inputs to apply at each step
            self.input_effects_sum = 0.0

        self.freeze_attrs()

    def update_decomposed(self):
        '''update the LP solution and, if it's feasible, get its solution, for GUARD_DECOMPOSED'''

        cur_time_mat = self.star.time_elapse.cur_time_elapse_mat
        row = self.key_dir_offset

        self.noinput_lpi.update_time_elapse_matrix(cur_time_mat[row:row+1])
        self.noinput_lpi.commit_cur_time_rows()

        noinput_result = np.zeros((self.dims + 1), dtype=float)
        direction = np.array([1.0], dtype=float)
        self.noinput_lpi.minimize(direction, noinput_result)
        noinput_value = noinput_result[self.dims]

        # add input effects for the current step (if it exists)
        if self.star.time_elapse.cur_input_effects_matrix is not None:
            scale = 1000
            input_effects_mat = scale * self.star.time_elapse.cur_input_effects_matrix

            self.input_lpi.update_time_elapse_matrix(input_effects_mat[row:row+1])
            self.input_lpi.commit_cur_time_rows()

            result = np.zeros((self.inputs + 1,), dtype=float)
            self.input_lpi.minimize(direction, result)

            self.input_history += [i for i in result[0:self.inputs]]
            self.input_effects_sum += result[self.inputs] / scale

        if noinput_value + self.input_effects_sum <= self.transition.guard_rhs[0]:
            # violation! reconstruct result
            total = noinput_value + self.input_effects_sum

            rv = np.concatenate((noinput_result[0:self.dims], [total], self.input_history))
        else:
            rv = None

        return rv

    def update_full_lp(self):
        '''update the LP solution and, if it's feasible, get its solution, for GUARD_FULL_LP'''

        cur_time_mat = self.star.time_elapse.cur_time_elapse_mat

        # start and end rows of key-dir matrices
        start = self.key_dir_offset
        end = self.key_dir_offset + self.num_constraints

        self.lpi.update_time_elapse_matrix(cur_time_mat[start:end])

        # add input effects for the current step (if it exists)
        if self.star.time_elapse.cur_input_effects_matrix is not None:
            input_effects_mat = self.star.time_elapse.cur_input_effects_matrix
            self.lpi.add_input_effects_matrix(input_effects_mat[start:end])

        self.lpi.commit_cur_time_rows()

        result_len = self.num_constraints + self.star.dims

        if self.inputs > 0:
            num_steps = self.star.time_elapse.next_step - 1
            result_len += self.inputs * num_steps

        result = np.zeros((result_len), dtype=float)
        direction = np.zeros((self.num_constraints,), dtype=float)

        is_feasible = self.lpi.minimize(direction, result, error_if_infeasible=False)

        return result if is_feasible else None

    def get_updated_lp_solution(self):
        '''update the LP solution and, if it's feasible, get its solution'''

        if self.settings.simulation.guard_mode == SimulationSettings.GUARD_FULL_LP:
            rv = self.update_full_lp()
        else:
            rv = self.update_decomposed()

        return rv
