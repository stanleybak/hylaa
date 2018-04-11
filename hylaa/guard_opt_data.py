'''
Stanley Bak
Hylaa-Continuous Guard Optimization Logic
July 2017
'''

import numpy as np
import scipy as sp

from scipy.sparse import csc_matrix

from hylaa.util import Freezable
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.settings import PlotSettings
from hylaa.glpk_interface import LpInstance
from hylaa.timerutil import Timers

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
            self.lpi.set_input_constraints_csc(csc_matrix(star.mode.u_constraints_csr), star.mode.u_constraints_rhs)

        if self.num_output_vars == 1 and len(self.transition.guard_rhs) == 1:
            self.guard_norm = sp.sparse.linalg.norm(self.transition.guard_matrix_csr[0, :], ord=np.inf)

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
        result_len += self.inputs * (self.star.time_elapse.next_step - 1)

        result = np.zeros((result_len), dtype=float)
        direction = np.zeros((self.num_output_vars,), dtype=float)

        # self.lpi.reset_lp()

        is_feasible = self.lpi.minimize(direction, result, error_if_infeasible=False)

        return result if is_feasible else None

    def get_guard_lpi(self):
        '''get the current full lp instance for this guard'''

        return self.lpi

    def get_optimized_lp_solution(self):
        '''gets the lp solution without calling an lp solver
        this is only possible if the initial set has only range conditions for each initial dimension,
        and the output-space has only a single condition

        This returns either an lp solution (np.ndarray) or None if infeasible
        '''

        Timers.tic('get_optimized_lp_solution')

        init_ranges = self.star.init_range_tuples
        basis_mat = self.star.time_elapse.cur_basis_mat

        assert len(init_ranges) == basis_mat.shape[1]
        assert len(self.transition.guard_rhs) == 1
        assert self.num_output_vars == 1

        guard_threshold = self.transition.guard_rhs[0]
        guard_val = 0

        for init_index in xrange(len(init_ranges)):
            basis_val = basis_mat[0][init_index]
            min_init = init_ranges[init_index][0]
            max_init = init_ranges[init_index][1]

            val1 = min_init * basis_val
            val2 = max_init * basis_val

            # take the minimum of val1 and val2, since guard is CONDITION <= RHS
            if val1 < val2:
                guard_val += val1
            else:
                guard_val += val2

        if guard_val <= guard_threshold:
            # reconstruct result
            result_len = self.num_output_vars + self.star.num_init_vars
            result_len += self.inputs * (self.star.time_elapse.next_step - 1)

            result = np.zeros((result_len), dtype=float)
            result[-1] = guard_val # single output variable

            for init_index in xrange(len(init_ranges)):
                basis_val = basis_mat[0][init_index]
                min_init = init_ranges[init_index][0]
                max_init = init_ranges[init_index][1]

                val1 = min_init * basis_val
                val2 = max_init * basis_val

                result[init_index] = min_init if val1 < val2 else max_init

            rv = result
        else:
            rv = None

        Timers.toc('get_optimized_lp_solution')

        return rv

    def get_updated_lp_solution(self):
        '''update the LP solution and, if it's feasible, get its solution'''

        if self.star.settings.interval_guard_optimization and self.star.init_range_tuples is not None and \
                                                              self.num_output_vars == 1:
            # LP can be decomposed column-by-column (optimization)
            rv = self.get_optimized_lp_solution()
        else:
            rv = self.update_full_lp()

        return rv
