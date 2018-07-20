'''
Generalized Star and other Star data structures
Stanley Bak
Aug 2016
'''

import numpy as np

from hylaa import lputil

from hylaa.hybrid_automaton import Mode
from hylaa.timerutil import Timers
from hylaa.util import Freezable
from hylaa.lpinstance import LpInstance

from hylaa import lpplot

class StateSet(Freezable):
    '''
    A set of states with a common mode.
    '''

    def __init__(self, lpi, mode, cur_step_since_start=0):
        assert isinstance(lpi, LpInstance)
        assert isinstance(mode, Mode)

        self.mode = mode
        self.lpi = lpi

        self.cur_step_in_mode = 0
        self.cur_step_since_start = cur_step_since_start

        self.invariant_constraint_rows = None # the LP row of the strongest constraint for each invariant condition

        self.basis_matrix = np.identity(mode.a_csr.shape[0])

        self._verts = None # cached vertices at the current step

        self.freeze_attrs()

    def __str__(self):
        'short string representation of this state set'

        return "[StateSet in '{}']".format(self.mode.name)

    def step(self):
        'update the star based on values from a new simulation time instant'

        self.cur_step_in_mode += 1
        self.cur_step_since_start += 1

        self.basis_matrix, _ = self.mode.time_elapse.get_basis_matrix(self.cur_step_in_mode)

        lputil.set_basis_matrix(self.lpi, self.basis_matrix)

        # update each transition's basis matrix
        for t in self.mode.transitions:
            lputil.set_basis_matrix(t.lpi,self.basis_matrix)

        self._verts = None # cached vertices no longer valid

    def verts(self, plotman):
        'get the vertices for plotting this state set, wraps around so rv[0] == rv[-1]'

        Timers.tic('verts')

        if self._verts is None:
            xdim = plotman.settings.xdim_dir
            ydim = plotman.settings.ydim_dir
            cur_time = self.cur_step_since_start * plotman.core.settings.step_size

            self._verts = lpplot.get_verts(self.lpi, xdim=xdim, ydim=ydim, plot_vecs=plotman.plot_vecs, \
                                           cur_time=cur_time)
            
        Timers.toc('verts')

        return self._verts

    def add_constraint(self, lc):
        '''add a constraint to the lpi (and the lpi of all transitions)
        returns the row of the newly-added constraint
        '''

        vec = lc.csr.toarray()[0] # this gets multiplied by the basis matrix anyways, so dense is fine

        row = lputil.add_init_constraint(self.lpi, vec, lc.rhs, basis_matrix=self.basis_matrix)

        for t in self.mode.transitions:
            if t.lpi is not None:
                lputil.add_init_constraint(t.lpi, vec, lc.rhs, basis_matrix=self.basis_matrix)

        return row

    def try_replace_constraint(self, lc, old_row):
        '''try stengthening an existing constraint with a new linear constraint condition
        returns the row of the newly-added constraint, or old_row if the old constraint is replaced
        '''

        vec = lc.csr.toarray()[0] # this gets multiplied by the basis matrix anyways, so dense is fine

        row = lputil.try_replace_init_constraint(self.lpi, old_row, vec, lc.rhs, basis_mat=self.basis_matrix)

        for t in self.mode.transitions:
            if row == old_row:
                # constraint was replaced
                t_row = lputil.try_replace_init_constraint(t.lpi, old_row, vec, lc.rhs, basis_mat=self.basis_matrix)
                assert t_row == row, "constraint should be replaced at same row"
            else:
                # new constraint was added
                lputil.add_init_constraint(t.lpi, vec, lc.rhs, basis_matrix=self.basis_matrix)

        return row

    def intersect_invariant(self):
        '''intersect the current state set with the mode invariant

        returns whether the state set is still feasbile after intersection'''

        if self.invariant_constraint_rows is None:
            self.invariant_constraint_rows = [None] * len(self.mode.inv_list)

        for i, inv_lc in enumerate(self.mode.inv_list):
            if lputil.check_intersection(self.lpi, inv_lc.negate()):
                if self.invariant_constraint_rows[i] is None:
                    # new constriant
                    row = self.add_constraint(inv_lc)
                    self.invariant_constraint_rows[i] = row
                else:
                    # strengthen existing constraint possibly
                    row = self.try_replace_constraint(inv_lc, self.invariant_constraint_rows[i])
                    self.invariant_constraint_rows[i] = row

        is_feasible = self.lpi.minimize(columns=[], fail_on_unsat=False) is not None

        return is_feasible
