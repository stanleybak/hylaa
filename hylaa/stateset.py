'''
Generalized Star and other Star data structures
Stanley Bak
Aug 2016
'''

from hylaa import lputil

from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.timerutil import Timers as Timers
from hylaa.util import Freezable

class StateSet(Freezable):
    '''
    A set of states with a common mode.
    '''

    def __init__(self, core, lpi, mode):
        assert isinstance(lpi. LpInstance)
        assert isinstance(mode, LinearAutomatonMode)

        self.mode = mode
        self.lpi = lpi

        self.cur_step_in_mode = 0
        self.cur_step_since_start = 0

        self._verts = None # cached vertices at the current step

        self.freeze_attrs()

    def step(self):
        'update the star based on values from a new simulation time instant'

        basis_matrix, _ = self.mode.get_basis_matrix(self.cur_step_in_mode)

        self.cur_step_in_mode += 1
        self.cur_step_since_start += 1

        lputil.set_basis_matrix(self.lpi, basis_matrix)
        self._verts = None # cached vertices no longer valid

    def verts(self):
        'get the vertices for plotting this state set, wraps around so rv[0] == rv[-1]'

        Timers.tic('verts')

        if self._verts is None:
            dims = mode.a_matrix.shape[0]
            
            self._verts = lpplot.get_verts(self.lpi, num_dims=dims, xdim=0, ydim=1, plot_vecs=None, cur_time=0)
            
        Timers.toc('verts')

        return self._verts
