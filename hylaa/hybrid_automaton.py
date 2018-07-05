'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

import numpy as np

from scipy.sparse import csr_matrix

from hylaa.util import Freezable
from hylaa.time_elapse import TimeElapser

class Mode(Freezable):
    '''
    A single mode of a hybrid automaton with dynamics x' = Ax + b. 

    The dynamics should be set_dynamics and (optionally) set_inputs. If they are not set, the mode will be
    considered an error mode.
    '''

    def __init__(self, parent, name):
        assert isinstance(parent, HybridAutomaton)

        self.parent = parent
        self.name = name

        # dynamics are x' = Ax + Bu
        self.a_matrix = None
        self.b_matrix = None

        self.u_constraints_csr = None # csr_matrix
        self.u_constraints_rhs = None # np.ndarray

        self.transitions = [] # outgoing transitions

        self.time_elapse = None # a TimeElapse object... initialized on init_time_elapse()

        self.freeze_attrs()

    def set_inputs(self, b_matrix, u_constraints_csr, u_constraints_rhs):
        'sets the time-varying / uncertain inputs for the mode (optional)'
    
        assert self.a_matrix is not None, "set_dynamics should be done before set_inputs"
        assert isinstance(b_matrix, np.ndarray)
        assert isinstance(u_constraints_csr, csr_matrix)
        assert isinstance(u_constraints_rhs, np.ndarray)
        u_constraints_rhs.shape = (len(u_constraints_rhs), ) # flatten init_rhs into a 1-d array

        assert u_constraints_csr.shape[0] == u_constraints_rhs.shape[0], "u_constraints rows shoud match rhs len"
        assert u_constraints_csr.shape[1] == b_matrix.shape[1], "u_constraints cols should match b.width"

        assert b_matrix.shape[0] == self.a_matrix.shape[0], \
                "B-mat shape {} incompatible with A-mat shape {}".format(b_matrix.shape, self.a_matrix.shape)

        self.b_matrix = b_matrix
        self.u_constraints_csr = u_constraints_csr
        self.u_constraints_rhs = u_constraints_rhs

        raise RuntimeError("Inputs not currently supported")

    def set_dynamics(self, a_matrix):
        'sets the autonomous system dynamics'

        assert isinstance(a_matrix, np.ndarray), "dynamics a_matrix should be a dense matrix"
        assert len(a_matrix.shape) == 2
        assert a_matrix.shape[0] == a_matrix.shape[1]

        self.a_matrix = a_matrix

    def init_time_elapse(self, step_size):
        'initialize the time elapse object for this mode'

        if self.a_matrix is not None:
            self.time_elapse = TimeElapser(self, step_size)

    def __str__(self):
        return '[AutomatonMode with name:{}, a_matrix:{}]'.format(self.name, self.a_matrix)

    def __repr__(self):
        return str(self)

class Transition(Freezable):
    'A transition of a hybrid automaton'

    def __init__(self, parent, from_mode, to_mode):
        self.parent = parent
        self.from_mode = from_mode
        self.to_mode = to_mode

        self.guard_matrix_csr = None
        self.guard_rhs = None

        self.freeze_attrs()

        from_mode.transitions.append(self)

    def set_guard(self, matrix_csr, rhs):
        '''set the guard matrix and right-hand side.
        '''

        assert isinstance(matrix_csr, csr_matrix)
        assert isinstance(rhs, np.ndarray)

        assert rhs.shape == (matrix_csr.shape[0],)

        self.guard_matrix_csr = matrix_csr
        self.guard_rhs = rhs

    def __str__(self):
        return self.from_mode.name + " -> " + self.to_mode.name

class HybridAutomaton(Freezable):
    'The hybrid automaton'

    def __init__(self, name='HybridAutomaton'):
        self.name = name
        self.modes = {}
        self.transitions = []

        self.freeze_attrs()

    def new_mode(self, name):
        '''add a mode'''
        m = Mode(self, name)
        self.modes[m.name] = m
        return m

    def new_transition(self, from_mode, to_mode):
        '''add a transition'''
        t = Transition(self, from_mode, to_mode)
        self.transitions.append(t)

        return t
