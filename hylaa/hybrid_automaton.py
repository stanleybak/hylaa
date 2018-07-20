'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

import numpy as np

import scipy as sp
from scipy.sparse import csr_matrix

from hylaa.util import Freezable
from hylaa.time_elapse import TimeElapser

from hylaa import lputil

class LinearConstraint(Freezable):
    'a single csr sparse linear constraint: csr_vec * x <= rhs'

    def __init__(self, csr_vec, rhs):
        if not isinstance(csr_vec, csr_matrix):
            csr_vec = csr_matrix(csr_vec)
            
        assert csr_vec.shape[0] == 1, "expected single row constraint vector"
        
        self.csr = csr_vec
        self.rhs = float(rhs)

        self.freeze_attrs()

    def almost_equals(self, other, tol):
        'equality up to a tolerance'
        assert isinstance(other, LinearConstraint)

        rv = True

        if abs(self.rhs - other.rhs) > tol:
            rv = False
        elif self.csr.shape != other.csr.shape:
            rv = False
        elif not (self.csr.indices == other.csr.indices).all():
            rv = False
        elif not (self.csr.indptr == self.csr.indptr).all():
            rv = False
        else:
            for i in range(len(self.csr.data)):
                a = self.csr.data[i]
                b = other.csr.data[i]

                if abs(a - b) > tol:
                    rv = False
                    break

        return rv

    def clone(self):
        'create a deep copy of this LinearConstraints object'

        return LinearConstraint(self.csr.copy(), self.rhs)

    def negate(self):
        'return the negation of the condition'

        return LinearConstraint(-1 * self.csr, -self.rhs)

    def __str__(self):
        return '[LinearConstraint: {} * x <= {}]'.format(self.csr.toarray(), self.rhs)

    def __repr__(self):
        return 'LinearConstraint({}, {})'.format(repr(self.csr), repr(self.rhs))

class Mode(Freezable):
    '''
    A single mode of a hybrid automaton with dynamics x' = Ax + b. 

    The dynamics should be set_dynamics and (optionally) set_inputs. If they are not set, the mode will be
    considered an error mode.
    '''

    def __init__(self, ha, name, mode_id):
        assert isinstance(ha, HybridAutomaton)

        self.ha = ha # pylint: disable=invalid-name
        self.name = name
        self.mode_id = mode_id # unique int identified for this mode

        # dynamics are x' = Ax + Bu
        self.a_csr = None
        self.b_csr = None

        # constraints on input
        self.u_constraints_csr = None # csr_matrix
        self.u_constraints_rhs = None # 1-d np.ndarray

        self.transitions = [] # outgoing Transition objects

        self.inv_list = [] # a list of LinearConstraint, if all are true then the invariant is true

        self.time_elapse = None # a TimeElapse object... initialized on init_time_elapse()

        self.freeze_attrs()

    def set_invariant(self, constraints_csr, constraints_rhs):
        'sets the invariant'

        assert self.a_csr is not None, "A matrix must be set first"
        
        if not isinstance(constraints_csr, csr_matrix):
            constraints_csr = csr_matrix(constraints_csr)
            
        if not isinstance(constraints_rhs, np.ndarray):
            constraints_rhs = np.array(constraints_rhs, dtype=float)

        constraints_rhs.shape = (len(constraints_rhs), ) # flatten rhs into a 1-d array
        assert constraints_csr.shape[1] == self.a_csr.shape[0], \
            "width of invaraiant constraints({}) must equal A matrix size({})".format( \
            constraints_csr.shape[1], self.a_csr.shape[0])
        assert constraints_csr.shape[0] == len(constraints_rhs)

        # for efficiency in checking, the invariant is split into a list of individual constraints
        for row, rhs in enumerate(constraints_rhs):
            inds = []
            data = []
            indptr = [0]

            for i in range(constraints_csr.indptr[row], constraints_csr.indptr[row+1]):
                column = constraints_csr.indices[i]
                d = constraints_csr.data[i]

                inds.append(column)
                data.append(d)

            indptr.append(len(data))

            constraint_vec = csr_matrix((data, inds, indptr), dtype=float, shape=(1, constraints_csr.shape[1]))
            
            self.inv_list.append(LinearConstraint(constraint_vec, rhs))

    def set_inputs(self, b_csr, u_constraints_csr, u_constraints_rhs):
        'sets the time-varying / uncertain inputs for the mode (optional)'
    
        assert self.a_csr is not None, "set_dynamics should be done before set_inputs"
        assert isinstance(b_csr, csr_matrix)
        assert isinstance(u_constraints_csr, csr_matrix)
        assert isinstance(u_constraints_rhs, np.ndarray)
        u_constraints_rhs.shape = (len(u_constraints_rhs), ) # flatten init_rhs into a 1-d array

        assert u_constraints_csr.shape[0] == u_constraints_rhs.shape[0], "u_constraints rows shoud match rhs len"
        assert u_constraints_csr.shape[1] == b_csr.shape[1], "u_constraints cols should match b.width"

        assert b_csr.shape[0] == self.a_csr.shape[0], \
                "B-mat shape {} incompatible with A-mat shape {}".format(b_csr.shape, self.a_csr.shape)

        self.b_csr = b_csr
        self.u_constraints_csr = u_constraints_csr
        self.u_constraints_rhs = u_constraints_rhs

        raise RuntimeError("Inputs not currently supported")

    def set_dynamics(self, a_csr):
        'sets the autonomous system dynamics (A matrix)'

        if not isinstance(a_csr, csr_matrix):
            a_csr = csr_matrix(a_csr)

        assert a_csr.shape[0] == a_csr.shape[1]

        self.a_csr = a_csr

    def init_time_elapse(self, step_size):
        'initialize the time elapse object for this mode (called by verification core)'

        if self.a_csr is not None:
            self.time_elapse = TimeElapser(self, step_size)

    def __str__(self):
        return '[AutomatonMode with name:{}, a_matrix:{}]'.format(self.name, self.a_csr.toarray())

    def __repr__(self):
        return str(self)

class Transition(Freezable):
    'A transition of a hybrid automaton'

    def __init__(self, ha, from_mode, to_mode, name=''):
        assert isinstance(ha, HybridAutomaton)
        self.ha = ha # pylint: disable=invalid-name
        self.from_mode = from_mode
        self.to_mode = to_mode

        self.guard_csr = None
        self.guard_rhs = None

        self.reset_csr = None
        self.reset_minkowski_csr = None
        self.reset_minkowski_constraints_csr = None
        self.reset_minkowski_constraints_rhs = None

        self.name = name

        self.lpi = None # assinged upon continuous post

        self.freeze_attrs()

        from_mode.transitions.append(self)

    def set_guard(self, guard_csr, guard_rhs):
        '''set the guard'''
        
        if not isinstance(guard_csr, csr_matrix):
            guard_csr = csr_matrix(guard_csr, dtype=float)
            
        if not isinstance(guard_rhs, np.ndarray):
            guard_rhs = np.array(guard_rhs, dtype=float)
        
        assert self.from_mode.a_csr is not None, "A-matrix not assigned in predecessor mode {}".format(self.from_mode)
        assert guard_csr.shape[1] == self.from_mode.a_csr.shape[0], "guard matrix expected {} columns, got {}".format(
            self.from_mode.a_csr.shape[0], guard_csr.shape[1])

        guard_rhs.shape = (len(guard_rhs), ) # flatten into a 1-d array
        assert guard_rhs.shape[0] == guard_csr.shape[0]

        self.guard_csr = guard_csr
        self.guard_rhs = guard_rhs

    def set_reset(self, reset_csr=None, reset_minkowski_csr=None, reset_minkowski_constraints_csr=None,
                  reset_minkowski_constraints_rhs=None):
        '''resets are of the form x' = Rx + My, Cy <= rhs, where y are fresh variables
        the reset_minowski variables can be None if no new variables are needed. If unassigned, the identity
        reset is assumed
 
        x' are the new variables
        x are the old variables       
        reset_csr is R
        reset_minkowski_csr is M
        reset_minkowski_constraints_csr is C
        reset_minkowski_constraints_rhs is rhs
        '''

        assert self.from_mode.a_csr is not None, "A matrix not assigned in predecessor mode {}".format(self.from_mode)
        assert self.to_mode.a_csr is not None, "A matrix not assigned in successor mode {}".format(self.to_mode)

        if reset_csr is None:
            assert self.from_mode.a_csr.shape[0] == self.to_mode.a_csr.shape[0], "identity reset but num dims changes"
            reset_csr = sp.sparse.identity(self.from_mode.a_csr.shape[0], dtype=float, format='csr')

        if not isinstance(reset_csr, csr_matrix):
            reset_csr = csr_matrix(reset_csr)

        if reset_minkowski_csr is not None and not isinstance(reset_minkowski_csr, csr_matrix):
            reset_minkowski_csr = csr_matrix(reset_minkowski_csr)

        if reset_minkowski_constraints_csr is not None and not isinstance(reset_minkowski_constraints_csr, csr_matrix):
            reset_minkowski_constraints_csr = csr_matrix(reset_minkowski_constraints_csr)

        if reset_minkowski_constraints_rhs is not None and not isinstance(reset_minkowski_constraints_rhs, np.ndarray):
            reset_minkowski_constraints_rhs = np.ndarray(reset_minkowski_constraints_rhs)
        
        assert reset_csr.shape[1] == self.from_mode.a_csr.shape[0], "reset matrix expected {} columns, got {}".format(
            self.from_mode.a_csr.shape[0], reset_csr.shape[1])
        assert reset_csr.shape[0] == self.to_mode.a_csr.shape[0], "reset matrix expected {} rows, got {}".format(
            self.to_mode.a_csr.shape[0], reset_csr.shape[0])

        if reset_minkowski_constraints_rhs is not None:
            assert len(reset_minkowski_constraints_rhs.shape) == 1
            assert reset_minkowski_constraints_csr is not None
            assert reset_minkowski_constraints_csr.shape[0] == reset_minkowski_constraints_rhs[0]

            new_vars = reset_minkowski_constraints_csr.shape[1]

            if reset_minkowski_csr is None:
                reset_minkowski_csr = sp.sparse.identity(new_vars, dtype=float, format='csr')

            assert isinstance(reset_minkowski_csr, csr_matrix)
            assert reset_minkowski_csr.shape[0] == self.to_mode.a_csr.shape[0]
            assert reset_minkowski_csr.shape[1] == new_vars

        self.reset_csr = reset_csr
        self.reset_minkowski_csr = reset_minkowski_csr
        self.reset_minkowski_constraints_csr = reset_minkowski_constraints_csr
        self.reset_minkowski_constraints_rhs = reset_minkowski_constraints_rhs

    def make_lpi(self, from_state):
        'make the lpi instance for this transition, from the given state'

        self.lpi = from_state.lpi.clone() 

        # add the guard condition
        lputil.add_curtime_constraints(self.lpi, self.guard_csr, self.guard_rhs)

    def __str__(self):
        return self.from_mode.name + " -> " + self.to_mode.name

class HybridAutomaton(Freezable):
    'The hybrid automaton'

    def __init__(self, name='HybridAutomaton'):
        self.name = name
        self.modes = {} # map name -> mode
        self.transitions = []

        self.freeze_attrs()

    def new_mode(self, name):
        '''add a mode'''
        m = Mode(self, name, len(self.modes))
        self.modes[m.name] = m
        return m

    def new_transition(self, from_mode, to_mode, name=None):
        '''add a transition'''
        t = Transition(self, from_mode, to_mode, name=name)
        self.transitions.append(t)

        return t

    def do_guard_strengthening(self):
        '''
        Strengthen the guards to include the invariants of target modes
        '''

        for t in self.transitions:
            if t.from_mode == t.to_mode:
                continue

            inv_list = t.to_mode.inv_list
            add_guard_list = []

            for inv_constraint in inv_list:
                already_in_cond_list = False

                for row in range(t.guard_csr.shape[0]):
                    inds = []
                    data = []
                    indptr = [0]

                    for i in range(t.guard_csr.indptr[row], t.guard_csr.indptr[row + 1]):
                        inds.append(t.guard_csr.indices[i])
                        data.append(t.guard_csr.data[i])

                    indptr.append(len(data))
                    csr_vec = csr_matrix((data, inds, indptr), dtype=float, shape=(1, t.guard_csr.shape[1]))
                    guard_constraint = LinearConstraint(csr_vec, t.guard_rhs[row])
                    
                    if guard_constraint.almost_equals(inv_constraint, 1e-13):
                        already_in_cond_list = True
                        break

                if not already_in_cond_list:
                    add_guard_list.append(inv_constraint)

            # add all consrtaints in add_guard_list to the transition's guard (construct a new guard matrix)
            if add_guard_list:
                inds = [n for n in t.guard_csr.indices]
                data = [x for x in t.guard_csr.data]
                indptr = [n for n in t.guard_csr.indptr]
                rhs = [x for x in t.guard_rhs]

                for new_guard_condition in add_guard_list:
                    inds += [n for n in new_guard_condition.csr.indices]
                    data += [x for x in new_guard_condition.csr.data]
                    rhs.append(new_guard_condition.rhs)

                    indptr.append(len(data))

                height = t.guard_csr.shape[0] + len(add_guard_list)
                width = t.guard_csr.shape[1]
                new_guard_csr = csr_matrix((data, inds, indptr), dtype=float, shape=(height, width))
                new_guard_rhs = np.array(rhs, dtype=float)

                t.guard_csr = new_guard_csr
                t.guard_rhs = new_guard_rhs
