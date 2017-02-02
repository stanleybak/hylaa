'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

import itertools

import numpy as np


class HyperRectangle(object):
    'An n-dimensional box'

    def __init__(self, dims):
        self.dims = dims # list of tuples

        for d in xrange(len(dims)):
            low = dims[d][0]
            high = dims[d][1]
            assert low <= high, "Invalid Hyperrectange: dim #{} min({}) > max({})".format(
                d, low, high)

    def center(self):
        'Returns a point in the center of the box'
        rv = []

        for d in self.dims:
            rv.append((d[0] + d[1]) / 2.0)

        return rv

    def star(self):
        'Returns a list of the so-called star points of this box (2*dims of them)'
        center = self.center()
        num_dims = len(self.dims)
        rv = []
        
        for index in xrange(num_dims):
            # min edge in dimension d
            pt = list(center)
            pt[index] = self.dims[index][0]
            rv.append(pt)

            # max edge in dimension d
            pt = list(center)
            pt[index] = self.dims[index][1]
            rv.append(pt)

        return rv

    def unique_corners(self, tol=1e-9):
        'Returns a list of the unique corner points of this box (up to 2^dims of them)'
        rv = []
        num_dims = len(self.dims)

        # compute max iterator index and make is_flat list
        max_iterator = 1
        is_flat = []

        for d in xrange(num_dims):
            if abs(self.dims[d][0] - self.dims[d][1]) > tol:
                is_flat.append(False)
                max_iterator *= 2
            else:
                is_flat.append(True)

        for it in xrange(max_iterator):
            point = []
            
            # construct point
            for d in xrange(num_dims):
                if is_flat[d]:
                    point.append(self.dims[d][0])
                else:
                    min_max_index = it % 2
                    point.append(self.dims[d][min_max_index])
                    it /= 2

            # append constructed point
            rv.append(point)

        return rv

class LinearConstraint(object):
    'a single linear constraint: vector * x <= value'

    def __init__(self, vector, value):
        self.vector = np.array(vector, dtype=float)
        self.value = float(value)

    def almost_equals(self, other, tol):
        'equality up to a tolerance'
        assert isinstance(other, LinearConstraint)

        rv = True

        if abs(self.value - other.value) > tol:
            rv = False
        else:
            for i in xrange(self.vector.shape[0]):
                a = self.vector[i]
                b = other.vector[i]

                if abs(a - b) > tol:
                    rv = False
                    break

        return rv

    def __str__(self):
        return '[LinearConstraint: {} * x <= {}]'.format(self.vector, self.value)

class LinearAutomatonMode(object):
    'A single mode of a hybrid automaton'

    def __init__(self, parent, name):

        self.a_matrix = None # dynamics are x' = Ax + b
        self.b_vector = None # dynamics are x' = Ax + b

        self.inv_list = [] # a list of LinearConstraint, if all are true then the invariant is true

        self.parent = parent
        self.name = name
        self.transitions = []

        self._null_dynamics = None # gets set on first call to is_null_dynamcics
        
    def set_dynamics(self, a_matrix, b_vector=None):
        'sets the non-inputs dynamics. c_vector can be None, in which case zeros are used'

        assert len(self.parent.variables) > 0, "automaton.variables should be set before dynamics"

        if b_vector is None:
            b_vector = np.zeros(len(self.parent.variables))

        assert isinstance(a_matrix, np.ndarray)
        assert len(a_matrix.shape) == 2

        assert isinstance(b_vector, np.ndarray)
        assert len(b_vector.shape) == 1

        assert a_matrix.shape[0] == b_vector.shape[0]
        assert a_matrix.shape[1] == len(self.parent.variables)

        self.a_matrix = a_matrix
        self.b_vector = b_vector

    def is_null_dynamics(self):
        '''is this mode's dynamics null?'''

        rv = self._null_dynamics

        if rv is None:
            rv = True

            for val in itertools.chain(self.b_vector, self.a_matrix.flat):
                if val != 0:
                    rv = False
                    break

            self._null_dynamics = rv

        return rv

    def __str__(self):
        return '[LinearAutomatonMode: ' + self.name + ']'

class LinearAutomatonTransition(object):
    'A transition of a hybrid automaton'

    def __init__(self, parent, from_mode, to_mode):
        self.parent = parent
        self.from_mode = from_mode
        self.to_mode = to_mode
        self.guard_list = [] # a list of LinearConstraint, if all are true then the guard is enabled
        from_mode.transitions.append(self)

        self.reset_matrix = None # a matrix (x_post := Ax + b)
        self.reset_vector = None # a vector (x_post := Ax + b)

    def __str__(self):
        return self.from_mode.name + " -> " + self.to_mode.name

class LinearHybridAutomaton(object):
    'The hybrid automaton'

    def __init__(self, name='HybridAutomaton'):
        self.name = name
        self.modes = {}
        self.transitions = []
        self.variables = [] # list of strings

    def new_mode(self, name):
        '''add a mode'''
        m = LinearAutomatonMode(self, name)
        self.modes[m.name] = m
        return m

    def new_transition(self, from_mode, to_mode):
        '''add a transition'''
        t = LinearAutomatonTransition(self, from_mode, to_mode)
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
            guard_list = t.guard_list

            for inv_constraint in inv_list:
                already_in_guard_list = False

                for guard_constraint in guard_list:
                    if guard_constraint.almost_equals(inv_constraint, 1e-13):
                        already_in_guard_list = True
                        break

                if not already_in_guard_list:
                    guard_list.append(inv_constraint)

