'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

from scipy.sparse import csr_matrix

from hylaa.util import Freezable

class SparseLinearConstraint(Freezable):
    'a single linear constraint: vector * x <= value'

    def __init__(self, vector, value):
        self.vector = csr_matrix(vector, dtype=float)
        self.value = float(value)

        self.freeze_attrs()

    def almost_equals(self, other, tol):
        'equality up to a tolerance'
        assert isinstance(other, SparseLinearConstraint)

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

    def clone(self):
        'create a deep copy of this LinearConstraints object'

        return SparseLinearConstraint(self.vector.copy(), self.value)

    def __str__(self):
        return '[SparseLinearConstraint: {} * x <= {}]'.format(self.vector, self.value)

    def __repr__(self):
        return 'SparseLinearConstraint({}, {})'.format(repr(self.vector), repr(self.value))

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

    def diamond(self):
        'Returns a list of the so-called diamond points of this box (2*dims of them)'
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

class LinearAutomatonMode(Freezable):
    'A single mode of a hybrid automaton'

    def __init__(self, parent, name):
        self.name = name

        self.a_matrix = None # dynamics are x' = Ax

        self.parent = parent
        self.transitions = [] # outgoing transitions

        self.freeze_attrs()

    def set_dynamics(self, a_matrix):
        'sets the autonomous system dynamics'

        assert isinstance(a_matrix, csr_matrix)
        assert len(a_matrix.shape) == 2

        assert a_matrix.shape[0] == self.parent.dims and a_matrix.shape[1] == self.parent.dims, \
            "Hybrid Automaton set to {} dimensions, but a_matrix.shape was {}".format(self.parent.dims, a_matrix.shape)

        self.a_matrix = a_matrix

    def __str__(self):
        return '[LinearAutomatonMode: {}]'.format(self.name)

class LinearAutomatonTransition(Freezable):
    'A transition of a hybrid automaton'

    def __init__(self, parent, from_mode, to_mode):
        self.parent = parent
        self.from_mode = from_mode
        self.to_mode = to_mode

        self.condition_list = [] # a list of LinearConstraint, if all are true then the guard is enabled
        from_mode.transitions.append(self)

        self.freeze_attrs()

    def __str__(self):
        return self.from_mode.name + " -> " + self.to_mode.name

class LinearHybridAutomaton(Freezable):
    'The hybrid automaton'

    def __init__(self, dims, name='HybridAutomaton'):
        self.name = name
        self.modes = {}
        self.transitions = []
        self.dims = dims

        self.freeze_attrs()

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
