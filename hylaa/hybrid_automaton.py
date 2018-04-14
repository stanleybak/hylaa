'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.util import Freezable

def bounds_list_to_init(bounds_list):
    '''convert a list of upper and lower bound tuples for each dimension into:

    (init_space, init_mat, init_mat_rhs, init_range_tuples)
    '''

    dims = len(bounds_list)
    fixed_dims = []
    init_range_tuples = []

    space_data = []
    space_inds = []
    space_indptrs = [0]
    mat_data = []
    mat_inds = []
    mat_indptrs = [0]
    rhs = []

    # each dimension with a range gets its own vector
    # all the fixed dimensions get a single vector
    for dim in xrange(dims):
        lb, ub = bounds_list[dim]

        assert lb <= ub, "expected lower bound ({}) <= upper bonud ({}) in dim {}".format(lb, ub, dim)

        if lb == ub and lb != 0:
            fixed_dims.append(dim)
        elif lb != ub:
            init_range_tuples.append((lb, ub))
            cur_space_dimension = len(space_data)
            space_data.append(1)
            space_inds.append(dim)
            space_indptrs.append(len(space_data))

            mat_data.append(1)
            mat_inds.append(cur_space_dimension)
            mat_indptrs.append(len(mat_data))
            rhs.append(ub)

            mat_data.append(-1)
            mat_inds.append(cur_space_dimension)
            mat_indptrs.append(len(mat_data))
            rhs.append(-lb)

    # if there were non-zero fixed dimensions, add one space dimension for that
    if len(fixed_dims) > 0:
        init_range_tuples.append((1, 1))
        cur_space_dimension = len(space_data)

        for dim in fixed_dims:
            space_data.append(bounds_list[dim][0])
            space_inds.append(dim)

        space_indptrs.append(len(space_data))

        mat_data.append(1)
        mat_inds.append(cur_space_dimension)
        mat_indptrs.append(len(mat_data))
        rhs.append(1)

        mat_data.append(-1)
        mat_inds.append(cur_space_dimension)
        mat_indptrs.append(len(mat_data))
        rhs.append(-1)

    space_dims = len(space_indptrs) - 1
    init_space = csc_matrix((space_data, space_inds, space_indptrs), dtype=float, shape=(dims, space_dims))

    mat_height = len(mat_indptrs) - 1
    init_mat = csr_matrix((mat_data, mat_inds, mat_indptrs), dtype=float, shape=(mat_height, space_dims))

    init_mat_rhs = np.array(rhs, dtype=float)

    return (init_space, init_mat, init_mat_rhs, init_range_tuples)

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
        assert isinstance(parent, LinearHybridAutomaton)

        self.name = name

        # dynamics are x' = Ax + Bu
        self.a_matrix_csr = None
        self.b_matrix_csc = None

        self.u_constraints_csr = None # csr_matrix
        self.u_constraints_rhs = None # np.ndarray

        self.parent = parent
        self.transitions = [] # outgoing transitions
        self.output_space_csr = None

        self.freeze_attrs()

    def set_output_space(self, output_space_csr):
        'sets the output space for the mode'

        assert self.a_matrix_csr is not None, "set_dynamics should be called before set_output_space"
        dims = self.a_matrix_csr.shape[0]
        assert isinstance(output_space_csr, csr_matrix)
        assert output_space_csr.shape[1] == dims, "output space width {} should equal dims {}".format(
            output_space_csr.shape[1], dims)
        assert self.output_space_csr is None, "output space assigned twice (shouldn't be changed after being set)"

        self.output_space_csr = output_space_csr

    def set_inputs(self, b_matrix_csc, u_constraints_csr, u_constraints_rhs):
        'sets the time-varying / uncertain inputs for the mode (optional)'

        assert self.a_matrix_csr is not None, "set_dynamics should be done before set_inputs"
        assert isinstance(b_matrix_csc, csc_matrix)
        assert isinstance(u_constraints_csr, csr_matrix)
        assert isinstance(u_constraints_rhs, np.ndarray)
        u_constraints_rhs.shape = (len(u_constraints_rhs), ) # flatten init_rhs into a 1-d array

        assert u_constraints_csr.shape[0] == u_constraints_rhs.shape[0], "u_constraints rows shoud match rhs len"
        assert u_constraints_csr.shape[1] == b_matrix_csc.shape[1], "u_constraints cols should match b.width"

        assert b_matrix_csc.shape[0] == self.a_matrix_csr.shape[0], \
                "B-mat shape {} incompatible with A-mat shape {}".format(b_matrix_csc.shape, self.a_matrix_csr.shape)

        self.b_matrix_csc = b_matrix_csc
        self.u_constraints_csr = u_constraints_csr
        self.u_constraints_rhs = u_constraints_rhs

    def set_dynamics(self, a_matrix_csr):
        'sets the autonomous system dynamics'

        assert not isinstance(a_matrix_csr, np.ndarray), "dynamics a_matrix should be be sparse matrix"
        assert len(a_matrix_csr.shape) == 2
        assert a_matrix_csr.shape[0] == a_matrix_csr.shape[1]

        self.a_matrix_csr = a_matrix_csr

    def __str__(self):
        return '[LinearAutomatonMode: {}]'.format(self.name)

class LinearAutomatonTransition(Freezable):
    'A transition of a hybrid automaton'

    def __init__(self, parent, from_mode, to_mode):
        self.parent = parent
        self.from_mode = from_mode
        self.to_mode = to_mode

        # matrix * (output_space * var_list) <= rhs
        self.guard_matrix_csr = None
        self.guard_rhs = None

        self.freeze_attrs()

        from_mode.transitions.append(self)

    def set_guard(self, matrix_csr, rhs):
        '''set the guard matrix and right-hand side. The transition is enabled if
        matrix * (output_space * var_list) <= rhs
        '''

        assert isinstance(matrix_csr, csr_matrix)
        assert isinstance(rhs, np.ndarray)

        assert rhs.shape == (matrix_csr.shape[0],)
        assert self.from_mode.output_space_csr is not None, "output_space_csr of mode should be set before set_guard"
        assert self.from_mode.output_space_csr.shape[0] == matrix_csr.shape[1]

        self.guard_matrix_csr = matrix_csr
        self.guard_rhs = rhs

    def __str__(self):
        return self.from_mode.name + " -> " + self.to_mode.name

class LinearHybridAutomaton(Freezable):
    'The hybrid automaton'

    def __init__(self, name='HybridAutomaton'):
        self.name = name
        self.modes = {}
        self.transitions = []

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
