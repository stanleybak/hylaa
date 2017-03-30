'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

import numpy as np

from hylaa.util import Freezable
from hylaa.simutil import SimulationBundle
from hylaa.containers import HylaaSettings

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

    def clone(self):
        'create a deep copy of this LinearConstraints object'

        return LinearConstraint(self.vector.copy(), self.value)

    def __str__(self):
        return '[LinearConstraint: {} * x <= {}]'.format(self.vector, self.value)

    def __repr__(self):
        return 'LinearConstraint({}, {})'.format(repr(self.vector), repr(self.value))

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

        self.a_matrix = None # dynamics are x' = Ax + Bu + c
        self.c_vector = None # dynamics are x' = Ax + Bu + c

        # inputs. set using set_inputs
        self.b_matrix = None # dynamics are x' = Ax + Bu + c
        self.u_constraints_a = None # linear constraints on inputs u are Au <= b
        self.u_constraints_a_t = None
        self.u_constraints_b = None # linear constraints on inputs u are Au <= b
        self.num_inputs = 0

        self.inv_list = [] # a list of LinearConstraint, if all are true then the invariant is true

        self.parent = parent
        self.name = name
        self.transitions = [] # outgoing transitions

        self.is_error = False # is this an error mode of the automaton?

        self.sim_settings = None # assigned first time get_sim_bundle() is called
        self._sim_bundle = None # assigned first time get_sim_bundle() is called
        self._gbt_matrix = None # assigned first time get_gb_t() is called
        self.freeze_attrs()

    def get_sim_bundle(self, settings, star, max_steps_remaining):
        '''
        Get the simulation bundle associated with the dynamics in this mode.

        sim_settings - instance of SimulationSettings
        star - the current star that was popped (used for presimulation)
        max_steps_remaining - maximum number of remaining simulation steps (used for presimulation)
        '''

        assert isinstance(settings, HylaaSettings)

        if self._sim_bundle is None:

            self.sim_settings = settings.simulation

            if settings.print_output is False:
                self.sim_settings.stdout = False

            self._sim_bundle = SimulationBundle(self.a_matrix, self.c_vector, self.sim_settings)

            if self.sim_settings.use_presimulation:
                self.presimulate(star, max_steps_remaining)

        return self._sim_bundle

    def get_existing_sim_bundle(self):
        'get the already-created simulation bundle for this mode'

        assert self._sim_bundle is not None

        return self._sim_bundle

    def presimulate(self, star, max_steps_remaining):
        'this is an optimation where we try to guess the dwell time using a simulation, so we avoid repeated calls'

        sim_bundle = self._sim_bundle

        if len(self.inv_list) == 0:
            num_presimulation_steps = max_steps_remaining
        else:
            pt_in_star = np.array(star.get_feasible_point(), dtype=float)
            origin_sim = sim_bundle.sim_until_inv_violated(pt_in_star, self.inv_list, max_steps_remaining)
            num_presimulation_steps = int(len(origin_sim) * 1.2) # simulate slightly past where the point leaves inv

        if num_presimulation_steps > max_steps_remaining:
            num_presimulation_steps = max_steps_remaining

        sim_bundle.presimulate(num_presimulation_steps)

    def get_gb_t(self):
        ''''get the transpose of G(A, h) * B, where G(A, h) is defined as:

        G(A,h) = A^{-1}(e^{Ah} - I)

        The actual computation of this, however, is done using simulations.
        '''

        assert self.sim_settings is not None, "get_sim_bundle() must be called before get_gb_t()"

        if self._gbt_matrix is None:
            self._gbt_matrix = self._sim_bundle.compute_gbt(self.b_matrix)

        return self._gbt_matrix

    def set_dynamics(self, a_matrix, c_vector=None):
        'sets the non-inputs dynamics. c_vector can be None, in which case zeros are used'

        assert len(self.parent.variables) > 0, "automaton.variables should be set before dynamics"

        if c_vector is None:
            c_vector = np.zeros(len(self.parent.variables))

        assert isinstance(a_matrix, np.ndarray)
        assert len(a_matrix.shape) == 2

        assert isinstance(c_vector, np.ndarray)
        assert len(c_vector.shape) == 1

        assert a_matrix.shape[0] == c_vector.shape[0]
        assert a_matrix.shape[1] == len(self.parent.variables)

        self.a_matrix = a_matrix
        self.c_vector = c_vector

    def set_inputs(self, u_constraints_a, u_constraints_b, b_matrix=None):
        'sets the input dynamics. B matrix can be None, in which case identity matrix is used.'

        assert self.a_matrix is not None, 'dynamics should be set before inputs'

        if b_matrix is None:
            b_matrix = np.identity(len(self.parent.variables))

        assert isinstance(b_matrix, np.ndarray)
        assert isinstance(u_constraints_a, np.ndarray)
        assert isinstance(u_constraints_b, np.ndarray)

        assert len(b_matrix.shape) == 2, "input B matrix should be 2-d"
        assert len(u_constraints_a.shape) == 2
        assert len(u_constraints_b.shape) == 1

        # number of rows of b_matrix should match number of variables for x' = Ax + Bu + c to make sense
        assert b_matrix.shape[0] == len(self.parent.variables), ("the number of rows in the input B matrix ({}) " + \
            "must be equal to the number of variables in the automaton ({})").format(\
            b_matrix.shape[0], len(self.parent.variables))
        self.num_inputs = b_matrix.shape[1]

        assert u_constraints_a.shape[0] == u_constraints_b.shape[0], \
            "input constraints a-matrix and b-vector must have the same number of rows"
        assert u_constraints_a.shape[1] == self.num_inputs, ("the number of columns in the input constraint " + \
            "a-matrix ({}) must equal the number of variables in the input B matrix ({})").format( \
            u_constraints_a.shape[1], self.num_inputs)

        self.b_matrix = b_matrix
        self.u_constraints_a = u_constraints_a
        self.u_constraints_a_t = u_constraints_a.transpose().copy()
        self.u_constraints_b = u_constraints_b

    def __str__(self):
        extra = ' (error mode)' if self.is_error else ''

        return '[LinearAutomatonMode: ' + self.name + extra + ']'

class LinearAutomatonTransition(Freezable):
    'A transition of a hybrid automaton'

    def __init__(self, parent, from_mode, to_mode):
        self.parent = parent
        self.from_mode = from_mode
        self.to_mode = to_mode

        self.condition_list = [] # a list of LinearConstraint, if all are true then the guard is enabled
        from_mode.transitions.append(self)

        self.reset_matrix = None # a matrix (x_post := Ax + b)
        self.reset_vector = None # a vector (x_post := Ax + b)

        self.freeze_attrs()

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
            condition_list = t.condition_list

            for inv_constraint in inv_list:
                already_in_cond_list = False

                for guard_constraint in condition_list:
                    if guard_constraint.almost_equals(inv_constraint, 1e-13):
                        already_in_cond_list = True
                        break

                if not already_in_cond_list:
                    condition_list.append(inv_constraint)
