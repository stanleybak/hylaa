'''
Generalized Star and other Star data structures
Stanley Bak
Aug 2016
'''

import math
import time

import numpy as np
from numpy import array_repr
from numpy.linalg import lstsq
from numpy.testing import assert_array_almost_equal

from hylaa.glpk_interface import LpInstance
from hylaa.hybrid_automaton import HyperRectangle, LinearAutomatonTransition, LinearAutomatonMode, LinearConstraint
from hylaa.timerutil import Timers as Timers
from hylaa.util import Freezable
from hylaa.starutil import GuardOptData, InitParent
from hylaa.containers import PlotSettings, HylaaSettings

class InputStar(Freezable):
    '''
    A sort of light-weight generalized star, used to represent the influence of inputs at a certain step.

    This has linear constraints on the input variables: input_a_matrix * u <= input_b_vector,
    it also has an input_basis_matrix, which is an n rows by m cols matrix, computed by taking the
    parent star's basis matrix at a particular step, and multiplying by B (from x' = Ax + Bu + c).

    B.T (m by n) * star_basis_matrix (n by n) = input_basis_matrix (m by n) [this is transposed]
    '''

    def __init__(self, mode, star_basis_matrix):
        Timers.tic('InputStar constructor')
        input_a_matrix_t = mode.u_constraints_a_t
        input_b_vector = mode.u_constraints_b

        assert isinstance(input_a_matrix_t, np.ndarray)
        assert isinstance(input_b_vector, np.ndarray)
        assert isinstance(star_basis_matrix, np.ndarray)

        self.a_matrix_t = input_a_matrix_t
        self.b_vector = input_b_vector
        gb_t = mode.get_gb_t()

        Timers.tic('InputStar constructor (dot)')
        self.input_basis_matrix = np.dot(gb_t, star_basis_matrix)
        Timers.toc('InputStar constructor (dot)')
        #print "input_basis_matrix = \n{}\n *\n {}\n = \n{}".format(gb_t, star_basis_matrix, self.input_basis_matrix)
        Timers.toc('InputStar constructor')

        self.freeze_attrs()

class Star(Freezable):
    '''
    A generalized star where linear constraints specify the containment predicate.

    A point alpha (expressed in the basis vectors) is in the star if each
    LinearConstraint is satisfied.

    The star's center is directly assigned from the origin simulation vector at each step.

    The star's basis_matrix (collection of basis vectors, one for each row) is a product of:
    - start_basis_matrix (if None then use identity matrix)
    - sim_basis_matrix (not stored in the star)

    During computation, center gets set from the origin vector simulation, and
    basis_matrix gets updated from the orthonormal vector simulation, by assigning
    basis_matrix := dot(sim_basis_matrix, start_basis_matrix). These updates occur in
    star.update_from_sim().

    For invariant trimming, constraints get added to the star's constraint list. This involves
    computing the inverse of the basis matrix (may lead to numerical issues).

    The other element, start_basis_matrix get assigned upon taking a transition.
    The predecessor star's basis_matrix gets set to the new star's start_basis_matrix, and
    the predecessor star's center gets taken into account by shifting each of the constraints in the new star.

    A star also can contain an input_star list, each of which is a star. This corresponds to the sets which
    get Minkowski added to the current star, when there are inputs to the system. Inputs and discrete
    transitions are currently not compatible.
    '''

    def __init__(self, settings, center, basis_matrix, constraint_list, parent, mode, extra_init=None):

        assert isinstance(center, np.ndarray)
        assert len(constraint_list) > 0
        assert isinstance(settings, HylaaSettings)

        self.settings = settings
        self.center = center
        self.num_dims = len(center)

        assert isinstance(basis_matrix, np.ndarray), "Expected basis matrix to be np.ndarray, got {}".format(
            type(basis_matrix))
        self.basis_matrix = basis_matrix # each row is a basis vector

        self.start_basis_matrix = None

        self.parent = parent

        assert isinstance(mode, LinearAutomatonMode)
        self.mode = mode # the LinearAutomatonMode

        for lc in constraint_list:
            isinstance(lc, LinearConstraint), "constraint_list should be a list of LinearConstraint objects"
            assert lc.vector.shape[0] == self.num_dims, "each star's constraint's size should match num_dims"

        self.constraint_list = [c.clone() for c in constraint_list] # copy constraints

        self.total_steps = 0
        self.fast_forward_steps = 0

        if settings.opt_decompose_lp and settings.opt_warm_start_lp and \
                                         settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.input_stars = None
        else:
            self.input_stars = [] # list of InputStars

        if extra_init is not None:
            start_basis_matrix, total_steps, fast_forward_steps = extra_init
            self.init_post_jump_data(start_basis_matrix, total_steps, fast_forward_steps)

        ###################################
        ## private member initialization ##
        ###################################
        self._star_lpi = None # LpInstance for plotting and non-guard operations
        self._guard_opt_data = GuardOptData(self) # contains LP instance(s) for guard checks
        self._verts = None # for plotting optimization, a cached copy of this star's projected polygon verts

        self.freeze_attrs()

    def init_post_jump_data(self, start_basis_matrix, total_steps, fast_forward_steps):
        '''initialize extra data after a discrete post occurs'''

        if start_basis_matrix is None:
            self.start_basis_matrix = None
        else:
            assert isinstance(start_basis_matrix, np.ndarray)
            self.start_basis_matrix = start_basis_matrix

        self.total_steps = total_steps # number of continuous post steps completed
        self.fast_forward_steps = fast_forward_steps

    def get_lpi(self):
        'get (maybe create) the LpInstance object for this star + inputs, and return it'

        rv = self._star_lpi

        if rv is None:
            rv = LpInstance(self.num_dims, self.num_dims)
            rv.update_basis_matrix(self.basis_matrix)

            for lc in self.constraint_list:
                rv.add_basis_constraint(lc.vector, lc.value)

            # add the influence of the inputs
            if self.input_stars is not None:
                for input_star in self.input_stars:
                    rv.add_input_star(input_star.a_matrix_t, input_star.b_vector, input_star.input_basis_matrix)

            self._star_lpi = rv

        return rv

    def update_from_sim(self, new_basis_matrix, new_center):
        'update the star based on values from a new simulation time instant'

        assert isinstance(new_basis_matrix, np.ndarray)
        assert isinstance(new_center, np.ndarray)

        Timers.tic('star.update_from_sim')
        prev_basis_matrix = self.basis_matrix

        Timers.tic('star.update_from_sim (make input star)')
        # add input star using the current basis matrix (before updating)
        input_star = None
        if self.mode.num_inputs > 0:
            mode = self.mode
            input_star = InputStar(mode, prev_basis_matrix)

            if self.input_stars is not None:
                self.input_stars.append(input_star)

            if self._star_lpi is not None:
                self._star_lpi.add_input_star(input_star.a_matrix_t, input_star.b_vector, input_star.input_basis_matrix)

        Timers.toc('star.update_from_sim (make input star)')
        Timers.tic('star.update_from_sim (update basis matrix)')

        # update the current step's basis matrix
        if self.start_basis_matrix is None:
            self.basis_matrix = new_basis_matrix
        else:
            self.basis_matrix = np.dot(self.start_basis_matrix, new_basis_matrix)

        self.center = new_center

        if self._star_lpi is not None:
            self._star_lpi.update_basis_matrix(self.basis_matrix)

        self._verts = None # cached vertices for plotting are no longer valid

        Timers.toc('star.update_from_sim (update basis matrix)')

        Timers.toc('star.update_from_sim')

        Timers.tic('guard_opt_data.update_from_sim')
        self._guard_opt_data.update_from_sim(input_star)
        Timers.toc('guard_opt_data.update_from_sim')

    def center_into_constraints(self, basis_center):
        '''
        convert the current sim_center into the star's constraints. This
        creates a new copy of the constraint matrix

        basis center is just: self.vector_to_star_basis(self.center), but it
        should be cached in the DiscreteSucessorParent, so there's not sense in
        computing it twice.
        '''

        new_vals = []

        for lc in self.constraint_list:
            offset = np.dot(lc.vector, basis_center)
            lc.value += offset

            new_vals.append(lc.value)

        new_vals = np.array(new_vals, dtype=float)

        # reset center_sim to 0
        self.center = np.zeros((self.num_dims), dtype=float)

        if self._star_lpi is not None:
            self._star_lpi.set_basis_constraint_values(new_vals)

        if self._guard_opt_data is not None:
            self._guard_opt_data.set_basis_constraint_values(new_vals)

    def point_to_star_basis(self, standard_pt):
        '''convert a point in the standard basis to the star's basis'''

        vec = standard_pt - self.center

        return self.vector_to_star_basis(vec)

    def vector_to_star_basis(self, standard_vec):
        '''
        convert a vector in the standard basis to a point in the star's basis.

        This solves basis_matrix * rv = input, which is essentially computing the
        inverse of basis_matrix, which can become ill-conditioned.
        '''

        Timers.tic("vector_to_star_basis()")

        rv = np.linalg.solve(self.basis_matrix.T, standard_vec)

        #rv = lstsq(self.basis_matrix.T, np.array(standard_vec, dtype=float))[0]

        # double-check that we've found the solution within some tolerance
        if not np.allclose(np.dot(self.basis_matrix.T, rv), standard_vec):
            raise RuntimeError("basis matrix was ill-conditioned, vector_to_star_basis() failed")

        Timers.toc("vector_to_star_basis()")

        assert isinstance(rv, np.ndarray)

        return rv

    def get_feasible_point(self, standard_dir=None):
        '''
        if it is feasible, this returns a point which is feasible, otherwise returns None
        '''

        lpi = self.get_lpi()

        dims = self.num_dims
        num_inputs = self.mode.num_inputs
        input_dims = self.total_steps * num_inputs
        result = np.zeros(2 * dims + input_dims)
        opt_direction = -1 * np.array(standard_dir, dtype=float) if standard_dir is not None else np.zeros(dims)

        if lpi.minimize(opt_direction, result, error_if_infeasible=False):
            rv = result[0:dims] + self.center
        else:
            rv = None

        return rv

    def get_guard_intersection(self, index):
        '''
        get the intersection (if it exists) with the guard with the given index

        returns the optimal lp solution vector, or None if the intersection doesn't exist
        '''

        is_error_intersection = self.mode.transitions[index].to_mode.is_error

        return self._guard_opt_data.get_guard_intersection(index, is_error_intersection)

    def is_feasible(self):
        'check if a star is feasible (not empty set)'

        return self.get_feasible_point() is not None

    def add_basis_constraint(self, lc):
        '''add a linear constraint, in the star's basis, to the star's predicate'''

        assert self.mode.num_inputs == 0, "add_basis_constraint() w/ time-varying inputs not yet supported"

        # add to predicate list
        self.constraint_list.append(lc)

        # add to guard opt data
        self._guard_opt_data.add_basis_constraint(lc)

        if self._star_lpi is not None:
            self._star_lpi.add_basis_constraint(lc.vector, lc.value)

        if self._verts is not None:
            self._verts = None

    def trim_to_invariant(self):
        '''
        trim the star to the mode's invariant.

        returns (is_still_feasible, inv_vio_star_list)
        '''

        still_feasible = True
        inv_vio_star_list = []

        if len(self.mode.inv_list) > 0:
            assert self.mode.num_inputs == 0, "mode invariants + dynamics with time-varying inputs not yet supported"

            # check each invariant condition to see if it is violated
            lpi = self.get_lpi()

            for lin_con in self.mode.inv_list:
                objective = np.array([-ele for ele in lin_con.vector], dtype=float)
                result = np.zeros(2 * self.num_dims)

                lpi.minimize(objective, result, error_if_infeasible=True)

                offset = result[0:self.num_dims]
                point = self.center + offset

                val = np.dot(point, lin_con.vector)

                if val > lin_con.value:
                    # add the constraint to the star's constraints
                    # first, convert the condition to the star's basis

                    # basis vectors (non-transpose) * standard_condition
                    basis_condition = np.dot(self.basis_matrix, lin_con.vector)
                    center_value = np.dot(self.center, lin_con.vector)
                    remaining_value = lin_con.value - center_value

                    basis_lc = LinearConstraint(basis_condition, remaining_value)

                    if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
                        # use the inverse of the invariant constraint for plotting

                        inv_lc = LinearConstraint([-1 * ele for ele in basis_lc.vector], -basis_lc.value)

                        inv_vio_star = self.clone()
                        inv_vio_star.add_basis_constraint(inv_lc)

                        # re-check for feasibility after adding the constraint
                        if inv_vio_star.is_feasible():
                            inv_vio_star_list.append(inv_vio_star)

                    # add the constraint AFTER making the plot violation star
                    self.add_basis_constraint(basis_lc)

                # we added a new constraint to the star, check if it's still feasible
                if not self.is_feasible():
                    still_feasible = False
                    break # don't check the remaining invariant linear conditions

        return (still_feasible, inv_vio_star_list)

    def add_std_constraint_direction(self, standard_direction):
        '''
        add a constraint direction, given in the standard basis to the star
        '''

        assert isinstance(standard_direction, np.ndarray)
        assert standard_direction.shape == (self.num_dims,)

        lpi = self.get_lpi()
        basis_direction = np.dot(self.basis_matrix, standard_direction)

        result = np.zeros((2 * self.num_dims))

        # multiplying by -1 turns it into a maximization
        lpi.minimize(-1 * standard_direction, result, error_if_infeasible=True)

        opt_pt = result[:self.num_dims]
        basis_pt = self.vector_to_star_basis(opt_pt)

        opt_val = np.dot(basis_pt, basis_direction)

        # offset the multiple to account for the stars' centers
        opt_val -= np.dot(basis_direction, self.center)

        lc = LinearConstraint(basis_direction, opt_val)

        self.add_basis_constraint(lc)

    def eat_star(self, other_star):
        '''
        merge the other star into this star, changing this star's linear basis constraints values
        until all the points in both stars satisfy all the constraints
        '''

        # This may be possible without inverting the basis matrix... but I couldn't figure it out
        # may not matter; new intersection will probably need to use combined LP

        changed = False

        # first update total_steps to be the minimum
        self.total_steps = min(self.total_steps, other_star.total_steps)

        assert self.input_stars is None or len(self.input_stars) == 0
        assert isinstance(other_star, Star)
        assert self.num_dims == other_star.num_dims

        lpi = other_star.get_lpi()
        result = np.zeros((2 * self.num_dims))

        # possibly increase every constraint
        lc_vals = []

        for lc in self.constraint_list:
            # maximize the basis constraint direction in other_star
            standard_direction = np.dot(np.linalg.inv(self.basis_matrix), lc.vector)

            # multiplying by -1 turns it into a maximization
            lpi.minimize(-1 * standard_direction, result, error_if_infeasible=True)

            opt_pt = result[:self.num_dims]

            basis_pt = self.vector_to_star_basis(opt_pt)

            opt_val = np.dot(basis_pt, lc.vector)

            # offset the multiple to account for the stars' centers
            opt_val += np.dot(lc.vector, other_star.center)
            opt_val -= np.dot(lc.vector, self.center)

            # and update if the new constraint is looser
            if opt_val > lc.value:
                lc.value = opt_val
                changed = True

            lc_vals.append(lc.value)

        # reset cached values if the star's constraints were changed
        if changed:
            self._verts = None

            lc_vals = np.array(lc_vals, dtype=float)

            if self._star_lpi is not None:
                self._star_lpi.set_basis_constraint_values(lc_vals)

            if self._guard_opt_data is not None:
                self._guard_opt_data.set_basis_constraint_values(lc_vals)

    def clone(self):
        'return a copy of the star'

        assert self.input_stars is None or len(self.input_stars) == 0, "clone() not supported with input stars"

        rv = Star(self.settings, self.center, self.basis_matrix, self.constraint_list, self.parent, self.mode)

        rv.init_post_jump_data(self.start_basis_matrix, self.total_steps, self.fast_forward_steps)

        return rv

    def __repr__(self):
        '''
        this does not print parent. mode is printed as ha.modes['name']
        '''

        return "Star(HylaaSettings({}, {}), {}, {}, {}, None, ha.modes['{}'], extra_init=({}, {}, {}))".format(
            self.settings.step,
            self.settings.step * self.settings.num_steps,
            array_repr(self.center),
            array_repr(self.basis_matrix),
            self.constraint_list,
            self.mode.name,
            array_repr(self.start_basis_matrix) if self.start_basis_matrix is not None else None,
            self.total_steps,
            self.fast_forward_steps
            )

    def contains_basis(self, basis_vals, atol=0):
        ''''
        is the passed-in point (already offset by the star's center and
        expressed as basis vectors) contained in the set?
        '''

        a_mat = np.zeros((len(self.constraint_list), self.num_dims))

        for i in xrange(len(self.constraint_list)):
            a_mat[i, :] = self.constraint_list[i].vector

        result = np.dot(a_mat, basis_vals)
        rv = True

        for i in xrange(len(result)):
            if result[i] > atol + self.constraint_list[i].value:
                rv = False
                break

        return rv

    def contains_point(self, standard_point, atol=0):
        'is the passed-in point (in the standard basis) contained in the set?'

        basis_vecs = self.point_to_star_basis(standard_point)

        return self.contains_basis(basis_vecs, atol=atol)

    def __str__(self):
        return "[Star: dims={}, center={}, basis_matrix=\n{}\n, constraint_list=\n{}\n, total_steps={}]".format(
            self.num_dims, self.center, self.basis_matrix, self.constraint_list, self.total_steps)

    ######### star plotting methods below ############

    # global
    plot_vecs = None # list of vectors to optimize in for plotting, assigned in Star.init_plot_vecs
    plot_settings = None # assigned in Star.init_plot_vecs
    high_vert_mode = False # reduce plotting directions if the set has lots of verticies (drawing optimization)

    @staticmethod
    def init_plot_vecs(num_dims, plot_settings):
        'initialize plot_vecs'

        assert num_dims >= 1
        Star.plot_settings = plot_settings

        xdim = plot_settings.xdim
        ydim = plot_settings.ydim

        Star.plot_vecs = []

        if xdim < 0 or xdim >= num_dims:
            raise RuntimeError('plot x dim out of bounds: {} not in [0, {}]'.format(xdim, num_dims-1))

        if ydim < 0 or ydim >= num_dims:
            raise RuntimeError('plot y dim out of bounds: {} not in [0, {}]'.format(ydim, num_dims-1))

        assert plot_settings.num_angles >= 3, "needed at least 3 directions in plot_settings.num_angles"

        step = 2.0 * math.pi / plot_settings.num_angles

        for theta in np.arange(0.0, 2.0*math.pi, step):
            x = math.cos(theta)
            y = math.sin(theta)

            vec = np.array([0.0] * num_dims, dtype=float)
            vec[xdim] = x
            vec[ydim] = y

            Star.plot_vecs.append(vec)

    def print_lp(self):
        '''print the star's lp to stdout'''

        lpi = self.get_lpi()
        lpi.print_lp()

    def verts(self):
        'get the verticies of the polygon projection of the star used for plotting'

        assert Star.plot_settings is not None, "init_plot_vecs() should be called before verts()"

        if self._verts is None:
            xdim = self.settings.plot.xdim
            ydim = self.settings.plot.ydim
            use_binary_search = True

            if Star.high_vert_mode:
                use_binary_search = False

            pts = self._find_star_boundaries(use_binary_search=use_binary_search)

            if len(pts) > len(Star.plot_vecs)/2 and not Star.high_vert_mode:
                # don't use binary search anymore, and reduce the number of directions being plotted

                Star.high_vert_mode = True
                new_vecs = []

                if len(Star.plot_vecs) > 32:
                    for i in xrange(len(Star.plot_vecs)):
                        if i % 4 == 0:
                            new_vecs.append(Star.plot_vecs[i])

                    Star.plot_vecs = new_vecs

            verts = [[pt[xdim], pt[ydim]] for pt in pts]

            # wrap polygon back to first point
            verts.append(verts[0])

            self._verts = verts

        return self._verts

    def _binary_search_star_boundaries(self, start, end, start_point, end_point):
        '''
        return all the optimized points in the star for the passed-in directions, between
        the start and end indices, exclusive

        points which match start_point or end_point are not returned
        '''

        star_lpi = self.get_lpi()

        dirs = Star.plot_vecs
        rv = []

        if start + 1 < end:
            mid = (start + end) / 2
            mid_point = np.zeros(self.num_dims)

            star_lpi.minimize(dirs[mid], mid_point, error_if_infeasible=True)

            not_start = not np.array_equal(start_point, mid_point)
            not_end = not np.array_equal(end_point, mid_point)

            if not_start:
                rv += self._binary_search_star_boundaries(start, mid, start_point, mid_point)

            if not_start and not_end:
                rv.append(mid_point)

            if not np.array_equal(end_point, mid_point):
                rv += self._binary_search_star_boundaries(mid, end, mid_point, end_point)

        return rv

    def _find_star_boundaries(self, use_binary_search=True):
        '''
        find a constaint-star's boundaries using Star.plot_vecs. This solves several LPs and
        returns a list of points on the boundary (in the standard basis) which maximize each
        of the passed-in directions
        '''

        star_lpi = self.get_lpi()

        standard_center = self.center
        point = np.zeros(self.num_dims)
        direction_list = Star.plot_vecs
        rv = []

        assert len(direction_list) > 2

        if not use_binary_search:
            # straightforward approach: minimize in each direction
            last_point = None

            for direction in direction_list:
                star_lpi.minimize(direction, point, error_if_infeasible=True)

                if last_point is None or not np.array_equal(point, last_point):
                    last_point = point.copy()
                    rv.append(standard_center + point)
        else:
            # optimized approach: do binary search to find changes
            star_lpi.minimize(direction_list[0], point, error_if_infeasible=True)
            rv.append(point.copy())

            # add it in thirds, to ensure we don't miss anything
            third = len(direction_list) / 3

            # 0 to 1/3
            star_lpi.minimize(direction_list[third], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(0, third, rv[-1], point)
                rv.append(point.copy())

            # 1/3 to 2/3
            star_lpi.minimize(direction_list[2*third], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(third, 2*third, rv[-1], point)
                rv.append(point.copy())

            # 2/3 to end
            star_lpi.minimize(direction_list[-1], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(2*third, len(direction_list) - 1, rv[-1], point)
                rv.append(point.copy())

            # finally offset all the points by standard_center
            rv = [standard_center + pt for pt in rv]

        # pop last point if it's the same as the first point
        if len(rv) > 1 and np.array_equal(rv[0], rv[-1]):
            rv.pop()

        return rv

def init_point_to_star(settings, pt, mode):
    'convert an n-d point to a LinearConstraintStar'

    dims = [(val, val) for val in pt]

    return init_hr_to_star(settings, HyperRectangle(dims), mode)

def init_hr_to_star(settings, hr, mode):
    'convert a HyperRectangle to a Star'

    assert isinstance(mode, LinearAutomatonMode)

    num_dims = len(hr.dims)

    constraint_list = []

    for i in xrange(num_dims):
        (low, high) = hr.dims[i]

        vector = np.array([1 if d == i else 0 for d in xrange(num_dims)], dtype=float)
        value = high
        constraint_list.append(LinearConstraint(vector, value))

        vector = np.array([-1 if d == i else 0 for d in xrange(num_dims)], dtype=float)
        value = -low
        constraint_list.append(LinearConstraint(vector, value))

    parent = InitParent(mode)
    basis_matrix = np.identity(num_dims, dtype=float)

    center = np.array([0.0] * num_dims, dtype=float)
    return Star(settings, center, basis_matrix, constraint_list, parent, mode)

def init_constraints_to_star(settings, constraint_list, mode):
    'convert a list of constraints to a Star'

    assert len(constraint_list) > 0

    num_dims = len(mode.parent.variables)
    parent = InitParent(mode)

    basis_matrix = np.identity(num_dims, dtype=float)
    center = np.array([0.0] * num_dims, dtype=float)

    return Star(settings, center, basis_matrix, constraint_list, parent, mode)
