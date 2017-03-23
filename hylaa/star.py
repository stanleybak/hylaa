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
from hylaa.containers import PlotSettings

class InputStar(Freezable):
    '''
    A sort of light-weight generalized star, used to represent the influence of inputs at a certain step.
    
    This has linear constraints on the input variables: input_a_matrix * u <= input_b_vector, 
    it also has an input_basis_matrix, which is an n rows by m cols matrix, computed by taking the 
    parent star's basis matrix at a particular step, and multiplying by B (from x' = Ax + Bu + c).

    B.T (m by n) * star_basis_matrix (n by n) = input_basis_matrix (m by n) [this is transposed]
    '''

    def __init__(self, mode, star_basis_matrix, time_step):
        input_a_matrix_t = mode.u_constraints_a_t
        input_b_vector = mode.u_constraints_b

        assert isinstance(input_a_matrix_t, np.ndarray)
        assert isinstance(input_b_vector, np.ndarray)
        assert isinstance(star_basis_matrix, np.ndarray)

        self.a_matrix_t = input_a_matrix_t
        self.b_vector = input_b_vector
        gb_t = mode.get_gb_t()

        self.input_basis_matrix = np.dot(gb_t, star_basis_matrix)
        #print "input_basis_matrix = \n{}\n *\n {}\n = \n{}".format(gb_t, star_basis_matrix, self.input_basis_matrix)

        self.freeze_attrs()

class Star(Freezable):
    '''
    A generalized star where linear constraints specify the containment functions.
    In this case, a point x (expressed in the basis vectors) is in the star 
    if, a * x <= b, for every a, b in a_list, b_list (a will be a numpy.ndarray)

    The star's sim_center is directly assigned from the origin simulation vector.

    The star's basis_matrix (collection of basis vectors, one for each row) is a product of:
    - start_basis_matrix (if None then use identity matrix)
    - sim_basis_matrix (not stored in the star)

    During computation, sim_center gets set from the origin vector simulation, and
    basis_matrix gets updated from the orthonormal vector simulation, by assigning
    basis_matrix := dot(sim_basis_matrix, start_basis_matrix). These updates occur in
    star.update_from_sim().

    The other element, start_basis_matrix get assigned upon taking a transition. 
    The predecessor star's basis_matrix gets set to the new star's start_basis_matrix, and
    the predecessor star's standard_center gets taken into account by shifting each of the constraints in the new star.

    A star also can contain an input_star list, each of which is a star. This corresponds to the sets which
    get Minkowski added to the current star, when there are inputs to the system
    '''

    def __init__(self, settings, sim_center, basis_matrix, a_list, b_list,
                 parent, mode, start_basis_matrix=None, total_steps=0, fast_forward_steps=0):

        assert isinstance(sim_center, np.ndarray)
        self.settings = settings
        self.sim_center = sim_center
        self.num_dims = len(sim_center)

        assert isinstance(basis_matrix, np.ndarray)
        self.basis_matrix = basis_matrix # each row is a basis vector

        if start_basis_matrix is None:
            self.start_basis_matrix = None
        else:
            assert isinstance(start_basis_matrix, np.ndarray)
            self.start_basis_matrix = start_basis_matrix

        self.parent = parent # object of type StarParent

        assert isinstance(mode, LinearAutomatonMode)
        self.mode = mode # the LinearAutomatonMode

        assert isinstance(a_list, list)
        assert isinstance(b_list, list)
        assert len(a_list) == len(b_list), "a_list and b_list should be of same length"

        self.a_list = []
        self.b_list = []

        for i in xrange(len(a_list)):
            assert isinstance(a_list[i], np.ndarray), "constraint a_vector shoud be np.array, got {}".format(
                type(a_list[i]))
            assert a_list[i].shape[0] == self.num_dims, "constraints a_list size should match num_dims"
            self.a_list.append(a_list[i])
            self.b_list.append(float(b_list[i]))
        
        self.total_steps = total_steps # number of continuous post steps completed
        self.fast_forward_steps = fast_forward_steps

        if settings.opt_decompose_lp and settings.opt_warm_start_lp and \
                        settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.input_stars = None
        else:
            self.input_stars = [] # list of InputStars

        self.plot_lpi = None # LpInstance for plotting, created when needed
        self.guard_opt_data = GuardOptData(self)
    
        self._verts = None # for plotting optimization, a cached copy of this star's projected polygon verts
        self.freeze_attrs()

    def create_plot_lpi(self):
        'create the LpInstance object for this star + inputs, and return it'

        rv = LpInstance(self.num_dims, self.num_dims)
        rv.update_basis_matrix(self.basis_matrix)

        for i in xrange(len(self.b_list)):
            rv.add_basis_constraint(self.a_list[i], self.b_list[i])

        # add the influence of the inputs
        for input_star in self.input_stars:
            rv.add_input_star(input_star.a_matrix_t, input_star.b_vector, input_star.input_basis_matrix)

        return rv

    def update_from_sim(self, sim_basis_matrix, sim_center, time_step):
        'update the star based on values from a new simulation time instant'

        assert isinstance(sim_basis_matrix, np.ndarray)
        assert isinstance(sim_center, np.ndarray)

        Timers.tic('star.update_from_sim')
        prev_basis_matrix = self.basis_matrix
        
        # add input star using the current basis matrix (before updating)
        input_star = None
        if self.mode.num_inputs > 0:
            mode = self.mode
            input_star = InputStar(mode, prev_basis_matrix, time_step)

            if self.input_stars is not None:
                self.input_stars.append(input_star)

            if self.plot_lpi is not None:
                self.plot_lpi.add_input_star(input_star.a_matrix_t, input_star.b_vector, input_star.input_basis_matrix)

        # update the current step's basis matrix
        if self.start_basis_matrix is None:
            self.basis_matrix = sim_basis_matrix
        else:
            self.basis_matrix = np.dot(self.start_basis_matrix, sim_basis_matrix)

        self.sim_center = sim_center

        if self.plot_lpi is not None:
            self.plot_lpi.update_basis_matrix(self.basis_matrix)

        self._verts = None
        Timers.toc('star.update_from_sim')

        Timers.tic('guard_opt_data.update_from_sim')
        self.guard_opt_data.update_from_sim(input_star)
        Timers.toc('guard_opt_data.update_from_sim')        

    def center(self):
        'get the center point in the standard basis'

        return self.sim_center

    def center_into_constraints(self, basis_center):
        '''
        convert the current sim_center into the star's constraints. This
        creates a new copy of the constraint matrix

        basis center is just: self.vector_to_star_basis(self.sim_center), but it
        should be cached in the DiscreteSucessorParent, so there's not sense in
        computing it twice.
        '''

        offset_vec = np.dot(self.a_mat, basis_center)
        self.b_vec = self.b_vec + offset_vec # (instances are shared between stars; do not modify in place)

        # for completeness, also offset the temp constraints
        for index in xrange(len(self.temp_constraints)):
            lc = self.temp_constraints[index]
            offset = np.dot(lc.vector, basis_center)

            # make a new instance (instances are shared between stars; do not modify in place)
            self.temp_constraints[index] = LinearConstraint(lc.vector, lc.value + offset)

        # reset center_sim to 0
        self.sim_center = np.array([0.0] * self.num_dims, dtype=float)

    def point_to_star_basis(self, standard_pt):
        '''convert a point in the standard basis to the star's basis'''

        vec = standard_pt - self.center()
        
        return self.vector_to_star_basis(vec)

    def vector_to_star_basis(self, standard_vec):
        '''convert a vector in the standard basis to a point in the star's basis'''

        # TODO: this could be optimized by solving a linear system first, and if
        # that fails, doing lstsq
        Timers.tic("vector_to_star_basis() [lstsq]")
        rv = lstsq(self.basis_matrix.T, np.array(standard_vec, dtype=float))[0]
        
        # double-check that we've found the solution within some tolerance
        #assert_array_almost_equal(np.dot(self.basis_matrix.T, rv), standard_vec)
        print ".todo, figure out why this check was failing for powertrain"

        Timers.toc("vector_to_star_basis() [lstsq]")

        assert isinstance(rv, np.ndarray)

        return rv   
 
    def get_feasible_point(self):
        '''
        if it is feasible, this returns a standard point, otherwise returns None
        '''

        lp_constraints = self.to_lp_constraints()

        # maximize the vector in the normal basis (and ignore transformed basis variables)
        c_list = [[0.0] * 2 * self.num_dims]

        result = optutil.optimize_multi(Star.solver, c_list, lp_constraints)[0]

        if result is not None:
            result = np.array([result[self.num_dims + d] for d in xrange(self.num_dims)], dtype=float)
            result += self.center()

        return result

    def is_feasible(self):
        'check if a star is feasible (not empty set)'
        return self.get_feasible_point() is not None

    def commit_temp_constraints(self):
        'convert the temp constraints to permanent constraints'

        if len(self.temp_constraints) > 0:
            temp_a = [lc.vector for lc in self.temp_constraints]
            temp_b = [lc.value for lc in self.temp_constraints]

            new_a_mat = np.append(self.a_mat, temp_a, axis=0)
            new_b_vec = np.append(self.b_vec, temp_b, axis=0)

            self.temp_constraints = []
            self.a_mat = new_a_mat
            self.b_vec = new_b_vec

    def trim_redundant_temp_constraints(self, tol=1e-9):
        'remove temp constraints from the star that are redundant'
        
        if len(self.temp_constraints) > 0:
            lpc = self.to_lp_constraints()
            c_list = []

            for i in xrange(len(lpc.b_temp_ub)):
                vec = lpc.a_temp_ub[i]
                c_list.append([-ele for ele in vec])

            result_list = optutil.optimize_multi(Star.solver, c_list, lpc)          

            for i in xrange(len(result_list)-1, -1, -1):
                result_basis = result_list[i][self.num_dims:]
                vec = lpc.a_temp_ub[i][self.num_dims:]
                result_val = np.dot(vec, result_basis)
                condition_val = lpc.b_temp_ub[i]

                # constraint was redundant
                if result_val + tol <= condition_val:
                    del self.temp_constraints[i]

    def trim_redundant_perm_constraints(self, tol=1e-9):
        'remove perm constraints from the star that are redundant'
        
        assert len(self.temp_constraints) == 0

        lpc = self.to_lp_constraints()
        c_list = []

        for i in xrange(len(lpc.b_perm_ub)):
            vec = lpc.a_perm_ub[i]
            c_list.append([-ele for ele in vec])

        optutil.MultiOpt.reset_per_mode_vars()
        result_list = optutil.optimize_multi(Star.solver, c_list, lpc)          
        optutil.MultiOpt.reset_per_mode_vars()

        new_a_matrix = []
        new_b_vector = []

        for i in xrange(len(result_list)):
            result_basis = result_list[i][self.num_dims:]
            vec = lpc.a_perm_ub[i][self.num_dims:]
            result_val = np.dot(vec, result_basis)
            condition_val = lpc.b_perm_ub[i]

            # if constaint was not redundant; keep it
            if result_val + tol > condition_val:
                new_a_matrix.append(vec)
                new_b_vector.append(condition_val)

        self.a_mat = np.array(new_a_matrix, dtype=float)
        self.b_vec = np.array(new_b_vector, dtype=float)

    def eat_star(self, other_star):
        '''
        merge the other star into this star, changing this star's linear constraints
        (self.a_mat * x <= self.b_vec) by increasing self.b_vec, until all the points in both stars
        satisfy the constraints
        
        This assumes origin_offset is 0 and there are no temp constraints (we're not in the middle 
        of a continuous post).
        '''

        # first update total_steps to be the minimum
        self.total_steps = min(self.total_steps, other_star.total_steps)

        # currently, this resets multi-opt optimization... we may not need to do this

        assert isinstance(other_star, Star)
        assert self.num_dims == other_star.num_dims
        assert (self.sim_center == np.array(self.num_dims * [0.])).all()
        assert (other_star.sim_center == np.array(self.num_dims * [0.])).all()
        assert len(other_star.temp_constraints) == 0
        assert len(self.temp_constraints) == 0

        dims = self.num_dims    

        # 3 copies of variables: (1) standard-basis, (2) self-basis, (3) other-basis
        lpc = optutil.LpConstraints()

        # encode relationship between self-basis and standard-basis
        # V1' * x' - x = 0
        for c in xrange(dims):
            constraint = [0.0 if c != d else -1.0 for d in xrange(dims)] # standard-basis                
            constraint += [ele for ele in self.basis_matrix[:, c]] # self-basis
            constraint += [0.0] * dims # other-basis

            lpc.a_basis_eq.append(constraint)
            lpc.b_basis_eq.append(0)

        # encode relationship between other-basis and standard-basis
        for c in xrange(dims):
            constraint = [0.0 if c != d else -1.0 for d in xrange(dims)] # standard-basis
            constraint += [0.0] * dims # self-basis
            constraint += [ele for ele in other_star.basis_matrix[:, c]] # other-basis

            lpc.a_basis_eq.append(constraint)
            lpc.b_basis_eq.append(0)

        # encode other-star constraints
        for r in xrange(other_star.a_mat.shape[0]):
            constraint = [0.0] * dims # standard-basis                
            constraint += [0.0] * dims # self-basis
            constraint += [other_star.a_mat[r][c] for c in xrange(dims)] # other-basis

            lpc.a_perm_ub.append(constraint)

            # these constraints take into account other_star.basis_center
            #val = other_star.b_vec[r] + np.dot(other_star.a_mat[r], other_star.basis_center)
            val = other_star.b_vec[r]
            lpc.b_perm_ub.append(val)

        c_list = []
 
        # optimize each constraint of the self star
        for row in self.a_mat:
            constraint = [0.0] * dims # standard-basis
            constraint += [-ele for ele in row] # basis, minimize negative = maximize
            constraint += [0.0] * dims # cur-basis

            c_list.append(constraint)

        # different stars can have a different constraint a_matrix, so a reset is required
        optutil.MultiOpt.reset_per_mode_vars()
        result_list = optutil.optimize_multi(Star.solver, c_list, lpc)

        # for each self-star constraint, take the maximum of the original value and the optimized value
        for i in xrange(len(c_list)):
            # real val is self_center_val + self.b_vec[i]

            result_point = result_list[i][dims:2*dims] # in self-basis
            result_val = np.dot(self.a_mat[i], result_point)

            self.b_vec[i] = max(self.b_vec[i], result_val)

        # Reset to clear out last optimization since the star may change
        optutil.MultiOpt.reset_per_mode_vars()

        # reset verts cache, since star may have changed
        self._verts = None

    def clone(self):
        'return a copy of the star'

        return Star(
            self.settings,
            self.sim_center, self.basis_matrix, self.a_list, self.b_list, self.parent, self.mode,
            start_basis_matrix=self.start_basis_matrix,  
            total_steps=self.total_steps, fast_forward_steps=self.fast_forward_steps)

    def __repr__(self):
        '''
        this does not print parent. mode is printed as ha.modes['name']  
        '''

        return "Star(HylaaSettings(), {}, {}, {}, {}, None, ha.modes['{}'], {}, {}, {})".format(
            array_repr(self.sim_center), 
            array_repr(self.basis_matrix), 
            array_repr(self.a_mat), 
            array_repr(self.b_vec),
            self.mode.name,
            array_repr(self.start_basis_matrix) if self.start_basis_matrix is not None else None,
            repr(self.temp_constraints),
            self.total_steps)

    def contains_basis(self, basis_vals, atol=0):
        ''''
        is the passed-in point (already offset by the star's center and 
        expressed as basis vectors) contained in the set?
        '''

        result = np.dot(self.a_mat, basis_vals)
        rv = True

        for i in xrange(len(result)):
            if result[i] > atol + self.b_vec[i]:
                rv = False
                break

        # also check the temporary constraints
        if rv:
            for lc in self.temp_constraints:
                if np.dot(lc.vector, basis_vals) > atol + lc.value:
                    rv = False
                    break

        return rv

    def contains_point(self, standard_point, atol=0):
        'is the passed-in point (in the standard basis) contained in the set?'

        basis_vecs = self.point_to_star_basis(standard_point)

        return self.contains_basis(basis_vecs, atol=atol)

    def __str__(self):
        return "[Star: dims={}, sim_center={}, basis_matrix=\n{}\n, a_mat=\n{}\n, b_vec={}, total_steps={}]".format(
            self.num_dims, self.sim_center, self.basis_matrix, self.a_mat, self.b_vec, self.total_steps) 

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

    def verts(self):
        'get the verticies of the polygon projection of the star used for plotting'

        if self._verts is None:
            xdim = Star.plot_settings.xdim
            ydim = Star.plot_settings.ydim
            use_binary_search = True

            if Star.high_vert_mode:
                use_binary_search = False

            pts = self._find_star_boundaries(use_binary_search=use_binary_search)

            #
            #print "---------------\nLP:"
            #self.plot_lpi.print_lp()

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

        dirs = Star.plot_vecs
        rv = []

        if start + 1 < end:
            mid = (start + end) / 2
            mid_point = np.zeros(self.num_dims)

            self.plot_lpi.minimize(dirs[mid], mid_point, error_if_infeasible=True)

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
        find a constaint-star's boundaries in the passed-in directions. This solves several LPs and
        returns a list of points on the boundary (in the standard basis) which maximize each 
            of the passed-in directions
        '''

        if self.plot_lpi is None:
            self.plot_lpi = self.create_plot_lpi()

        standard_center = self.center()
        point = np.zeros(self.num_dims)
        direction_list = Star.plot_vecs
        rv = []

        assert len(direction_list) > 2

        if not use_binary_search:
            # straightforward approach: minimize in each direction
            last_point = None

            for direction in direction_list:
                self.plot_lpi.minimize(direction, point, error_if_infeasible=True)

                if last_point is None or not np.array_equal(point, last_point):
                    last_point = point.copy()
                    rv.append(standard_center + point)
        else:
            # optimized approach: do binary search to find changes
            self.plot_lpi.minimize(direction_list[0], point, error_if_infeasible=True)
            rv.append(point.copy())

            # add it in thirds, to ensure we don't miss anything
            third = len(direction_list) / 3

            # 0 to 1/3
            self.plot_lpi.minimize(direction_list[third], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(0, third, rv[-1], point)
                rv.append(point.copy())

            # 1/3 to 2/3
            self.plot_lpi.minimize(direction_list[2*third], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(third, 2*third, rv[-1], point)
                rv.append(point.copy())

            # 2/3 to end
            self.plot_lpi.minimize(direction_list[-1], point, error_if_infeasible=True)

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

    b_list = []
    a_list = []

    for i in xrange(num_dims):
        (low, high) = hr.dims[i]

        a_list.append(np.array([1 if d == i else 0 for d in xrange(num_dims)], dtype=float))
        b_list.append(high)

        a_list.append(np.array([-1 if d == i else 0 for d in xrange(num_dims)], dtype=float))
        b_list.append(-low)

    parent = InitParent(mode)
    orthonormal_basis = [[1.0 if d == index else 0.0 for d in xrange(num_dims)] for index in xrange(num_dims)]
    basis_matrix = np.array(orthonormal_basis, dtype=float)

    center = np.array([0.0] * num_dims, dtype=float)
    return Star(settings, center, basis_matrix, a_list, b_list, parent, mode)

def init_constraints_to_star(settings, constraints, mode):
    'convert a list of constraints to a Star'

    assert len(constraints) > 0

    num_dims = len(mode.parent.variables)
    parent = InitParent(mode)

    orthonormal_basis = [[1.0 if d == index else 0.0 for d in xrange(num_dims)] for index in xrange(num_dims)]
    basis_matrix = np.array(orthonormal_basis, dtype=float)

    num_constarints = len(constraints)
    a_list = []
    b_list = []

    for row in xrange(num_constarints):
        lc = constraints[row]

        a_list.append(lc.vector)
        b_list.append(lc.value)

    center = np.array([0.0] * num_dims, dtype=float)
    return Star(settings, center, basis_matrix, a_list, b_list, parent, mode)










