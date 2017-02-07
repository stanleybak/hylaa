'''
Generalized Star and other Star data structures
Stanley Bak
Aug 2016
'''

import numpy as np

from numpy import array_repr
from numpy.linalg import lstsq
from numpy.testing import assert_array_almost_equal

from hylaa.hybrid_automaton import HyperRectangle, LinearAutomatonTransition, LinearAutomatonMode, LinearConstraint
import hylaa.optutil as optutil
from hylaa.timerutil import Timers as Timers

class Star(object):
    '''
    A generalized star where linear constraints specify the containment functions.
    In this case, a point x (expressed in the basis vectors) is in the star 
    if a_mat * x <= b_vec

    The star's sim_center is directly assigned from the origin simulation vector.

    The star's basis_matrix (collection of basis vectors) is a product of:
    - start_basis_matrix (if None then use identity matrix)
    - sim_basis_matrix (not stored in the star)

    During computation, sim_center gets set from the origin vector simulation, and
    basis_matrix gets updated from the orthonormal vector simulation, by assigning
    basis_matrix := dot(sim_basis_matrix, start_basis_matrix). These updates occur in
    star.update_from_sim().

    The other element, start_basis_matrix get assigned upon taking
    a transition. The prestar's basis_matrix gets set to start_basis_matrix, and
    standard_center gets taken into account by shifting each of the constraints.
    '''

    solver = None # optimization solver name, like 'cvxopt'

    def __init__(self, sim_center, basis_matrix, constraint_a_matrix, constraint_b_vector, 
                 parent, start_basis_matrix=None, temp_constraints=None, total_steps=0, fast_forward_steps=0):

        assert isinstance(sim_center, np.ndarray) 
        self.sim_center = sim_center
        self.num_dims = len(sim_center)

        assert isinstance(basis_matrix, np.ndarray)
        self.basis_matrix = basis_matrix # each row is a basis vector

        if start_basis_matrix is None:
            self.start_basis_matrix = None
        else:
            assert isinstance(start_basis_matrix, np.ndarray)
            self.start_basis_matrix = start_basis_matrix

        assert isinstance(parent, StarParent)
        self.parent = parent # object of type StarParent

        assert isinstance(constraint_a_matrix, np.ndarray)
        assert isinstance(constraint_b_vector, np.ndarray)

        self.a_mat = constraint_a_matrix
        self.b_vec = constraint_b_vector

        assert self.a_mat.shape[0] == self.b_vec.shape[0], "a_matrix height should match b_vector height"
        assert self.a_mat.shape[1] == self.num_dims, "a_matrix width should equal num_dims ({})".format(self.num_dims)

        # list of non-permanent LinearConstraint
        if temp_constraints is None:
            temp_constraints = []
       
        self.temp_constraints = []
 
        for lc in temp_constraints:
            assert isinstance(lc, LinearConstraint)
            self.temp_constraints.append(lc)

        self.verts = None # for plotting optimization, a cached copy of this star's projected polygon verts
        self.total_steps = total_steps # number of continuous post steps completed
        self.fast_forward_steps = fast_forward_steps

    def update_from_sim(self, sim_basis_matrix, sim_center):
        'update the star based on values from a new simulation time instant'

        assert isinstance(sim_basis_matrix, np.ndarray)
        assert isinstance(sim_center, np.ndarray)

        Timers.tic('update_from_sim')
        if self.start_basis_matrix is None:
            self.basis_matrix = sim_basis_matrix
        else:
            self.basis_matrix = np.dot(self.start_basis_matrix, sim_basis_matrix)

        self.sim_center = sim_center
        Timers.toc('update_from_sim')

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

        # TODO: this could be optimized by solving a linear program first, and if
        # that fails, doing lstsq
        Timers.tic("vector_to_star_basis() [lstsq]")
        rv = lstsq(self.basis_matrix.T, np.array(standard_vec, dtype=float))[0]
        
        # double-check that we've found the solution within some tolerance
        assert_array_almost_equal(np.dot(self.basis_matrix.T, rv), standard_vec, decimal=1)
        #print ".todo, figure out why this check was failing for powertrain"

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
        self.verts = None

    def clone(self):
        'return a copy of the star'

        return Star(
            self.sim_center, self.basis_matrix, self.a_mat, self.b_vec, self.parent, 
            start_basis_matrix=self.start_basis_matrix, temp_constraints=self.temp_constraints, 
            total_steps=self.total_steps, fast_forward_steps=self.fast_forward_steps)

    def __repr__(self):
        'this does not print parents'

        return "Star({}, {}, {}, {}, None, {}, {}, {})".format(
            array_repr(self.sim_center), 
            array_repr(self.basis_matrix), 
            array_repr(self.a_mat), 
            array_repr(self.b_vec),
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

    def to_lp_constraints(self):
        '''
        convert the star's constraints to an LpConstraints object
        '''

        dims = self.num_dims

        # two copies of variables [x1, x2]
        # x1 is the values in standard basis
        # x2 is the values in the star's basis
        a_basis_eq = []
        b_basis_eq = []

        a_constraints_ub = []    
        b_constraints_ub = []

        # first encode the relationship between the standard basis and star basis
        # V1' * x' - x = 0 as part of the LP
        for c in xrange(dims):
            constraint = [0.0 if c != d else -1.0 for d in xrange(dims)]

            constraint += [ele for ele in self.basis_matrix[:, c]]

            a_basis_eq.append(constraint)
            b_basis_eq.append(0)

        # next, add the constraints from the star
        for r in xrange(self.a_mat.shape[0]):
            constraint = [0.0] * dims + [self.a_mat[r][c] for c in xrange(dims)]

            a_constraints_ub.append(constraint)
            b_constraints_ub.append(self.b_vec[r])

        rv = optutil.LpConstraints()
        rv.a_perm_ub = a_constraints_ub
        rv.b_perm_ub = b_constraints_ub

        rv.a_basis_eq = a_basis_eq
        rv.b_basis_eq = b_basis_eq

        # finally add the manually-created temp ub constraints
        for lin_con in self.temp_constraints:
            constraint = [0.0] * self.num_dims + [lin_con.vector[i] for i in xrange(dims)]

            rv.a_temp_ub.append(constraint)
            rv.b_temp_ub.append(lin_con.value)

        return rv

    def print_constraints_as_lp(self):
        'print the star as an lp problem'

        lpc = self.to_lp_constraints()
        a_ub, b_ub = lpc.to_ub()
        c = [0.0] * 2 * self.num_dims

        print "optimize_single('cvxopt-glpk', {}, {}, {})".format(c, a_ub, b_ub)

    def __str__(self):
        return "[Star: dims={}, sim_center={}, basis_matrix=\n{}\n, a_mat=\n{}\n, b_vec={}, total_steps={}]".format(
            self.num_dims, self.sim_center, self.basis_matrix, self.a_mat, self.b_vec, self.total_steps) 

class StarParent(object):
    '''
    The parent object of a star. Used to track predecessors. This is a parent class for
    the more specific parent types: 
    InitParent, ContinuousPostParent, DiscretePostParent, and AggregationParent
    '''

    def __init__(self, mode):
        self.mode = mode

        assert isinstance(mode, LinearAutomatonMode)

class InitParent(StarParent):
    'a parent which is an initial state'

    def __init__(self, mode):
        assert isinstance(mode, LinearAutomatonMode)
        StarParent.__init__(self, mode)

class ContinuousPostParent(StarParent):
    'a parent of a star which came from a continuous post operation'

    def __init__(self, mode, star):
        StarParent.__init__(self, mode)
        self.star = star

        assert isinstance(star, Star)

class DiscretePostParent(StarParent):
    'a parent of a star which came from a discrete post operation'

    def __init__(self, premode, prestar, prestar_basis_center, transition):
        StarParent.__init__(self, premode)
        self.transition = transition
        self.prestar = prestar
        self.prestar_basis_center = prestar_basis_center

        assert isinstance(prestar, Star)
        assert isinstance(prestar_basis_center, np.ndarray)
        assert isinstance(transition, LinearAutomatonTransition)
        assert premode == transition.from_mode

class AggregationParent(StarParent):
    'a parent of a star which resulted from guard successor aggregation'

    def __init__(self, mode, stars):
        StarParent.__init__(self, mode)

        self.stars = stars

        assert len(stars) > 1, "aggregation successor should be of 2 or more stars"

        for star in stars:
            assert isinstance(star, Star)

def init_point_to_star(pt, mode):
    'convert an n-d point to a LinearConstraintStar'

    dims = [(val, val) for val in pt]

    return init_hr_to_star(HyperRectangle(dims), mode)

def init_hr_to_star(hr, mode):
    'convert a HyperRectangle to a Star'

    assert isinstance(mode, LinearAutomatonMode)

    num_dims = len(hr.dims)
    orthonormal_basis = [[1.0 if d == index else 0.0 for d in xrange(num_dims)] for index in xrange(num_dims)]

    # two constraints for every dimension
    num = 2 * num_dims
    
    b_vec = np.array([0.0] * num, dtype=float)
    a_matrix = np.zeros([num, num_dims], dtype=float)

    for i in xrange(num_dims):
        (low, high) = hr.dims[i]

        a_matrix[2*i][i] = 1.0
        b_vec[2*i] = high

        a_matrix[2*i+1][i] = -1.0
        b_vec[2*i+1] = -low

    parent = InitParent(mode)
    basis_matrix = np.array(orthonormal_basis, dtype=float)

    center = np.array([0.0] * num_dims, dtype=float)
    return Star(center, basis_matrix, a_matrix, b_vec, parent=parent)

def init_constraints_to_star(constraints, mode):
    'convert a list of constraints to a Star'

    assert len(constraints) > 0

    num_dims = len(mode.parent.variables)
    parent = InitParent(mode)

    orthonormal_basis = [[1.0 if d == index else 0.0 for d in xrange(num_dims)] for index in xrange(num_dims)]
    basis_matrix = np.array(orthonormal_basis, dtype=float)

    num_constarints = len(constraints)
    a_matrix = np.zeros([num_constarints, num_dims], dtype=float)
    b_vec = np.zeros([num_constarints], dtype=float)

    for row in xrange(num_constarints):
        lc = constraints[row]

        a_matrix[row, :] = lc.vector
        b_vec[row] = lc.value

    center = np.array([0.0] * num_dims, dtype=float)
    return Star(center, basis_matrix, a_matrix, b_vec, parent=parent)

def find_star_boundaries(star, direction_list):
    '''
    find a constaint-star's boundaries in the passed-in directions. This solves several LPs and

    returns a list of points on the boundary (in the standard basis) which maximize each 
        of the the passed-in directions
    '''

    if not isinstance(star, Star):
        raise RuntimeError("Expected Star object in find_star_boundaries()")

    standard_center = star.center()

    lp_constraints = star.to_lp_constraints()

    # maximize the vector in the normal basis (and ignore transformed basis variables)
    c_list = [[-ele for ele in direction] + [0.0] * star.num_dims for direction in direction_list]

    result_list = optutil.optimize_multi(Star.solver, c_list, lp_constraints)
    rv = []

    for result in result_list:
        if result is None:
            raise RuntimeError("Infeasible star constraints; cannot draw star")

        offset = np.array(result[0:star.num_dims], dtype=float)
        rv.append(standard_center + offset)

    return rv

# helper functions for adding constraints to a star
def add_guard_to_star(star, guard_lc_list):
    '''
    Add a guard's conditions to the passed-in star.

    star - the star to add the constraints to
    guard_lc_list - the list of linear constraints in the guard
    '''

    dims = star.num_dims
    c_list = []
    vec_list = []

    for lc in guard_lc_list:
        vec_list.append([ele for ele in lc.vector])
        c_list.append([-ele for ele in vec_list[-1]] + [0.0] * dims)

        vec_list.append([-ele for ele in lc.vector])
        c_list.append([-ele for ele in vec_list[-1]] + [0.0] * dims)

    lpc = star.to_lp_constraints()
    optutil.MultiOpt.reset_per_mode_vars()
    result_list = optutil.optimize_multi(Star.solver, c_list, lpc)
    standard_center = star.center()

    for i in xrange(len(vec_list)):
        vec = vec_list[i]
        result = result_list[i]
        offset = result[0:dims]
        point = standard_center + offset

        val = np.dot(point, vec)

        # convert the condition to the star's basis
        basis_influence = np.dot(star.basis_matrix, vec)
        center_value = np.dot(standard_center, vec)
        remaining_value = val - center_value

        lc = LinearConstraint(basis_influence, remaining_value)
        star.temp_constraints.append(lc)

def add_box_to_star(star):
    '''
    Add box constraints to the passed-in star.

    star - the star to add the constraints to
    '''

    dims = star.num_dims
    c_list = []
    vec_list = []

    ortho_vec_list = [[1.0 if d == index else 0.0 for d in xrange(dims)] for index in xrange(dims)]

    for vec in ortho_vec_list:
        vec_list.append([ele for ele in vec])
        c_list.append([-ele for ele in vec_list[-1]] + [0.0] * dims)

        vec_list.append([-ele for ele in vec])
        c_list.append([-ele for ele in vec_list[-1]] + [0.0] * dims)

    lpc = star.to_lp_constraints()
    optutil.MultiOpt.reset_per_mode_vars()
    result_list = optutil.optimize_multi(Star.solver, c_list, lpc)
    standard_center = star.center()

    for i in xrange(len(vec_list)):
        vec = vec_list[i]
        result = result_list[i]
        offset = result[0:dims]
        point = standard_center + offset

        val = np.dot(point, vec)

        # convert the condition to the star's basis
        basis_influence = np.dot(star.basis_matrix, vec)
        center_value = np.dot(standard_center, vec)
        remaining_value = val - center_value

        lc = LinearConstraint(basis_influence, remaining_value)
        star.temp_constraints.append(lc)
