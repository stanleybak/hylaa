'''
Hybrid Automaton generic definition for Hylaa
Stanley Bak (Sept 2016)
'''

import numpy as np

import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.util import Freezable
from hylaa.time_elapse import TimeElapser

from hylaa import lputil

class LinearConstraint(Freezable):
    'a single csr sparse linear constraint: csr_vec * x <= rhs'

    def __init__(self, csr_vec, rhs):
        if not isinstance(csr_vec, csr_matrix):
            csr_vec = csr_matrix(csr_vec, dtype=float)
            
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
        self.u_constraints_csc = None # csc_matrix
        self.u_constraints_rhs = None # 1-d np.ndarray

        self.transitions = [] # outgoing Transition objects

        self.inv_list = [] # a list of LinearConstraint, if all are true then the invariant is true

        self.time_elapse = None # a TimeElapse object... initialized on init_time_elapse()

        self.freeze_attrs()

    def is_error(self):
        'is this an error mode'

        return self.a_csr is None

    def set_invariant(self, constraints_csr, constraints_rhs):
        'sets the invariant'

        assert self.a_csr is not None, "A matrix must be set first"
        
        if not isinstance(constraints_csr, csr_matrix):
            constraints_csr = csr_matrix(constraints_csr, dtype=float)
            
        if not isinstance(constraints_rhs, np.ndarray):
            constraints_rhs = np.array(constraints_rhs, dtype=float)

        constraints_rhs.shape = (len(constraints_rhs), ) # flatten rhs into a 1-d array
        assert constraints_csr.shape[1] == self.a_csr.shape[0], \
            "width of invaraiant constraints({}) must equal A matrix size({})".format( \
            constraints_csr.shape[1], self.a_csr.shape[0])
        assert constraints_csr.shape[0] == len(constraints_rhs)

        assert lputil.is_feasible(constraints_csr, constraints_rhs), "invariant constraints were infeasible"

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

    def _check_inputs(self, b_csr, u_constraints_csr, u_constraints_rhs, allow_constants):
        'Run assersion checks on input matrices'

        assert u_constraints_csr.shape[0] == u_constraints_rhs.shape[0], "u_constraints rows shoud match rhs len"
        assert u_constraints_csr.shape[1] == b_csr.shape[1], "u_constraints cols should match b.width"

        assert b_csr.shape[0] == self.a_csr.shape[0], \
                "B-mat shape {} incompatible with A-mat shape {}".format(b_csr.shape, self.a_csr.shape)

        # make sure the input constraints are feasible
        assert lputil.is_feasible(u_constraints_csr, u_constraints_rhs), "input constraints were infeasible"

        #make sure there are not inputs that are fixed to a constant. This is for efficiency reasons. It is better 
        #to add an affine variable to the a matrix and including this as part of A.
        if not allow_constants:
            num_inputs = b_csr.shape[1]

            for i in range(num_inputs):
                # does this input only affect a single variable? --> does b_col have a single nonzero?
                b_col = b_csr[:, i].toarray()
                nonzeros = sum([1 if x != 0 else 0 for x in b_col])

                if nonzeros != 1:
                    continue

                # check if is there a fixed lower and upper bound for this input
                lb_row = np.array([0 if n != i else 1 for n in range(num_inputs)], dtype=float)
                ub_row = np.array([0 if n != i else -1 for n in range(num_inputs)], dtype=float)
                lb = ub = None

                for row, rhs in zip(u_constraints_csr, u_constraints_rhs):
                    row = row.toarray()
                    if np.array_equiv(lb_row, row):
                        lb = rhs
                    elif np.array_equiv(ub_row, row):
                        ub = -rhs

                if ub is None or lb is None:
                    continue

                assert abs(ub-lb) > 1e-9, ("Time-varying input #{} is fixed to {}. This is a (very) inefficient " + \
                    "way encode affine terms. Instead, introduce a fixed affine varible in the A matrix with a' = 0" + \
                    " and a(0) = 1, and refer to that variable in any differential equations that use affine " + \
                    "terms. This check can be disabled by using 'allow_constants=True' in set_inputs().").format(i, lb)
                

    def set_inputs(self, b_csr, u_constraints_csr, u_constraints_rhs, allow_constants=False):
        '''sets the time-varying / uncertain inputs for the mode (optional)

        if allow_constants is True, this will permit inputs that are fixed to constants. This is inefficient though,
        you should instead add an affine variables to the A matrix that's initially equal to 1 with derivative 0, and
        refer to that variable in the A matrix, rather than adding inputs.
        '''
    
        assert self.a_csr is not None, "set_dynamics should be done before set_inputs"
        if not isinstance(b_csr, csr_matrix):
            b_csr = csr_matrix(b_csr, dtype=float)
            
        if not isinstance(u_constraints_csr, csr_matrix):
            u_constraints_csr = csr_matrix(u_constraints_csr, dtype=float)
            
        if not isinstance(u_constraints_rhs, np.ndarray):
            u_constraints_rhs = np.array(u_constraints_rhs, dtype=float)
        
        u_constraints_rhs.shape = (len(u_constraints_rhs), ) # flatten init_rhs into a 1-d array

        # additional checks on inputs
        self._check_inputs(b_csr, u_constraints_csr, u_constraints_rhs, allow_constants)

        self.b_csr = b_csr

        # inputs will be assigned column-by-column, so store constraints in csc matrices
        self.u_constraints_csc = csc_matrix(u_constraints_csr)
        self.u_constraints_rhs = u_constraints_rhs

    def set_dynamics(self, a_csr):
        'sets the autonomous system dynamics (A matrix)'

        if not isinstance(a_csr, csr_matrix):
            a_csr = csr_matrix(a_csr, dtype=float)

        assert a_csr.shape[0] == a_csr.shape[1], "expected square dynamics matrix, got {}".format(a_csr.shape)

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

        self.time_triggered = False # assigned automatically if settings.optimize_tt_transitions == True

        self.transition_index = len(from_mode.transitions)
        from_mode.transitions.append(self)

        self.freeze_attrs()

    def set_guard_true(self):
        '''sets the guard to be True (always enabled)'''

        assert self.from_mode.a_csr is not None, "A-matrix not assigned in predecessor mode {}".format(self.from_mode)
        dims = self.from_mode.a_csr.shape[1]

        self.set_guard(csr_matrix((0, dims), dtype=float), [])

    def set_guard(self, guard_csr, guard_rhs):
        '''set the guard'''
        
        if not isinstance(guard_csr, csr_matrix):
            guard_csr = csr_matrix(guard_csr, dtype=float)
            
        if not isinstance(guard_rhs, np.ndarray):
            guard_rhs = np.array(guard_rhs, dtype=float)
        
        assert self.from_mode.a_csr is not None, "A-matrix not assigned in predecessor mode {}".format(self.from_mode)

        if guard_csr.shape[0] > 0:
            assert guard_csr.shape[1] == self.from_mode.a_csr.shape[0], "guard matrix expected {} columns, got {}" \
                                                          .format(self.from_mode.a_csr.shape[0], guard_csr.shape[1])

        assert lputil.is_feasible(guard_csr, guard_rhs), "guard constraints were infeasible"

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
            reset_csr = csr_matrix(reset_csr, dtype=float)

        if reset_minkowski_csr is not None and not isinstance(reset_minkowski_csr, csr_matrix):
            reset_minkowski_csr = csr_matrix(reset_minkowski_csr, dtype=float)

        if reset_minkowski_constraints_csr is not None and not isinstance(reset_minkowski_constraints_csr, csr_matrix):
            reset_minkowski_constraints_csr = csr_matrix(reset_minkowski_constraints_csr, dtype=float)

        if reset_minkowski_constraints_rhs is not None and not isinstance(reset_minkowski_constraints_rhs, np.ndarray):
            reset_minkowski_constraints_rhs = np.array(reset_minkowski_constraints_rhs, dtype=float)
        
        assert reset_csr.shape[1] == self.from_mode.a_csr.shape[0], "reset matrix expected {} columns, got {}".format(
            self.from_mode.a_csr.shape[0], reset_csr.shape[1])
        assert reset_csr.shape[0] == self.to_mode.a_csr.shape[0], "reset matrix expected {} rows, got {}".format(
            self.to_mode.a_csr.shape[0], reset_csr.shape[0])

        if reset_minkowski_constraints_rhs is not None:
            assert len(reset_minkowski_constraints_rhs.shape) == 1
            assert reset_minkowski_constraints_csr is not None
            assert reset_minkowski_constraints_csr.shape[0] == len(reset_minkowski_constraints_rhs)

            new_vars = reset_minkowski_constraints_csr.shape[1]

            if reset_minkowski_csr is None:
                reset_minkowski_csr = sp.sparse.identity(new_vars, dtype=float, format='csr')

            assert isinstance(reset_minkowski_csr, csr_matrix)
            assert reset_minkowski_csr.shape[0] == self.to_mode.a_csr.shape[0]
            assert reset_minkowski_csr.shape[1] == new_vars, \
                "expected num reset_minkowski columns({}) to match reset_minkowski_constraints columns({})".format( \
                reset_minkowski_csr.shape[1], new_vars)

            assert lputil.is_feasible(reset_minkowski_constraints_csr, reset_minkowski_constraints_rhs), \
                "reset minkowski variable constraints were infeasible"

        self.reset_csr = reset_csr
        self.reset_minkowski_csr = reset_minkowski_csr
        self.reset_minkowski_constraints_csr = reset_minkowski_constraints_csr
        self.reset_minkowski_constraints_rhs = reset_minkowski_constraints_rhs

    def get_guard_intersection(self, lpi):
        '''do an intersection between this guard and the passed-in lpi
        if there is an intersection, return a new lpi that equals the intersection, otherwise return None
        '''

        # optimized version: first check if every constraint is satisfiable, before checking that they're
        # all satisfied at the same time. The first part can be done quickly by only changing the objective function.
        rv = None
        all_sat = True

        for i, row in enumerate(self.guard_csr):
            # row is csr_matrix of a single row
            
            lpi.set_minimize_direction(row, is_csr=True)

            #print('.1 hybrid_automaton t={}, is_feasible before = {}'.format(i, lpi.is_feasible()))
            #print('.2 hybrid_automaton t={}, is_feasible before = {}'.format(i, lpi.is_feasible()))
            
            columns = [lpi.cur_vars_offset + i for i in row.indices]

            result = lpi.minimize(columns=columns, retry_on_unsat=True)

            dot_res = np.dot(result, row.data)

            if dot_res > self.guard_rhs[i]:
                all_sat = False
                break

        if all_sat:
            # make the full lpi including all the guard constraints simultaneously
            t_lpi = lpi.clone()

            # add the guard condition
            lputil.add_curtime_constraints(t_lpi, self.guard_csr, self.guard_rhs)

            if t_lpi.is_feasible():
                rv = t_lpi

        return rv

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

        assert not name in self.modes, "Mode with name '{}' already exists in the automaton".format(name)
        
        m = Mode(self, name, len(self.modes))
        self.modes[m.name] = m
        return m

    def new_transition(self, from_mode, to_mode, name=None):
        '''add a transition'''

        t = Transition(self, from_mode, to_mode, name=name)
        self.transitions.append(t)

        return t

    def check_transition_dimensions(self):
        '''
        check that transitios have appropriate resets if the number of variables changes. This is done automatically
        when set_reset is called, but sometimes this may not be called (identity resets). This will check these cases.
        '''

        for t in self.transitions:
            assert t.from_mode.a_csr is not None, \
                "Outgoing transition detected from error mode: {} (not allowed)".format(t)
            
            if t.to_mode.a_csr is None:
                continue
            
            premode_dims = t.from_mode.a_csr.shape[0]
            postmode_dims = t.to_mode.a_csr.shape[0]

            assert premode_dims == postmode_dims or t.reset_csr is not None, \
                "Transition {} premode has {} dims and postmode has {} dims, but no reset was assigned".format(
                    t, premode_dims, postmode_dims)

    def do_guard_strengthening(self):
        '''
        Strengthen the guards to include the invariants of target modes
        '''

        for t in self.transitions:
            if t.from_mode == t.to_mode or t.time_triggered:
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

    def detect_tt_transitions(self, print_func=print): # if you get a syntax error here, make sure you're using python3
        '''
        Mark all time-triggered transitions within the automaton.

        This checks for guards where one of the conditions is an inequality involving a single variable, x <= K,
        where the mode invariant has the opposite condition, x >= K (encoded as -x <= -K),
        and the derivative of variable x only depends on a single variable: x' == 5a,
        and the derivative of that variable is all zeros: a' == 0,

        at verification-time, a further check will make sure that a is a constant (not an interval), not equal to 0, 
        and that x is flat (not an interval)
        '''

        for mode in self.modes.values():
            if mode.a_csr is None: # skip error modes
                continue
            
            # find all variables with derivative equal to zero
            constant_vars = []
            
            for i in range(mode.a_csr.shape[0]):
                nonzeros = mode.a_csr[i].getnnz()

                if nonzeros == 0: # row of all zeros
                    constant_vars.append(i)

            if not constant_vars:
                continue

            # find all derivatives that only depend on constant vars
            tt_vars = []
            for row_index, row in enumerate(mode.a_csr):
                if row.getnnz() == 0: # skip constant variables
                    continue
                
                all_constant = True

                for i, col in enumerate(row.indices):
                    if row.data[i] != 0 and not col in constant_vars:
                        all_constant = False
                        break

                if all_constant:
                    tt_vars.append(row_index)

            print_func("Checking mode '{}', tt_vars = {}, num transitions = {}".format( \
                mode.name, tt_vars, len(mode.transitions)))

            if not tt_vars:
                continue

            # at this point, we know which variables are constantly changing... check guards / invariants
            for t in mode.transitions:
                if is_time_triggered(t, tt_vars, print_func):
                    t.time_triggered = True

def is_time_triggered(t, tt_vars, print_func):
    'is the passed-in transition time triggered?'

    rv = False

    print_func("checking transition {}, t.guard.rhs = {}".format(t, t.guard_rhs))

    if len(t.guard_rhs) == 1:
        all_tt_vars = True
        for i, col in enumerate(t.guard_csr.indices):
            if t.guard_csr.data[i] != 0 and not col in tt_vars:
                all_tt_vars = False
                break

        print_func("t.guard_csr = {}, all_tt_vars = {}".format(t.guard_csr.toarray(), all_tt_vars))

        if all_tt_vars:
            # check if there is a mode invariant with the opposite condition
            found_invariant = False

            for lc in t.from_mode.inv_list:
                print_func("checking invariant {} <= {}".format(lc.csr.toarray(), lc.rhs))
                
                if lc.rhs == -1 * t.guard_rhs[0] and (-1 * lc.csr != t.guard_csr[0]).nnz == 0:
                    print_func("found opposite invariant!")
                    found_invariant = True
                    break

            if found_invariant:
                rv = True

        print_func("Transition {} {} time triggered".format(t, "is" if rv else "is NOT"))
                   
    return rv

def was_tt_taken(state_lpi, t):
    '''do the run-time checks to see if this transition was a time-triggred one

    if we have x <= K, with x' = -5 * a,
    make sure that a is a constant (not an interval), not equal to 0, and that x is flat (not an interval)
    '''

    assert len(t.guard_rhs) == 1, "should true due to static-time check"
    dims = state_lpi.dims
    rv = False

    # first look at the guard, and see which variables it uses (find 'x' in x <= K, x' == a)
    guard_vars = []

    for i, col in enumerate(t.guard_csr.indices):
        if t.guard_csr.data[i] != 0:
            guard_vars.append(col)

    # next, find all the 'a' in x' == a
    affine_vars = {}
    a_csr = t.from_mode.a_csr

    for var in guard_vars:
        for index in range(a_csr.indptr[var], a_csr.indptr[var+1]):
            if a_csr.data[index] != 0:
                affine_vars[a_csr.indices[index]] = True # insert it into the dict

    all_affine_nonzero = True

    for var in affine_vars:
        min_dir = [1 if n == var else 0 for n in range(dims)]
        max_dir = [-1 if n == var else 0 for n in range(dims)]
        col = state_lpi.cur_vars_offset + var

        min_val = state_lpi.minimize(direction_vec=min_dir, columns=[col])[0]
        max_val = state_lpi.minimize(direction_vec=max_dir, columns=[col])[0]

        if abs(max_val - min_val) > 1e-9 or abs(max_val) < 1e-9:
            all_affine_nonzero = False
            break

    if all_affine_nonzero:
        # make sure x is flat... first find x

        state_cols = [state_lpi.cur_vars_offset + n for n in range(dims)]
        direction = t.guard_csr.toarray()[0]

        min_state = state_lpi.minimize(direction_vec=direction, columns=state_cols)
        max_state = state_lpi.minimize(direction_vec=-1 * direction, columns=state_cols)

        min_val = np.dot(min_state, direction)
        max_val = np.dot(max_state, direction)

        if abs(max_val - min_val) < 1e-9:
            rv = True

    return rv
