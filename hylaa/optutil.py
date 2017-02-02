'''
Optimization utility functions for Hylaa.

Stanley Bak
September 2016
'''

from cvxopt import matrix as cvxmatrix, solvers

import glpk.glpkpi as glpk # if fails do: 'sudo apt-get install python-glpk'

from hylaa.timerutil import Timers

class LpConstraints(object):
    '''
    Constraints for passing into optimize_multi.

    For practical optimization of the optimization routines, there are different types of constraints,
    permanent constarints ('perm', temporary constraints('temp'), equality constraints ('eq'), and 
    inequality constraints (upper-bound or 'ub').

    If constraints are permanent ('perm'), they shouldn't change with subsequent invocations, if they
    are temporary ('temp') they are removed after solving.

    All constraints have an a-vector part and a b-value part. For example, an upper-bound constraint is 
    expressed at a*x <= b, where x is the vector of variables.
    '''

    def __init__(self):
        self.a_perm_ub = []
        self.b_perm_ub = []
      
        self.a_temp_ub = []
        self.b_temp_ub = []
 
        self.a_basis_eq = []
        self.b_basis_eq = []

    def __str__(self):
        rv = "perm: {} * x <= {}".format(self.a_perm_ub, self.b_perm_ub)

        if len(self.a_temp_ub) > 0:
            rv += "\ntemp: {} * x <= {}".format(self.a_temp_ub, self.b_temp_ub)

        if len(self.a_basis_eq) > 0:
            rv += "\nbasis: {} * x <= {}".format(self.a_basis_eq, self.b_basis_eq)

        return rv

    def to_ub(self):
        '''
        convert all the constraints into a single a_matrix and b_vector

        returns the tuple (a_matrix, b_vector)
        '''
        
        a_ub = self.a_perm_ub + self.a_temp_ub 
        b_ub = self.b_perm_ub + self.b_temp_ub

        # split equality constraints into two inequality constraints
        for i in xrange(len(self.b_basis_eq)):
            row = self.a_basis_eq[i]
            val = self.b_basis_eq[i]

            a_ub.append(row)
            b_ub.append(val)

            # and the negation (since it's an equality constraint)
            a_ub.append([-ele for ele in row])
            b_ub.append(val)

        return (a_ub, b_ub)

def check_lp_problem(num_vars, lp_constraints):
    'assertion checks that lp is well-formed and compatible'
    
    c = lp_constraints

    if len(c.a_perm_ub) > 0:
        assert len(c.a_perm_ub[0]) == num_vars

    if len(c.a_temp_ub) > 0:
        assert len(c.a_temp_ub[0]) == num_vars

    if len(c.a_basis_eq) > 0:
        assert len(c.a_basis_eq[0]) == num_vars

    assert len(c.a_perm_ub) == len(c.b_perm_ub)
    assert len(c.a_temp_ub) == len(c.b_temp_ub)
    assert len(c.a_basis_eq) == len(c.b_basis_eq)

def optimize_multi(solver, c_list, lp_constraints):
    '''
    solve a linear optimization problem with the same set of constraints but multiple objective functions

    returns a list of solution points
    '''
    rv = []

    if len(c_list) > 0:
        check_lp_problem(len(c_list[0]), lp_constraints)

        if solver == 'glpk-multi':
            rv = MultiOpt.glpk_linprog_multi(c_list, lp_constraints)
        else:
            # make a single a lower-bound matrix
            a_ub, b_ub = lp_constraints.to_ub()

            rv = [optimize_single(solver, c, a_ub, b_ub) for c in c_list]

    return rv

def optimize_single(solver, c, a_ub, b_ub):
    '''
    solve an optimization problem with the passed-in solver. Minimize c st. A*x <= b
    
    returns the solution point
    '''

    Timers.tic("lp")

    result = None

    if solver == 'cvxopt-glpk':
        result = cvxopt_linprog(c, a_ub, b_ub, solver='glpk')
    # this one doesn't like under-constrained problems (see unit tests)
    #elif solver == 'cvxopt':
    #    result = cvxopt_linprog(c, a_ub, b_ub, solver=None)
    #elif solver == 'z3':
    #    result = z3_linprog(c, a_ub, b_ub)
    elif solver == 'glpk':
        result = glpk_linprog_single(c, a_ub, b_ub)
    else:
        raise RuntimeError('unknown lp solver: {}'.format(solver))

    Timers.toc("lp")

    return result    

def cvxopt_linprog(c, a_ub, b_ub, solver='glpk'):
    'use cvxopt to solve the linear programming problem'

    options = {'glpk':{'msg_lev': 'GLP_MSG_OFF'}}
        
    sol = solvers.lp(cvxmatrix(c), cvxmatrix(a_ub).T, cvxmatrix(b_ub), solver=solver, options=options)

    rv = None

    if sol['status'] == 'primal infeasible':
        rv = None
    elif sol['status'] == 'optimal':
        rv = [float(n) for n in sol['x']]
    else:
        raise RuntimeError("LP solver unbounded or failed: {}".format(sol['status']))

    return rv

def glpk_linprog_single(c, orig_a_ub, orig_b_ub):
    'solve a single lp using the direct glpk interface'

    Timers.tic("lp setup")

    num_vars = len(c)
    num_constraints = len(orig_b_ub)

    c = cvxmatrix(c)
    a_ub = cvxmatrix(orig_a_ub).T
    b_ub = cvxmatrix(orig_b_ub)

    prob = glpk.glp_create_prob()

    # minimize the objective
    glpk.glp_set_obj_dir(prob, glpk.GLP_MIN)
   
    # set number of constraints
    glpk.glp_add_rows(prob, num_constraints)

    # add upper bound for each constraint
    for i in xrange(num_constraints):
        glpk.glp_set_row_bnds(prob, i+1, glpk.GLP_UP, 0.0, b_ub[i])

    # number of variables
    glpk.glp_add_cols(prob, num_vars)

    # all variables are free (-inf to int)
    for i in xrange(num_vars):
        glpk.glp_set_col_bnds(prob, i+1, glpk.GLP_FR, 0.0, 0.0)

    # set objective    
    for i in xrange(num_vars):
        glpk.glp_set_obj_coef(prob, i+1, c[i])

    a_elements = num_vars * len(orig_a_ub)
    
    ia = glpk.intArray(a_elements + 1)
    ja = glpk.intArray(a_elements + 1)
    ar = glpk.doubleArray(a_elements + 1)

    # add each constraint
    constraint_index = 1 # start at index 1

    for row in xrange(num_constraints):
        for col in xrange(num_vars):
            val = a_ub[row, col]

            # only add non-zero constraints
            if val != 0:
                ia[constraint_index] = row+1
                ja[constraint_index] = col+1
                ar[constraint_index] = val

                constraint_index += 1

    assert constraint_index <= a_elements

    glpk.glp_load_matrix(prob, constraint_index - 1, ia, ja, ar)

    # set solver parameters
    smcp = glpk.glp_smcp()
    glpk.glp_init_smcp(smcp)
    smcp.msg_lev = glpk.GLP_MSG_OFF # turn off printing

    Timers.toc("lp setup")

    Timers.tic("lp solve")
    glpk.glp_simplex(prob, smcp)
    
    #####Z = glp_get_obj_val(prob)
    rv = None

    result = glpk.glp_get_status(prob)

    if result == glpk.GLP_OPT:
        rv = [glpk.glp_get_col_prim(prob, i+1) for i in xrange(num_vars)]
    elif result == glpk.GLP_NOFEAS:
        rv = None # infeasible
    else:
        msgs = {
            glpk.GLP_OPT: "solution is optimal",
            glpk.GLP_FEAS: "solution is feasible",
            glpk.GLP_INFEAS: "solution is infeasible",
            glpk.GLP_NOFEAS: "problem has no feasible solution",
            glpk.GLP_UNBND: "problem has unbounded solution",
            glpk.GLP_UNDEF: "solution is undefined"}

        msg = msgs.get(result)

        if msg is None:
            msg = "Unknown result code"

        raise RuntimeError("LP was unbounded / errored ({}): {}".format(result, msg))

    del prob
    Timers.toc("lp solve")

    return rv

class MultiOpt(object):
    'static test class for glpk opt'

    # these are one-time allocated / set
    num_vars = None
    vals = None # allocated once and never freed
    inds = None  # allocated once and never freed
    scmp = None

    # These are set on every mode (use 'reset_per_mode_vars()' to clear them)
    prob = None # allocated and freed per-mode
    num_existing_perm_constraints = None
    num_existing_basis_constraints = None

    @staticmethod
    def init_one_time_vars():
        'initialize the one-time variables'

        # solver parameters
        MultiOpt.smcp = glpk.glp_smcp()
        glpk.glp_init_smcp(MultiOpt.smcp)
        MultiOpt.smcp.msg_lev = glpk.GLP_MSG_OFF # turn off printing

    @staticmethod
    def init_per_mode_vars(num_vars, num_basis_constraints):
        'initialize per-mode variables'

        if num_vars != MultiOpt.num_vars:
            MultiOpt.num_vars = num_vars

            # allocate inds, vals, and single_row_index
            MultiOpt.inds = glpk.intArray(num_vars + 1)

            for i in xrange(num_vars):
                MultiOpt.inds[i+1] = i+1

            MultiOpt.vals = glpk.doubleArray(num_vars + 1)

        MultiOpt.prob = glpk.glp_create_prob()

        # minimize the objective
        glpk.glp_set_obj_dir(MultiOpt.prob, glpk.GLP_MIN)

        # number of variables
        glpk.glp_add_cols(MultiOpt.prob, MultiOpt.num_vars)

        # all variables are free (-inf to int)
        for i in xrange(MultiOpt.num_vars):
            glpk.glp_set_col_bnds(MultiOpt.prob, i+1, glpk.GLP_FR, 0.0, 0.0)

        MultiOpt.num_existing_perm_constraints = 0
        MultiOpt.num_existing_basis_constraints = num_basis_constraints

    @staticmethod
    def reset_per_mode_vars():
        'resets per mode variables'

        MultiOpt.num_existing_perm_constraints = 0
        MultiOpt.num_existing_basis_constraints = None

        if MultiOpt.prob is not None:
            glpk.glp_delete_prob(MultiOpt.prob)
            del MultiOpt.prob
            MultiOpt.prob = None

    @staticmethod
    def update_prob_rows(lp_constraints):
        'update the number of rows in the existing problem (delete/add rows)'

        c = lp_constraints

        # at this point, the one-time and per-mode variables are all initialized
        num_rows = glpk.glp_get_num_rows(MultiOpt.prob)

        # adjust the number of rows to match this specific problem
        num_constraints = len(c.b_perm_ub) + len(c.b_basis_eq) + len(c.b_temp_ub)

        if num_rows < num_constraints:
            new_rows = num_constraints - num_rows

            glpk.glp_add_rows(MultiOpt.prob, new_rows)
        elif num_rows > num_constraints:
            raise RuntimeError('Num constraints decreased? Should never happen.')

    @staticmethod
    def add_new_constraints(lp_constraints):
        'add new constraints to the lp problem (both new perm and temp constraints)'

        c = lp_constraints
        num_constraints = len(c.b_perm_ub) + len(c.b_temp_ub) + len(c.b_basis_eq)
        num_vars = MultiOpt.num_vars

        # set any new constraints
        # cvx matrix seems to be noticably faster than np.array or lists
        a_perm_ub = cvxmatrix(c.a_perm_ub).T
        a_temp_ub = cvxmatrix(c.a_temp_ub).T
        a_basis_eq = cvxmatrix(c.a_basis_eq).T

        # the first num_vars row are the variable-basis equality constraints
        row_index = 0
        for row in xrange(len(c.b_basis_eq)):
            for col in xrange(num_vars):
                MultiOpt.vals[col+1] = a_basis_eq[row, col]

            glpk.glp_set_mat_row(MultiOpt.prob, row_index+1, num_vars, MultiOpt.inds, MultiOpt.vals)
            glpk.glp_set_row_bnds(MultiOpt.prob, row_index+1, glpk.GLP_FX, c.b_basis_eq[row], c.b_basis_eq[row])
            row_index += 1

        # next, handle permanent constraints
        if len(c.b_perm_ub) < MultiOpt.num_existing_perm_constraints:
            raise RuntimeError('Permanent constraints can only be added. prev count={}; new count={}'.format(
                MultiOpt.num_existing_perm_constraints, len(c.b_perm_ub)))

        # set any NEW permanent constraints (assume they were added at the end)
        row_index = len(c.b_basis_eq) + MultiOpt.num_existing_perm_constraints

        for row in xrange(MultiOpt.num_existing_perm_constraints, len(c.b_perm_ub)):
            for col in xrange(num_vars):
                MultiOpt.vals[col+1] = a_perm_ub[row, col]

            glpk.glp_set_mat_row(MultiOpt.prob, row_index+1, num_vars, MultiOpt.inds, MultiOpt.vals)
            glpk.glp_set_row_bnds(MultiOpt.prob, row_index+1, glpk.GLP_UP, 0.0, c.b_perm_ub[row])
            row_index += 1
                       
        # update the number of permanent ub constraints
        MultiOpt.num_existing_perm_constraints = len(c.b_perm_ub)

        # set any temporary upper-bound constraints
        for row in xrange(len(c.b_temp_ub)):
            for col in xrange(num_vars):
                MultiOpt.vals[col+1] = a_temp_ub[row, col]

            glpk.glp_set_mat_row(MultiOpt.prob, row_index+1, num_vars, MultiOpt.inds, MultiOpt.vals)
            glpk.glp_set_row_bnds(MultiOpt.prob, row_index+1, glpk.GLP_UP, 0.0, c.b_temp_ub[row])
            row_index += 1

        assert row_index == num_constraints

    @staticmethod
    def glpk_linprog_multi(c_list, lp_constraints):
        '''
        Solve mutliple lps with different objective functions using the direct glpk interface.

        This method is optimized to understand permanent and temporary constraints in the passed-in
        LpConstraints instance. This serves to reduce the setup time with glpk. Also, problem instances 
        are reused when possible, which provides a warm-start feature that generally improves performance.
        '''

        Timers.tic("lp setup")
        num_vars = len(c_list[0])

        if MultiOpt.num_vars is None:
            MultiOpt.init_one_time_vars()

        if MultiOpt.prob is None:
            MultiOpt.init_per_mode_vars(num_vars, len(lp_constraints.b_basis_eq))

        if MultiOpt.num_vars != num_vars:
            raise RuntimeError("number of variables changed between lp calls.")
        
        if MultiOpt.num_existing_basis_constraints != len(lp_constraints.b_basis_eq):
            raise RuntimeError("num b_basis constraints ({}) changed between lp calls (was {}).".format(
                len(lp_constraints.b_basis_eq), MultiOpt.num_existing_basis_constraints))
        
        MultiOpt.update_prob_rows(lp_constraints)
        MultiOpt.add_new_constraints(lp_constraints)

        Timers.toc("lp setup")

        rv = MultiOpt.lp_solve(c_list)

        # if we used temporary constraints, reset the problem since we can't reuse the solution
        if len(lp_constraints.b_temp_ub) > 0:
            MultiOpt.reset_per_mode_vars()

        return rv

    @staticmethod
    def lp_solve(c_list):
        'solve the already-setup linear progam and accumulate the result'

        Timers.tic("lp solve")
        rv = []

        for c in c_list:
            # update objective    
            for i in xrange(MultiOpt.num_vars):
                glpk.glp_set_obj_coef(MultiOpt.prob, i+1, c[i])

            glpk.glp_simplex(MultiOpt.prob, MultiOpt.smcp)    

            result = glpk.glp_get_status(MultiOpt.prob)

            if result == glpk.GLP_OPT:
                rv.append([glpk.glp_get_col_prim(MultiOpt.prob, i+1) for i in xrange(MultiOpt.num_vars)])
            elif result == glpk.GLP_NOFEAS:
                rv.append(None) # infeasible
            else:
                msgs = {
                    glpk.GLP_OPT: "solution is optimal",
                    glpk.GLP_FEAS: "solution is feasible",
                    glpk.GLP_INFEAS: "solution is infeasible",
                    glpk.GLP_NOFEAS: "problem has no feasible solution",
                    glpk.GLP_UNBND: "problem has unbounded solution",
                    glpk.GLP_UNDEF: "solution is undefined"}

                msg = msgs.get(result)

                #print "Printing solution info to lp.txt"
                #glpk.glp_print_sol(MultiOpt.prob, 'lp.txt')

                if msg is None:
                    msg = "Unknown result code"

                raise RuntimeError("LP was unbounded / errored ({}): {}".format(result, msg))

        Timers.toc("lp solve")

        return rv








