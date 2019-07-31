'''
LP Utilities

This assumes a common LP structure, where the
first N columns correspond to the current-time variables, and
the first N rows are the current-time constraints (equality constraints equal to zero)
'''

import math

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix, hstack

import swiglpk as glpk
from hylaa.lpinstance import LpInstance, SwigArray
from hylaa.timerutil import Timers

def from_box(box_list, mode):
    'make a new lp instance from a passed-in box'

    rhs = []

    for lb, ub in box_list:
        assert lb <= ub, "lower bound ({}) > upper bound ({})".format(lb, ub)
        rhs.append(-lb)
        rhs.append(ub)

    # make constraints as csr_matrix
    dims = len(box_list)
    data = []
    inds = []
    indptr = [0]

    # -1 <= -lb
    # 1 <= ub
    for n in range(dims):
        data.append(-1)
        inds.append(n)
        indptr.append(len(data))

        data.append(1)
        inds.append(n)
        indptr.append(len(data))

    csr = csr_matrix((data, inds, indptr), shape=(2*dims, dims), dtype=float)
    csr.check_format()

    return from_constraints(csr, rhs, mode)

def from_zonotope(center, generator_list, mode):
    'make a new lp instance from the passed in zonotope'

    #variables [x1, x2, ..., center_var, alpha1, alpha2, ...]
    mat = []
    rhs = []

    cdims = len(center)
    dims = cdims + 1 + len(generator_list)

    # bounds on center_var (center_var = 1)
    mat.append([1 if d == cdims else 0 for d in range(dims)])
    rhs.append(1)

    mat.append([-1 if d == cdims else 0 for d in range(dims)])
    rhs.append(-1)

    for i, g in enumerate(generator_list):
        assert len(g) == cdims, "expected each generator to have the same number of dims as the center"
        
        # bounds on each alpha (-1 <= alpha <= 1)
        mat.append([1 if d == cdims + 1 + i else 0 for d in range(dims)])
        rhs.append(1)

        mat.append([-1 if d == cdims + 1 + i else 0 for d in range(dims)])
        rhs.append(1)

    # zonotope generator constraints x = c + alpha1 * g1 + alpha2 * g2 + ...
    for dim, c in enumerate(center):
        row = [0] * dims

        row[dim] = -1
        row[cdims] = c

        for gindex, generator in enumerate(generator_list):     
            row[cdims + 1 + gindex] = generator[dim]

        mat.append(row)
        rhs.append(0)

        mat.append([-1 * x for x in row])
        rhs.append(0)

    return from_constraints(mat, rhs, mode, dims=cdims)

def from_constraints(csr, rhs, mode, types=None, names=None, dims=None):
    '''make a new lp instance from a passed-in set of constraints and rhs

    if types/names is None, then assume all constraints are '<=' constriants

    if dims is None, assume the number of columns in the csr matrix is the number of variables
    otherwise, assume the left-most columns in the csr_matrix are the current-time variables 

    '''

    if not isinstance(csr, csr_matrix):
        csr = csr_matrix(csr, dtype=float)

    if not isinstance(rhs, np.ndarray):
        rhs = np.array(rhs, dtype=float)

    assert len(rhs.shape) == 1
    assert csr.shape[0] == len(rhs)
    assert is_feasible(csr, rhs), "initial constraints are not feasible"
    assert (types is None) == (names is None)

    if dims is None:
        dims = csr.shape[1]
    else:
        assert dims <= csr.shape[1]

    lpi = LpInstance()
    lpi.add_rows_equal_zero(dims)

    if names is None:
        names = ["m{}_i{}".format(mode.mode_id, var_index) for var_index in range(csr.shape[1])]
        
    names += ["m{}_c{}".format(mode.mode_id, var_index) for var_index in range(dims)]
    
    lpi.add_cols(names)

    if types is None:
        lpi.add_rows_less_equal(rhs)
    else:
        assert len(types) == len(rhs)
        lpi.add_rows_with_types(types, rhs)

    has_inputs = mode.b_csr is not None

    if has_inputs:
        names = ["m{}_ti{}".format(mode.mode_id, n) for n in range(dims)]
        lpi.add_cols(names)

    # make constraints as csr_matrix
    data = []
    inds = []
    indptr = [0]

    # I 0 -I for first n rows
    for n in range(dims):
        data.append(1)
        inds.append(n)

        data.append(-1)
        inds.append(csr.shape[1] + n)

        if has_inputs:
            data.append(1)
            inds.append(csr.shape[1] + dims + n)

        indptr.append(len(data))
        
    num_cols = csr.shape[1] + dims if not has_inputs else 2*dims + csr.shape[1]
    basis_constraints = csr_matrix((data, inds, indptr), shape=(dims, num_cols), dtype=float)
    basis_constraints.check_format()

    lpi.set_constraints_csr(basis_constraints)

    # add constraints on initial conditions, offset by <dims> rows
    lpi.set_constraints_csr(csr, offset=(dims, 0))

    # add total input effects rows
    if has_inputs:
        rows_before = lpi.get_num_rows()
        ie_pos = (rows_before, dims + csr.shape[1])
        lpi.add_rows_equal_zero(dims)

        # -I
        csr_ti = -1 * sp.sparse.identity(dims, dtype=float, format='csr')
        lpi.set_constraints_csr(csr_ti, offset=ie_pos)
    else:
        ie_pos = None

    cur_vars_offset = csr.shape[1]
    lpi.set_reach_vars(dims, (0, 0), cur_vars_offset, ie_pos)

    return lpi

def set_basis_matrix(lpi, basis_mat):
    'modify the lpi in place to set the basis matrix'

    assert basis_mat.shape[0] == basis_mat.shape[1], "expected square matrix"
    assert basis_mat.shape[0] == lpi.dims, \
      f"basis matrix wrong shape, expected ({lpi.dims}, {lpi.dims}), got {basis_mat.shape}"

    # this is done using the optimized swigvec interface in lpinstance

    data_vec_list = [] # list of swig doubleArray objects for each row

    # 0 BM 0 -I 0 (I? <- if inputs exist)
    for row in range(lpi.dims):
        data = []

        data += basis_mat[row].tolist()
        
        data.append(-1.0)

        if lpi.input_effects_offsets is not None:
            data.append(1.0)

        data_vec_list.append(SwigArray.as_double_array(data))

    entries_per_row = basis_mat.shape[0] + 1 + (1 if lpi.input_effects_offsets else 0)
    count_list = [entries_per_row] * basis_mat.shape[0]
        
    lpi.set_constraints_swigvec_rows(data_vec_list, lpi.bm_indices, count_list, lpi.basis_mat_pos[0])

def add_input_effects_matrix(lpi, input_mat, mode, lgg_beta=None):
    'add an input effects matrix to this lpi'

    assert lpi.input_effects_offsets is not None
    assert mode.b_csr is not None

    num_vars = mode.a_csr.shape[1]
    num_inputs = mode.b_csr.shape[1]
    num_constraints = len(mode.u_constraints_rhs)

    # if lgg approximation model is used, then input effects will also bloat in every dimension by beta
    # this makes the input effects matrix wider
    if lgg_beta is None:
        assert input_mat.shape[1] == num_inputs
    else:
        assert input_mat.shape[1] + num_inputs + num_vars
    
    assert input_mat.shape[0] == num_vars
    assert lpi.dims == num_vars

    # add new row/cols
    names = ["m{}_I{}".format(mode.mode_id, i) for i in range(num_inputs)]

    if lgg_beta is not None: # bloating variables
        names += ["m{}_b{}".format(mode.mode_id, i) for i in range(num_vars)]
    
    pre_cols = lpi.get_num_cols()
    lpi.add_cols(names)

    pre_rows = lpi.get_num_rows()
    lpi.add_rows_less_equal(mode.u_constraints_rhs)

    if lgg_beta is not None: # bloating constraints
        pre_bloat_rows = lpi.get_num_rows()
        lpi.add_rows_less_equal([lgg_beta] * (2 * num_vars))

    # set constraints on the rows/cols, as well as the input basis matrix using a csc_matrix
    data = []
    inds = []
    indptr = [0]

    for c in range(num_inputs):
        # input basis matrix column c
        for row in range(lpi.dims):
            data.append(input_mat[row, c])
            inds.append(row + lpi.input_effects_offsets[0])

        # constraints_csr column c
        for i in range(mode.u_constraints_csc.indptr[c], mode.u_constraints_csc.indptr[c+1]):
            row = mode.u_constraints_csc.indices[i]
            val = mode.u_constraints_csc.data[i]

            data.append(val)
            inds.append(pre_rows + row)

        indptr.append(len(data))

    if lgg_beta is not None: # constraints for bloating
        for v in range(num_vars):
            # input basis matrix column num_inputs + v
            for row in range(num_vars):
                data.append(input_mat[row, num_inputs + v])
                inds.append(row + lpi.input_effects_offsets[0])

            # constraints_csr column corresponding to variable v
            data.append(1)
            inds.append(pre_bloat_rows + 2*v)

            data.append(-1)
            inds.append(pre_bloat_rows + 2*v + 1)

            indptr.append(len(data))

    num_cols = num_inputs
    num_rows = pre_rows + num_constraints

    if lgg_beta is not None:
        num_rows += 2 * num_vars
        num_cols += num_vars
    
    csc = csc_matrix((data, inds, indptr), shape=(num_rows, num_cols), dtype=float)
    csc.check_format()

    lpi.set_constraints_csc(csc, offset=(0, pre_cols))

def check_intersection(lpi, lc, tol=1e-13):
    '''check if there is an intersection between the LP constriants and the LinearConstraint object lc

    This solves an LP optimizing in the given direction... without adding the constraint to the LP

    This returns True/False if an intersection is possible

    it also can return None if the lp is infeasible
    '''

    Timers.tic("check_intersection")

    lpi.set_minimize_direction(lc.csr, is_csr=True)

    columns = lc.csr.indices[0:lc.csr.indptr[1]]
    lp_columns = [lpi.cur_vars_offset + c for c in columns]

    lp_res = lpi.minimize(columns=lp_columns, fail_on_unsat=False)

    if lp_res is None:
        # sometimes, changing optimization direction makes lp infeasible (up to numerical accuracy)
        # this happens in gearbox with small time steps. In this case, return no intersection
        rv = None
    else:
        dot_res = np.dot(lc.csr.data, lp_res)
        rv = dot_res + tol <= lc.rhs

    Timers.toc("check_intersection")

    return rv

def add_init_constraint(lpi, vec, rhs, basis_matrix=None, input_effects_list=None, row_index=None):
    '''
    add a constraint to the lpi

    this replaces an existing row or adds a new one (if row_index is None), with constraints assigned to the 
    initial-time variables

    this returns the row of the newly-created constraint
    '''

    if basis_matrix is None:
        basis_matrix = get_basis_matrix(lpi)

    assert isinstance(basis_matrix, np.ndarray)

    # we need to project the basis matrix using the passed in direction vector
    preshape = vec.shape

    dims = basis_matrix.shape[0]
    vec.shape = (1, dims) # vec is now the projection matrix for this direction
    bm_projection = np.dot(vec, basis_matrix)

    rows = lpi.get_num_rows()
    cols = lpi.get_num_cols()

    if row_index is None:
        row_index = rows
        lpi.add_rows_less_equal([rhs])
    else:
        # overwrite the old rhs
        lpi.set_constraint_rhs(row_index, rhs)

    indptr = [0]

    # basis matrix
    inds = [lpi.basis_mat_pos[1] + i for i in range(dims)]
    data = [val for val in bm_projection[0]]

    # each of the input effects matrices
    if input_effects_list:
        num_inputs = input_effects_list[0].shape[1]
        offset = lpi.input_effects_offsets[1] + dims

        for ie_mat in input_effects_list:
            ie_projection = np.dot(vec, ie_mat)[0]

            assert len(ie_projection) == num_inputs, \
              f"len(ie_projection) ({len(ie_projection)}) != num_inputs ({num_inputs})"

            inds += [offset + i for i in range(num_inputs)]
            data += [val for val in ie_projection]
            offset += num_inputs

    indptr.append(len(data))

    # restore vector shape to what it was when passed in
    vec.shape = preshape

    csr_row_mat = csr_matrix((data, inds, indptr), dtype=float, shape=(1, cols))
    csr_row_mat.check_format()

    lpi.set_constraints_csr(csr_row_mat, offset=(row_index, 0))

    return row_index

def try_replace_init_constraint(lpi, old_row_index, direction, rhs, basis_mat=None, input_effects_list=None):
    '''replace the constraint in row_index by a new constraint, if the new constraint is stronger, otherwise
    create new constriant

    this is used for removing redundant invariant constraints

    This returns (row_index, is_stronger) where row_index is the index of the new constraint and is_stronger is
    a boolean indicating if the old constraint was completely replaced (the new constraint was strictly stronger)
    '''

    is_stronger = True

    if basis_mat is None:
        basis_mat = get_basis_matrix(lpi)

    # Improved algorithm to do this: Replace the constraint, then optimize in the direction of the old constraint to
    # check if the old constraint is still feasible.
    # this avoids deleting rows and therefore possibly ruining warm start if the constraint was non-basic

    # first extract the old constraint
    old_constraint = lpi.get_row(old_row_index)
    old_rhs = lpi.get_rhs([old_row_index])[0]
        
    # replace the old constraint with the new one
    new_row_index = add_init_constraint(lpi, direction, rhs, basis_matrix=basis_mat, \
                                        input_effects_list=input_effects_list, row_index=old_row_index)

    # optimize in the direction of the old constraint, to see if the old constraint is still feasible
    lpi.set_minimize_direction(-1 * old_constraint, is_csr=True, offset=0)

    res = lpi.minimize(fail_on_unsat=False)

    # if the lp was unsat, this means adding the new constraint makes it infeasible, so it's safe to replace
    if res is not None:
        res_dot = (old_constraint * np.array(res, dtype=float))

        res_dot = res_dot[0]

        # if res_dot < old_rhs, then the old constraint is no longer needed, otherwise, re-add it
        if res_dot >= old_rhs:
            lpi.add_rows_less_equal([old_rhs])
            rows = lpi.get_num_rows()

            lpi.set_constraints_csr(old_constraint, offset=(rows - 1, 0))
            is_stronger = False

    return new_row_index, is_stronger

def aggregate_chull(lpi_list, mode):
    '''
    perform aggregation using convex hull (non-recursive call)

    This uses the sop closed convex hull algorithm from Willem Hagemann. See his PhD Dissertation:
    "Symbolic Orthogonal Projections: A New Polyhedral Representation for Reachability Analysis of Hybrid Systems"
    '''

    if len(lpi_list) == 1:
        return lpi_list[0].clone()

    # make all the base LP's complete by adding the redudant row 0 * x <= 1
    new_lpi_list = []

    for old_lpi in lpi_list:
        lpi = old_lpi.clone()
        lpi.add_rows_less_equal([1])
        new_lpi_list.append(lpi)

    lpi = _aggregate_chull_recursive(new_lpi_list, mode)

    # need to essentially take snapshot variables
    csr = lpi.get_full_constraints()
    rhs = lpi.get_rhs()
    types = lpi.get_types()
    names = lpi.get_names()

    # from_constraints assumes left-most variables are current-time variables
    return from_constraints(csr, rhs, mode, types=types, names=names, dims=lpi.dims)

def _aggregate_chull_recursive(lpi_list, mode):
    '''
    perform aggregation using convex hull (recursive call)

    This uses the sop closed convex hull algorithm from Willem Hagemann. See his PhD Dissertation:
    "Symbolic Orthogonal Projections: A New Polyhedral Representation for Reachability Analysis of Hybrid Systems"
    '''

    assert len(lpi_list) >= 2

    # recursive cases:
    #if isinstance(lpi_list, LpInstance):
    #    return lpi_list

    if len(lpi_list) > 2:
        mid = len(lpi_list) // 2

        return aggregate_chull([aggregate_chull(lpi_list[:mid], mode), aggregate_chull(lpi_list[mid:], mode)], mode)

    # base case: exactly two lpis in lpi_list
    assert len(lpi_list) == 2

    dims = lpi_list[0].dims
    assert lpi_list[1].dims == dims

    lpi1 = lpi_list[0]
    csr1 = lpi1.get_full_constraints()
    rhs1 = lpi1.get_rhs()
    types1 = lpi1.get_types()

    # csr1 contains L_left A L_right, split it into these three in order to construct L
    l_left = csr1[:, 0:lpi1.cur_vars_offset]
    a1 = csr1[:, lpi1.cur_vars_offset:lpi1.cur_vars_offset+dims]
    l_right = csr1[:, lpi1.cur_vars_offset+dims:]

    l1 = hstack([l_left, l_right])

    # repeat for lpi2
    lpi2 = lpi_list[1]
    csr2 = lpi2.get_full_constraints()
    rhs2 = lpi2.get_rhs()
    types2 = lpi2.get_types()

    # csr1 contains L_left A L_right, split it into these three in order to construct L
    l_left = csr2[:, 0:lpi2.cur_vars_offset]
    a2 = csr2[:, lpi2.cur_vars_offset:lpi2.cur_vars_offset+dims]
    l_right = csr2[:, lpi2.cur_vars_offset+dims:]
    l2 = hstack([l_left, l_right])

    lpi = LpInstance()
    #lpi.add_rows_equal_zero(dims)

    types = types1 + types2
    rhs = [n for n in rhs1] + ([0] * len(rhs2))

    lpi.add_rows_with_types(types, rhs)

    cols = []

    cols += [f"A1_{i}" for i in range(dims)]
    cols += [f"A2_{i}" for i in range(dims)]
    cols += [f"L1_{i}" for i in range(l1.shape[1])]
    cols += [f"L2_{i}" for i in range(l2.shape[1])]
    cols += ["a"]

    lpi.add_cols(cols)

    # set constraints
    l2_zero = csr_matrix((a1.shape[0], l2.shape[1])) # the 0 above L2
    l1_zero = csr_matrix((a2.shape[0], l1.shape[1])) # the 0 below L1
    a1_zero = csr_matrix((a2.shape[0], a1.shape[1])) # the 0 below A1

    rhs1_vmat = csr_matrix(np.array([[num] for num in rhs1]))
    rhs2_vmat = csr_matrix(np.array([[num] for num in rhs2]))

    top = csr_matrix(hstack([a1, a1, l1, l2_zero, rhs1_vmat]))
    lpi.set_constraints_csr(top)

    bottom = csr_matrix(hstack([a1_zero, -a2, l1_zero, l2, -rhs2_vmat]))
    lpi.set_constraints_csr(bottom, offset=(top.shape[0], 0))

    lpi.dims = dims
    lpi.cur_vars_offset = 0 # left-most variables are current time variables

    return lpi

def aggregate(lpi_list, direction_matrix, mode):
    '''
    return a new lpi consisting of an aggregation of the passed-in lpi list

    This creates a template polytope using the passed-in directions (passed in as rows of direction_matrix).

    use lputil.make_direction_matrix() to create the direction_matrix with arnoldi directions
    '''

    assert isinstance(direction_matrix, np.ndarray)
    assert direction_matrix.dtype == float
    assert direction_matrix.shape[0] >= direction_matrix.shape[1], "expected num directions >= dims"
    assert len(lpi_list) > 1, "expected more than one lpi to perform an aggregation"

    inds = []
    data = []
    indptrs = [0]
    rhs = []

    # for each direction, minimize and maximize it within the list
    for direction in direction_matrix:
        if abs(np.linalg.norm(direction)) < 1e-6:
            continue
        #assert abs(np.linalg.norm(direction) - 1) < 1e-9, "expected normalized directions, got {}".format(direction)

        dir_inds = [i for i, x in enumerate(direction) if x != 0]
        dir_data = [x for x in direction if x != 0]
        dir_neg_data = [-x for x in dir_data]

        max_val = -np.inf
        min_val = np.inf
       
        for lpi in lpi_list:
            assert direction_matrix.shape[1] == lpi.dims

            dir_columns = [i + lpi.cur_vars_offset for i in dir_inds]

            result = lpi.minimize(direction_vec=-direction, columns=dir_columns)
            max_val = max(max_val, np.dot(result, dir_data))
            
            result = lpi.minimize(direction_vec=direction, columns=dir_columns)
            min_val = min(min_val, np.dot(result, dir_data))

        inds += dir_inds
        data += dir_data
        indptrs.append(len(data))
        rhs.append(max_val)

        inds += dir_inds
        data += dir_neg_data
        indptrs.append(len(data))
        rhs.append(-min_val)

    rows = len(indptrs) - 1
    cols = direction_matrix.shape[1]
    csr_mat = csr_matrix((data, inds, indptrs), dtype=float, shape=(rows, cols))
    csr_mat.check_format()
    
    rv = from_constraints(csr_mat, rhs, mode)

    return rv

def get_basis_matrix(lpi):
    'get the basis matrix from the lpi'

    row = lpi.basis_mat_pos[0]
    col = lpi.basis_mat_pos[1]

    # get_dense_constraints takes in x and y
    return lpi.get_dense_constraints(col, row, lpi.dims, lpi.dims)

def scale_with_bm(lpi, amount):
    '''
    scale the current set using the basis matrix
    '''

    bm = get_basis_matrix(lpi)
    set_basis_matrix(lpi, amount * bm)

def bloat(lpi, amount, var_name_prefix='bloat'):
    '''
    bloat the current set of states
    '''

    assert amount >= 0

    # strategy add n variables with bounds -amount <= x <= amount
    names = [f"{var_name_prefix}{n}" for n in range(lpi.dims)]
    precols = lpi.get_num_cols()
    prerows = lpi.get_num_rows()

    data = []
    inds = []
    indptr = [0]

    for n in range(lpi.dims):
        # 1.0 entry in the basis matrix row for variable n
        data.append(1)
        inds.append(lpi.basis_mat_pos[0] + n)
        
        # x <= amount
        data.append(1)
        inds.append(prerows + 2*n)

        # -x <= amount ---> x >= -amount
        data.append(-1)
        inds.append(prerows + 2*n + 1)
        indptr.append(len(data))

    mat = csc_matrix((data, inds, indptr), shape=(prerows + 2*lpi.dims, lpi.dims), dtype=float)

    rhs = [amount] * (2 * lpi.dims)
    lpi.add_rows_less_equal(rhs)
    lpi.add_cols(names)

    lpi.set_constraints_csc(mat, offset=(0, precols))

def add_reset_variables(lpi, mode_id, transition_index, # pylint: disable=too-many-locals, too-many-statements
                        reset_csr=None, minkowski_csr=None,
                        minkowski_constraints_csr=None, minkowski_constraints_rhs=None, successor_has_inputs=False): 
    '''
    add variables associated with a reset

    general resets are of the form x' = Rx + My, Cy <= rhs, where y are fresh variables
    the reset_minkowski variables can be None if no new variables are needed. If unassigned, the identity
    reset is assumed

    x' are the new variables
    x are the old variables       
    reset_csr is R (None -> identity)
    minkowski_csr is M
    minkowski_constraints_csr is C
    minkowski_constraints_rhs is rhs

    this function adds new variables for both the initial states and the current states in the new mode
    '''

    old_dims = lpi.dims
    cols = lpi.get_num_cols()
    rows = lpi.get_num_rows()

    if reset_csr is None:
        reset_csr = sp.sparse.identity(old_dims, dtype=float, format='csr')

    if minkowski_csr is None:
        minkowski_csr = csr_matrix((0, 0))
        minkowski_constraints_csr = csr_matrix((0, 0))
        minkowski_constraints_rhs = np.array([])

    assert isinstance(reset_csr, csr_matrix)
    assert isinstance(minkowski_csr, csr_matrix)
    assert old_dims == reset_csr.shape[1], "Reset matrix shape is wrong (expected {} cols)".format(old_dims)

    # it may be possible to change the number of dimensions between modes
    new_dims = reset_csr.shape[0]

    min_vars = minkowski_csr.shape[1]

    names = ["reset{}".format(min_var) for min_var in range(min_vars)]
    names += ["m{}_i0_t{}".format(mode_id, transition_index)]

    names += ["m{}_i{}".format(mode_id, d) for d in range(1, new_dims)]
    names += ["m{}_c{}".format(mode_id, d) for d in range(new_dims)]

    if successor_has_inputs:
        names += ["m{}_ti{}".format(mode_id, d) for d in range(new_dims)]
    
    lpi.add_cols(names)

    lpi.add_rows_equal_zero(2*new_dims)

    lpi.add_rows_less_equal(minkowski_constraints_rhs)

    if successor_has_inputs:
        lpi.add_rows_equal_zero(new_dims)

    data = []
    inds = []
    indptrs = [0]

    # new_init_vars = reset_mat * old_cur_vars + minkow_csr * minkow_vars:
    # -I for new mode initial vars, RM for old mode cur_vars, MK for minkow_vars
    for dim in range(new_dims):

        # old cur_vars
        for index in range(reset_csr.indptr[dim], reset_csr.indptr[dim + 1]):
            rm_col = reset_csr.indices[index]
            value = reset_csr.data[index]
            
            data.append(value)
            inds.append(lpi.cur_vars_offset + rm_col)

        # minkow_vars
        if minkowski_csr.shape[1] > 0:
            for index in range(minkowski_csr.indptr[dim], minkowski_csr.indptr[dim + 1]):
                minkowski_col = minkowski_csr.indices[index]
                value = minkowski_csr.data[index]

                data.append(value)
                inds.append(cols + minkowski_col)

        # new mode initial vars
        data.append(-1)
        inds.append(cols + min_vars + dim)

        indptrs.append(len(data))

    # new_cur_vars = BM * new_init_vars: -I for new cur vars, BM (initially identity) for new init vars
    for dim in range(new_dims):
        data.append(1)
        inds.append(cols + min_vars + dim)

        data.append(-1)
        inds.append(cols + min_vars + new_dims + dim)

        if successor_has_inputs:
            data.append(1)
            inds.append(cols + min_vars + 2*new_dims + dim)

        indptrs.append(len(data))

    # encode minkowski constraint rows
    for row in range(minkowski_constraints_csr.shape[0]):
        for index in range(minkowski_constraints_csr.indptr[row], minkowski_constraints_csr.indptr[row + 1]):
            col = minkowski_constraints_csr.indices[index]
            value = minkowski_constraints_csr.data[index]
            
            data.append(value)
            inds.append(cols + col)

        indptrs.append(len(data))

    # encode total input effects
    if successor_has_inputs:
        for dim in range(new_dims):
            data.append(-1)
            inds.append(cols + min_vars + 2*new_dims + dim)

            indptrs.append(len(data))

    height = 2*new_dims + len(minkowski_constraints_rhs)
    width = cols + 2*new_dims + min_vars

    if successor_has_inputs:
        height += new_dims
        width += new_dims

    mat = csr_matrix((data, inds, indptrs), dtype=float, \
                     shape=(height, width))
    mat.check_format()

    lpi.set_constraints_csr(mat, offset=(rows, 0))

    # input effects position
    if successor_has_inputs:
        ie_x = cols + min_vars + 2*new_dims
        ie_y = rows + 2*new_dims + len(minkowski_constraints_rhs)
        ie_offsets = (ie_y, ie_x)
    else:
        ie_offsets = None

    basis_mat_pos = (rows+new_dims, cols + minkowski_csr.shape[1])
    cur_vars_offset = cols + minkowski_csr.shape[1] + new_dims
    
    lpi.set_reach_vars(new_dims, basis_mat_pos, cur_vars_offset, ie_offsets)

def add_curtime_constraints(lpi, csr, rhs_vec):
    '''
    add constraints to the lpi

    this adds them on the current time variables (not the initial time variables)
    '''

    assert isinstance(csr, csr_matrix)

    prerows = lpi.get_num_rows()
    lpi.add_rows_less_equal(rhs_vec)

    lpi.set_constraints_csr(csr, offset=(prerows, lpi.cur_vars_offset))

def get_box_center(lpi):
    '''get the center of the box overapproximation of the passed-in lpi

    may return None if lp solving fails (numerical issues)
    '''

    Timers.tic('get_box_center')

    dims = lpi.dims
    pt = []

    for dim in range(dims):
        col = lpi.cur_vars_offset + dim
        min_dir = [1 if i == dim else 0 for i in range(dims)]
        max_dir = [-1 if i == dim else 0 for i in range(dims)]
        
        min_val = lpi.minimize(direction_vec=min_dir, columns=[col], fail_on_unsat=False)
        max_val = lpi.minimize(direction_vec=max_dir, columns=[col], fail_on_unsat=False)

        if min_val is None or max_val is None:
            pt = None
            break
        
        min_val = min_val[0]
        max_val = max_val[0]

        pt.append((min_val + max_val) / 2.0)

    Timers.toc('get_box_center')

    return pt

def make_direction_matrix(point, a_csr):
    '''make the direction matrix for arnoldi aggregation

    this is a set of full rank, linearly-independent vectors, extracted from the dynamics using something
    similar to the arnoldi iteration

    the null-space vectors first try to be filled with the orthonormal directions, and then by random vectors

    point is the point where to sample the dynamics
    a_csr is the dynamics matrix
    '''

    Timers.tic('make_direction_matrix')

    assert isinstance(a_csr, csr_matrix)
    cur_vec = np.array(point, dtype=float)
    
    assert len(point) == a_csr.shape[1], "expected point dims({}) to equal A-matrix dims({})".format( \
                len(point), a_csr.shape[1])
    
    dims = len(point)
    rv = []

    while len(rv) < dims:
        if cur_vec is None: # inside the null space
            # first try to pick orthonormal directions if we can
            for d in range(dims):
                found_nonzero = False

                for vec in rv:
                    if vec[d] != 0:
                        found_nonzero = True
                        break

                if found_nonzero is False:
                    cur_vec = np.array([1 if n == d else 0 for n in range(dims)], dtype=float)
                    break

            # if that didn't work, just a random vector
            if cur_vec is None:
                cur_vec = np.random.rand(dims,)
        else:
            cur_vec = a_csr * cur_vec

        # project out the previous vectors
        for prev_vec in rv:
            dot_val = np.dot(prev_vec, cur_vec)

            cur_vec -= prev_vec * dot_val

        norm = np.linalg.norm(cur_vec, 2)

        assert not math.isinf(norm) and not math.isnan(norm), "vector norm was infinite in arnoldi"

        if norm < 1e-6:
            # super small norm... basically it's in the subspace spanned by previous vectors, restart
            cur_vec = None
        else:
            cur_vec = cur_vec / norm

            rv.append(cur_vec)

    Timers.toc('make_direction_matrix')

    return np.array(rv, dtype=float)

def reorthogonalize_matrix(mat, dims):
    '''given an input matrix (one 'dims'-dimensional vector per row), return a new matrix such that the vectors are in 
    the same order, but orthonormal (project out earlier vectors and scale), with the passed-in number of dimensions 
    (a smaller matrix may be returned, or new vectors may be generated to fill the nullspace if the dims > dim(mat)'''

    if isinstance(mat, list):
        mat = np.array(mat, dtype=float)

    assert mat.shape[1] == dims, "mat should have width equal to dims({})".format(dims)

    # take approach similar to arnoldi, except without the matrix-vector multiplication (see make_direction_matrix)

    Timers.tic('reorthogonalize_matrix')

    rv = []

    next_index = 0

    while len(rv) < dims:
        if next_index >= mat.shape[0]:
            # first try to pick orthonormal directions if we can
            for d in range(dims):
                found_nonzero = False

                for vec in rv:
                    if vec[d] != 0:
                        found_nonzero = True
                        break

                if found_nonzero is False:
                    cur_vec = np.array([1 if n == d else 0 for n in range(dims)], dtype=float)
                    break

            # if that didn't work, just a random vector
            if cur_vec is None:
                cur_vec = np.random.rand(dims,)
        else:
            cur_vec = mat[next_index]
            next_index += 1

        # project out the previous vectors
        for prev_vec in rv:
            dot_val = np.dot(prev_vec, cur_vec)

            cur_vec -= prev_vec * dot_val

        norm = np.linalg.norm(cur_vec, 2)

        assert not math.isinf(norm) and not math.isnan(norm), "vector norm was infinite in arnoldi"

        if norm < 1e-6:
            # super small norm... basically it's in the subspace spanned by previous vectors, restart
            cur_vec = None
        else:
            cur_vec = cur_vec / norm

            rv.append(cur_vec)

    Timers.toc('reorthogonalize_matrix')

    return np.array(rv, dtype=float)

def is_feasible(csr, rhs):
    'are the passed in constraints feasible?'

    if not isinstance(csr, csr_matrix):
        csr = csr_matrix(csr, dtype=float)

    assert len(rhs) == csr.shape[0], "constraints RHS differs from number of rows"

    lpi = LpInstance()
    names = ["x{}".format(n) for n in range(csr.shape[1])]
    lpi.add_cols(names)
    lpi.add_rows_less_equal(rhs)

    lpi.set_constraints_csr(csr)

    lpi.cur_vars_offset = 0
    lpi.dims = 0
    lpi.basis_mat_pos = (0, 0)

    return lpi.is_feasible()

def is_point_in_lpi(point, orig_lpi):
    '''is the passed-in point in the lpi?

    This function is strictly for unit testing as it's slow (copies the lpi). 
    A warning is printed to stdout to reflect this and discourage usage in other places.
    '''

    print("Warning: Using testing function lputil.is_point_in_lpi (slow)")

    assert len(point) <= orig_lpi.dims

    inds = []
    data = []
    indptr = [0]
    rhs = []

    for i, x in enumerate(point):
        inds.append(i)
        data.append(1)
        indptr.append(len(data))
        rhs.append(x)

        inds.append(i)
        data.append(-1)
        indptr.append(len(data))
        rhs.append(-x)

    rows = len(indptr) - 1
    cols = len(point)
    csr = csr_matrix((data, inds, indptr), dtype=float, shape=(rows, cols))

    lpi = orig_lpi.clone()
    add_curtime_constraints(lpi, csr, rhs)

    return lpi.is_feasible()

def compute_radius_inf(lpi):
    '''
    compute the max ||x||, x in lpi, according to the infinity norm

    This uses 2*n LPs and then returns the maximum over all components
    '''

    max_val = -float('inf')

    for n in range(lpi.dims):
        for posneg in [1.0, -1.0]:
            dir_vec = [posneg if n == d else 0.0 for d in range(lpi.dims)]

            lp_result = lpi.minimize(dir_vec)

            max_val = max(max_val, abs(lp_result[n]))

    return max_val

def minkowski_sum(lpi_list, mode):
    '''
    perform a minkowski sum of the passed-in sets, and return the resultant lpi
    '''

    for lpi in lpi_list:
        assert lpi.dims == lpi_list[0].dims, "dimension mismatch during minkowski sum"

    dims = lpi_list[0].dims

    csr_list = []
    combined_rhs = [0] * dims
    combined_types = [glpk.GLP_FX] * dims
    combined_names = [f"c{n}" for n in range(dims)]

    total_new_vars = dims

    for i, lpi in enumerate(lpi_list):
        csr = lpi.get_full_constraints()
        csr_list.append(csr)
        combined_rhs += [v for v in lpi.get_rhs()]
        combined_types += lpi.get_types()

        total_new_vars += csr.shape[1]
        combined_names += [f"l{i}_{v}" for v in range(csr.shape[1])]

    # create combined_csr constraints
    data = []
    indices = []
    indptr = [0]

    for d in range(dims):
        data.append(1)
        indices.append(d)
        col_offset = dims

        for lpi in lpi_list:
            data.append(-1)
            indices.append(col_offset + lpi.cur_vars_offset + d)

            col_offset += lpi.get_num_cols()

        indptr.append(len(data))

    # copy constraints from each lpi
    col_offset = dims
    indptr_offset = indptr[-1]
    
    for csr in csr_list:
        data += [d for d in csr.data]
        indices += [col_offset + i for i in csr.indices]
        indptr += [indptr_offset + i for i in csr.indptr[1:]]

        col_offset += csr.shape[1]
        indptr_offset = indptr[-1]

    rows = len(combined_rhs)
    cols = col_offset
    combined_csr = csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=float)

    # from_constraints assumes left-most variables are current-time variables
    return from_constraints(combined_csr, combined_rhs, mode, types=combined_types, names=combined_names, dims=dims)

def from_input_constraints(b_mat, u_constraints, u_rhs, mode):
    'create an lpi from input constraints (B matrix and constraints on U)'

    if not isinstance(b_mat, csr_matrix):
        b_mat = csr_matrix(b_mat, dtype=float)

    if not isinstance(u_constraints, csr_matrix):
        u_constraints = csr_matrix(u_constraints, dtype=float)

    if not isinstance(u_rhs, np.ndarray):
        u_rhs = np.array(u_rhs, dtype=float)

    # V = B U
    dims = b_mat.shape[0]

    combined_rhs = [0] * dims
    combined_types = [glpk.GLP_FX] * dims
    combined_names = [f"c{n}" for n in range(dims)]

    combined_rhs += [v for v in u_rhs]
    combined_types += [glpk.GLP_UP] * u_constraints.shape[0]
    combined_names += [f"u{n}" for n in range(b_mat.shape[1])]

    # create combined_csr constraints
    data = []
    indices = []
    indptr = [0]

    for d in range(dims):
        data.append(-1)
        indices.append(d)

        data += [d for d in b_mat.data[b_mat.indptr[d]:b_mat.indptr[d+1]]]
        indices += [dims + i for i in b_mat.indices[b_mat.indptr[d]:b_mat.indptr[d+1]]]

        indptr.append(len(data))

    indptr_offset = indptr[-1]
    
    data += [d for d in u_constraints.data]
    indices += [dims + i for i in u_constraints.indices]
    indptr += [indptr_offset + i for i in u_constraints.indptr[1:]]

    rows = dims + u_constraints.shape[0]
    cols = dims + b_mat.shape[1]
    combined_csr = csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=float)

    # from_constraints assumes left-most variables are current-time variables
    return from_constraints(combined_csr, combined_rhs, mode, types=combined_types, names=combined_names, dims=dims)
