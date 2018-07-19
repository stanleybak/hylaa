'''
LP Utilities

This assumes a common LP structure, where the
first N columns correspond to the current-time variables, and
the first N rows are the current-time constraints (equality constraints equal to zero)
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.lpinstance import LpInstance

def from_box(box_list, mode):
    'make a new lp instance from a passed-in box'

    lpi = LpInstance()

    dims = len(box_list)

    lpi.add_rows_equal_zero(dims)

    names = ["m{}_i{}".format(mode.mode_id, var_index) for var_index in range(dims)]
    names += ["m{}_c{}".format(mode.mode_id, var_index) for var_index in range(dims)]
    
    lpi.add_cols(names)

    rhs = []

    for lb, ub in box_list:
        assert lb <= ub, "lower bound ({}) > upper bound ({})".format(lb, ub)
        rhs.append(-lb)
        rhs.append(ub)

    lpi.add_rows_less_equal(rhs)

    # make constraints as csr_matrix
    data = []
    inds = []
    indptr = [0]

    # I -I for first n rows
    for n in range(dims):
        data.append(1)
        inds.append(n)

        data.append(-1)
        inds.append(dims + n)

        indptr.append(len(data))

    # -1 <= -lb
    # 1 <= ub

    for n in range(dims):
        data.append(-1)
        inds.append(n)
        indptr.append(len(data))

        data.append(1)
        inds.append(n)
        indptr.append(len(data))

    mat = csr_matrix((data, inds, indptr), shape=(dims + 2*dims, 2*dims), dtype=float)
    mat.check_format()

    lpi.set_constraints_csr(mat)

    lpi.set_reach_vars(dims, (0, 0))

    return lpi

def set_basis_matrix(lpi, basis_mat):
    'modify the lpi in place to set the basis matrix'

    assert basis_mat.shape[0] == basis_mat.shape[1], "expected square matrix"
    assert basis_mat.shape[0] == lpi.dims, "basis matrix wrong shape"

    # do it row by row, assume -I is first part, and last N is basis matrix

    # make constraints as csr_matrix
    data = []
    inds = []
    indptr = [0]

    # BM -I
    for row in range(lpi.dims):
        for col in range(lpi.dims):
            data.append(basis_mat[row, col])
            inds.append(col)
            
        data.append(-1)
        inds.append(lpi.dims + row)

        indptr.append(len(data))

    mat = csr_matrix((data, inds, indptr), shape=(lpi.dims, 2 * lpi.dims), dtype=float)
    mat.check_format()
    lpi.set_constraints_csr(mat, offset=lpi.basis_mat_pos)

def check_intersection(lpi, lc, tol=1e-13):
    '''check if there is an intersection between the LP constriants and the LinearConstraint object lc

    This solves an LP optimizing in the given direction... without adding the constraint to the LP
    '''

    lpi.set_minimize_direction(lc.csr, is_csr=True)

    columns = lc.csr.indices[0:lc.csr.indptr[1]]
    lp_columns = [lpi.cur_vars_offset + c for c in columns]

    lp_res = lpi.minimize(columns=lp_columns)

    lp_res_csr = csr_matrix((lp_res, columns, [0, len(columns)]), dtype=float, shape=(1, lc.csr.shape[1]))
    lp_res_csr.check_format()

    dot_res = (lp_res_csr * lc.csr.T)[0, 0]

    return dot_res + tol <= lc.rhs

def add_init_constraint(lpi, vec, rhs, basis_matrix=None):
    '''
    add a constraint to the lpi

    this adds a new row, with constraints assigned to the right-most variables (where the basis matrix is)

    this returns the row of the newly-created constraint
    '''

    if basis_matrix is None:
        basis_matrix = get_basis_matrix(lpi)

    assert isinstance(basis_matrix, np.ndarray)

    # we need to convert the passed-in vector using the basis matrix
    preshape = vec.shape
    dims = basis_matrix.shape[0]
    vec.shape = (1, dims)
    new_vec = np.dot(vec, basis_matrix)
    vec.shape = preshape

    lpi.add_rows_less_equal([rhs])

    rows = lpi.get_num_rows()

    indptr = [0, dims]
    inds = [i for i in range(dims)]
    data = new_vec
    data.shape = (dims,)

    csr_row_mat = csr_matrix((data, inds, indptr), dtype=float, shape=(1, dims))
    csr_row_mat.check_format()

    lpi.set_constraints_csr(csr_row_mat, offset=(rows-1, lpi.basis_mat_pos[1]))

    return rows - 1

def try_replace_init_constraint(lpi, old_row_index, direction, rhs, basis_mat=None):
    '''replace the constraint in row_index by a new constraint, if the new constraint is stronger, otherwise
    create new constriant

    this is used for removing redundant invariant constraints

    This returns row_index, if the constriant is replaced, or the new row index of the new constraint
    '''

    if basis_mat is None:
        basis_mat = get_basis_matrix(lpi)

    # how can we check if the passed in constraint is stronger than the existing one?
    # if negating the existing constraint, and adding the new one is UNSAT
    rv = None

    lpi.flip_constraint(old_row_index)

    new_row_index = add_init_constraint(lpi, direction, rhs, basis_matrix=basis_mat)

    is_sat = lpi.minimize(columns=[], fail_on_unsat=False) is not None

    lpi.flip_constraint(old_row_index) # flip it back

    if is_sat:
        # keep both constraints
        rv = new_row_index
    else:
        # keep only the new constraint
        
        # delete new constraint row
        lpi.del_constraint(new_row_index)

        # replace the old constraint row with the new constraint condition
        dims = basis_mat.shape[0]

        indptr = [0, dims]
        inds = [i for i in range(dims)]
        new_vec = np.dot(direction, basis_mat) # convert the constraint using the basis matrix
        data = new_vec
        data.shape = (dims,)

        csr_row_mat = csr_matrix((data, inds, indptr), dtype=float, shape=(1, dims))
        csr_row_mat.check_format()

        lpi.set_constraints_csr(csr_row_mat, offset=(old_row_index, lpi.basis_mat_pos[1]))
        lpi.set_constraint_rhs(old_row_index, rhs)

        rv = old_row_index

    return rv

def aggregate(lpi_list, direction_matrix):
    '''
    return a new lpi consisting of an aggregation of the passed-in list

    This uses minkowski sum with template directions.
    '''

    assert direction_matrix.shape[0] == direction_matrix.shape[1], "expected square direction matrix"
    assert len(lpi_list) > 1, "expected more than one lpi to perform an aggregation"

    middle_index = len(lpi_list) // 2
    middle_lpi = lpi_list[middle_index]
    
    # for each direction, minimize and maximize it within the list
    num_directions = direction_matrix.shape[0]
     
    mins = [np.inf] * num_directions
    mid_mins = [np.inf] * num_directions
    maxes = [-np.inf] * num_directions
    mid_maxes = [-np.inf] * num_directions

    for i in range(num_directions):
        direction = direction_matrix[i]
        assert abs(np.linalg.norm(direction) - 1) < 1e-9, "expected normalized directions, got {}".format(direction)

        for lpi in lpi_list:
            assert direction_matrix.shape[0] == lpi.dims
   
            columns = np.array([lpi.cur_vars_offset + i for i in range(lpi.dims)], dtype=int)
            
            result = lpi.minimize(direction_vec=direction, columns=columns)
            min_val = np.dot(result, direction)
            mins[i] = min(mins[i], min_val)

            result = lpi.minimize(direction_vec=-direction, columns=columns)
            max_val = np.dot(-result, -direction)
            maxes[i] = max(maxes[i], max_val)

            if lpi == middle_lpi:
                mid_mins[i] = min_val
                mid_maxes[i] = max_val

    rows = middle_lpi.get_num_rows()
    cols = middle_lpi.get_num_cols()
    dims = middle_lpi.dims

    rv = middle_lpi.clone()

    # add n new columns and 2n new rows, for the minkowski sum constriants
    names = ["agg{}".format(i) for i in range(dims)]
    rv.add_cols(names)

    # csc matrix with constriants
    data = []
    inds = []
    indptrs = [0]
    rhs = []

    for dim in range(dims):
        direction = direction_matrix[dim]
        
        # column is direction[dim]
        for i, d in enumerate(direction):
            data.append(d)
            inds.append(i)

        data.append(1.0) # <= constraint
        inds.append(rows + 2*dim)
        rhs.append(maxes[dim] - mid_maxes[dim])

        data.append(-1.0) # >= constraint
        inds.append(rows + 2*dim + 1)
        rhs.append(-(mins[dim] - mid_mins[dim]))

        indptrs.append(len(data))

    rv.add_rows_less_equal(rhs)

    constraints = csc_matrix((data, inds, indptrs), dtype=float, shape=(rows + 2*dims, dims))
    constraints.check_format()

    rv.set_constraints_csc(constraints, offset=(0, cols))

    add_snapshot_variables(rv)

    return rv

def get_basis_matrix(lpi):
    'get the basis matrix from the lpi'

    return lpi.get_dense_constraints(lpi.basis_mat_pos[0], lpi.basis_mat_pos[1], lpi.dims, lpi.dims)

def add_snapshot_variables(lpi):
    '''
    add snapshot variables to the existing lpi

    this adds n new variables (the post-snapshot variables), which is assigned with new rows to have:
    I in the columns of the old cur-time variables (this is also the new basis matrix position)
    -I in the new columns
    0 everywhere else
    '''

    dims = lpi.dims
    cols = lpi.get_num_cols()
    rows = lpi.get_num_rows()

    names = ["ss{}".format(d) for d in range(dims)]
    lpi.add_cols(names)
    lpi.add_rows_equal_zero(dims)
    
    data = []
    inds = []
    indptrs = [0]
    
    # set constraints for the first <dims> rows
    data = []
    inds = []
    indptrs = [0]

    # set constraints for the last <dims> rows
    for dim in range(dims):
        # I at the previous cur_time vars (basis matrix)
        data.append(1)
        inds.append(lpi.cur_vars_offset + dim)

        # -I at the end
        data.append(-1)
        inds.append(cols + dim)

        indptrs.append(len(data))

    mat = csr_matrix((data, inds, indptrs), shape=(dims, cols + dims), dtype=float)
    mat.check_format()

    lpi.set_constraints_csr(mat, offset=(rows, 0))
    lpi.set_reach_vars(lpi.dims, (rows-1, lpi.cur_vars_offset))

def add_curtime_constraints(lpi, csr, rhs_vec):
    '''
    add constraints to the lpi

    this adds them on the current time variables (not the initial time variables)
    '''

    assert isinstance(csr, csr_matrix)
    assert isinstance(rhs_vec, np.ndarray)

    prerows = lpi.get_num_rows()
    lpi.add_rows_less_equal(rhs_vec)

    lpi.set_constraints_csr(csr, offset=(prerows, lpi.cur_vars_offset))
