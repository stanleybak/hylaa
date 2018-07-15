'''
LP Utilities

This assumes a common LP structure, where the
first N columns correspond to the current-time variables, and
the first N rows are the current-time constraints (equality constraints equal to zero)
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.glpk.python_sparse_glpk import LpInstance

def from_box(box_list):
    'make a new lp instance from a passed-in box'

    lpi = LpInstance()

    dims = len(box_list)

    lpi.add_rows_equal_zero(dims)
    lpi.add_cols(2*dims)

    rhs = []

    for lb, ub in box_list:
        rhs.append(-lb)
        rhs.append(ub)

    lpi.add_rows_less_equal(rhs)

    # make constraints as csr_matrix
    data = []
    inds = []
    indptr = [0]

    # -I I for first n rows
    for n in range(dims):
        data.append(-1)
        inds.append(n)
        
        data.append(1)
        inds.append(dims + n)

        indptr.append(len(data))

    # -1 <= -lb
    # 1 <= ub

    for n in range(dims):
        data.append(-1)
        inds.append(dims + n)
        indptr.append(len(data))

        data.append(1)
        inds.append(dims + n)
        indptr.append(len(data))

    mat = csr_matrix((data, inds, indptr), shape=(dims + 2*dims, 2*dims), dtype=float)
    mat.check_format()

    lpi.set_constraints_csr(mat)

    return lpi

def set_basis_matrix(lpi, basis_mat):
    'modify the lpi in place to set the basis matrix'

    dims = basis_mat.shape[0]
    assert dims == basis_mat.shape[1], "expected square matrix"

    # do it row by row, assume -I is first part, and last N is basis matrix

    # make constraints as csr_matrix
    data = []
    inds = []
    indptr = [0]

    num_cols = lpi.get_num_cols()

    # -I I for first n rows
    for n in range(dims):
        data.append(-1)
        inds.append(n)

        for col in range(dims):
            data.append(basis_mat[n, col])
            inds.append(num_cols - dims + col)

        indptr.append(len(data))

    mat = csr_matrix((data, inds, indptr), shape=(dims, num_cols), dtype=float)
    mat.check_format()
    
    lpi.set_constraints_csr(mat)

def check_intersection(lpi, vec, rhs):
    '''check if there is an intersection between the LP constriants and vec <= rhs

    This solves an LP optimizing in the given direction... without adding the constraint to the LP
    '''

    lpi.set_minimize_direction(vec)

    columns = np.array([i for i in range(len(vec))], dtype=int) # get the first len(vec) columns
    result = lpi.minimize_partial_result(columns, fail_on_unsat=True)

    return np.dot(result, vec) <= rhs

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

    cols = lpi.get_num_cols()
    rows = lpi.get_num_rows()

    indptr = [0, dims]
    inds = [i for i in range(dims)]
    data = new_vec
    data.shape = (dims,)

    csr_row_mat = csr_matrix((data, inds, indptr), dtype=float, shape=(1, dims))
    csr_row_mat.check_format()

    lpi.set_constraints_csr(csr_row_mat, offset=(rows-1, cols-dims))

    return rows - 1

def try_replace_constraint(lpi, old_row_index, direction, rhs):
    '''replace the constraint in row_index by a new constraint, if the new constraint is stronger, otherwise
    create new constriant

    this is used for removing redundant invariant constraints

    This returns row_index, if the constriant is replaced, or the new row index of the new constraint
    '''

    # how can we check if the passed in constraint is stronger than the existing one?
    # if negating the existing constraint, and adding the new one is UNSAT

    rv = None
    basis_mat = get_basis_matrix(lpi)

    lpi.flip_constraint(old_row_index)

    new_row_index = add_init_constraint(lpi, direction, rhs, basis_matrix=basis_mat)

    is_sat = lpi.minimize_partial_result([], fail_on_unsat=False) is not None

    lpi.flip_constraint(old_row_index) # flip it back

    if is_sat:
        # keep both constraints
        rv = new_row_index
    else:
        # keep only the new constraint
        
        # delete new constraint row
        lpi.del_constraint(new_row_index)

        # replace the old constraint row with the new constraint condition
        cols = lpi.get_num_cols()
        dims = basis_mat.shape[0]

        indptr = [0, dims]
        inds = [i for i in range(dims)]
        new_vec = np.dot(direction, basis_mat) # convert the constraint using the basis matrix
        data = new_vec
        data.shape = (dims,)

        csr_row_mat = csr_matrix((data, inds, indptr), dtype=float, shape=(1, dims))
        csr_row_mat.check_format()

        lpi.set_constraints_csr(csr_row_mat, offset=(old_row_index, cols-dims))
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
    dims = num_directions = direction_matrix.shape[0]
    columns = np.array([i for i in range(num_directions)], dtype=int) # get the first len(vec) columns
    
    mins = [np.inf] * num_directions
    mid_mins = [np.inf] * num_directions
    maxes = [-np.inf] * num_directions
    mid_maxes = [-np.inf] * num_directions

    for i in range(num_directions):
        direction = direction_matrix[i]
        assert abs(np.linalg.norm(direction) - 1) < 1e-9, "expected normalized directions, got {}".format(direction)
        
        for lpi in lpi_list:
            lpi.set_minimize_direction(direction)
            result = lpi.minimize_partial_result(columns)
            min_val = np.dot(result, direction)
            mins[i] = min(mins[i], min_val)

            lpi.set_minimize_direction(-direction)
            result = lpi.minimize_partial_result(columns)
            max_val = np.dot(-result, -direction)
            maxes[i] = max(maxes[i], max_val)

            if lpi == middle_lpi:
                mid_mins[i] = min_val
                mid_maxes[i] = max_val

    rows = middle_lpi.get_num_rows()
    cols = middle_lpi.get_num_cols()

    rv = LpInstance(middle_lpi) # copy it

    # add n new columns and 2n new rows, for the minkowski sum constriants
    rv.add_cols(dims)

    # csc matrix with constriants
    data = []
    inds = []
    indptrs = [0]
    rhs = []

    for dim in range(dims):
        direction = direction_matrix[dim]
        
        # column is direction[dim]
        for i in range(len(direction)):
            data.append(direction[i])
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

def get_dims(lpi):
    '''get the number of dimensions of the lpi by counting '==' constriants'''

    # first figure out the dimensions of the basis matrix by counting the number of '==' constraints
    types = lpi.get_types()

    dims = 0

    for t in types:
        if t != LpInstance.GLP_FX:
            break

        dims += 1

    return dims

def get_basis_matrix(lpi):
    'get the basis matrix from the lpi'

    dims = get_dims(lpi)
    cols = lpi.get_num_cols()

    return lpi.get_subconstraints(cols-dims, 0, dims, dims)

def add_snapshot_variables(lpi):
    '''
    add snapshot variables to the existing lpi

    this adds n new variables (the pre-snapshot variables), which is assigned with new rows to
    0 BM -I = 0

    and sets the first rows to a fresh setup:
    -I 0 I = 0
    '''

    dims = get_dims(lpi)
    cols = lpi.get_num_cols()
    rows = lpi.get_num_rows()

    lpi.add_cols(dims)
    lpi.add_rows_equal_zero(dims)
    
    data = []
    inds = []
    indptrs = [0]

    # set constraints for the last <dims> rows
    for dim in range(dims):
        row_csr = lpi.get_row(dim)

        for i in range(len(row_csr.data)):

            if row_csr.indices[i] < dims:
                assert row_csr.data[i] == -1, "expected negative identity in row {}".format(dim)
            else:
                data.append(row_csr.data[i])
                inds.append(row_csr.indices[i])

        # add -I at the end
        data.append(-1)
        inds.append(cols + dim)

        indptrs.append(len(data))

    mat = csr_matrix((data, inds, indptrs), shape=(dims, cols + dims), dtype=float)
    mat.check_format()
    
    lpi.set_constraints_csr(mat, offset=(rows, 0))
    
    # set constraints for the first <dims> rows
    data = []
    inds = []
    indptrs = [0]

    # set constraints for the last <dims> rows
    for dim in range(dims):
        # -I at the start
        data.append(-1)
        inds.append(dim)

        # I at the end (basis matrix)
        data.append(1)
        inds.append(cols + dim)

        indptrs.append(len(data))

    mat = csr_matrix((data, inds, indptrs), shape=(dims, cols + dims), dtype=float)
    mat.check_format()
    
    lpi.set_constraints_csr(mat)
