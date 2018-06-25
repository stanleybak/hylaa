'''
LP Utilities

This assumes a common LP structure, where the
first N columns correspond to the current-time variables, and
the first N rows are the current-time constraints (equality constraints equal to zero)
'''

import numpy as np
from scipy.sparse import csr_matrix

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

def add_constraint(lpi, basis_matrix, vec, rhs):
    '''
    add a constraint to the lpi

    this adds a new row, with constraints assigned to the right-most variables (where the basis matrix is)

    this returns the row of the newly-created constraint
    '''

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

def try_replace_constraint(lpi, old_row_index, basis_mat, direction, rhs):
    '''replace the constraint in row_index by a new constraint, if the new constraint is stronger, otherwise
    create new constriant

    this is used for removing redundant invariant constraints

    This returns row_index, if the constriant is replaced, or the new row index of the new constraint
    '''

    # how can we check if the passed in constraint is stronger than the existing one?
    # if negating the existing constraint, and adding the new one is UNSAT

    rv = None

    lpi.flip_constraint(old_row_index)

    new_row_index = add_constraint(lpi, basis_mat, direction, rhs)

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
