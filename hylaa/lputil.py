'''
LP Utilities

This assumes a common LP structure, where the
first N columns correspond to the current-time variables, and
the first N rows are the current-time constraints (equality constraints equal to zero)
'''

from hylaa.glpk.python_sparse_glpk import LpInstance

from scipy.sparse import csr_matrix

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
