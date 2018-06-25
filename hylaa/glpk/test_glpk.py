'''
Stanley Bak
Unit tests for python_glpk_sparse
May 2018
'''

import cvxopt
import numpy as np

from scipy.sparse import csr_matrix, csc_matrix

from python_sparse_glpk import LpInstance

def test_cpp():
    'runs the c++ test() function for the glpk interface'

    assert LpInstance.test() == 0

def compare_opt(a_ub, b_ub, direction):
    'compare cvx opt versus our glpk interface (both csc and csr)'

    # make sure we're using floats not ints
    a_ub = [[float(x) for x in row] for row in a_ub]
    b_ub = [float(x) for x in b_ub]
    c = [float(x) for x in direction]

    num_vars = len(a_ub[0])

    # solve it with cvxopt
    options = {'show_progress': False}
    sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(a_ub).T, cvxopt.matrix(b_ub), options=options)

    #if sol['status'] == 'primal infeasible':
    #    res_cvxopt = None

    if sol['status'] != 'optimal':
        raise RuntimeError("cvxopt LP failed: {}".format(sol['status']))

    res_cvxopt = [float(n) for n in sol['x']]

    #print "cvxopt value = {}, result = {}".format(np.dot(res_cvxopt, c), repr(res_cvxopt))

    # solve it with the glpk <-> c++ interface
    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    res_glpk = lp.minimize(direction)

    #print "glpk interface value = {}, result = {}".format(np.dot(res_glpk, c), repr(res_glpk))

    assert num_vars == len(res_cvxopt)
    assert abs(np.dot(res_glpk, c) - np.dot(res_cvxopt, c)) < 1e-6

    # try again with csc constraints
    lp = LpInstance()

    a_csc = csc_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csc(a_csc)

    lp.set_minimize_direction(direction)
    res_glpk = lp.minimize_full_result()

    assert num_vars == len(res_cvxopt)
    assert abs(np.dot(res_glpk, c) - np.dot(res_cvxopt, c)) < 1e-6

def test_simple():
    '''test consistency with cvxopt on a simple problem'''

    # max 0.6x + 0.5y st.
    # x + 2y <= 1
    # 3x + y <= 2

    a_ub = [[1, 2], [3, 1]]
    b_ub = [1, 2]
    c = [-0.6, -0.5]

    compare_opt(a_ub, b_ub, c)

def test_underconstrained():
    'test an underconstrained case (fails for cvxopt)'

    a_ub = [[1.0, 0.0], [-1.0, 0.0]]
    b_ub = [1.0, 1.0]
    direction = [1.0, 0.0]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    lp.set_minimize_direction(direction)
    res_glpk = lp.minimize_full_result()

    assert abs(res_glpk[0] - (-1) < 1e-6)

def test_tricky():
    '''test consistency with cvxopt on a tricky problem (scipy.linprog fails)'''

    a_ub = [[-1.0, 0.0, 0.0, -2.1954134149515525e-08, 1.0000000097476742, 0.0],
            [1.0, -0.0, -0.0, 2.1954134149515525e-08, -1.0000000097476742, -0.0],
            [0.0, -1.0, 0.0, -1.000000006962809, 2.5063524589086228e-08, 0.0],
            [-0.0, 1.0, -0.0, 1.000000006962809, -2.5063524589086228e-08, -0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 1.0000000000000009],
            [-0.0, -0.0, 1.0, -0.0, -0.0, -1.0000000000000009],
            [0., 0., 0., 1.0, 0.0, 0.0],
            [0., 0., 0., -1.0, 0.0, 0.0],
            [0., 0., 0., 0.0, 1.0, 0.0],
            [0., 0., 0., 0.0, -1.0, 0.0],
            [0., 0., 0., 0.0, 0.0, 1.0],
            [0., 0., 0., 0.0, 0.0, -1.0]]

    b_ub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, -0.0]

    num_vars = len(a_ub[0])
    c = [1.0 if i % 2 == 0 else 0.0 for i in range(num_vars)]

    compare_opt(a_ub, b_ub, c)

def test_add_constraints():
    'test incrementally adding constraints'

    a1_mat = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    a1_vec = np.array([-10, 10.2, 0, 0])

    a1_csr = csr_matrix(np.array(a1_mat, dtype=float))

    lp = LpInstance()
    lp.add_cols(a1_csr.shape[1])
    lp.add_rows_less_equal(a1_vec)
    lp.set_constraints_csr(a1_csr)

    direction = [1, 0]

    a2_mat = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    a2_vec = np.array([0, 0, 9.81, -9.81], dtype=float)

    zero_mat = np.zeros((4, 2))
    a_csc = csc_matrix(np.vstack((zero_mat, a2_mat)))

    lp.add_cols(a_csc.shape[1])
    lp.add_rows_less_equal(a2_vec)

    lp.set_constraints_csc(a_csc, offset=(0, 2))
    direction = [1, 0, 0, 0]

    lp.set_minimize_direction(direction)
    res = lp.minimize_full_result()

    assert res is not None

def test_get_iterations():
    '''tests get_iterations (requires newer version of GLPK, otherwise OSError occurs, see README)'''

    # max 0.6x + 0.5y st.
    # x + 2y <= 1
    # 3x + y <= 2

    a_ub = [[1, 2], [3, 1]]
    b_ub = [1, 2]
    c = [-0.6, -0.5]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    assert lp.get_num_rows() == len(b_ub)
    assert lp.get_num_cols() == a_csr.shape[1]

    lp.set_constraints_csr(a_csr)

    assert lp.get_iterations() == 0
    lp.set_minimize_direction(c)
    lp.minimize_full_result()
    assert lp.get_iterations() > 0

def test_partial_result():
    'tests minimize with partial result'

    # max 0.6x + 0.5y st.
    # x + 2y <= 1
    # 3x + y <= 2

    a_ub = [[1, 2], [3, 1]]
    b_ub = [1, 2]
    c = [-0.6, -0.5]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    assert lp.get_num_rows() == len(b_ub)
    assert lp.get_num_cols() == a_csr.shape[1]

    lp.set_constraints_csr(a_csr)

    lp.set_minimize_direction(c)
    full_result = lp.minimize_full_result()

    col1 = lp.minimize_partial_result([0])
    col2 = lp.minimize_partial_result([1])

    assert full_result[0] == col1[0]
    assert full_result[1] == col2[0]

    cols12 = lp.minimize_partial_result([0, 1])

    assert full_result[0] == cols12[0]
    assert full_result[1] == cols12[1]

    cols21 = lp.minimize_partial_result([1, 0])

    assert full_result[0] == cols21[1]
    assert full_result[1] == cols21[0]

def test_get_matrix():
    '''tests get_matrix testing function'''

    # x + 2y <= 1
    # 3x + y <= 2

    a_ub = [[1, 2], [3, 1]]
    b_ub = [1, 2]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    mat, vec = lp.get_matrix()

    expected_mat = np.array([[1, 2], [3, 1]], dtype=float)
    expected_vec = np.array([1, 2.0], dtype=float)
    
    assert np.allclose(mat, expected_mat)

    assert np.allclose(vec, expected_vec)

def test_min_partial_result():
    'test minimize partial result'

    # x + 2y <= 1
    # x + 2y >= -2

    a_ub = [[1, 2], [-1, -2]]
    b_ub = [1, 2]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    result = lp.minimize_partial_result([])

    assert result is not None and len(result) == 0

def test_unsat():
    'test unsat case returning None'

    # x + 2y <= 1
    # x + 2y >= 2

    a_ub = [[1, 2], [-1, -2]]
    b_ub = [1, -2]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    result = lp.minimize_partial_result([], fail_on_unsat=False)

    assert result is None

def test_bad_col():
    'test minimize getting a non-integer column'

    # x + 2y <= 1
    # x + 2y >= 2

    a_ub = [[1, 2], [-1, -2]]
    b_ub = [1, -2]

    lp = LpInstance()

    a_csr = csr_matrix(np.array(a_ub, dtype=float))
    lp.add_cols(a_csr.shape[1])
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(a_csr)

    try:
        lp.minimize_partial_result([1.5])
        assert False, "expected exception to be raised"
    except RuntimeError:
        pass

def test_csr_with_coloffset():
    '''tests set csr constraints with column offset'''

    # 0 + 2y <= 1
    # 0 + y <= 2

    b_ub = [1, 2]

    lp = LpInstance()

    col2_csr = csr_matrix(np.array([[2], [1]], dtype=float))
    
    lp.add_cols(2)
    lp.add_rows_less_equal(b_ub)

    lp.set_constraints_csr(col2_csr, offset=(0,1))

    mat, vec = lp.get_matrix()

    expected_mat = np.array([[0, 2], [0, 1]], dtype=float)
    expected_vec = np.array([1, 2.0], dtype=float)
    
    assert np.allclose(mat, expected_mat)

    assert np.allclose(vec, expected_vec)

def test_flip_constraint():
    '''test changing constraint direction from <= to >='''

    # 0 <= x <= 1
    # extra constraint x <= 0.5... changed to x >= 0.5 and then back

    lp = LpInstance()

    lp.add_cols(1)
    lp.add_rows_less_equal([1, 0])
    lp.set_constraints_csr(csr_matrix(np.array([[1], [-1]], dtype=float)))

    # add x <= 0.5 constraint
    lp.add_rows_less_equal([0.5])
    lp.set_constraints_csr(csr_matrix(np.array([[1]], dtype=float)), offset=(2, 0))

    max_x = lp.minimize([-1])[0]
    assert max_x == 0.5

    min_x = lp.minimize([1])[0]
    assert min_x == 0.0

    # flip the constraint
    is_lesser = lp.flip_constraint(2)

    assert not is_lesser, "constraint should now be a '>=' constraint"

    max_x = lp.minimize([-1])[0]
    assert max_x == 1.0

    min_x = lp.minimize([1])[0]
    assert min_x == 0.5

def test_pop_row():
    '''test removing a row'''

    # 0 <= x <= 1

    lp = LpInstance()

    lp.add_cols(1)
    lp.add_rows_less_equal([1, 0])
    lp.set_constraints_csr(csr_matrix(np.array([[1], [-1]], dtype=float)))

    # add x <= 0.5 constraint
    lp.add_rows_less_equal([0.5])
    lp.set_constraints_csr(csr_matrix(np.array([[1]], dtype=float)), offset=(2, 0))

    max_x = lp.minimize([-1])[0]
    assert max_x == 0.5

    min_x = lp.minimize([1])[0]
    assert min_x == 0.0

    assert lp.get_num_rows() == 3

    # remove the constraint
    lp.del_constraint(2)

    assert lp.get_num_rows() == 2

    max_x = lp.minimize([-1])[0]
    assert max_x == 1.0

    min_x = lp.minimize([1])[0]
    assert min_x == 0.0

def test_set_constraint_rhs():
    '''test changing rhs of constraint'''

    # 0 <= x <= 1
    # extra constraint x <= 0.5... changed to x <= 0.9

    lp = LpInstance()

    lp.add_cols(1)
    lp.add_rows_less_equal([1, 0])
    lp.set_constraints_csr(csr_matrix(np.array([[1], [-1]], dtype=float)))

    # add x <= 0.5 constraint
    lp.add_rows_less_equal([0.5])
    lp.set_constraints_csr(csr_matrix(np.array([[1]], dtype=float)), offset=(2, 0))

    max_x = lp.minimize([-1])[0]
    assert max_x == 0.5

    min_x = lp.minimize([1])[0]
    assert min_x == 0.0

    assert lp.get_num_rows() == 3

    # change constraint to <= 0.9
    lp.set_constraint_rhs(2, 0.9)

    assert lp.get_num_rows() == 3

    max_x = lp.minimize([-1])[0]
    assert max_x == 0.9

    min_x = lp.minimize([1])[0]
    assert min_x == 0.0
