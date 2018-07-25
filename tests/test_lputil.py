'''
Tests for LP operations. Made for use with py.test
'''

import math
import numpy as np

from scipy.linalg import expm
from scipy.sparse import csr_matrix

import swiglpk as glpk

from hylaa import lputil, lpplot
from hylaa.hybrid_automaton import HybridAutomaton, LinearConstraint
from hylaa.timerutil import Timers

def test_from_box():
    'tests from_box'
    
    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    assert lpi.basis_mat_pos == (0, 0)
    assert lpi.dims == 2

    mat = lpi.get_full_constraints()
    types = lpi.get_types()
    rhs = lpi.get_rhs()
    names = lpi.get_names()

    expected_mat = np.array([\
        [1, 0, -1, 0], \
        [0, 1, 0, -1], \
        [-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, -1, 0, 0], \
        [0, 1, 0, 0]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1], dtype=float)

    fx = glpk.GLP_FX
    up = glpk.GLP_UP
    expected_types = [fx, fx, up, up, up, up]

    expected_names = ["m0_i0", "m0_i1", "m0_c0", "m0_c1"]

    assert np.allclose(rhs, expected_vec)
    assert types == expected_types
    assert np.allclose(mat.toarray(), expected_mat)
    assert names == expected_names

def test_print_lp():
    'test printing the lp to stdout'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))
    assert str(lpi) is not None

def test_set_basis_matrix():
    'tests lputil set_basis_matrix on harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    assert np.allclose(lputil.get_basis_matrix(lpi), basis)

    mat, vec = lpi.get_full_constraints(), lpi.get_rhs()

    expected_mat = np.array([\
        [0, 1, -1, 0], \
        [-1, 0, 0, -1], \
        [-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, -1, 0, 0], \
        [0, 1, 0, 0]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1], dtype=float)

    assert np.allclose(vec, expected_vec)

    assert np.allclose(mat.toarray(), expected_mat)

def test_check_intersection():
    'tests check_intersection on the harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    # check if initially y >= 4.5 is possible (should be false)
    direction = np.array([0, -1], dtype=float)
    lc = LinearConstraint(direction, -4.5)

    assert not lputil.check_intersection(lpi, lc)

    # after basis matrix update
    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    # now check if y >= 4.5 is possible (should be true)
    assert lputil.check_intersection(lpi, lc)
    
def test_verts():
    'tests verts'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    plot_vecs = lpplot.make_plot_vecs(4, offset=(math.pi / 4.0))
    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)

    assert len(verts) == 5
    
    assert [-5.0, 0.] in verts
    assert [-5.0, 1.] in verts
    assert [-4.0, 1.] in verts
    assert [-4.0, 0.] in verts
    assert verts[0] == verts[-1]

def test_add_init_constraint():
    'tests add_init_constraint on the harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    # update basis matrix
    basis_mat = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis_mat)

    # minimize y should give 4.0
    miny = lpi.minimize([0, 1], columns=[lpi.cur_vars_offset + 1])[0]
    assert abs(miny - 4.0) < 1e-6

    # add constraint: y >= 4.5
    direction = np.array([0, -1], dtype=float)

    new_row = lputil.add_init_constraint(lpi, direction, -4.5)

    assert new_row == 6, "new constraint should have been added in row index 6"

    # minimize y should give 4.5
    miny = lpi.minimize([0, 1], columns=[lpi.cur_vars_offset + 1])[0]
    assert abs(miny - 4.5) < 1e-6

    # check verts()
    verts = lpplot.get_verts(lpi)

    assert len(verts) == 5
    
    assert [0.0, 5.0] in verts
    assert [1.0, 5.0] in verts
    assert [0.0, 4.5] in verts
    assert [1.0, 4.5] in verts
    assert verts[0] == verts[-1]

def test_try_replace_init_constraint():
    'tests try_replace_init_constraint on the harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    # update basis matrix
    basis_mat = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis_mat)

    # minimize y should give 4.0
    miny = lpi.minimize([0, 1], columns=[lpi.cur_vars_offset + 1])[0]
    assert abs(miny - 4.0) < 1e-6

    # add constraint: y >= 4.5
    direction = np.array([0, -1], dtype=float)

    row_index = lputil.add_init_constraint(lpi, direction, -4.5)

    # minimize y should give 4.5
    miny = lpi.minimize([0, 1], columns=[lpi.cur_vars_offset + 1])[0]
    assert abs(miny - 4.5) < 1e-6

    assert lpi.get_num_rows() == 7

    # try to replace constraint y >= 4.6 (should be stronger than 4.5)
    row_index = lputil.try_replace_init_constraint(lpi, row_index, direction, -4.6)

    assert row_index == 6
    assert lpi.get_num_rows() == 7

    # try to replace constraint x <= 0.9 (should be incomparable)
    xdir = np.array([1, 0], dtype=float)
    row_index = lputil.try_replace_init_constraint(lpi, row_index, xdir, 0.9)

    assert row_index == 7
    assert lpi.get_num_rows() == 8

    # check verts()
    verts = lpplot.get_verts(lpi)

    assert len(verts) == 5
    
    assert [0.0, 5.0] in verts
    assert [0.9, 5.0] in verts
    assert [0.0, 4.6] in verts
    assert [0.9, 4.6] in verts
    assert verts[0] == verts[-1]

def test_box_aggregate2():
    'tests box aggregation'

    lpi1 = lputil.from_box([[0, 1], [0, 1]], HybridAutomaton().new_mode('mode_name'))
    lpi2 = lputil.from_box([[1, 2], [1, 2]], HybridAutomaton().new_mode('mode_name'))

    agg_dirs = np.array([[1, 0], [0, 1]], dtype=float)

    # box aggregation
    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs)

    verts = lpplot.get_verts(lpi)

    assert len(verts) == 5
    
    assert [0., 0.] in verts
    assert [0, 2.] in verts
    assert [2., 0.] in verts
    assert [2., 2.] in verts
    
    assert verts[0] == verts[-1]

def pair_almost_in(pair, pair_list, tol=1e-9):
    'check if a pair is in a pair list (up to small tolerance)'

    rv = False

    for a, b in pair_list:
        if abs(a - pair[0]) < tol and abs(b - pair[1]) < tol:
            rv = True
            break

    return rv

def test_rotated_aggregate():
    'tests rotated aggregation'

    lpi1 = lputil.from_box([[0, 1], [0, 1]], HybridAutomaton().new_mode('mode_name'))
    lpi2 = lputil.from_box([[1, 2], [1, 2]], HybridAutomaton().new_mode('mode_name'))

    sq2 = math.sqrt(2) / 2.0

    agg_dirs = np.array([[sq2, sq2], [sq2, -sq2]], dtype=float)

    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs)

    verts = lpplot.get_verts(lpi)

    assert len(verts) == 7

    assert pair_almost_in([0., 0.], verts)
    assert pair_almost_in([1., 0.], verts)
    assert pair_almost_in([2., 1.], verts)
    assert pair_almost_in([2., 2.], verts)
    assert pair_almost_in([1., 2.], verts)
    assert pair_almost_in([0., 1.], verts)

    assert verts[0] == verts[-1]

def test_get_basis_matrix():
    'tests lputil get_basis_matrix on harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    mat = lputil.get_basis_matrix(lpi)

    assert np.allclose(mat, basis)

def test_box_aggregate3():
    'tests box aggregation with 3 boxes'

    lpi1 = lputil.from_box([[-2, -1], [-0.5, 0.5]], HybridAutomaton().new_mode('mode_name'))
    lpi2 = lpi1.clone()
    lpi3 = lpi1.clone()

    basis2 = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi2, basis2)

    basis3 = np.array([[-1, 0], [0, -1]], dtype=float)
    lputil.set_basis_matrix(lpi3, basis3)

    plot_vecs = lpplot.make_plot_vecs(256, offset=0.1) # use an offset to prevent LP dir from being aligned with axis

    # bounds for lpi1 should be [[-2, -1], [-0.5, 0.5]]
    verts = lpplot.get_verts(lpi1, plot_vecs=plot_vecs)

    assert len(verts) == 5
    assert verts[0] == verts[-1]

    assert [-2, -0.5] in verts
    assert [-2, 0.5] in verts
    assert [-1, 0.5] in verts
    assert [-1, -0.5] in verts

    # bounds for lpi2 should be [[-0.5, 0.5], [1, 2]]
    verts = lpplot.get_verts(lpi2, plot_vecs=plot_vecs)

    assert len(verts) == 5
    assert verts[0] == verts[-1]

    assert [-0.5, 2] in verts
    assert [-0.5, 1] in verts
    assert [0.5, 1] in verts
    assert [0.5, 2] in verts

    # bounds for lpi3 should be [[2, 1], [-0.5, 0.5]]
    verts = lpplot.get_verts(lpi3, plot_vecs=plot_vecs)

    assert len(verts) == 5
    assert verts[0] == verts[-1]

    assert [2, -0.5] in verts
    assert [2, 0.5] in verts
    assert [1, 0.5] in verts
    assert [1, -0.5] in verts
    
    agg_dirs = np.array([[1, 0], [0, 1]], dtype=float)

    # box aggregation, bounds should be [[-2, 2], [-0.5, 2]]
    lpi = lputil.aggregate([lpi1, lpi2, lpi3], agg_dirs)
    assert lpi.cur_vars_offset == 6, "cur_vars_offset{} should be 6 (snapshot var columns)".format(lpi.cur_vars_offset)
 
    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)

    assert len(verts) == 5
    assert verts[0] == verts[-1]

    assert [-2., -0.5] in verts
    assert [-2, 2.] in verts
    assert [2., 2.] in verts
    assert [2., -0.5] in verts

def test_add_curtime_constraints():
    'tests add_curtime_constraints'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    # new constraint to be added, x <= 3.14, y <= 10
    csr_constraint = csr_matrix(np.array([[1, 0], [0, 1]], dtype=float))
    rhs = np.array([3.14, 10], dtype=float)

    lputil.add_curtime_constraints(lpi, csr_constraint, rhs)

    mat = lpi.get_full_constraints()
    vec = lpi.get_rhs()
    types = lpi.get_types()

    expected_mat = np.array([\
        [1, 0, -1, 0], \
        [0, 1, 0, -1], \
        [-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, -1, 0, 0], \
        [0, 1, 0, 0], \
        [0, 0, 1, 0], \
        [0, 0, 0, 1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1, 3.14, 10], dtype=float)

    fx = glpk.GLP_FX
    up = glpk.GLP_UP
    expected_types = [fx, fx, up, up, up, up, up, up]

    assert np.allclose(vec, expected_vec)
    assert types == expected_types
    assert np.allclose(mat.toarray(), expected_mat)

def test_add_reset_variables():
    'tests add_reset_variables'
    
    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    reset_csr = csr_matrix(2 * np.identity(2))
    mode_id = 1
    transition_id = 13
    lputil.add_reset_variables(lpi, mode_id, transition_id, reset_csr=reset_csr)

    assert lpi.dims == 2

    mat = lpi.get_full_constraints()
    types = lpi.get_types()
    rhs = lpi.get_rhs()
    names = lpi.get_names()

    expected_mat = np.array([\
        [1, 0, -1, 0, 0, 0, 0, 0], \
        [0, 1, 0, -1, 0, 0, 0, 0], \
        [-1, 0, 0, 0, 0, 0, 0, 0], \
        [1, 0, 0, 0, 0, 0, 0, 0], \
        [0, -1, 0, 0, 0, 0, 0, 0], \
        [0, 1, 0, 0, 0, 0, 0, 0], \
        [0, 0, 2, 0, -1, 0, 0, 0], \
        [0, 0, 0, 2, 0, -1, 0, 0], \
        [0, 0, 0, 0, 1, 0, -1, 0], \
        [0, 0, 0, 0, 0, 1, 0, -1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1, 0, 0, 0, 0], dtype=float)

    fx = glpk.GLP_FX
    up = glpk.GLP_UP
    expected_types = [fx, fx, up, up, up, up, fx, fx, fx, fx]

    expected_names = ["m0_i0", "m0_i1", "m0_c0", "m0_c1", "m1_i0_t13", "m1_i1", "m1_c0", "m1_c1"]

    assert np.allclose(rhs, expected_vec)
    assert types == expected_types
    assert np.allclose(mat.toarray(), expected_mat)
    assert names == expected_names

    assert lpi.basis_mat_pos == (8, 4)

    plot_vecs = lpplot.make_plot_vecs(4, offset=(math.pi / 4.0))
    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)

    assert len(verts) == 5
    
    assert [-10.0, 0.] in verts
    assert [-10.0, 2.] in verts
    assert [-8.0, 2.] in verts
    assert [-8.0, 0.] in verts
    assert verts[0] == verts[-1]

    # update the basis matrix to rotate quarter circle
    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)
    assert len(verts) == 5
    
    assert [0.0, 10.] in verts
    assert [0.0, 8.] in verts
    assert [2.0, 8.] in verts
    assert [2.0, 10.] in verts
    assert verts[0] == verts[-1]

def test_reset_less_dims():
    '''tests a reset to a mode with less dimensions
    project onto just the y variable multiplied by 0.5
    '''
    
    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    reset_csr = csr_matrix(np.array([[0, 0.5]], dtype=float))
    mode_id = 1
    transition_id = 13
    lputil.add_reset_variables(lpi, mode_id, transition_id, reset_csr=reset_csr)

    mat = lpi.get_full_constraints()
    types = lpi.get_types()
    rhs = lpi.get_rhs()
    names = lpi.get_names()

    expected_mat = np.array([\
        [1, 0, -1, 0, 0, 0], \
        [0, 1, 0, -1, 0, 0], \
        [-1, 0, 0, 0, 0, 0], \
        [1, 0, 0, 0, 0, 0], \
        [0, -1, 0, 0, 0, 0], \
        [0, 1, 0, 0, 0, 0], \
        [0, 0, 0, 0.5, -1, 0], \
        [0, 0, 0, 0, 1, -1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1, 0, 0], dtype=float)

    fx = glpk.GLP_FX
    up = glpk.GLP_UP
    expected_types = [fx, fx, up, up, up, up, fx, fx]

    expected_names = ["m0_i0", "m0_i1", "m0_c0", "m0_c1", "m1_i0_t13", "m1_c0"]

    assert np.allclose(rhs, expected_vec)
    assert types == expected_types
    assert np.allclose(mat.toarray(), expected_mat)
    assert names == expected_names

    assert lpi.basis_mat_pos == (7, 4)
    assert lpi.dims == 1

    plot_vecs = lpplot.make_plot_vecs(4, offset=(math.pi / 4.0))

    verts = lpplot.get_verts(lpi, xdim=0, ydim=None, plot_vecs=plot_vecs, cur_time=0)

    assert len(verts) == 3
    
    assert [0.5, 0] in verts
    assert [0, 0] in verts
    assert verts[0] == verts[-1]

    # update the basis matrix
    basis = np.array([[2]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    verts = lpplot.get_verts(lpi, xdim=0, ydim=None, plot_vecs=plot_vecs, cur_time=0)
    assert len(verts) == 3

    assert [1.0, 0] in verts
    assert [0, 0] in verts
    assert verts[0] == verts[-1]

def test_reset_minkowski():
    '''tests reset with a minkowski sum term and a new variable

    pre reset we have x = [-5, -4], y = [0, 1]
    post reset we have x = [-15, -14] (-10), y = [0, 1], t' = [0, 5]

    reset_matrix is [[1, 0], [0, 1], [0, 0]]
    minkowski_csr is [[1, 0], [0, 0], [0, 1]]
    minkowski_constraints_csr is [[1, 0], [-1, 0], [0, 1], [0, -1]]
    minkowski_constraints_rhs is [-10, 10, 5, 0]

    '''
    
    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    reset_csr = csr_matrix([[1, 0], [0, 1], [0, 0]], dtype=float)
    mode_id = 1
    transition_id = 13

    minkowski_csr = csr_matrix([[1, 0], [0, 0], [0, 1]], dtype=float)
    constraints_csr = csr_matrix([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    constraints_rhs = np.array([-10, 10, 5, 0], dtype=float)
    
    lputil.add_reset_variables(lpi, mode_id, transition_id, reset_csr=reset_csr, minkowski_csr=minkowski_csr, \
                               minkowski_constraints_csr=constraints_csr, minkowski_constraints_rhs=constraints_rhs)

    assert lpi.dims == 3

    # basis matrix should be at 9, 6
    assert lpi.basis_mat_pos == (9, 6)

    expected_names = ["m0_i0", "m0_i1", "m0_c0", "m0_c1", "a0", "a1", "m1_i0_t13", "m1_i1", "m1_i2", \
                          "m1_c0", "m1_c1", "m1_c2"]

    assert lpi.get_names() == expected_names

    plot_vecs = lpplot.make_plot_vecs(4, offset=(math.pi / 4.0))
    verts = lpplot.get_verts(lpi, xdim=0, ydim=1, plot_vecs=plot_vecs)

    assert len(verts) == 5
    
    assert [-15.0, 0.] in verts
    assert [-15.0, 1.] in verts
    assert [-14.0, 1.] in verts
    assert [-14.0, 0.] in verts
    assert verts[0] == verts[-1]

    verts = lpplot.get_verts(lpi, xdim=2, ydim=None, plot_vecs=plot_vecs, cur_time=0.0)

    assert len(verts) == 3
    
    assert [0, 0.] in verts
    assert [5, 0.] in verts
    assert verts[0] == verts[-1]

    lputil.set_basis_matrix(lpi, 3 * np.identity(3))

    verts = lpplot.get_verts(lpi, xdim=2, ydim=None, plot_vecs=plot_vecs, cur_time=0.0)

    assert len(verts) == 3
    
    assert [0, 0.] in verts
    assert [15, 0.] in verts
    assert verts[0] == verts[-1]

def test_init_triangle():
    'tests initialization from a non-box initial set of states'

    # x + y < 1, x > 0, y > 0

    constraints_mat = [[1, 1], [-1, 0], [0, -1]]
    constraints_rhs = [1, 0, 0]

    lpi = lputil.from_constraints(constraints_mat, constraints_rhs, HybridAutomaton().new_mode('mode_name'))

    mat = lpi.get_full_constraints()
    types = lpi.get_types()
    rhs = lpi.get_rhs()
    names = lpi.get_names()

    expected_mat = np.array([\
        [1, 0, -1, 0], \
        [0, 1, 0, -1], \
        [1, 1, 0, 0], \
        [-1, 0, 0, 0], \
        [0, -1, 0, 0]], dtype=float)

    expected_vec = np.array([0, 0, 1, 0, 0], dtype=float)

    fx = glpk.GLP_FX
    up = glpk.GLP_UP
    expected_types = [fx, fx, up, up, up]

    expected_names = ["m0_i0", "m0_i1", "m0_c0", "m0_c1"]

    assert np.allclose(rhs, expected_vec)
    assert types == expected_types
    assert np.allclose(mat.toarray(), expected_mat)
    assert names == expected_names

    # check verts

    plot_vecs = lpplot.make_plot_vecs(4, offset=(math.pi / 4.0))
    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)

    assert len(verts) == 4
    
    assert [0., 1.] in verts
    assert [0., 0.] in verts
    assert [1., 0] in verts
    assert verts[0] == verts[-1]

def test_get_box_center():
    'test get_box_center'

    lpi = lputil.from_box([[-5, -4], [0, 1]], HybridAutomaton().new_mode('mode_name'))

    pt = lputil.get_box_center(lpi)
    assert len(pt) == 2
    assert abs(pt[0] - (-4.5)) < 1e-4
    assert abs(pt[1] - (0.5)) < 1e-4

    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    pt = lputil.get_box_center(lpi)
    assert len(pt) == 2
    assert abs(pt[0] - (0.5)) < 1e-4
    assert abs(pt[1] - (4.5)) < 1e-4

    # try it rotated 1/4 around the circle
    a_mat = np.array([[0, 1], [-1, 0]], dtype=float)

    bm = expm(a_mat * math.pi / 4)
    lputil.set_basis_matrix(lpi, bm)

    expected = np.dot(bm, np.array([[-4.5], [0.5]], dtype=float))

    pt = lputil.get_box_center(lpi)

    assert len(pt) == 2
    assert abs(pt[0] - expected[0][0]) < 1e-4
    assert abs(pt[1] - expected[1][0]) < 1e-4


def test_make_direction_matrix():
    '''
    test making the direction matrix on a 2d example. The first vector should match the dynamics as the passed-in 
    point, while the second should be orthogonal to the first
    '''

    a_csr = csr_matrix(np.array([1, 0], [0, 1], dtype=float))
    pt = [2, 2] # derivative should be <2, 2>

    mat = lputil.make_direction_matrix(pt, a_csr)

    assert mat.shape == (2, 2)

    # first row should be <1, 1> (normalized version of derivative)
    assert abs(mat[0][0] - 1.0) < 1e-6
    assert abs(mat[0][1] - 1.0) < 1e-6

    for row_a in mat:
        assert abs(np.linalg.norm(row_a) - 1.0) < 1e-6, "rows should be normalized"

        for row_b in mat:
            if row_a == row_b:
                continue

            assert np.dot(row_a, row_b) < 1e-6, "rows should be orthononal"

    assert False

def test_direction_matrix_empty():
    '''test make_direction matrix when the dynamics matrix is empty

    it should just return any orthonormal set of vectors of the appropriate dimension
    '''

    a_csr = csr_matrix(np.zeros((5, 5)))
    pt = [0] * 5

    mat = lputil.make_direction_matrix(pt, a_csr)

    assert mat.shape == (3, 3)

    for row_a in mat:
        assert abs(np.linalg.norm(row_a) - 1.0) < 1e-6, "rows should be normalized"

        for row_b in mat:
            if row_a == row_b:
                continue

            assert np.dot(row_a, row_b) < 1e-6, "rows should be orthononal"

test_direction_matrix_empty()

def test_aggregate_on_subspace():
    '''
    test aggregation when the dynamics and sets are only on a subspace. The aggregation minkowski variables 
    should not include directions on the subspace, since they aren't necessary. 
    '''

    # dynamics are x' == 1, y' == 0, a' == 0
    # lpi1 is [0, 1] x [0, 1] x [1, 1]
    # lpi2 is [3, 4] x [0, 1] x [1, 1]

    # aggregation shouldn't need to introduce a variable along the y direction

    mode = HybridAutomaton().new_mode('mode_name')
    lpi1 = lputil.from_box([[0, 1], [0, 1], [1, 1]], mode)
    lpi2 = lputil.from_box([[4, 5], [0, 1], [1, 1]], mode)

    #a_csr = csr_matrix(np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float))
    agg_dirs = np.array([[1, 0, 0], [0, 1, 1], [0, 1, -1]], dtype=float)

    # box aggregation
    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs)

    verts = lpplot.get_verts(lpi)

    assert len(verts) == 5

    assert pair_almost_in([0., 0.], verts)
    assert pair_almost_in([0., 1.], verts)
    assert pair_almost_in([5., 1.], verts)
    assert pair_almost_in([5., 0.], verts)

    assert verts[0] == verts[-1]

    # make sure only one aggregation variable was introduced
    names = lpi.get_names()

    expected_names = ["m0_i0", "m0_i1", "m0_i3", "m0_c0", "m0_c1", "m0_c2", "agg0", "m0_c0", "m0_c1", "m0_c2"]

    assert names == expected_names
