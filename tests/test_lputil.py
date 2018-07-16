'''
Tests for LP operations. Made for use with py.test
'''

import math
import numpy as np

import swiglpk as glpk

from hylaa import lputil, lpplot
from hylaa.lpinstance import LpInstance

def test_from_box():
    'tests from_box'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    mat, types, vec = lpi.get_constraints()

    expected_mat = np.array([\
        [-1, 0, 1, 0], \
        [0, -1, 0, 1], \
        [0, 0, -1, 0], \
        [0, 0, 1, 0], \
        [0, 0, 0, -1], \
        [0, 0, 0, 1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1], dtype=float)

    fx = glpk.GLP_FX
    up = glpk.GLP_UP
    expected_types = np.array([fx, fx, up, up, up, up], dtype=np.int32)

    assert np.allclose(vec, expected_vec)
    assert np.allclose(types, expected_types)
    assert np.allclose(mat.toarray(), expected_mat)

def test_print_lp():
    'test printing the lp to stdout'

    lpi = lputil.from_box([[-5, -4], [0, 1]])
    assert str(lpi) is not None

def test_set_basis_matrix():
    'tests lputil set_basis_matrix on harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    mat, _, vec = lpi.get_constraints()

    expected_mat = np.array([\
        [-1, 0, 0, 1], \
        [0, -1, -1, 0], \
        [0, 0, -1, 0], \
        [0, 0, 1, 0], \
        [0, 0, 0, -1], \
        [0, 0, 0, 1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1], dtype=float)

    assert np.allclose(vec, expected_vec)

    assert np.allclose(mat.toarray(), expected_mat)

def test_check_intersection():
    'tests check_intersection on the harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    # check if initially y >= 4.5 is possible (should be false)
    direction = np.array([0, -1], dtype=float)

    assert not lputil.check_intersection(lpi, direction, -4.5)

    # after basis matrix update
    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    # now check if y >= 4.5 is possible (should be true)
    assert lputil.check_intersection(lpi, direction, -4.5)

def test_verts():
    'tests verts'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

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

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    # update basis matrix
    basis_mat = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis_mat)

    # minimize y should give 4.0
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.0) < 1e-6

    # add constraint: y >= 4.5
    direction = np.array([0, -1], dtype=float)

    new_row = lputil.add_init_constraint(lpi, direction, -4.5)

    assert new_row == 6, "new constraint should have been added in row index 6"

    # minimize y should give 4.5
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.5) < 1e-6

    # check verts()
    verts = lpplot.get_verts(lpi)

    assert len(verts) == 5
    
    assert [0.0, 5.0] in verts
    assert [1.0, 5.0] in verts
    assert [0.0, 4.5] in verts
    assert [1.0, 4.5] in verts
    assert verts[0] == verts[-1]

def test_try_replace_constraint():
    'tests try_replace_constraint on the harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    # update basis matrix
    basis_mat = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis_mat)

    # minimize y should give 4.0
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.0) < 1e-6

    # add constraint: y >= 4.5
    direction = np.array([0, -1], dtype=float)

    row_index = lputil.add_init_constraint(lpi, direction, -4.5)

    # minimize y should give 4.5
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.5) < 1e-6

    assert lpi.get_num_rows() == 7

    # try to replace constraint y >= 4.6 (should be stronger than 4.5)
    row_index = lputil.try_replace_constraint(lpi, row_index, direction, -4.6)

    assert row_index == 6
    assert lpi.get_num_rows() == 7

    # try to replace constraint x <= 0.9 (should be incomparable)
    xdir = np.array([1, 0], dtype=float)
    row_index = lputil.try_replace_constraint(lpi, row_index, xdir, 0.9)

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

def test_box_aggregate():
    'tests box aggregation'

    lpi1 = lputil.from_box([[0, 1], [0, 1]])
    lpi2 = lputil.from_box([[1, 2], [1, 2]])

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

    lpi1 = lputil.from_box([[0, 1], [0, 1]])
    lpi2 = lputil.from_box([[1, 2], [1, 2]])

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

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    mat = lputil.get_basis_matrix(lpi)

    assert np.allclose(mat, basis)

def test_box_aggregate3():
    'tests box aggregation with 3 boxes'

    lpi1 = lputil.from_box([[-2, -1], [-0.5, 0.5]])
    lpi2 = LpInstance(lpi1)
    lpi3 = LpInstance(lpi1)

    basis2 = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi2, basis2)

    basis3 = np.array([[-1, 0], [0, -1]], dtype=float)
    lputil.set_basis_matrix(lpi3, basis3)

    agg_dirs = np.array([[1, 0], [0, 1]], dtype=float)

    # box aggregation
    lpi = lputil.aggregate([lpi1, lpi2, lpi3], agg_dirs)

    plot_vecs = lpplot.make_plot_vecs(256, offset=0.1) # use an offset to prevent LP dir from being aligned with axis
    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)

    assert len(verts) == 5

    assert [-2., -0.5] in verts
    assert [-2, 2.] in verts
    assert [2., 2.] in verts
    assert [2., -0.5] in verts
    
    assert verts[0] == verts[-1]
