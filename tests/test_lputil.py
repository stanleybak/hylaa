'''
Tests for LP operations.
'''

from hylaa import lputil, lpplot

import numpy as np

def test_from_box():
    'tests from_box'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    mat, vec = lpi.get_matrix()

    expected_mat = np.array([\
        [-1, 0, 1, 0], \
        [0, -1, 0, 1], \
        [0, 0, -1, 0], \
        [0, 0, 1, 0], \
        [0, 0, 0, -1], \
        [0, 0, 0, 1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1], dtype=float)

    assert np.allclose(vec, expected_vec)
    assert np.allclose(mat, expected_mat)

def test_set_basis_matrix():
    'tests lputil set_basis_matrix on harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    basis = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis)

    mat, vec = lpi.get_matrix()

    expected_mat = np.array([\
        [-1, 0, 0, 1], \
        [0, -1, -1, 0], \
        [0, 0, -1, 0], \
        [0, 0, 1, 0], \
        [0, 0, 0, -1], \
        [0, 0, 0, 1]], dtype=float)

    expected_vec = np.array([0, 0, 5, -4, 0, 1], dtype=float)

    assert np.allclose(vec, expected_vec)
        
    assert np.allclose(mat, expected_mat)

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

    verts = lpplot.get_verts(lpi, 2)

    assert len(verts) == 5
    
    assert [-5.0, 0.] in verts
    assert [-5.0, 1.] in verts
    assert [-4.0, 1.] in verts
    assert [-4.0, 0.] in verts
    assert verts[0] == verts[-1]

def test_add_constraint():
    'tests add_constraint on the harmonic oscillator example'

    lpi = lputil.from_box([[-5, -4], [0, 1]])

    # update basis matrix
    basis_mat = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi, basis_mat)

    # minimize y should give 4.0
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.0) < 1e-6

    # add constraint: y >= 4.5
    direction = np.array([0, -1], dtype=float)

    new_row = lputil.add_constraint(lpi, basis_mat, direction, -4.5)

    assert new_row == 6, "new constraint should have been added in row index 6"

    # minimize y should give 4.5
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.5) < 1e-6

    # check verts()
    verts = lpplot.get_verts(lpi, 2)

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

    row_index = lputil.add_constraint(lpi, basis_mat, direction, -4.5)

    # minimize y should give 4.5
    miny = lpi.minimize([0, 1, 0, 0])[1]
    assert abs(miny - 4.5) < 1e-6

    assert lpi.get_num_rows() == 7

    # try to replace constraint y >= 4.6 (should be stronger than 4.5)
    row_index = lputil.try_replace_constraint(lpi, row_index, basis_mat, direction, -4.6)

    assert row_index == 6
    assert lpi.get_num_rows() == 7

    # try to replace constraint x <= 0.9 (should be incomparable)
    xdir = np.array([1, 0], dtype=float)
    row_index = lputil.try_replace_constraint(lpi, row_index, basis_mat, xdir, 0.9)

    assert row_index == 7
    assert lpi.get_num_rows() == 8

    # check verts()
    verts = lpplot.get_verts(lpi, 2)

    assert len(verts) == 5
    
    assert [0.0, 5.0] in verts
    assert [0.9, 5.0] in verts
    assert [0.0, 4.6] in verts
    assert [0.9, 4.6] in verts
    assert verts[0] == verts[-1]

def test_convex_hull():
    'tests convex_hull'

    lpi1 = lputil.from_box([[-5, -4], [0, 1]])
    lpi2 = lputil.from_box([[0, 1], [4, 5]])

    lpi = lputil.convex_hull([lpi1, lpi2])

    verts = lpplot.get_verts(lpi, 2)

    assert len(verts) == 7
    
    assert [-5.0, 0.] in verts
    assert [-5.0, 1.] in verts
    #assert [-4.0, 1.] in verts
    assert [-4.0, 0.] in verts

    #assert [0., 4.] in verts
    assert [0., 5.] in verts
    assert [1., 5.] in verts
    assert [1., 4.] in verts
    
    assert verts[0] == verts[-1]
