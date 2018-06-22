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
    
