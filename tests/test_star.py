'''
Unit tests for star.py
Stanley Bak
August 2016
'''

import unittest
import math
import matplotlib.pyplot as plt

import numpy as np
from numpy.testing import assert_allclose

from hylaa.star import init_hr_to_star, init_constraints_to_star
from hylaa.star import Star, InitParent
from hylaa.hybrid_automaton import HyperRectangle, LinearHybridAutomaton, LinearConstraint
from hylaa.timerutil import Timers
from hylaa.glpk_interface import LpInstance
from hylaa.containers import HylaaSettings
from hylaa.plotutil import debug_plot_star as plot_star

def make_settings():
    'make the default settings'

    settings = HylaaSettings(0.1, 1.0)
    settings.plot.xdim = 0
    settings.plot.ydim = 1
    settings.plot.num_angles = 128

    return settings

def make_debug_mode():
    'make an AutomatonMode object'

    ha = LinearHybridAutomaton('Test Automaton')
    ha.variables = ["x", "y"]
    loc = ha.new_mode('test_mode')

    return loc

def make_star(center, basis, a_list, b_list):
    'make a star without a parent'

    constraint_list = []

    assert len(a_list) == len(b_list)

    for i in xrange(len(a_list)):
        vector = np.array(a_list[i], dtype=float)
        constraint_list.append(LinearConstraint(vector, b_list[i]))

    mode = make_debug_mode()
    parent = InitParent(mode)

    return Star(make_settings(), np.array(center, dtype=float), np.array(basis, dtype=float), constraint_list,
                parent, mode)

def make_star_a_mat(star):
    'extract the constraints a-matrix (vectors) from a star'

    num_constraints = len(star.constraint_list)
    num_dims = star.constraint_list[0].vector.shape[0]

    a_mat = np.zeros((num_constraints, num_dims))

    for c in xrange(num_constraints):
        a_mat[c, :] = star.constraint_list[c].vector

    return a_mat

def make_star_b_vec(star):
    'extract the constraints b-vector (values) from a star'

    num_constraints = len(star.constraint_list)

    b_vec = np.zeros((num_constraints))

    for c in xrange(num_constraints):
        b_vec[c] = star.constraint_list[c].value

    return b_vec

class TestStar(unittest.TestCase):
    'Unit tests for star'

    loc = make_debug_mode()
    plot_settings = make_settings().plot

    def setUp(self):
        'setup function'
        Timers.reset()

        Star.init_plot_vecs(2, make_settings().plot)

    def test_boundaries_optimized(self):
        'test finding the boundaries in an optimized fashion'

        Star.init_plot_vecs(2, TestStar.plot_settings)

        hr = HyperRectangle([(0, 1), (0, 1),])
        star = init_hr_to_star(make_settings(), hr, TestStar.loc)

        start_op = LpInstance.total_optimizations()

        verts = star.verts()

        num_op = LpInstance.total_optimizations() - start_op

        self.assertEqual(len(verts), 5)
        self.assertLess(num_op, 100)

    def test_boundaries_simple(self):
        'check the boundaries for a harmonic oscillator after some time elapses'

        Star.init_plot_vecs(2, TestStar.plot_settings)
        center = [0, 0]

        # basis vectors after some time elapsed
        vec1 = [0.98510371, -0.09884004]
        vec2 = [0.09884004, 0.98510371]

        basis = [vec1, vec2]

        a_list = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        b_list = [-4, 5, 1, 0]

        star = make_star(center, basis, a_list, b_list)

        Star.plot_vecs = [np.array([0, -1], dtype=float), # max(y)
                          np.array([-1, 0], dtype=float), # max(x)
                          np.array([1, 0], dtype=float)] # min(x)

        verts = star.verts()

        self.assertGreater(verts[0][1], 1.0)

        self.assertLess(verts[1][0], -3.0)
        self.assertGreater(verts[1][0], -4.0)

        self.assertGreater(verts[2][0], -5.0)

    def test_hr_to_star(self):
        'test hr converstion to a star'

        hr = HyperRectangle([(1, 2), (2, 2), (-6, -3)])

        star = init_hr_to_star(make_settings(), hr, TestStar.loc)

        self.assertTrue(isinstance(star, Star))

        expected_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert_allclose(star.basis_matrix, expected_basis, rtol=1e-10, atol=1e-10)

        expected_a_mat = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])

        a_mat = make_star_a_mat(star)
        assert_allclose(a_mat, expected_a_mat, rtol=1e-10, atol=1e-10)

        expected_b = np.array([2, -1, 2, -2, -3, 6])
        b_vec = make_star_b_vec(star)
        assert_allclose(b_vec, expected_b, rtol=1e-10, atol=1e-10)

    def test_update_from_sim(self):
        'test the update_from_sim function'

        basis = [[1., 0.], [0., 1.]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [0.5, 0.5, 0.5, 0.5]
        center = [-5, 0.5]

        star = make_star(center, basis, a_mat, b_vec)
        center = star.center
        self.assertEqual(2, len(center))

        assert_allclose(center, np.array([-5, 0.5]), rtol=1e-6, atol=1e-6)

        # at this point, star is centered at (-5, 0.5)
        # let's rotate around by half a cycle
        star.update_from_sim(np.array([[-1., 0.], [0., -1.]]), np.array([5.0, -0.5]))
        center = star.center
        self.assertEqual(2, len(center))

        assert_allclose(center, np.array([5.0, -0.5]), rtol=1e-6, atol=1e-6)

    def test_find_star_boundary_square(self):
        'test the lp solving to find boundary points with square-shaped star'

        basis = [[1., 0.], [0., 1.]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [0.5, 0.5, 0.5, 0.5]
        center = [-5, 0.5]

        star = make_star(center, basis, a_mat, b_vec)
        vec = np.array([0.70710678, -0.70710678])

        Star.plot_vecs = [-1 * vec] * 3 # multiple by -1 because we minimize the direction
        pt = star._find_star_boundaries()[0]

        self.assertEquals(pt.shape, (2,))
        assert_allclose(pt, np.array([-4.5, 0.0]), rtol=1e-6, atol=1e-6)

        # check that we've achieved the limit
        pt_inside = pt - 1e-2 * vec
        pt_outside = pt + 1e-2 * vec

        self.assertTrue(star.contains_point(pt_inside),
                        "boundary point {} is not in the star".format(pt_inside))

        self.assertFalse(star.contains_point(pt_outside),
                         "boundary point is not extreme, {} is better".format(pt_outside))

    def test_find_star_boundary_diamond(self):
        'test the lp solving to find boundary points with diamond-shaped star'

        basis = [[1, 2], [-1, 2]]
        a_mat = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        mag = 0.5
        b_vec = [mag, mag, mag, mag]
        center = [0, 0]

        star = make_star(center, basis, a_mat, b_vec)
        vec = np.array([1, 0], dtype=float)

        Star.plot_vecs = [-1 * vec] * 3 # multiply by -1 because we minimize the direction
        pt = star._find_star_boundaries()[0]

        # check that we've achieved the limit
        pt_inside = pt - 1e-2 * vec
        pt_outside = pt + 1e-2 * vec

        self.assertTrue(star.contains_point(pt_inside),
                        "boundary point {} is not in the star".format(pt_inside))

        self.assertFalse(star.contains_point(pt_outside),
                         "boundary point is not extreme, {} is better".format(pt_outside))

        self.assertEquals(pt.shape, (2,))
        assert_allclose(pt, np.array([1.0, 0.0]), rtol=1e-6, atol=1e-6)

    def check_stars_equal(self, star1, star2):
        'test for star equality using a sampling-based approach'

        self.assertEqual(star1.num_dims, star2.num_dims)
        dims = star1.num_dims

        # try each orthonormal direction
        dirs = [[1.0 if d == index else 0.0 for d in xrange(dims)] for index in xrange(dims)]

        # and its negation
        dirs += [[-1.0 if d == index else 0.0 for d in xrange(dims)] for index in xrange(dims)]

        # and try diagonal
        dirs.append([1.0] * dims)

        result = np.zeros((2*dims), dtype=float)
        msg = None
        tol = 1e-6

        for i in xrange(len(dirs)):
            direction = np.array(dirs[i], dtype=float)

            star1.get_lpi().minimize(direction, result, error_if_infeasible=True)
            val1 = np.dot(direction, result[0:dims])

            star2.get_lpi().minimize(direction, result, error_if_infeasible=True)
            val2 = np.dot(direction, result[0:dims])

            if abs(val1 - val2) > tol:
                msg = "stars are not equal in direction {}. star1={}, star2={}. set show_plot=True to see plots".format(
                    direction, val1, val2)
                break

        show_plot = False

        if show_plot:
            print "Note: show_plot was set to True (set it to False after you finished debugging)"

        if show_plot and msg is not None and dims == 2:
            print msg
            print "Plotting... star1 = blue, star2 = red"

            plot_star(star1, col='b-')
            plot_star(star2, col='r--')

            plt.xlim([-5, 5])
            plt.ylim([-5, 5])
            plt.show()

        self.assertEqual(msg, None, msg=msg)

    def test_eat_simple_1d(self):
        '''
        test star.eat_star() on a simple 1d example:
        star1 is [3, 5] with center 4
        star2 is [8, 10] with center 8

        expected result is that star1 becomes [3, 10] with center 4, so constraints become:
        -x <= -1
        x <= 6
        '''

        ha = LinearHybridAutomaton('Test Automaton')
        ha.variables = ["x"]
        mode = ha.new_mode('test_mode')

        constraint_list = []
        constraint_list.append(LinearConstraint(np.array([-1.0]), -1))
        constraint_list.append(LinearConstraint(np.array([1.0]), 1))
        star1 = init_constraints_to_star(make_settings(), constraint_list, mode)
        star1.center = np.array([4.0])

        constraint_list = []
        constraint_list.append(LinearConstraint(np.array([-1.0]), 0))
        constraint_list.append(LinearConstraint(np.array([1.0]), 2))
        star2 = init_constraints_to_star(make_settings(), constraint_list, mode)
        star2.center = np.array([8.0])

        star1.eat_star(star2)

        self.assertAlmostEqual(star1.center[0], 4.0)

        self.assertEqual(len(star1.constraint_list), 2)
        self.assertAlmostEqual(star1.constraint_list[0].value, -1.0)
        self.assertAlmostEqual(star1.constraint_list[1].value, 6.0)

    def test_eat_self(self):
        'test star.eat_star() on an equal star (should not change constraints)'

        hr = HyperRectangle([(1, 2), (2, 2), (-6, -3)])

        star = init_hr_to_star(make_settings(), hr, TestStar.loc)
        star2 = init_hr_to_star(make_settings(), hr, TestStar.loc)

        star.eat_star(star2)
        tol = 1e-6

        # constraints should be the same as before
        a_mat = make_star_a_mat(star)
        a_mat2 = make_star_a_mat(star2)
        b_vec = make_star_b_vec(star)
        b_vec2 = make_star_b_vec(star2)

        self.assertEqual(len(b_vec), len(b_vec2))

        for row_index in xrange(len(b_vec)):
            assert_allclose(a_mat[row_index], a_mat2[row_index], rtol=tol, atol=tol)
            self.assertAlmostEqual(b_vec[row_index], b_vec2[row_index])

        self.check_stars_equal(star, star2)

    def test_eat_other(self):
        'test star.eat_star() on a non-equal star: [1,2] eating [2, 3] results in [1, 3]'

        star = init_hr_to_star(make_settings(), HyperRectangle([(1, 2)]), TestStar.loc)
        star2 = init_hr_to_star(make_settings(), HyperRectangle([(2, 3)]), TestStar.loc)

        expected = init_hr_to_star(make_settings(), HyperRectangle([(1, 3)]), TestStar.loc)

        star.eat_star(star2)

        self.check_stars_equal(star, expected)

    def test_eat_other_2d(self):
        'test star.eat_star() on a 2-d non-equal star'

        star = init_hr_to_star(make_settings(), HyperRectangle([(-5, -2), (10, 20)]), TestStar.loc)
        star2 = init_hr_to_star(make_settings(), HyperRectangle([(20, 34), (-3, -2)]), TestStar.loc)
        expected = init_hr_to_star(make_settings(), HyperRectangle([(-5, 34), (-3, 20)]), TestStar.loc)

        star.eat_star(star2)
        self.check_stars_equal(star, expected)

    def test_eat_unit_rotated(self):
        'test star.eat_star() on a unit square eating rotated unit-square'

        star = init_hr_to_star(make_settings(), HyperRectangle([(-1, 1), (-1, 1)]), TestStar.loc)

        size = math.sqrt(2.0) / 2.0
        basis = [[size, size], [size, -size]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [1, 1, 1, 1]
        center = [0, 0]

        star2 = make_star(center, basis, a_mat, b_vec)

        root2 = math.sqrt(2)

        expected = init_hr_to_star(make_settings(), HyperRectangle([(-root2, root2), (-root2, root2)]), TestStar.loc)

        star.eat_star(star2)
        self.check_stars_equal(star, expected)

    def test_rotated_check_stars_equal(self):
        'test star.check_stars_equal on a rotated unit-square'

        size = math.sqrt(2.0) / 2.0
        basis = [[size, size], [size, -size]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [1, 1, 1, 1]
        center = [0, 0]

        star = make_star(center, basis, a_mat, b_vec)

        self.check_stars_equal(star, star)

    def test_eat_rotated_unit(self):
        'test star.eat_star() on a rotated unit-square eating the unit square'

        size = math.sqrt(2.0) / 2.0
        basis = [[size, size], [size, -size]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [1, 1, 1, 1]
        center = [0, 0]

        star = make_star(center, basis, a_mat, b_vec)
        star2 = init_hr_to_star(make_settings(), HyperRectangle([(-1, 1), (-1, 1)]), TestStar.loc)

        plot_star(star2, col='k:')

        size = 1
        basis = [[size, size], [size, -size]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [1, 1, 1, 1]
        center = [0, 0]

        expected = make_star(center, basis, a_mat, b_vec)

        star.eat_star(star2)

        self.check_stars_equal(star, expected)

    def test_rotated_eat_offset(self):
        'test star.eat_star() on a rotated unit-square eating an offset-star at (2, 0)'

        size = math.sqrt(2.0) / 2.0
        basis = [[size, size], [size, -size]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [1, 1, 1, 1]
        center = [0, 0]

        star = make_star(center, basis, a_mat, b_vec)
        star2 = init_hr_to_star(make_settings(), HyperRectangle([(2, 2), (0, 0)]), TestStar.loc)

        basis = [[size, size], [size, -size]]
        a_mat = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
        b_vec = [math.sqrt(2), 1, math.sqrt(2), 1]
        center = [0, 0]

        expected = make_star(center, basis, a_mat, b_vec)

        star.eat_star(star2)

        self.check_stars_equal(star, expected)

    def test_square_eat_parallelogram(self):
        'test square star eating parallelogram star'

        star = init_hr_to_star(make_settings(), HyperRectangle([(0, 1), (0, 1)]), TestStar.loc)

        basis = [[1, 0], [1, 1]]
        a_mat = [[1., 0.], [-1, 0.], [0., 1.], [0., -1.]]
        b_vec = [1, 0, 1, 0]
        center = [0, 0]
        par = make_star(center, basis, a_mat, b_vec)

        expected = init_hr_to_star(make_settings(), HyperRectangle([(0, 2), (0, 1)]), TestStar.loc)

        star.eat_star(par)

        self.check_stars_equal(star, expected)

    def test_offset_dif_basis(self):
        'test a tricky case an offset square eating another one, with non-standard basis'

        basis = [[0, 1], [1, 0]]
        a_mat = [[0, 1], [-1, 0], [1, 0], [0, -1]]
        b_vec = [4, 1, 1, -2]
        center = [0, 0]
        a2 = make_star(center, basis, a_mat, b_vec)

        basis = [[0, 1], [1, 0]]
        a_mat = [[0, 1], [-1, 0], [1, 0], [0, -1]]
        b_vec = [1, 1, 1, 1]
        center = [0, 0]
        b = make_star(center, basis, a_mat, b_vec)

        a2.eat_star(b)

        expected = init_hr_to_star(make_settings(), HyperRectangle([(-1, 4), (-1, 1)]), TestStar.loc)

        self.check_stars_equal(a2, expected)

    def test_init_constraints(self):
        'test hylaa initialization using constraints'

        hr = HyperRectangle([(-1, 4), (-1, 1)])
        star = init_hr_to_star(make_settings(), hr, TestStar.loc)

        # now try a list of LinearConstraints
        constraints = [LinearConstraint([-1, 0], 1), LinearConstraint([1, 0], 4),
                       LinearConstraint([0, -1], 1), LinearConstraint([0, 1], 1)]

        star2 = init_constraints_to_star(make_settings(), constraints, TestStar.loc)

        self.check_stars_equal(star, star2)

if __name__ == '__main__':
    unittest.main()
