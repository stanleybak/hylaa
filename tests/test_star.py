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
    settings.plot.num_angles = 512

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
            val1 += np.dot(star1.center, direction)

            star2.get_lpi().minimize(direction, result, error_if_infeasible=True)
            val2 = np.dot(direction, result[0:dims])
            val2 += np.dot(star2.center, direction)

            if abs(val1 - val2) > tol:
                msg = "stars are not equal in direction {}. star1={}, star2={}. set show_plot=True to see plots".format(
                    -1 * direction, val1, val2)
                break

        show_plot = False

        if show_plot:
            print "Note: show_plot was set to True (set it to False after you finished debugging)"

        if show_plot and msg is not None and dims == 2:
            print msg
            print "Plotting... star1 = blue, star2 = red"

            plot_star(star1, col='b-')
            plot_star(star2, col='r--')

            #plt.xlim([-5, 5])
            #plt.ylim([-5, 5])
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
        'test a case an offset square eating another one, with non-standard basis'

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

    def test_clone(self):
        'test that clone is working as expected'

        mode = make_debug_mode()

        array = np.array

        star = Star(HylaaSettings(0.1, 1.0), array([-0.11060623, -0.62105371]),
                    array([[-0.12748444, -6.33082449], [0.06330825, -0.3807174]]),
                    [LinearConstraint(array([1., 0.]), -0.76877967521149881),
                     LinearConstraint(array([-1., 0.]), 0.8687796752114989),
                     LinearConstraint(array([0., 1.]), -1.2821810092782602),
                     LinearConstraint(array([0., -1.]), 1.4821810092782604)],
                    None, mode, extra_init=(None, 20, 0))

        self.check_stars_equal(star, star.clone())

    def test_center_into_constraints(self):
        '''test that center_into_constraints doesn't modify the stars'''

        mode = make_debug_mode()
        array = np.array

        star = Star(HylaaSettings(0.1, 1.0),
                    array([5.0, 5.0]), array([[0.707, 0.707], [-0.707, 0.707]]),
                    [LinearConstraint(array([1., 0.]), 1), LinearConstraint(array([-1., 0.]), 0.),
                     LinearConstraint(array([0., 1.]), 1), LinearConstraint(array([0., -1.]), 0)],
                    None, mode, extra_init=(None, 20, 0))

        star2 = star.clone()
        star2.center_into_constraints(star2.vector_to_star_basis(star2.center))

        self.check_stars_equal(star, star2)

    def test_center_into_constraints_complex(self):
        '''test that center_into_constraints doesn't modify the stars'''

        mode = make_debug_mode()
        array = np.array

        star = Star(HylaaSettings(0.01, 2.0),
                    array([-0.11060623, -0.62105371]), array([[-0.12748444, -6.33082449], [0.06330825, -0.3807174]]),
                    [LinearConstraint(array([1., 0.]), -0.95), LinearConstraint(array([-1., 0.]), 1.05),
                     LinearConstraint(array([0., 1.]), 0.1), LinearConstraint(array([0., -1.]), 0.1),
                     LinearConstraint(array([6.33082449, 0.3807174]), -0.6210537093866693),
                     LinearConstraint(array([0.12748444, -0.06330825]), -0.11060623185202963),
                     LinearConstraint(array([-0.12748444, 0.06330825]), 1.1106062318520296)],
                    None, mode, extra_init=(None, 20, 0))

        star2 = star.clone()
        star2.center_into_constraints(star2.vector_to_star_basis(star2.center))

        self.check_stars_equal(star, star2)

    def test_eat_star_bs1_2d(self):
        'test 2d eat-star derived from example-1 of the ball_string system'

        array = np.array
        mode = make_debug_mode()
        center = array([0., 0.])

        basis_matrix = array([[1.0, 0], [0, 0.5]])

        cur_star = Star(HylaaSettings(0.01, 2.0), center, basis_matrix, [ \
           LinearConstraint(array([1, 0]), 1), \
           LinearConstraint(array([-1., 0.]), 0), \
           LinearConstraint(array([0., 1.]), 1.0), \
           LinearConstraint(array([0., -1.]), 0.0)], \
           None, mode)

        new_star = Star(HylaaSettings(0.01, 2.0), center, basis_matrix, [ \
           LinearConstraint(array([1, 0]), 1), \
           LinearConstraint(array([-1, 0]), 0), \
           LinearConstraint(array([0., 1.]), 3.0), \
           LinearConstraint(array([0., -1.]), -2.0)], \
           None, mode)

        cur_star.eat_star(new_star)

        new_point = new_star.get_feasible_point()
        self.assertTrue(cur_star.contains_point(new_point))

    def test_eat_star_bs_tricky(self):
        'test 2d eat-star derived from a tricky example with the ball_string system'

        array = np.array
        mode = make_debug_mode()

        cur_star = Star(HylaaSettings(0.01, 2.0), array([0., 0.]), array([[-1.39364934, -6.33082449],\
           [-0.01283523, -0.3807174]]), [LinearConstraint(array([1., 0.]), -0.65858417090053001), \
           LinearConstraint(array([-1., 0.]), 0.7585841709005301), \
           LinearConstraint(array([0., 1.]), 2.0388429332115034), \
           LinearConstraint(array([0., -1.]), -1.8388429332115035), \
           LinearConstraint(array([6.33082449, 0.3807174]), 1.9619999999999997), \
           LinearConstraint(array([0.12748444, -0.06330825]), -0.19619999994184489), \
           LinearConstraint(array([-0.12748444, 0.06330825]), 1.1961999999418449), \
           LinearConstraint(array([1.39364934, 0.01283523]), -1.0000000000000002)], \
           None, mode, extra_init=(array([[-1.39364934, -6.33082449],\
           [-0.01283523, -0.3807174]]), 40, 0))

        new_star = Star(HylaaSettings(0.01, 2.0), array([0., 0.]), array([[-1.45695758, -6.33082449],\
           [-0.0166424, -0.3807174]]), [LinearConstraint(array([1., 0.]), -0.66180203160723172), \
           LinearConstraint(array([-1., 0.]), 0.76180203160723181), \
           LinearConstraint(array([0., 1.]), 2.3500231188180134), \
           LinearConstraint(array([0., -1.]), -2.1500231188180132), \
           LinearConstraint(array([6.33082449, 0.3807174]), 2.0600999999999994), \
           LinearConstraint(array([0.12748444, -0.06330825]), -0.21631049992724288), \
           LinearConstraint(array([-0.12748444, 0.06330825]), 1.2163104999272427), \
           LinearConstraint(array([-1.39364934, -0.01283523]), 1.0004904999853983), \
           LinearConstraint(array([1.45695758, 0.0166424]), -1.0000000000000002)], \
           None, mode, extra_init=(array([[-1.45695758, -6.33082449],\
           [-0.0166424, -0.3807174]]), 41, 0))

        cur_star.eat_star(new_star)

        new_point = new_star.get_feasible_point(standard_dir=[5, -1])
        self.assertTrue(cur_star.contains_point(new_point))

    def test_eat_star_bs2_2d(self):
        'test 2d eat-star derived from example-2 of the ball_string system'

        array = np.array
        mode = make_debug_mode()
        settings = HylaaSettings(0.01, 2.0)
        center = array([0., 0.])

        basis_matrix = array([[1.0, 0], [0, 1.0]], dtype=float)

        cur_star = Star(settings, center, basis_matrix, [\
           LinearConstraint(array([-1., 0.]), 1.0), \
           LinearConstraint(array([0., 1.]), 2.1), \
           LinearConstraint(array([0., -1.]), -2.0), \
           LinearConstraint(array([2.0, 0.]), -1),\
           ], \
           None, mode)

        new_star = Star(settings, center, basis_matrix, [\
           LinearConstraint(array([-1., 0.]), 0.7), \
           LinearConstraint(array([1., 0.]), -0.6), \
           LinearConstraint(array([0., 1.]), 2.3), \
           LinearConstraint(array([0., -1.]), -2.2), \
           ], \
           None, mode)

        cur_star.eat_star(new_star)

        new_point = new_star.get_feasible_point()
        self.assertTrue(cur_star.contains_point(new_point))

    def test_eat_star_bs1_1d(self):
        'test 1d eat-star with example-1 derived from the ball_string system'

        settings = HylaaSettings(0.01, 2.0)
        array = np.array
        ha = LinearHybridAutomaton('Test Automaton')
        ha.variables = ["x"]
        mode = ha.new_mode('test_mode')

        center = array([0.])

        # x = 0.5 * alpha    (2x = alpha)

        # 0 <= alpha <= 1.0   -> (0 <= x <= 0.5)
        cur_star = Star(settings, center, array([[0.5]]), [ \
           LinearConstraint(array([1.]), 1.0), \
           LinearConstraint(array([-1.]), 0.0)], \
           None, mode)

        # 2.0 <= alpha <= 3.0    -> (1.0 <= x <= 1.5)
        new_star = Star(settings, center, array([[0.5]]), [ \
           LinearConstraint(array([1.]), 3.0), \
           LinearConstraint(array([-1.]), -2.0)], \
           None, mode)

        cur_star.eat_star(new_star)

        # should be 0.0 <= alpha <= 3.0
        self.assertAlmostEqual(cur_star.constraint_list[0].value, 3.0)
        self.assertAlmostEqual(cur_star.constraint_list[1].value, 0.0)

    def test_eat_star_bs2_1d(self):
        'test 1d eat-star derived from example-2 of the ball_string system'

        array = np.array
        settings = HylaaSettings(0.01, 2.0)
        ha = LinearHybridAutomaton('Test Automaton')
        ha.variables = ["x"]
        mode = ha.new_mode('test_mode')
        center = array([0.])
        basis_matrix = array([[1.0]], dtype=float)

        # -alpha <= 1.0  ----> -1.0 <= alpha
        # 2 * alpha <= -1   ----> alpha <= -0.5
        cur_star = Star(settings, center, basis_matrix, [\
           LinearConstraint(array([-1.]), 1.0), \
           LinearConstraint(array([2.0]), -1.0),\
           ], \
           None, mode)

        # -0.7 <= alpha <= -0.6
        new_star = Star(settings, center, basis_matrix, [\
           LinearConstraint(array([-1.]), 0.7), \
           LinearConstraint(array([1.]), -0.6), \
           ], \
           None, mode)

        cur_star.eat_star(new_star)

        # should be unchanged: -0.0 <= alpha and 2 * alpha <= 1.0
        self.assertAlmostEqual(cur_star.constraint_list[0].value, 1.0)
        self.assertAlmostEqual(cur_star.constraint_list[1].value, -1.0)

    def test_eat_star_bs3_2d(self):
        'test 2d eat-star derived from example 3 with the ball_string system'

        array = np.array
        mode = make_debug_mode()
        settings = HylaaSettings(0.01, 2.0)
        center = array([0., 0.])

        bm = array([[1.0, 1.0], [0., 1.0]])

        cur_star = Star(settings, center, bm, [\
           LinearConstraint(array([-1., 0.]), 1.), \
           LinearConstraint(array([1., 0.]), 0.),\
           LinearConstraint(array([0., -1.]), 0.), \
           LinearConstraint(array([0., 1.]), 1.), \
           ], \
           None, mode)

        new_star = Star(settings, center, bm, [\
           LinearConstraint(array([-1., 0.]), 1.), \
           LinearConstraint(array([1., 0.]), 0.), \
           LinearConstraint(array([0., -1]), -2.), \
           LinearConstraint(array([0., 1.]), 3.), \
           ], \
           None, mode)

        cur_star.eat_star(new_star)

        # check that constraint #4 was expanded correctly
        self.assertAlmostEqual(cur_star.constraint_list[0].value, 1.0)
        self.assertAlmostEqual(cur_star.constraint_list[1].value, 0.0)
        self.assertAlmostEqual(cur_star.constraint_list[2].value, 0.0)
        self.assertAlmostEqual(cur_star.constraint_list[3].value, 3.0)

        new_point = new_star.get_feasible_point()
        self.assertTrue(cur_star.contains_point(new_point))

    def test_eat_tricky2(self):
        'test 2d eat-star derived from a tricky example with the ball_string system'

        array = np.array
        mode = make_debug_mode()
        settings = HylaaSettings(0.01, 2.0)
        center = array([0., 0.])

        basis_matrix = array([[-1., -1.], [-0.01, -0.4]])

        cur_star = Star(settings, center, basis_matrix, [\
           LinearConstraint(array([-1., 0.]), 0.5), \
           LinearConstraint(array([0., -1.]), -1.), \
           LinearConstraint(array([14., 0.1]), -5.)], \
           None, mode)

        new_star = Star(settings, center, basis_matrix, [\
           LinearConstraint(array([-1., 0.]), 0.5), \
           LinearConstraint(array([0., -1.]), -1.), \
           LinearConstraint(array([14.5, 0.15]), 0.),\
           ], \
           None, mode)

        cur_star.eat_star(new_star)

        new_point = new_star.get_feasible_point(standard_dir=[-1, 0])

        # move towards inside a little
        new_point -= 1e-6 * np.array([-1., 0.], dtype=float)

        self.assertTrue(cur_star.contains_point(new_point))

    def test_add_box_dir(self):
        'test 2d eat-star with add box directions'

        array = np.array
        mode = make_debug_mode()

        cur_star = Star(HylaaSettings(0.01, 2.0), array([0., 0.]), array([[-1.39364934, -6.33082449],\
           [-0.01283523, -0.3807174]]), [LinearConstraint(array([1., 0.]), -0.65858417090053001), \
           LinearConstraint(array([-1., 0.]), 0.7585841709005301), \
           LinearConstraint(array([0., 1.]), 2.0388429332115034), \
           LinearConstraint(array([0., -1.]), -1.8388429332115035), \
           LinearConstraint(array([6.33082449, 0.3807174]), 1.9619999999999997), \
           LinearConstraint(array([0.12748444, -0.06330825]), -0.19619999994184489), \
           LinearConstraint(array([-0.12748444, 0.06330825]), 1.1961999999418449), \
           LinearConstraint(array([1.39364934, 0.01283523]), -1.0000000000000002)], \
           None, mode, extra_init=(array([[-1.39364934, -6.33082449],\
           [-0.01283523, -0.3807174]]), 40, 0))

        new_star = Star(HylaaSettings(0.01, 2.0), array([0., 0.]), array([[-1.45695758, -6.33082449],\
           [-0.0166424, -0.3807174]]), [LinearConstraint(array([1., 0.]), -0.66180203160723172), \
           LinearConstraint(array([-1., 0.]), 0.76180203160723181), \
           LinearConstraint(array([0., 1.]), 2.3500231188180134), \
           LinearConstraint(array([0., -1.]), -2.1500231188180132), \
           LinearConstraint(array([6.33082449, 0.3807174]), 2.0600999999999994), \
           LinearConstraint(array([0.12748444, -0.06330825]), -0.21631049992724288), \
           LinearConstraint(array([-0.12748444, 0.06330825]), 1.2163104999272427), \
           LinearConstraint(array([-1.39364934, -0.01283523]), 1.0004904999853983), \
           LinearConstraint(array([1.45695758, 0.0166424]), -1.0000000000000002)], \
           None, mode, extra_init=(array([[-1.45695758, -6.33082449],\
           [-0.0166424, -0.3807174]]), 41, 0))

        # add box constraints
        for dim in xrange(cur_star.num_dims):
            vector = np.array([1.0 if d == dim else 0.0 for d in xrange(cur_star.num_dims)], dtype=float)
            cur_star.add_std_constraint_direction(vector)
            cur_star.add_std_constraint_direction(-1 * vector)

        cur_star.eat_star(new_star)

        new_point = new_star.get_feasible_point(standard_dir=[5, -1])
        self.assertTrue(cur_star.contains_point(new_point))

if __name__ == '__main__':
    unittest.main()
