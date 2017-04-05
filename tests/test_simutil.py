'''
Unit tests for simutil (exact step version).
'''

import unittest
import math

import numpy as np
from numpy.testing import assert_array_almost_equal
from hylaa.containers import SimulationSettings

from hylaa.timerutil import Timers
from hylaa.simutil import SimulationBundle
from hylaa.openblas import OpenBlasThreads

def make_settings(step_time):
    'make simulation settings object for testing'

    sett = SimulationSettings(step_time)
    sett.stdout = False
    sett.threads = 1

    return sett

def compute_gbt_series(a_matrix, b_matrix, step_size, tol=1e-12, max_val=1e100, max_terms=2500, stdout=False):
    '''
    compute the transpose of G(A, h) * B, where G(A, h) is computed using the series expansion:

    G(A,h) = (1/1!)*I*h + (1/2!)*A*h^2 + (1/3!)*A*A*h^3 + ...

    This has issues if A^n gets too large (floating-point error)
    '''

    num_vars = a_matrix.shape[0]

    h = float(step_size) # h
    ah_matrix = np.array(a_matrix, dtype=float) * h
    term = np.array(np.identity(num_vars), dtype=float) * h

    g_sum = term # (1/1!)*I*h
    term_num = 2

    while True:
        term = np.dot(term, ah_matrix) # h * (h*A)^{term_num - 1}
        term /= term_num # 1 / (term_num)!

        delta = max(abs(term.max()), abs(term.min()))
        g_sum += term

        if stdout:
            print "Computing G(A, h) - Term #{}, delta = {}".format(term_num, delta)

        if delta > max_val:
            raise RuntimeError(('Computing G(A, h) * U exceeded limit value of {} after ' + \
                '{} terms (value = {}). Consider reducing time step.').format(max_val, term_num, delta))

        if delta < tol:
            break

        if term_num >= max_terms:
            raise RuntimeError(('Computing G(A, h) * U did not converge within a tolerance of {} after ' + \
                '{} terms. Delta of last term was {}').format(tol, max_terms, delta))

        term_num += 1

    if stdout:
        print "G(A, g) matrix = {}".format(g_sum)
        print "G(A, g) * B = {}".format(np.dot(g_sum, b_matrix))

    return np.array(np.dot(g_sum, b_matrix).transpose(), dtype=float)

class TestSimUtil(unittest.TestCase):
    'Unit tests object'

    def setUp(self):
        Timers.reset()

    def test_sim_ha(self):
        '''test simulating the harmonic oscillator'''

        # x' = y,   y' = -x
        a_matrix = [[0.0, 1.0], [-1.0, 0.0]]
        b_vector = [0.0, 0.0]
        step_time = 0.1
        max_steps = 10

        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))
        vals = None

        for s in xrange(max_steps+1):

            vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertAlmostEquals(center[0], 0)
            self.assertAlmostEquals(center[1], 0)

            self.assertTrue(isinstance(vals, np.ndarray))

            self.assertEquals(vals.shape[0], 2)
            self.assertEquals(vals.shape[1], 2)

            t = step_time * s

            vec1 = np.array([math.cos(t), math.sin(t)], dtype=float)
            vec2 = np.array([-math.sin(t), math.cos(t)], dtype=float)

            # result should be transpose of basis vectors
            self.assertAlmostEqual(vals[0][0], vec1[0])
            self.assertAlmostEqual(vals[1][0], vec1[1])

            self.assertAlmostEqual(vals[0][1], vec2[0])
            self.assertAlmostEqual(vals[1][1], vec2[1])

    def test_sim_ha_parallel(self):
        '''test simulating the harmonic oscillator in a multithreaded fashion'''

        # x' = y,   y' = -x
        a_matrix = [[0.0, 1.0], [-1.0, 0.0]]
        b_vector = [0.0, 0.0]
        step_time = 0.1
        max_steps = 10

        sett = make_settings(step_time)
        sett.threads = 2
        bundle = SimulationBundle(a_matrix, b_vector, sett)
        vals = None

        for s in xrange(max_steps+1):

            vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertAlmostEquals(center[0], 0)
            self.assertAlmostEquals(center[1], 0)

            self.assertTrue(isinstance(vals, np.ndarray))

            self.assertEquals(vals.shape[0], 2)
            self.assertEquals(vals.shape[1], 2)

            t = step_time * s

            vec1 = np.array([math.cos(t), math.sin(t)], dtype=float)
            vec2 = np.array([-math.sin(t), math.cos(t)], dtype=float)

            # result should be transpose of basis vectors
            self.assertAlmostEqual(vals[0][0], vec1[0])
            self.assertAlmostEqual(vals[1][0], vec1[1])

            self.assertAlmostEqual(vals[0][1], vec2[0])
            self.assertAlmostEqual(vals[1][1], vec2[1])

    def test_sim_timer(self):
        '''test simulating a timer'''

        # t' = 1,   y' = -x
        a_matrix = [[0.0]]
        b_vector = [1.0]
        step_time = 0.1
        max_steps = 5

        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))
        vals = None

        for s in xrange(max_steps+1):

            vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertAlmostEquals(center[0], s * step_time)

            self.assertAlmostEqual(vals[0][0], 1.0)

    def test_sim_exact(self):
        '''test simutil result'''

        # x' = x,   y' = 1
        a_matrix = [[1.0, 0.0], [0.0, 0.0]]
        b_vector = [0.0, 1.0]
        step_time = 0.25
        max_steps = 5

        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))
        vals = None

        for s in xrange(max_steps+1):

            vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertAlmostEquals(center[1], s * 0.25)

            self.assertTrue(isinstance(vals, np.ndarray))

            self.assertEquals(vals.shape[0], 2)
            self.assertEquals(vals.shape[1], 2)

            t = step_time * s
            assert_array_almost_equal(vals[0], np.array([math.exp(t), 0.]))
            assert_array_almost_equal(vals[1], np.array([0., 1.]))

        # x(t) = e^t
        # y(t) = t

    def test_with_presimulation(self):
        '''test simutil with presimulation'''

        # x' = x,   y' = 1
        a_matrix = [[1.0, 0.0], [0.0, 0.0]]
        b_vector = [0.0, 1.0]
        step_time = 0.25
        max_steps = 5

        origin = np.array([1.0, 1.5])
        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))
        vals = None

        # presimulation logic
        origin_sim = bundle.sim_until_inv_violated(origin, [], max_steps)

        assert len(origin_sim) == max_steps + 1

        num_presimulation_steps = len(origin_sim)

        if num_presimulation_steps > max_steps:
            num_presimulation_steps = max_steps

        bundle.presimulate(num_presimulation_steps)

        # simulate past the presimulation number of steps, which will force a re-simulation
        for s in xrange(max_steps+2):
            vals, _ = bundle.get_vecs_origin_at_step(s, max_steps+2)

            self.assertTrue(isinstance(vals, np.ndarray))

            self.assertEquals(vals.shape[0], 2)
            self.assertEquals(vals.shape[1], 2)

            t = step_time * s
            assert_array_almost_equal(vals[0], np.array([math.exp(t), 0.]))
            assert_array_almost_equal(vals[1], np.array([0., 1.]))

        # x(t) = e^t
        # y(t) = t

    def test_sim_methods(self):
        '''test the simulation approach vs exmp approach vs one-step approach'''

        # harmonic oscialltor
        # x' = y,   y' = -x
        a_matrix = [[0.0, 1.0], [-1.0, 0.0]]
        b_vector = [0.0, 0.0]
        step_time = 0.1
        max_steps = 10

        sim_bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))

        expm_settings = make_settings(step_time)
        expm_settings.sim_mode = SimulationSettings.MATRIX_EXP
        expm_bundle = SimulationBundle(a_matrix, b_vector, expm_settings)

        vals = None

        for s in xrange(max_steps+1):

            vals_sim, center_sim = sim_bundle.get_vecs_origin_at_step(s, max_steps)
            vals_expm, center_expm = expm_bundle.get_vecs_origin_at_step(s, max_steps)

            vals = [vals_sim, vals_expm]
            centers = [center_sim, center_expm]

            for index in xrange(len(centers[0])):
                self.assertAlmostEquals(centers[0][index], centers[1][index], places=5)

            self.assertEquals(vals[0].shape[0], vals[1].shape[0])

            # result should be transpose of basis vectors
            for x in xrange(2):
                for y in xrange(2):
                    self.assertAlmostEqual(vals[0][x][y], vals[1][x][y], places=5)

    def test_sim_step_one(self):
        '''test simulating the harmonic oscillator requesting step 1 first'''

        # x' = y,   y' = -x
        a_matrix = [[0.0, 1.0], [-1.0, 0.0]]
        b_vector = [0.0, 0.0]
        step_time = 0.1
        max_steps = 10

        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))
        vals = None

        # start at step 1
        for s in xrange(1, max_steps+1):

            vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertAlmostEquals(center[0], 0)
            self.assertAlmostEquals(center[1], 0)

            self.assertTrue(isinstance(vals, np.ndarray))

            self.assertEquals(vals.shape[0], 2)
            self.assertEquals(vals.shape[1], 2)

            t = step_time * s

            vec1 = np.array([math.cos(t), math.sin(t)], dtype=float)
            vec2 = np.array([-math.sin(t), math.cos(t)], dtype=float)

            # result should be transpose of basis vectors
            self.assertAlmostEqual(vals[0][0], vec1[0])
            self.assertAlmostEqual(vals[1][0], vec1[1])

            self.assertAlmostEqual(vals[0][1], vec2[0])
            self.assertAlmostEqual(vals[1][1], vec2[1])

    def test_input_sim(self):
        '''test the computation of G(A, h) * B using simulations'''

        # x' = y + u1 + 2*u3,   y' = -x + 0.5*u2 + u3
        a_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
        c_vector = [1.0, 0.0]
        b_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 0.5, 1.0]], dtype=float)
        step_time = 0.1

        bundle = SimulationBundle(a_matrix, c_vector, make_settings(step_time))

        # make sure it matches the series computation
        series_result = compute_gbt_series(a_matrix, b_matrix, step_time)
        sim_result = bundle.compute_gbt(b_matrix)

        self.assertEquals(sim_result.shape[0], series_result.shape[0])
        self.assertEquals(sim_result.shape[1], series_result.shape[1])

        assert_array_almost_equal(sim_result, series_result)

    def test_sim_types_shapes(self):
        '''check that the types and shapes returned by get_vecs_origin_at_step are correct'''

        # harmonic oscillator: x' = y,   y' = -x
        a_matrix = [[0.0, 1.0], [-1.0, 0.0]]
        b_vector = [0.0, 0.0]
        step_time = 0.1
        max_steps = 10

        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))

        for s in xrange(max_steps+1):
            vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertTrue(isinstance(vals, np.ndarray))
            self.assertTrue(isinstance(center, np.ndarray))

            self.assertTrue(vals.shape == (2, 2))
            self.assertTrue(center.shape == (2,))

    def test_sim_ha_restart(self):
        '''test simulating the harmonic oscillator, with a restart from step 0 half way'''

        # x' = y,   y' = -x
        a_matrix = [[0.0, 1.0], [-1.0, 0.0]]
        b_vector = [0.0, 0.0]
        step_time = 0.1
        max_steps = 10

        bundle = SimulationBundle(a_matrix, b_vector, make_settings(step_time))
        vals = None

        for _ in xrange(2):
            for s in xrange(1, max_steps+1):

                vals, center = bundle.get_vecs_origin_at_step(s, max_steps)

                self.assertAlmostEquals(center[0], 0)
                self.assertAlmostEquals(center[1], 0)

                self.assertTrue(isinstance(vals, np.ndarray))

                self.assertEquals(vals.shape[0], 2)
                self.assertEquals(vals.shape[1], 2)

                t = step_time * s

                vec1 = np.array([math.cos(t), math.sin(t)], dtype=float)
                vec2 = np.array([-math.sin(t), math.cos(t)], dtype=float)

                # result should be transpose of basis vectors
                self.assertAlmostEqual(vals[0][0], vec1[0])
                self.assertAlmostEqual(vals[1][0], vec1[1])

                self.assertAlmostEqual(vals[0][1], vec2[0])
                self.assertAlmostEqual(vals[1][1], vec2[1])
                
    def test_openblas(self):
        'test openblas context object'
        
        size = 100
        a_mat = np.random.rand(size, size)
        b_mat = np.random.rand(size, size)
        
        prod1 = np.dot(a_mat, b_mat)

        with OpenBlasThreads(1):
            
            prod2 = np.dot(a_mat, b_mat)
        
        for y in xrange(size):
            for x in xrange(size):
                self.assertAlmostEquals(prod1[y,x], prod2[y,x])
        
        
if __name__ == '__main__':
    unittest.main()
