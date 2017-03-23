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

def make_settings(step_time):
    'make simulation settings object for testing'

    sett = SimulationSettings(step_time)
    sett.stdout = False
    sett.threads = 1

    return sett

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

if __name__ == '__main__':
    unittest.main()
