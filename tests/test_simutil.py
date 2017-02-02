
'''
Unit tests for simutil (exact step version).
'''

import unittest
import math

import numpy as np
from numpy.testing import assert_array_almost_equal

from hylaa.simutil import SimulationBundle

class TestSimUtil(unittest.TestCase):
    'Unit tests for hypy'

    def test_sim_exact(self):
        '''test simutil result'''

        # x' = x,   y' = 1
        a_matrix = [[1.0, 0.0], [0.0, 0.0]]
        b_vector = [0.0, 1.0]
        step_time = 0.25
        max_steps = 5

        bundle = SimulationBundle(a_matrix, b_vector, step_time)
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
        bundle = SimulationBundle(a_matrix, b_vector, step_time)
        vals = None

        # presimulation logic
        origin_sim = bundle.sim_until_inv_violated(origin, [], max_steps)
        num_presimulation_steps = int(len(origin_sim) * 1.2)

        if num_presimulation_steps > max_steps:
            num_presimulation_steps = max_steps

        bundle.get_vecs_origin_at_step(num_presimulation_steps, max_steps)

        for s in xrange(max_steps+1):

            vals, _ = bundle.get_vecs_origin_at_step(s, max_steps)

            self.assertTrue(isinstance(vals, np.ndarray))

            self.assertEquals(vals.shape[0], 2)
            self.assertEquals(vals.shape[1], 2)

            t = step_time * s
            assert_array_almost_equal(vals[0], np.array([math.exp(t), 0.]))
            assert_array_almost_equal(vals[1], np.array([0., 1.]))

        # x(t) = e^t
        # y(t) = t

if __name__ == '__main__':
    unittest.main()






