'''
Unit tests for Hylass's engine.py
Stanley Bak
September 2016
'''

import unittest
import math
import numpy as np

from hylaa.hybrid_automaton import HyperRectangle, LinearHybridAutomaton
from hylaa.engine import HylaaEngine, HylaaSettings
from hylaa.plotutil import PlotSettings
from hylaa.timerutil import Timers

class TestEngine(unittest.TestCase):
    'Unit tests for hylaa engine'

    def setUp(self):
        'setup function'
        Timers.reset()

    def test_rectangular(self):
        '''test integration of x' = 1, y' = 2'''

        ha = LinearHybridAutomaton('Harmonic Oscillator')
        ha.variables = ["x", "y"]

        # x' = x
        loc1 = ha.new_mode('loc')
        loc1.a_matrix = np.array([[0, 0], [0, 0]])
        loc1.c_vector = np.array([1, 2])

        # x(0) = 1, y(0) = 2
        init_list = [(ha.modes['loc'], HyperRectangle([(0.99, 1.01), (1.99, 2.01)]))]

        plot_settings = PlotSettings()
        plot_settings.plot_mode = PlotSettings.PLOT_NONE
        settings = HylaaSettings(step=0.1, max_time=1.1, plot_settings=plot_settings)
        settings.print_output = False

        engine = HylaaEngine(ha, settings)

        engine.load_waiting_list(init_list)
        # x(t) = 1 + t; y(t) = 2 + 2*t

        # pop from waiting_list (doesn't advance state)
        engine.do_step()

        for i in xrange(10):
            engine.do_step()

            t = 0.1 * (i+1)
            star = engine.cur_state
            point = [1 + t, 2 + 2 * t]

            self.assertTrue(star.contains_point(point))

    def test_exp(self):
        '''test integration of x' = x'''

        ha = LinearHybridAutomaton('Harmonic Oscillator')
        ha.variables = ["x"]

        # x' = x
        a_matrix = np.array([[1]], dtype=float)
        c_vector = np.array([0], dtype=float)

        loc1 = ha.new_mode('loc')
        loc1.set_dynamics(a_matrix, c_vector)

        # x(0) = 1
        init_list = [(ha.modes['loc'], HyperRectangle([(0.99, 1.01)]))]

        plot_settings = PlotSettings()
        plot_settings.plot_mode = PlotSettings.PLOT_NONE
        settings = HylaaSettings(step=0.1, max_time=1.1, plot_settings=plot_settings)
        settings.print_output = False

        engine = HylaaEngine(ha, settings)

        engine.load_waiting_list(init_list)

        # pop from waiting_list (doesn't advance state)
        engine.do_step()

        # x(t) should be e^t
        for i in xrange(10):
            engine.do_step()

            t = 0.1 * (i+1)
            star = engine.cur_state

            self.assertTrue(star.contains_point([math.exp(t)]))

    def test_exp_plus_one(self):
        '''test integration of x' = x + 1; x(t) should be 2*e^t - 1'''

        ha = LinearHybridAutomaton('Harmonic Oscillator')
        ha.variables = ["x"]

        # x' = x + 1
        loc1 = ha.new_mode('loc')
        loc1.a_matrix = np.array([[1]])
        loc1.c_vector = np.array([1])

        # x(0) = 1
        init_list = [(ha.modes['loc'], HyperRectangle([(0.99, 1.01)]))]

        plot_settings = PlotSettings()
        plot_settings.plot_mode = PlotSettings.PLOT_NONE
        settings = HylaaSettings(step=0.1, max_time=1.1, plot_settings=plot_settings)
        settings.print_output = False

        engine = HylaaEngine(ha, settings)

        engine.load_waiting_list(init_list)

        # pop from waiting_list (doesn't advance state)
        engine.do_step()

        # x(t) should be 2*e^t - 1
        for i in xrange(10):
            engine.do_step()

            t = 0.1 * (i+1)
            star = engine.cur_state

            self.assertTrue(star.contains_point([2 * math.exp(t) - 1]))

if __name__ == '__main__':
    unittest.main()
