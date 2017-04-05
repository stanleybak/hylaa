'''
Hylaa Regression Tests. These are a series of tests which demonstrate bugs, and used to verify their fixes.

Stanley Bak
October 2016
'''

import unittest
#import matplotlib.pyplot as plt

from hylaa.plotutil import PlotSettings
from hylaa.engine import HylaaEngine
from hylaa.timerutil import Timers

from models import ball_string, drivetrain, sync_motor

class TestRegression(unittest.TestCase):
    'Regression tests'

    def setUp(self):
        'setup function'
        Timers.reset()

    def test_sync_motor(self):
        '''test sync motor with input system'''

        model = sync_motor
        ha = model.define_ha()

        init_list = model.define_init_states(ha)
        settings = model.define_settings()

        engine = HylaaEngine(ha, settings)
        engine.run(init_list)

        self.assertTrue(engine.reached_error)

    def test_violation_extends_axis(self):
        '''invariant limits should also extend axis draw limits'''

        ha = ball_string.define_ha()
        init_list = ball_string.define_init_states(ha)
        settings = ball_string.define_settings()
        plot_settings = settings.plot

        plot_settings.plot_mode = PlotSettings.PLOT_INTERACTIVE
        plot_settings.skip_frames = 21
        plot_settings.skip_show_gui = True
        plot_settings.num_angles = 256
        settings.print_output = False

        engine = HylaaEngine(ha, settings)
        engine.run(init_list)

        self.assertTrue(engine.plotman.drawn_limits.xmax > 0)

    def test_drivetrain(self):
        'test running a single step on the drivetrain model (zonotope initialization)'

        ha = drivetrain.define_ha()
        init_list = drivetrain.define_init_states(ha)

        self.assertEquals(len(init_list), 1)

        settings = drivetrain.define_settings()
        plot_settings = settings.plot

        settings.print_output = False
        plot_settings.skip_frames = 10 # run 10 frames
        plot_settings.skip_show_gui = True

        engine = HylaaEngine(ha, settings)
        engine.run(init_list)

if __name__ == '__main__':
    unittest.main()







