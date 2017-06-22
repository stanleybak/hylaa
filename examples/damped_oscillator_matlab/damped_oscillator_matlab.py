'''
Damped oscillator model, with dynamics:

x' == -0.1 * x + y
y' == -x - 0.1 * y
'''

import math
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('loc1')
    a_matrix = np.array([[-0.1, 1], [-1, -0.1]])
    c_vector = np.array([0, 0], dtype=float)
    loc1.set_dynamics(a_matrix, c_vector)
    

    return ha

def define_init_states(ha):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []

    r = HyperRectangle([(-6, -5), (0, 1)])
    rv.append((ha.modes['loc1'], r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_MATLAB
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    s = HylaaSettings(step=0.2, max_time=20.0, plot_settings=plot_settings)
    
    return s

def run_hylaa(settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha = define_ha()
    init = define_init_states(ha)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    settings = define_settings()
    run_hylaa(settings)

