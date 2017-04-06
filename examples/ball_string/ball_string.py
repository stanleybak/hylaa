'''
The first two modes of Goran's ball-string example. This is a simple model demonstrating guards and transitions.
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "v"]

    extension = ha.new_mode('extension')
    extension.a_matrix = np.array([[0.0, 1.0], [-100.0, -4.0]], dtype=float)
    extension.c_vector = np.array([0.0, -9.81], dtype=float)
    extension.inv_list.append(LinearConstraint([1.0, 0.0], 0)) # x <= 0

    freefall = ha.new_mode('freefall')
    freefall.a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    freefall.c_vector = np.array([0.0, -9.81], dtype=float)
    freefall.inv_list.append(LinearConstraint([-1.0, 0.0], 0.0)) # 0 <= x
    freefall.inv_list.append(LinearConstraint([1.0, 0.0], 1.0)) # x <= 1

    trans = ha.new_transition(extension, freefall)
    trans.condition_list.append(LinearConstraint([-0.0, -1.0], -0.0)) # v >= 0

    return ha

def define_init_states(ha):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = []

    rv.append((ha.modes['extension'], HyperRectangle([(-1.05, -0.95), (-0.1, 0.1)])))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1
    #plot_settings.label.axes_limits = (0.9, 1.1, 2.0, 6.0)

    settings = HylaaSettings(step=0.01, max_time=2.0, plot_settings=plot_settings)
    
    return settings

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

