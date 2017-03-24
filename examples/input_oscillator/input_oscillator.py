'''
Created by Hyst v1.3
Hybrid Automaton in Hylaa
Converted from file: /home/stan/Desktop/repositories/hylaa/examples/input_oscillator/osc_spaceex.xml
Command Line arguments: -tool hylaa "" -output /home/stan/Desktop/repositories/hylaa/examples/input_oscillator/osc_spaceex_converted.py -input /home/stan/Desktop/repositories/hylaa/examples/input_oscillator/osc_spaceex.xml /home/stan/Desktop/repositories/hylaa/examples/input_oscillator/osc_spaceex.cfg
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y"]

    # input variable order: [u1, u2]

    loc1 = ha.new_mode('loc1')
    a_matrix = np.array([ \
        [0, 1], \
        [-1, 0], \
        ], dtype=float)
    c_vector = np.array([0, 0], dtype=float)
    loc1.set_dynamics(a_matrix, c_vector)
    
    # -0.5 <= u1
    # u1 <= 0.5
    # -0.5 <= u2
    # u2 <= 0.5
    u_constraints_a = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    u_constraints_b = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
    b_matrix = np.array([[1, 0], [0, 1]], dtype=float)
    loc1.set_inputs(u_constraints_a, u_constraints_b, b_matrix)

    return ha

def define_init_states(ha):
    '''returns a list of (mode, list(LinearConstraint])'''
    # Variable ordering: [x, y]
    rv = []
    
    constraints = []
    constraints.append(LinearConstraint([-1, 0], 5)) # -5.0 <= x
    constraints.append(LinearConstraint([1, 0], -5)) # x <= -5.0
    constraints.append(LinearConstraint([0, -1], 0)) # 0.0 <= y
    constraints.append(LinearConstraint([0, 1], 0)) # y <= 0.0
    rv.append((ha.modes['loc1'], constraints))
    
    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.1, max_time=3.2, plot_settings=plot_settings)

    return settings

def run_hylaa(settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha = define_ha()
    init = define_init_states(ha)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(define_settings())

