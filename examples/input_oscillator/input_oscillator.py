'''
Harmonic oscialltor with inputs, with dynamics:

x' == y + [-0.5, 0.5]
y' == -x + [-0.5, 0.5]

from (-5, 0) for reach_time = pi
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('loc1')
    a_matrix = np.array([[0, 1], [-1, 0]], dtype=float)
    c_vector = np.array([1.0, 0.0], dtype=float)
    loc1.set_dynamics(a_matrix, c_vector)

    u_constraints_a = np.array([[1], [-1]], dtype=float)
    u_constraints_b = np.array([0.5, 0.5], dtype=float)
    b_matrix = np.array([[1.0], [0.0]], dtype=float)
    loc1.set_inputs(u_constraints_a, u_constraints_b, b_matrix)

    error_loc = ha.new_mode('error')
    error_loc.is_error = True

    trans = ha.new_transition(loc1, error_loc)
    trans.condition_list.append(LinearConstraint([-1, 0], -6.0)) # x >= 6.0

    return ha

def define_init_states(ha):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []

    r = HyperRectangle([(-5, -5), (0, 0)])
    #r = HyperRectangle([(0, 0), (0, 0)])
    rv.append((ha.modes['loc1'], r))

    return rv

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_FULL
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    #plot_settings.make_video('harmonic_osc_inputs.avi', 45, 50)
    #return HylaaSettings(step=0.001, max_time=6, plot_settings=plot_settings)
    settings = HylaaSettings(step=0.5, max_time=3.5, plot_settings=plot_settings)

    #settings = HylaaSettings(step=0.001, max_time=5, plot_settings=plot_settings)
    # 5.5 secs (5.1 in minimize 2600 calls)
    # optimized: 0.46 secs; Guard_opt_data.update_from_sim Time (2522 calls): 0.30 sec (65.5%)
    
    settings.print_step_times = False
    settings.opt_decompose_lp = False
    settings.opt_warm_start_lp = False
    
    settings.simulation.sparse=True
    
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
