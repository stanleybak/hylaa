'''
Created by Hyst v1.3
Hybrid Automaton in Hylaa
Converted from file: model.xml
Command Line arguments: null
'''
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "t"]

    # input variable order: [u1, u2]

    Model = ha.new_mode('Model')
    a_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, -1.0865, 8487.2, 0, 0, 0, 0, 0, 0], [-2592.1, -21.119, -698.91, -141399, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1.0865, 8487.2, 0, 0], [0, 0, 0, 0, -2592.1, -21.119, -698.91, -141399, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    Model.set_dynamics(a_matrix, c_vector)
    
    # u1 >= 0.16
    # u1 <= 0.3
    # u2 >= 0.2
    # u2 <= 0.4
    u_constraints_a = np.array([[-1, -0], [1, 0], [-0, -1], [0, 1]], dtype=float)
    u_constraints_b = np.array([-0.16, 0.3, -0.2, 0.4], dtype=float)
    b_matrix = np.array([[0, 0], [0, 0], [0, 0], [-1, 0], [0, 0], [0, 0], [0, 0], [0, -1], [0, 0]], dtype=float)
    Model.set_inputs(u_constraints_a, u_constraints_b, b_matrix)

    _error = ha.new_mode('_error')
    _error.is_error = True

    trans = ha.new_transition(Model, _error)
    trans.condition_list.append(LinearConstraint([0, 0, 0, 0, -1, 0, 0, 0, 0], -0.001501)) # 0.45 <= x5

    return ha

def define_init_states(ha):
    '''returns a list of (mode, list(LinearConstraint])'''
    # Variable ordering: [x1, x2, x3, x4, x5, x6, x7, x8, t]
    rv = []
    
    constraints = []
    constraints.append(LinearConstraint([-1, -0, -0, -0, -0, -0, -0, -0, -0], -0.002)) # x1 >= 0.002
    constraints.append(LinearConstraint([1, 0, 0, 0, 0, 0, 0, 0, 0], 0.0025)) # x1 <= 0.0025
    constraints.append(LinearConstraint([-0, -1, -0, -0, -0, -0, -0, -0, -0], -0)) # x2 >= 0.0
    constraints.append(LinearConstraint([0, 1, 0, 0, 0, 0, 0, 0, 0], 0)) # x2 <= 0.0
    constraints.append(LinearConstraint([-0, -0, -1, -0, -0, -0, -0, -0, -0], -0)) # x3 >= 0.0
    constraints.append(LinearConstraint([0, 0, 1, 0, 0, 0, 0, 0, 0], 0)) # x3 <= 0.0
    constraints.append(LinearConstraint([-0, -0, -0, -1, -0, -0, -0, -0, -0], -0)) # x4 >= 0.0
    constraints.append(LinearConstraint([0, 0, 0, 1, 0, 0, 0, 0, 0], 0)) # x4 <= 0.0
    constraints.append(LinearConstraint([-0, -0, -0, -0, -1, -0, -0, -0, -0], -0.001)) # x5 >= 0.001
    constraints.append(LinearConstraint([0, 0, 0, 0, 1, 0, 0, 0, 0], 0.0015)) # x5 <= 0.0015
    constraints.append(LinearConstraint([-0, -0, -0, -0, -0, -1, -0, -0, -0], -0)) # x6 >= 0.0
    constraints.append(LinearConstraint([0, 0, 0, 0, 0, 1, 0, 0, 0], 0)) # x6 <= 0.0
    constraints.append(LinearConstraint([-0, -0, -0, -0, -0, -0, -1, -0, -0], -0)) # x7 >= 0.0
    constraints.append(LinearConstraint([0, 0, 0, 0, 0, 0, 1, 0, 0], 0)) # x7 <= 0.0
    constraints.append(LinearConstraint([-0, -0, -0, -0, -0, -0, -0, -1, -0], -0)) # x8 >= 0.0
    constraints.append(LinearConstraint([0, 0, 0, 0, 0, 0, 0, 1, 0], 0)) # x8 <= 0.0
    constraints.append(LinearConstraint([0, 0, 0, 0, 0, 0, 0, 0, 1], 0)) # t = 0.0
    constraints.append(LinearConstraint([-0, -0, -0, -0, -0, -0, -0, -0, -1], -0)) # t = 0.0
    rv.append((ha.modes['Model'], constraints))
    
    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE
    plot_settings.xdim = 0
    plot_settings.ydim = 4

    settings = HylaaSettings(step=0.001, max_time=0.001, plot_settings=plot_settings)
    settings.print_output = False
    settings.counter_example_filename = None

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



