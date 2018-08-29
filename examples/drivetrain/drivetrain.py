'''
Created by Hyst v1.5
Hybrid Automaton in Hylaa2
Converted from file: 
Command Line arguments: -gen drivetrain "-theta 2 -init_scale 1.0 -reverse_errors" -passes sub_constants "" simplify -p -tool hylaa2 ""
'''

import heapq

import numpy as np

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings, AggregationSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = HybridAutomaton('Drivetrain')

    # dynamics variable order: [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t, affine]

    negAngle = ha.new_mode('negAngle')
    a_matrix = [ \
        [0, 0, 0, 0, 0, 0, 0.0833333333333333, 0, -1, 0, 0, 0, 0], \
        [13828.8888888889, -26.6666666666667, 60, 60, 0, 0, -5, -60, 0, 0, 0, 0, 716.666666666667], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5], \
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, -714.285714285714, -0.04, 0, 0, 0, 714.285714285714, 0, 0, 0], \
        [-2777.77777777778, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -83.3333333333333], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [100, 0, 0, 0, 0, 0, 0, -1000, -0.01, 1000, 0, 0, 3], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 1000, 0, 0, 1000, 0, -2000, -0.01, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        ]
    negAngle.set_dynamics(a_matrix)
    # x1 <= -0.03
    negAngle.set_invariant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], [-0.03, ])

    deadzone = ha.new_mode('deadzone')
    a_matrix = [ \
        [0, 0, 0, 0, 0, 0, 0.0833333333333333, 0, -1, 0, 0, 0, 0], \
        [-60, -26.6666666666667, 60, 60, 0, 0, -5, -60, 0, 0, 0, 0, 300], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5], \
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, -714.285714285714, -0.04, 0, 0, 0, 714.285714285714, 0, 0, 0], \
        [0, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, -1000, -0.01, 1000, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 1000, 0, 0, 1000, 0, -2000, -0.01, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        ]
    deadzone.set_dynamics(a_matrix)
    # -0.03 <= x1 & x1 <= 0.03
    deadzone.set_invariant([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], [0.03, 0.03, ])

    posAngle = ha.new_mode('posAngle')
    a_matrix = [ \
        [0, 0, 0, 0, 0, 0, 0.0833333333333333, 0, -1, 0, 0, 0, 0], \
        [13828.8888888889, -26.6666666666667, 60, 60, 0, 0, -5, -60, 0, 0, 0, 0, -116.666666666667], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5], \
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, -714.285714285714, -0.04, 0, 0, 0, 714.285714285714, 0, 0, 0], \
        [-2777.77777777778, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83.3333333333333], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [100, 0, 0, 0, 0, 0, 0, -1000, -0.01, 1000, 0, 0, -3], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 1000, 0, 0, 1000, 0, -2000, -0.01, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        ]
    posAngle.set_dynamics(a_matrix)
    # 0.03 <= x1
    posAngle.set_invariant([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], [-0.03, ])

    negAngleInit = ha.new_mode('negAngleInit')
    a_matrix = [ \
        [0, 0, 0, 0, 0, 0, 0.0833333333333333, 0, -1, 0, 0, 0, 0], \
        [13828.8888888889, -26.6666666666667, 60, 60, 0, 0, -5, -60, 0, 0, 0, 0, 116.666666666667], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5], \
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, -714.285714285714, -0.04, 0, 0, 0, 714.285714285714, 0, 0, 0], \
        [-2777.77777777778, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -83.3333333333333], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [100, 0, 0, 0, 0, 0, 0, -1000, -0.01, 1000, 0, 0, 3], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 1000, 0, 0, 1000, 0, -2000, -0.01, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        ]
    negAngleInit.set_dynamics(a_matrix)
    # t <= 0.2
    negAngleInit.set_invariant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], ], [0.2, ])

    error = ha.new_mode('error')

    trans = ha.new_transition(negAngleInit, negAngle)
    # t >= 0.2
    trans.set_guard([[-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -1, -0], ], [-0.2, ])

    trans = ha.new_transition(negAngle, deadzone)
    # x1 >= -0.03
    trans.set_guard([[-1, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0], ], [0.03, ])

    trans = ha.new_transition(deadzone, posAngle)
    # x1 >= 0.03
    trans.set_guard([[-1, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0], ], [-0.03, ])

    trans = ha.new_transition(deadzone, error)
    # x1 <= -0.03
    trans.set_guard([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], [-0.03, ])

    trans = ha.new_transition(posAngle, error)
    # x1 <= 0.03
    trans.set_guard([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], [0.03, ])

    return ha

def define_init_states(ha):
    '''returns a list of StateSet objects'''
    # Variable ordering: [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t, affine]
    rv = []
    
    # 1.0 * x1 + 0.0488 = 0.00056 * x4 - 0.0112 & 1.0 * x2 + 15.67 = 0.46699999999999997 * x4 - 9.34 & x3 = 0.0 & x5 = 0.0 & 1.0 * x6 - 20.0 = 1.0 * x4 - 20.0 & 1.0 * x7 - 240.0 = 12.0 * x4 - 240.0 & 1.0 * x8 + 0.0019199999999999998 = 0.00006 * x4 - 0.0012 & 1.0 * x9 - 20.0 = 1.0 * x4 - 20.0 & 1.0 * x10 + 0.0019199999999999998 = 0.00006 * x4 - 0.0012 & 1.0 * x11 - 20.0 = 1.0 * x4 - 20.0 & 20.0 <= x4 & x4 <= 40.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['negAngleInit']
    mat = [[1, 0, 0, -0.00056, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [-1, -0, -0, 0.00056, -0, -0, -0, -0, -0, -0, -0, -0, -0], \
        [0, 1, 0, -0.46699999999999997, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [-0, -1, -0, 0.46699999999999997, -0, -0, -0, -0, -0, -0, -0, -0, -0], \
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [-0, -0, -1, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0], \
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], \
        [-0, -0, -0, -0, -1, -0, -0, -0, -0, -0, -0, -0, -0], \
        [0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [-0, -0, -0, 1, -0, -1, -0, -0, -0, -0, -0, -0, -0], \
        [0, 0, 0, -12, 0, 0, 1, 0, 0, 0, 0, 0, 0], \
        [-0, -0, -0, 12, -0, -0, -1, -0, -0, -0, -0, -0, -0], \
        [0, 0, 0, -0.00006, 0, 0, 0, 1, 0, 0, 0, 0, 0], \
        [-0, -0, -0, 0.00006, -0, -0, -0, -1, -0, -0, -0, -0, -0], \
        [0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [-0, -0, -0, 1, -0, -0, -0, -0, -1, -0, -0, -0, -0], \
        [0, 0, 0, -0.00006, 0, 0, 0, 0, 0, 1, 0, 0, 0], \
        [-0, -0, -0, 0.00006, -0, -0, -0, -0, -0, -1, -0, -0, -0], \
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [-0, -0, -0, 1, -0, -0, -0, -0, -0, -0, -1, -0, -0], \
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], \
        [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -1, -0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
        [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -1], ]
    rhs = [-0.060000000000000005, 0.060000000000000005, -25.009999999999998, 25.009999999999998, 0, -0, 0, -0, 0, -0, 0, -0, -0.0031199999999999995, 0.0031199999999999995, 0, -0, -0.0031199999999999995, 0.0031199999999999995, 0, -0, -20, 40, 0, -0, 1, -1, ]
    rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    return rv


def define_settings():
    '''get the hylaa settings object
    see hylaa/settings.py for a complete list of reachability settings'''

    # step_size = 5.0E-4, max_time = 2.0
    settings = HylaaSettings(5.0E-4, 2.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_INTERACTIVE
    settings.plot.xdim_dir = [0, None, None]

    #x0_dir = np.array([0, 0, 0, 0, 0, 0, 0.0833333333333333, 0, -1, 0, 0, 0, 0], dtype=float)
    settings.plot.ydim_dir = [2, 6, 8]

    settings.stop_on_error = False
    settings.plot.draw_stride = 10
    #settings.aggregation.agg_mode = AggregationSettings.AGG_NONE

    def custom_pop_func(waiting_list):
        'custom pop function for aggregation'

        mode = waiting_list[0].mode

        state_list = [state for state in waiting_list if state.mode == mode]

        num = 2 # performing clustering with this number of items

        return heapq.nsmallest(num, state_list, lambda s: s.cur_steps_since_start)
    

    settings.aggregation.custom_pop_func = custom_pop_func

    return settings

def run_hylaa():
    'runs hylaa, returning a HylaaResult object'
    
    ha = define_ha()
    init = define_init_states(ha)
    settings = define_settings()

    result = Core(ha, settings).run(init)

    return result

if __name__ == '__main__':
    run_hylaa()
