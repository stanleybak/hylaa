'''
Created by Hyst v1.5
Hybrid Automaton in Hylaa2
Converted from file: 
Command Line arguments: -gen nav "-matrix -1.2 0.1 0.2 -1.2 -i_list 4 6 6 6 4 4 A 1 1 4 4 2 1 1 4 3 0 0 B 4 3 0 0 6 6 -width 5 -startx 3.5 -starty 3.5 -noise 0.0" -passes simplify -python -o nav_fig2b.py -tool hylaa2 ""
'''

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = HybridAutomaton()

    # dynamics variable order: [x, y, xvel, yvel, affine]

    mode_0_0 = ha.new_mode('mode_0_0')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.9192388155425117], \
        [0, 0, 0.2, -1.2, -0.9899494936611665], \
        [0, 0, 0, 0, 0], \
        ]
    mode_0_0.set_dynamics(a_matrix)
    # x <= 1.0 & y <= 1.0
    mode_0_0.set_invariant([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], ], [1, 1, ])

    mode_1_0 = ha.new_mode('mode_1_0')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -0.1], \
        [0, 0, 0.2, -1.2, 1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_1_0.set_dynamics(a_matrix)
    # x >= 1.0 & x <= 2.0 & y <= 1.0
    mode_1_0.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], ], [-1, 2, 1, ])

    mode_2_0 = ha.new_mode('mode_2_0')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -0.1], \
        [0, 0, 0.2, -1.2, 1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_2_0.set_dynamics(a_matrix)
    # x >= 2.0 & x <= 3.0 & y <= 1.0
    mode_2_0.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], ], [-2, 3, 1, ])

    mode_3_0 = ha.new_mode('mode_3_0')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -1.2], \
        [0, 0, 0.2, -1.2, 0.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_3_0.set_dynamics(a_matrix)
    # x >= 3.0 & x <= 4.0 & y <= 1.0
    mode_3_0.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], ], [-3, 4, 1, ])

    mode_4_0 = ha.new_mode('mode_4_0')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -1.2], \
        [0, 0, 0.2, -1.2, 0.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_4_0.set_dynamics(a_matrix)
    # x >= 4.0 & y <= 1.0
    mode_4_0.set_invariant([[-1, -0, -0, -0, -0], [0, 1, 0, 0, 0], ], [-4, 1, ])

    mode_0_1 = ha.new_mode('mode_0_1')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.9192388155425117], \
        [0, 0, 0.2, -1.2, -0.9899494936611665], \
        [0, 0, 0, 0, 0], \
        ]
    mode_0_1.set_dynamics(a_matrix)
    # x <= 1.0 & y >= 1.0 & y <= 2.0
    mode_0_1.set_invariant([[1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [1, -1, 2, ])

    mode_1_1 = ha.new_mode('mode_1_1')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -0.1], \
        [0, 0, 0.2, -1.2, 1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_1_1.set_dynamics(a_matrix)
    # x >= 1.0 & x <= 2.0 & y >= 1.0 & y <= 2.0
    mode_1_1.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-1, 2, -1, 2, ])

    mode_2_1 = ha.new_mode('mode_2_1')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -0.1], \
        [0, 0, 0.2, -1.2, 1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_2_1.set_dynamics(a_matrix)
    # x >= 2.0 & x <= 3.0 & y >= 1.0 & y <= 2.0
    mode_2_1.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-2, 3, -1, 2, ])

    mode_3_1 = ha.new_mode('mode_3_1')
    a_matrix = [ \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        ]
    mode_3_1.set_dynamics(a_matrix)
    # x >= 3.0 & x <= 4.0 & y >= 1.0 & y <= 2.0
    mode_3_1.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-3, 4, -1, 2, ])

    mode_4_1 = ha.new_mode('mode_4_1')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_4_1.set_dynamics(a_matrix)
    # x >= 4.0 & y >= 1.0 & y <= 2.0
    mode_4_1.set_invariant([[-1, -0, -0, -0, -0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-4, -1, 2, ])

    mode_0_2 = ha.new_mode('mode_0_2')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_0_2.set_dynamics(a_matrix)
    # x <= 1.0 & y >= 2.0 & y <= 3.0
    mode_0_2.set_invariant([[1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [1, -2, 3, ])

    mode_1_2 = ha.new_mode('mode_1_2')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 1.2], \
        [0, 0, 0.2, -1.2, -0.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_1_2.set_dynamics(a_matrix)
    # x >= 1.0 & x <= 2.0 & y >= 2.0 & y <= 3.0
    mode_1_2.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-1, 2, -2, 3, ])

    mode_2_2 = ha.new_mode('mode_2_2')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.7778174593052022], \
        [0, 0, 0.2, -1.2, 0.7071067811865476], \
        [0, 0, 0, 0, 0], \
        ]
    mode_2_2.set_dynamics(a_matrix)
    # x >= 2.0 & x <= 3.0 & y >= 2.0 & y <= 3.0
    mode_2_2.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-2, 3, -2, 3, ])

    mode_3_2 = ha.new_mode('mode_3_2')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.7778174593052022], \
        [0, 0, 0.2, -1.2, 0.7071067811865476], \
        [0, 0, 0, 0, 0], \
        ]
    mode_3_2.set_dynamics(a_matrix)
    # x >= 3.0 & x <= 4.0 & y >= 2.0 & y <= 3.0
    mode_3_2.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-3, 4, -2, 3, ])

    mode_4_2 = ha.new_mode('mode_4_2')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_4_2.set_dynamics(a_matrix)
    # x >= 4.0 & y >= 2.0 & y <= 3.0
    mode_4_2.set_invariant([[-1, -0, -0, -0, -0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-4, -2, 3, ])

    mode_0_3 = ha.new_mode('mode_0_3')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_0_3.set_dynamics(a_matrix)
    # x <= 1.0 & y >= 3.0 & y <= 4.0
    mode_0_3.set_invariant([[1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [1, -3, 4, ])

    mode_1_3 = ha.new_mode('mode_1_3')
    a_matrix = [ \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], \
        ]
    mode_1_3.set_dynamics(a_matrix)
    # x >= 1.0 & x <= 2.0 & y >= 3.0 & y <= 4.0
    mode_1_3.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-1, 2, -3, 4, ])

    mode_2_3 = ha.new_mode('mode_2_3')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.7778174593052022], \
        [0, 0, 0.2, -1.2, 0.7071067811865476], \
        [0, 0, 0, 0, 0], \
        ]
    mode_2_3.set_dynamics(a_matrix)
    # x >= 2.0 & x <= 3.0 & y >= 3.0 & y <= 4.0
    mode_2_3.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-2, 3, -3, 4, ])

    mode_3_3 = ha.new_mode('mode_3_3')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.7778174593052022], \
        [0, 0, 0.2, -1.2, 0.7071067811865476], \
        [0, 0, 0, 0, 0], \
        ]
    mode_3_3.set_dynamics(a_matrix)
    # x >= 3.0 & x <= 4.0 & y >= 3.0 & y <= 4.0
    mode_3_3.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-3, 4, -3, 4, ])

    mode_4_3 = ha.new_mode('mode_4_3')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_4_3.set_dynamics(a_matrix)
    # x >= 4.0 & y >= 3.0 & y <= 4.0
    mode_4_3.set_invariant([[-1, -0, -0, -0, -0], [-0, -1, -0, -0, -0], [0, 1, 0, 0, 0], ], [-4, -3, 4, ])

    mode_0_4 = ha.new_mode('mode_0_4')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_0_4.set_dynamics(a_matrix)
    # x <= 1.0 & y >= 4.0
    mode_0_4.set_invariant([[1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], ], [1, -4, ])

    mode_1_4 = ha.new_mode('mode_1_4')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -1.2], \
        [0, 0, 0.2, -1.2, 0.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_1_4.set_dynamics(a_matrix)
    # x >= 1.0 & x <= 2.0 & y >= 4.0
    mode_1_4.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], ], [-1, 2, -4, ])

    mode_2_4 = ha.new_mode('mode_2_4')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -1.2], \
        [0, 0, 0.2, -1.2, 0.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_2_4.set_dynamics(a_matrix)
    # x >= 2.0 & x <= 3.0 & y >= 4.0
    mode_2_4.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], ], [-2, 3, -4, ])

    mode_3_4 = ha.new_mode('mode_3_4')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, -1.2], \
        [0, 0, 0.2, -1.2, 0.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_3_4.set_dynamics(a_matrix)
    # x >= 3.0 & x <= 4.0 & y >= 4.0
    mode_3_4.set_invariant([[-1, -0, -0, -0, -0], [1, 0, 0, 0, 0], [-0, -1, -0, -0, -0], ], [-3, 4, -4, ])

    mode_4_4 = ha.new_mode('mode_4_4')
    a_matrix = [ \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, -1.2, 0.1, 0.1], \
        [0, 0, 0.2, -1.2, -1.2], \
        [0, 0, 0, 0, 0], \
        ]
    mode_4_4.set_dynamics(a_matrix)
    # x >= 4.0 & y >= 4.0
    mode_4_4.set_invariant([[-1, -0, -0, -0, -0], [-0, -1, -0, -0, -0], ], [-4, -4, ])

    _error = ha.new_mode('_error')

    trans = ha.new_transition(mode_0_0, mode_1_0)
    # x >= 1.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_0_0, mode_0_1)
    # y >= 1.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_1_0, mode_0_0)
    # x <= 1.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_1_0, mode_2_0)
    # x >= 2.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_1_0, mode_1_1)
    # y >= 1.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_2_0, mode_1_0)
    # x <= 2.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_2_0, mode_3_0)
    # x >= 3.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_2_0, mode_2_1)
    # y >= 1.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_3_0, mode_2_0)
    # x <= 3.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_3_0, mode_4_0)
    # x >= 4.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_3_0, mode_3_1)
    # y >= 1.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_4_0, mode_3_0)
    # x <= 4.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_4_0, mode_4_1)
    # y >= 1.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_0_1, mode_1_1)
    # x >= 1.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_0_1, mode_0_0)
    # y <= 1.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_0_1, mode_0_2)
    # y >= 2.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_1_1, mode_0_1)
    # x <= 1.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_1_1, mode_2_1)
    # x >= 2.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_1_1, mode_1_0)
    # y <= 1.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_1_1, mode_1_2)
    # y >= 2.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_2_1, mode_1_1)
    # x <= 2.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_2_1, mode_3_1)
    # x >= 3.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_2_1, mode_2_0)
    # y <= 1.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_2_1, mode_2_2)
    # y >= 2.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_3_1, mode_2_1)
    # x <= 3.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_3_1, mode_4_1)
    # x >= 4.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_3_1, mode_3_0)
    # y <= 1.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_3_1, mode_3_2)
    # y >= 2.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_4_1, mode_3_1)
    # x <= 4.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_4_1, mode_4_0)
    # y <= 1.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_4_1, mode_4_2)
    # y >= 2.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_0_2, mode_1_2)
    # x >= 1.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_0_2, mode_0_1)
    # y <= 2.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_0_2, mode_0_3)
    # y >= 3.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_1_2, mode_0_2)
    # x <= 1.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_1_2, mode_2_2)
    # x >= 2.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_1_2, mode_1_1)
    # y <= 2.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_1_2, mode_1_3)
    # y >= 3.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_2_2, mode_1_2)
    # x <= 2.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_2_2, mode_3_2)
    # x >= 3.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_2_2, mode_2_1)
    # y <= 2.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_2_2, mode_2_3)
    # y >= 3.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_3_2, mode_2_2)
    # x <= 3.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_3_2, mode_4_2)
    # x >= 4.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_3_2, mode_3_1)
    # y <= 2.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_3_2, mode_3_3)
    # y >= 3.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_4_2, mode_3_2)
    # x <= 4.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_4_2, mode_4_1)
    # y <= 2.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_4_2, mode_4_3)
    # y >= 3.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_0_3, mode_1_3)
    # x >= 1.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_0_3, mode_0_2)
    # y <= 3.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_0_3, mode_0_4)
    # y >= 4.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_1_3, mode_0_3)
    # x <= 1.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_1_3, mode_2_3)
    # x >= 2.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_1_3, mode_1_2)
    # y <= 3.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_1_3, mode_1_4)
    # y >= 4.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_2_3, mode_1_3)
    # x <= 2.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_2_3, mode_3_3)
    # x >= 3.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_2_3, mode_2_2)
    # y <= 3.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_2_3, mode_2_4)
    # y >= 4.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_3_3, mode_2_3)
    # x <= 3.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_3_3, mode_4_3)
    # x >= 4.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_3_3, mode_3_2)
    # y <= 3.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_3_3, mode_3_4)
    # y >= 4.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_4_3, mode_3_3)
    # x <= 4.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_4_3, mode_4_2)
    # y <= 3.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_4_3, mode_4_4)
    # y >= 4.0
    trans.set_guard([[-0, -1, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_0_4, mode_1_4)
    # x >= 1.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-1, ])

    trans = ha.new_transition(mode_0_4, mode_0_3)
    # y <= 4.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_1_4, mode_0_4)
    # x <= 1.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [1, ])

    trans = ha.new_transition(mode_1_4, mode_2_4)
    # x >= 2.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-2, ])

    trans = ha.new_transition(mode_1_4, mode_1_3)
    # y <= 4.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_2_4, mode_1_4)
    # x <= 2.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [2, ])

    trans = ha.new_transition(mode_2_4, mode_3_4)
    # x >= 3.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-3, ])

    trans = ha.new_transition(mode_2_4, mode_2_3)
    # y <= 4.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_3_4, mode_2_4)
    # x <= 3.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [3, ])

    trans = ha.new_transition(mode_3_4, mode_4_4)
    # x >= 4.0
    trans.set_guard([[-1, -0, -0, -0, -0], ], [-4, ])

    trans = ha.new_transition(mode_3_4, mode_3_3)
    # y <= 4.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_4_4, mode_3_4)
    # x <= 4.0
    trans.set_guard([[1, 0, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_4_4, mode_4_3)
    # y <= 4.0
    trans.set_guard([[0, 1, 0, 0, 0], ], [4, ])

    trans = ha.new_transition(mode_3_1, _error)
    trans.set_guard_true()

    return ha

def define_init_states(ha):
    '''returns a list of StateSet objects'''
    # Variable ordering: [x, y, xvel, yvel, affine]
    rv = []
    
    # 3.5 <= x & x <= 3.5 & 3.5 <= y & y <= 3.5 & -1.0 <= xvel & xvel <= 1.0 & -1.0 <= yvel & yvel <= 1.0 & affine = 1.0
    mode = ha.modes['mode_3_3']
    mat = [[-1, 0, 0, 0, 0], \
        [1, 0, 0, 0, 0], \
        [0, -1, 0, 0, 0], \
        [0, 1, 0, 0, 0], \
        [0, 0, -1, 0, 0], \
        [0, 0, 1, 0, 0], \
        [0, 0, 0, -1, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, 0, 0, 1], \
        [-0, -0, -0, -0, -1], ]
    rhs = [-3.5, 3.5, -3.5, 3.5, 1, 1, 1, 1, 1, -1, ]
    rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    return rv


def define_settings():
    '''get the hylaa settings object
    see hylaa/settings.py for a complete list of reachability settings'''

    # step_size = 0.1, max_time = 10.0
    settings = HylaaSettings(0.1, 10.0)
    settings.plot.plot_mode = PlotSettings.PLOT_IMAGE
    settings.plot.xdim_dir = 0
    settings.plot.ydim_dir = 1

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

