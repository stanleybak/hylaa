'''
Fuzzy PD controller example, from Aaron Fifarek's MS Thesis

D gain is only nonzero near the setpoint

Model was originally converted with Hyst
'''

from matplotlib import animation

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = HybridAutomaton()

    # dynamics variable order: [x1, x2, t, affine]

    loc2 = ha.new_mode('loc2')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [18300, -20, 0, -9562.5], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc2.set_dynamics(a_matrix)
    # -x1 > -0.51 & -x1 <= -0.5
    loc2.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [0.51, -0.5, ])

    loc3 = ha.new_mode('loc3')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [-825, -20, 0, 0], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc3.set_dynamics(a_matrix)
    # -x1 > -0.5 & -x1 <= -0.09
    loc3.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [0.5, -0.09, ])

    loc4 = ha.new_mode('loc4')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [30675, -695, 0, -2835], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc4.set_dynamics(a_matrix)
    # -x1 > -0.09 & -x1 <= -0.08
    loc4.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [0.09, -0.08, ])

    loc1 = ha.new_mode('loc1')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [-450, -20, 0, 0], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc1.set_dynamics(a_matrix)
    # -x1 <= -0.51
    loc1.set_invariant([[-1, 0, 0, 0], ], [-0.51, ])

    loc5 = ha.new_mode('loc5')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [-4762.5, -95, 0, 0], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc5.set_dynamics(a_matrix)
    # -x1 > -0.08 & -x1 < 0.08
    loc5.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [0.08, 0.08, ])

    loc6 = ha.new_mode('loc6')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [30675, -695, 0, 2835], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc6.set_dynamics(a_matrix)
    # -x1 >= 0.08 & -x1 < 0.09
    loc6.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [-0.08, 0.09, ])

    loc7 = ha.new_mode('loc7')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [-825, -20, 0, 0], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc7.set_dynamics(a_matrix)
    # -x1 >= 0.09 & -x1 < 0.5
    loc7.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [-0.09, 0.5, ])

    loc8 = ha.new_mode('loc8')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [18300, -20, 0, 9562.5], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc8.set_dynamics(a_matrix)
    # -x1 >= 0.5 & -x1 < 0.51
    loc8.set_invariant([[1, -0, -0, -0], [-1, 0, 0, 0], ], [-0.5, 0.51, ])

    loc9 = ha.new_mode('loc9')
    a_matrix = [ \
        [0, 1, 0, 0], \
        [-450, -20, 0, 0], \
        [0, 0, 0, 1], \
        [0, 0, 0, 0], \
        ]
    loc9.set_dynamics(a_matrix)
    # -x1 >= 0.51
    loc9.set_invariant([[1, -0, -0, -0], ], [-0.51, ])

    trans = ha.new_transition(loc2, loc1)
    # -x1 <= -0.51
    trans.set_guard([[-1, 0, 0, 0], ], [-0.51, ])

    trans = ha.new_transition(loc2, loc3)
    # -x1 > -0.5
    trans.set_guard([[1, -0, -0, -0], ], [0.5, ])

    trans = ha.new_transition(loc3, loc4)
    # -x1 > -0.09
    trans.set_guard([[1, -0, -0, -0], ], [0.09, ])

    trans = ha.new_transition(loc3, loc2)
    # -x1 <= -0.5
    trans.set_guard([[-1, 0, 0, 0], ], [-0.5, ])

    trans = ha.new_transition(loc4, loc5)
    # -x1 > -0.08
    trans.set_guard([[1, -0, -0, -0], ], [0.08, ])

    trans = ha.new_transition(loc4, loc3)
    # -x1 <= -0.09
    trans.set_guard([[-1, 0, 0, 0], ], [-0.09, ])

    trans = ha.new_transition(loc1, loc2)
    # -x1 > -0.51
    trans.set_guard([[1, -0, -0, -0], ], [0.51, ])

    trans = ha.new_transition(loc5, loc6)
    # -x1 >= 0.0801
    trans.set_guard([[1, -0, -0, -0], ], [-0.0801, ])

    trans = ha.new_transition(loc5, loc4)
    # -x1 <= -0.0801
    trans.set_guard([[-1, 0, 0, 0], ], [-0.0801, ])

    trans = ha.new_transition(loc6, loc7)
    # -x1 >= 0.09
    trans.set_guard([[1, -0, -0, -0], ], [-0.09, ])

    trans = ha.new_transition(loc6, loc5)
    # -x1 < 0.08
    trans.set_guard([[-1, 0, 0, 0], ], [0.08, ])

    trans = ha.new_transition(loc7, loc8)
    # -x1 >= 0.5
    trans.set_guard([[1, -0, -0, -0], ], [-0.5, ])

    trans = ha.new_transition(loc7, loc6)
    # -x1 < 0.09
    trans.set_guard([[-1, 0, 0, 0], ], [0.09, ])

    trans = ha.new_transition(loc8, loc9)
    # -x1 >= 0.51
    trans.set_guard([[1, -0, -0, -0], ], [-0.51, ])

    trans = ha.new_transition(loc8, loc7)
    # -x1 < 0.5
    trans.set_guard([[-1, 0, 0, 0], ], [0.5, ])

    trans = ha.new_transition(loc9, loc8)
    # -x1 < 0.51
    trans.set_guard([[-1, 0, 0, 0], ], [0.51, ])

    return ha

def define_init_states(ha):
    '''returns a list of StateSet objects'''
    # Variable ordering: [x1, x2, t, affine]
    rv = []

    should_init = lambda name: True

    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc2']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc3']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc4']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc1']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc5']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc6']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc7']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc8']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    # -1.0 <= x1 & x1 <= 1.0 & x2 = 0.0 & t = 0.0 & affine = 1.0
    mode = ha.modes['loc9']
    mat = [[-1, 0, 0, 0], \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [-0, -1, -0, -0], \
        [0, 0, 1, 0], \
        [-0, -0, -1, -0], \
        [0, 0, 0, 1], \
        [-0, -0, -0, -1], ]
    rhs = [1, 1, 0, -0, 0, -0, 1, -1, ]

    if should_init(mode.name):
        rv.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))
    
    return rv


def define_settings():
    '''get the hylaa settings object
    see hylaa/settings.py for a complete list of reachability settings'''

    # step_size = 0.001, max_time = 0.75
    settings = HylaaSettings(0.001, 0.15)
    settings.plot.plot_mode = PlotSettings.PLOT_IMAGE # try PLOT_VIDEO (takes 10 minutes)
    settings.plot.xdim_dir = 2
    settings.plot.ydim_dir = 0
    settings.plot.label.title = "Fuzzy PD Controller"
    settings.plot.label.axes_limits = (-0.01, 0.3, -1.1, 1.1) 
    settings.stdout = HylaaSettings.STDOUT_VERBOSE

    #settings.aggregation.require_same_path=False
    #settings.aggregation.pop_strategy=AggregationSettings.POP_LARGEST_MAXTIME
    #self.aggstrat = aggstrat.Aggregated()

    # custom settings for video export
    def make_video_writer():
        'returns the Writer to create a video for export'

        writer_class = animation.writers['ffmpeg']
        return writer_class(fps=50, metadata=dict(artist='Me'), bitrate=1800)

    settings.plot.make_video_writer_func = make_video_writer
    
    return settings

def run_hylaa():
    'runs hylaa, returning a HylaaResult object'
    ha = define_ha()
    init = define_init_states(ha)
    settings = define_settings()

    core = Core(ha, settings)
    result = core.run(init)

    #core.aggdag.show()

    return result

if __name__ == '__main__':
    run_hylaa()
