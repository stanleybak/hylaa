'''
Harmonic Oscillator example with deaggreagtion

2-d dynamics:

x' == y
y' == -x

after a semi-circle, the dynamics in the second mode are y' == -1, x' == 0

'''

import math

from matplotlib import collections

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.aggstrat import Aggregated
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha(unsafe_box):
    '''make the hybrid automaton'''

    ha = HybridAutomaton()

    # dynamics: x' = y, y' = -x, t' == a
    a_mat = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]

    one = ha.new_mode('one')
    one.set_dynamics(a_mat)
    one.set_invariant([[0, 0, 1, 0]], [math.pi - 1e-6]) # t <= pi

    two = ha.new_mode('two')
    two.set_dynamics([[0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 1], [0, 0, 0, 0]])
    two.set_invariant([[0, -1, 0, 0]], [3]) # y >= -3

    t = ha.new_transition(one, two)
    t.set_guard_true()

    error = ha.new_mode('error')
    t = ha.new_transition(two, error)

    unsafe_rhs = [-unsafe_box[0][0], unsafe_box[0][1], -unsafe_box[1][0], unsafe_box[1][1]]
    t.set_guard([[-1, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 1, 0, 0]], unsafe_rhs) 

    return ha

def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['one']
    # init states: x in [-5, -4], y in [0, 1]
    init_lpi = lputil.from_box([[-5, -4], [-0.5, 0.5], [0, 0], [1, 1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list

def define_settings(unsafe_box):
    'get the hylaa settings object'

    step = math.pi/6
    max_time = 3 * math.pi
    settings = HylaaSettings(step, max_time)

    settings.process_urgent_guards = True
    settings.aggstrat.deaggregate = True # use deaggregation
    settings.aggstrat.deagg_preference = Aggregated.DEAGG_LEAVES_FIRST
    settings.aggstrat.agg_type = Aggregated.AGG_CONVEX_HULL

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    plot_settings.label.x_label = '$x$'
    plot_settings.label.y_label = '$y$'

    cols = []
    line = [(-10, -3), (10, -3)]
    cols.append(collections.LineCollection([line], animated=True, colors=('gray'), linewidths=(2), linestyle='dashed'))

    line = []
    line.append((unsafe_box[0][0], unsafe_box[1][0]))
    line.append((unsafe_box[0][1], unsafe_box[1][0]))
    line.append((unsafe_box[0][1], unsafe_box[1][1]))
    line.append((unsafe_box[0][0], unsafe_box[1][1]))
    line.append((unsafe_box[0][0], unsafe_box[1][0]))

    cols.append(collections.LineCollection([line], animated=True, colors=('red'), linewidths=(2), linestyle='dashed'))
    
    settings.plot.extra_collections = cols

    plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    plot_settings.filename = 'ha_deagg.mp4'
    plot_settings.video_fps = 4
    plot_settings.video_extra_frames = 12 # extra frames at the end of a video so it doesn't end so abruptly
    plot_settings.video_pause_frames = 2 # frames to render in video whenever a 'pause' occurs

    plot_settings.label.axes_limits = [-6, 6, -4, 6]
    
    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Deaggregation Demo'

    return settings

def run_hylaa():
    'Runs hylaa with the given settings'

    unsafe_box = [[0.7, 1.3], [-2, -1]]

    ha = define_ha(unsafe_box)
    settings = define_settings(unsafe_box)
    init_states = make_init(ha)

    Core(ha, settings).run(init_states)

if __name__ == '__main__':
    run_hylaa()
