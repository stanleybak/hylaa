'''
Deaggregation Model from Hylaa

A simple demo of deaggregation occuring.
'''

from matplotlib import collections

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings, LabelSettings
from hylaa.core import Core
from hylaa.aggstrat import Aggregated
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton(unsafe_box):
    'make the hybrid automaton'

    ha = HybridAutomaton('Deaggregation Example')

    # x' = 2
    m1 = ha.new_mode('mode0_right')
    m1.set_dynamics([[0, 0, 2], [0, 0, 0], [0, 0, 0]])
    m1.set_invariant([[1, 0, 0]], [3.5]) # x <= 3.5

    # y' == 2
    m2 = ha.new_mode('mode1_up')
    m2.set_dynamics([[0, 0, 0], [0, 0, 2], [0, 0, 0]])
    m2.set_invariant([0., 1., 0], [3.5]) # y <= 3.5

    # x' == 2 
    m3 = ha.new_mode('mode2_right')
    m3.set_dynamics([[0, 0, 2], [0, 0, -0], [0, 0, 0]])
    m3.set_invariant([1., 0, 0], [7]) # x <= 7

    t = ha.new_transition(m1, m2)
    t.set_guard_true()

    t = ha.new_transition(m2, m3)
    t.set_guard_true()

    error = ha.new_mode('error')
    t = ha.new_transition(m3, error)

    unsafe_rhs = [-unsafe_box[0][0], unsafe_box[0][1], -unsafe_box[1][0], unsafe_box[1][1]]
    
    # x >= 1.1 x <= 1.9, y >= 2.7, y <= 4.3
    t.set_guard([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], unsafe_rhs)

    t = ha.new_transition(m2, error)
    # x >= 1.1 x <= 1.9, y >= 2.7, y <= 4.3
    t.set_guard([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], unsafe_rhs) 
    
    return ha

def make_init(ha):
    'make the initial states'

    mode = ha.modes['mode0_right']
    init_lpi = lputil.from_box([(0, 1), (0, 1.0), (1.0, 1.0)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings(unsafe_box):
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(1.0, 20.0)
    settings.process_urgent_guards = True
    settings.stdout = HylaaSettings.STDOUT_VERBOSE

    settings.plot.video_pause_frames = 5
    settings.plot.video_fps = 5
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.interactive_skip_count = 5

    #settings.plot.plot_mode = PlotSettings.PLOT_VIDEO
    #settings.plot.filename = "deagg_example.mp4"

    settings.stop_on_aggregated_error = False
    settings.aggstrat.deaggregate = True # use deaggregation
    settings.aggstrat.deagg_preference = Aggregated.DEAGG_LEAVES_FIRST

    settings.plot.extra_collections = []
    settings.plot.label = []

    ls = LabelSettings()
    ls.axes_limits = [-1, 12, -1, 6]
    settings.plot.label.append(ls)

    ls.big(size=24)

    ls.x_label = '$x$'
    ls.y_label = '$y$'

    cols = []

    line = [(3.5, -20), (3.5, 20)]
    cols.append(collections.LineCollection([line], animated=True, colors=('gray'), linewidths=(2), linestyle='dashed'))

    line = [(-20, 3.5), (20, 3.5)]
    cols.append(collections.LineCollection([line], animated=True, colors=('gray'), linewidths=(2), linestyle='dashed'))

    # x >= 1.1 x <= 1.9, y >= 2.7, y <= 4.3
    line = []
    line.append((unsafe_box[0][0], unsafe_box[1][0]))
    line.append((unsafe_box[0][1], unsafe_box[1][0]))
    line.append((unsafe_box[0][1], unsafe_box[1][1]))
    line.append((unsafe_box[0][0], unsafe_box[1][1]))
    line.append((unsafe_box[0][0], unsafe_box[1][0]))

    cols.append(collections.LineCollection([line], animated=True, colors=('red'), linewidths=(2), linestyle='dashed'))

    settings.plot.extra_collections.append(cols)

    return settings

def run_hylaa():
    'main entry point'

    unsafe_box = [[5.1, 5.9], [4.1, 4.9]]

    ha = make_automaton(unsafe_box)

    init_states = make_init(ha)

    settings = make_settings(unsafe_box)

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    run_hylaa()
