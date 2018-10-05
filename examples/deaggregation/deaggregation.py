'''
Deaggregation Model from Hylaa
'''

from matplotlib import collections

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings, LabelSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton('Deaggregation Example')

    # x' = 2
    m1 = ha.new_mode('green')
    m1.set_dynamics([[0, 0, 2], [0, 0, 0], [0, 0, 0]])
    m1.set_invariant([[1, 0, 0]], [3.5]) # x <= 3.5

    # y' == 2
    m2 = ha.new_mode('cyan')
    m2.set_dynamics([[0, 0, 0], [0, 0, 2], [0, 0, 0]])
    m2.set_invariant([0., 1., 0], [3.5]) # y <= 3.5

    # x' == 1 
    m3 = ha.new_mode('orange')
    m3.set_dynamics([[0, 0, 1], [0, 0, -0], [0, 0, 0]])
    m3.set_invariant([1., 0, 0], [10]) # x <= 10

    t = ha.new_transition(m1, m2)
    t.set_guard_true()

    t = ha.new_transition(m2, m3)
    t.set_guard_true()

    error = ha.new_mode('error')
    t = ha.new_transition(m3, error)
    t.set_guard([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], [-1, 2, -3.1, 3.9]) # x >= 1 x <= 2, y >= 1.1, y <= 1.9
    
    return ha

def make_init(ha):
    'make the initial states'

    mode = ha.modes['green']
    init_lpi = lputil.from_box([(0, 1), (0, 1.0), (1.0, 1.0)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(1.0, 20.0)
    settings.process_urgent_guards = True
    settings.plot.plot_mode = PlotSettings.PLOT_INTERACTIVE
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.filename = "deaggregation.png"

    settings.stop_on_aggregated_error = False
    settings.aggstrat.deaggregate = True # use deaggregation

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

    line = [(2, 3.9), (1, 3.9), (1, 3.1), (2, 3.1), (2, 3.9)]
    cols.append(collections.LineCollection([line], animated=True, colors=('red'), linewidths=(2), linestyle='dashed'))

    settings.plot.extra_collections.append(cols)

    return settings

def main():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    main()
