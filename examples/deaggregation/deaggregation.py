'''
Deaggregation Model from Hylaa
'''

import math

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton('Deaggregation Example')

    m1 = ha.new_mode('green')
    m1.set_dynamics([[0, 1], [-1, 0]])
    m1.set_invariant([[0., -1.]], [0.5]) # y >= -0.5

    m2 = ha.new_mode('cyan')
    m2.set_dynamics([[0, 0, 0], [0, 0, -2], [0, 0, 0]])
    m2.set_invariant([0., -1., 0], [2.5]) # y >= 2.5

    m3 = ha.new_mode('orange')
    m3.set_dynamics([[0, 0, 0], [0, 0, -2], [0, 0, 0]])
    m3.set_invariant([0., -1., 0], [4]) # y >= 4

    t1 = ha.new_transition(m1, m2)
    t1.set_guard([[0., -1.]], [0.0]) # y >= 0
    reset_mat = [[1, 0], [0, 1], [0, 0]]
    t1.set_reset(reset_mat, [[0], [0], [1]], [[1], [-1]], [1, -1]) # create 3rd variable with a0 = 1

    t2 = ha.new_transition(m2, m3)
    t2.set_guard([[0, 1, 0], [1, 0, 0], [-1, 0, 0]], [0, 0.5, 0.5]) # y <= 0 & x <= 0.5 & x >= -0.5
 
    return ha

def make_init(ha):
    'make the initial states'

    mode = ha.modes['green']
    init_lpi = lputil.from_box([(-5.5, -4.5), (0, 1.0)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(1.0, 2.0)
    settings.process_urgent_guards = True
    settings.plot.plot_mode = PlotSettings.PLOT_INTERACTIVE
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.filename = "deaggregation.png"

    return settings

def main():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    main()
