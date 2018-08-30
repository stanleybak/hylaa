'''
Tests for Hylaa deaggregation. Made for use with py.test
'''

import math

import matplotlib.pyplot as plt

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.stateset import StateSet
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa import lputil
from hylaa.core import Core

from util import assert_verts_is_box

def fail_deagg_counterexample():
    'test that aggregation with a counterexample'
    # init: x0, y0 \in [0, 1], step = 1.0
    # 
    # m1 dynamics: x' == 1, y' == 0, 
    # m1 invariant: x <= 3
    # m1 -> m2 guard: True
    # m2 dynamics: x' == 0, y' == 1
    # m2 -> error: y >= 3

    ha = HybridAutomaton()

    # mode one: x' = 1, y' = 0, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

    # mode two: x' = 0, y' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 0, 0], [0, 0, 1], [0, 0, 0]])

    # invariant: x <= 3.0
    m1.set_invariant([[1, 0, 0]], [3.0])

    # guard: True
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard_true()

    error = ha.new_mode('error')
    trans2 = ha.new_transition(m2, error, 'trans2')
    trans2.set_guard([[0, -1, 0]], [-3]) # y >= 3

    # initial set has x0 = [0, 1], t = [0, 1], a = 1
    init_lpi = lputil.from_box([(0, 1), (0, 1), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 10.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.aggregation.deaggregation = True

    core = Core(ha, settings)

    core.setup(init_list)
    
    core.step() # pop

    for _ in range(4):
        core.step() # continuous post in m1

    core.step() # pop

    # at this point, the state should be aggregated, but the aggregation maintains one concrete state that's feasible

    #result = core.run(init_list)

    #assert not result.counterexample

def fail_recursive_aggregation():
    'tests recursive aggregation process on Harmonic oscillator example'

    for agg_recursive in [True, False]:
        ha = HybridAutomaton()

        # x' = y, y' = -x
        m1 = ha.new_mode('green')
        m1.set_dynamics([[0, 1], [-1, 0]])
        m1.set_invariant([0, -1], [0.75]) # y >= -0.75

        # x' == 1
        m2 = ha.new_mode('cyan')
        m2.set_dynamics([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

        t1 = ha.new_transition(m1, m2)
        t1.set_guard_true()
        reset_mat = [[1, 0], [0, 1], [0, 0]]
        t1.set_reset(reset_mat, [[0], [0], [1]], [[1], [-1]], [1, -1]) # create 3rd variable with a0 = 1

        mode = ha.modes['green']
        init_lpi = lputil.from_box([(-2, -1), (-0.5, 0.5)], mode) # -2, -1

        init_list = [StateSet(init_lpi, mode)]

        step = math.pi/2.0
        settings = HylaaSettings(step, 10*step)
        settings.process_urgent_guards = True
        settings.plot.plot_mode = PlotSettings.PLOT_NONE
        settings.stdout = HylaaSettings.STDOUT_DEBUG

        core = Core(ha, settings)
        core.setup(init_list)

        for _ in range(4): # step 0 -> 3
            core.do_step() # pop
            xs, ys = zip(*core.cur_state.verts(core.plotman))
            plt.plot(xs, ys, 'k:')
        
        core.do_step() # invariant became false
        assert core.cur_state is None
        
        core.do_step() # pop
        xs, ys = zip(*core.cur_state.verts(core.plotman))
        plt.plot(xs, ys, 'g:', lw=4)
        
        state = core.cur_state

        plt.show()

        assert_verts_is_box(state.verts(core.plotman), [(-2, 2), (-2, 2)], tol=1e-5)

        assert isinstance(state.predecessor, AggregationPredecessor)

        if agg_recursive:
            assert len(state.predecessor.states) == 2
        else:
            assert len(state.predecessor.states) == 4
