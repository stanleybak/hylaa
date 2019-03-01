'''
Tests for Hylaa deaggregation. Made for use with py.test
'''

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.stateset import StateSet
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa import lputil
from hylaa.core import Core
from hylaa.aggdag import OpTransition, OpInvIntersect

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

    core.do_step() # pop

    assert core.aggdag.cur_node is not None

    for _ in range(5): # 4 + 1 step to leave invariant
        core.do_step() # continuous post in m1

    core.do_step() # pop

    # at this point, the state should be aggregated, but we should maintain one concrete state that's feasible
    cur_node = core.aggdag.cur_node
    assert cur_node.aggregated_state == core.aggdag.get_cur_state()

    assert cur_node.concrete_state is not None
    
    assert len(core.aggdag.roots) == 1
    root = core.aggdag.roots[0]

    assert root.op_list
    assert isinstance(root.op_list[0], OpTransition) and root.op_list[0].step == 1 # transition from x=[1, 2]
    assert isinstance(root.op_list[1], OpTransition) and root.op_list[1].step == 2 # transition from x=[2, 3]
    assert isinstance(root.op_list[2], OpTransition) and root.op_list[2].step == 3 # transition from x=[3, 4]
    assert isinstance(root.op_list[3], OpTransition) and root.op_list[3].step == 4 # transition from x=[4, 5]

    for s in range(4):
        assert root.op_list[s].transition == trans1 and root.op_list[s].poststate.is_concrete

    assert isinstance(root.op_list[4], OpInvIntersect)
    op4 = root.op_list[4]
    assert op4.step == 5 and op4.node == root and op4.i_index == 0 and not op4.is_stronger

    assert isinstance(root.op_list[5], OpTransition) and root.op_list[5].step == 5 # transition from x=[4, 4]

    assert isinstance(root.op_list[6], OpInvIntersect)
    op6 = root.op_list[6]
    assert op6.step == 6 and op6.node == root and op6.i_index == 0 and op6.is_stronger

    assert len(root.op_list) == 7

    core.run_to_completion()

    assert root.op_list[0].child_node == cur_node
    assert root.op_list[3].child_node == cur_node

    #result = core.run(init_list)

    #assert not result.counterexample
