'''
Tests for Hylaa core object. Made for use with py.test
'''

import math
import numpy as np
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet, TransitionPredecessor, AggregationPredecessor
from hylaa import lputil
from hylaa.lpinstance import LpInstance

def test_guard_strengthening():
    'simple 2-mode, 2-guard, 2d system with 1st guard A->B is x <= 2, 2nd guard A->B is y <= 2, and inv(B) is y <= 2'

    ha = HybridAutomaton()

    mode_a = ha.new_mode('A')
    mode_a.set_dynamics(np.identity(2))

    mode_b = ha.new_mode('B')
    mode_b.set_dynamics(np.identity(2))
    mode_b.set_invariant([[0, 1]], [2])

    trans1 = ha.new_transition(mode_a, mode_b, 'first')
    trans1.set_guard([[1, 0]], [2])

    trans2 = ha.new_transition(mode_a, mode_b, 'second')
    trans2.set_guard([[0, 1]], [2])

    ha.do_guard_strengthening()

    # trans1 should now have 2 conditions
    assert (trans1.guard_csr.toarray() == np.array([[1, 0], [0, 1]], dtype=float)).all()
    assert (trans1.guard_rhs == np.array([2, 2], dtype=float)).all()

    # trans2 should still have 1 condition since invariant was redundant
    assert (trans2.guard_csr.toarray() == np.array([[0, 1]], dtype=float)).all()

def test_ha_line_arch18():
    'test for the harmonic oscillator example with line initial set (from ARCH 2018 paper)'

    ha = HybridAutomaton()

    # with time and affine variable
    mode = ha.new_mode('mode')
    mode.set_dynamics([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

    error = ha.new_mode('error')

    trans1 = ha.new_transition(mode, error)
    trans1.set_guard([[1., 0, 0, 0], [-1., 0, 0, 0]], [4.0, -4.0])

    # initial set
    init_lpi = lputil.from_box([(-5, -5), (0, 1), (0, 0), (1, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # settings
    settings = HylaaSettings(math.pi/4, 2*math.pi)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    
    core = Core(ha, settings)
    result = core.run(init_list)

    assert not result.safe

    ce = result.counterexample[0]

    # [-5.0, 0.6568542494923828, 0.0, 1.0] -> [4.0, 3.0710678118654737, 2.356194490192345, 1.0]

    assert ce.mode == mode
    assert np.allclose(ce.start, np.array([-5, 0.65685, 0, 1], dtype=float))
    assert np.allclose(ce.end, np.array([4, 3.07106, 2.35619, 1], dtype=float))

    # check the reachable state (should always have x <= 3.5)
    polys = result.mode_to_polys[mode.name]

    for poly in polys:
        for vert in poly:
            x, _ = vert

            assert x <= 4.9

def test_plot_over_time():
    'test doing a plot over time'

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
    mode.set_dynamics([[0, 1], [-1, 0]])

    # initial set
    init_lpi = lputil.from_box([(-5, -4), (0, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # settings
    settings = HylaaSettings(math.pi/4, math.pi)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.ydim_dir = None # y dimension will be time

    result = Core(ha, settings).run(init_list)

    assert result.safe

    # check the reachable state
    # we would expect at the end that x = [4, 5], t = pi
    polys = result.mode_to_polys[mode.name]

    for vert in polys[0]:
        x, y = vert

        assert abs(y) < 1e-6, "initial poly time is wrong"
        assert abs(-5 - x) < 1e-6 or abs(-4 - x) < 1e-6

    for vert in polys[-1]:
        x, y = vert

        assert abs(math.pi - y) < 1e-6, "final poly time is wrong"
        assert abs(5 - x) < 1e-6 or abs(4 - x) < 1e-6

def assert_verts_is_box(verts, box, tol=1e-5):
    '''check that a list of verts is almost equal to the passed-in box using assertions

    box is [[xmin, xmax], [ymin, ymax]]
    '''

    is_flat = abs(box[0][0] - box[0][1]) < tol or abs(box[1][0] - box[1][1]) < tol

    expected_verts = 3 if is_flat else 5

    assert len(verts) == expected_verts and verts[0] == verts[-1]

    pts = [(box[0][0], box[1][0]), (box[0][1], box[1][0]), (box[0][1], box[1][1]), (box[0][0], box[1][1])]

    for pt in pts:
        found = False

        for vert in verts:
            x, y = vert

            if abs(x - pt[0]) < tol and abs(y - pt[1]) < tol:
                found = True
                break

        assert found, "Point {} was not found in verts: {}".format(pt, verts)

def test_init_outside_invariant():
    'test when initial state is outside of the mode invariant'

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
    mode.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]]) # x' = 1, y' = 1, a' = 0

    # x <= 2.5
    mode.set_invariant([[1, 0, 0]], [2.5])

    # initial set, x = [3, 4]
    init_lpi = lputil.from_box([(3, 4), (0, 1), (1, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # transition to error if x >= 10
    error = ha.new_mode('error')
    trans = ha.new_transition(mode, error)
    trans.set_guard([[-1., 0, 0],], [-10]) 

    # settings
    settings = HylaaSettings(1.0, 5.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE

    try:
        Core(ha, settings).run(init_list)
        assert False, "running with initial state outside of invariant did not raise RuntimeError"
    except RuntimeError:
        pass

def test_invariants():
    'test invariant trimming'

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
 
    # dynamics: x' = 1, y' = 1, a' = 0
    mode.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    # invariant: x <= 2.5
    mode.set_invariant([[1, 0, 0]], [2.5])

    # initial set has x0 = [0, 1]
    init_lpi = lputil.from_box([(0, 1), (0, 1), (1, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 5.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.store_plot_result = True

    result = Core(ha, settings).run(init_list)

    # check the reachable state
    polys = result.mode_to_polys[mode.name]

    # 4 steps because invariant is allowed to be false for the final step
    assert len(polys) == 4, "expected invariant to become false after 4 steps"

    assert_verts_is_box(polys[0], [[0, 1], [0, 1]])

    assert_verts_is_box(polys[1], [[1, 2], [1, 2]])

    assert_verts_is_box(polys[2], [[2, 3], [2, 3]])

    assert_verts_is_box(polys[3], [[3, 3.5], [3, 4]])

def test_transition():
    'test a discrete transition'

    ha = HybridAutomaton()

    # mode one: x' = 1, t' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    # mode two: x' = -1, t' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 0, -1], [0, 0, 1], [0, 0, 0]])

    # invariant: t <= 2.5
    m1.set_invariant([[0, 1, 0]], [2.5])

    # guard: t >= 2.5
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard([[0, -1, 0]], [-2.5])

    # error t >= 4.5
    error = ha.new_mode('error')
    trans2 = ha.new_transition(m2, error, "to_error")
    trans2.set_guard([[0, -1, 0]], [-4.5])

    # initial set has x0 = [0, 1], t = [0, 0.2], a = 1
    init_lpi = lputil.from_box([(0, 1), (0, 0.2), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 10.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True

    result = Core(ha, settings).run(init_list)
    ce = result.counterexample

    assert len(ce) == 2
    assert ce[0].mode.name == 'm1'
    assert ce[0].outgoing_transition.name == 'trans1'

    assert ce[1].mode.name == 'm2'
    assert ce[1].outgoing_transition.name == 'to_error'

    assert ce[1].start[0] + 1e-9 >= 3.0
    assert ce[1].end[0] - 1e-9 <= 2.0

    assert len(result.mode_to_polys['m1']) == 4
    assert len(result.mode_to_polys['m2']) == 3

    assert result.last_cur_state.cur_step_since_start == 5

def test_time_triggered():
    'test to make sure exact time-triggered guards only have a single sucessor state'

    ha = HybridAutomaton()

    # mode one: x' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 1], [0, 0]])

    # mode two: x' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 1], [0, 0]])

    # invariant: x <= 2.0
    m1.set_invariant([[1, 0]], [2.0])

    # guard: x >= 2.0
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard([[-1, 0]], [-2.0])

    # error x >= 4.0
    error = ha.new_mode('error')
    trans2 = ha.new_transition(m2, error, "to_error")
    trans2.set_guard([[-1, 0]], [-4.0])


    # manually run ha.detect_tt_transitions() and check the result
    ha.detect_tt_transitions()

    assert trans1.time_triggered
    assert not trans2.time_triggered # not time-triggered because invariant of m2 is True

    # initial set has x = 0, a = 1
    init_lpi = lputil.from_box([(0, 0), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 10.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True

    result = Core(ha, settings).run(init_list)
    ce = result.counterexample

    assert len(ce) == 2
    assert ce[0].mode.name == 'm1'
    assert ce[0].outgoing_transition.name == 'trans1'

    assert ce[1].mode.name == 'm2'
    assert ce[1].outgoing_transition.name == 'to_error'

    assert abs(ce[0].start[0] - 0.0) < 1e-5
    assert abs(ce[0].end[0] - 2.0) < 1e-5

    assert abs(ce[1].start[0] - 2.0) < 1e-5
    assert abs(ce[1].end[0] - 4.0) < 1e-5

    assert len(result.mode_to_polys['m1']) == 3 # time 0, 1, 2
    assert len(result.mode_to_polys['m2']) == 3 # times 2, 3, 4

def test_aggregation():
    'test the aggregation of states across discrete transitions'

    # m1 dynamics: x' == 1, y' == 0, x0, y0: [0, 1], step: 1.0
    # m1 invariant: x <= 3
    # m1 -> m2 guard: True
    # m2 dynamics: x' == 0, y' == 1
    # time bound: 4
    # excepted final states to be: x: [0, 4], y: [4,5]
    # x is [1, 4] because no transitions are allowed at step 0 (simulation-equiv semantics) and a transition is
    #        allowed one step after the invariant becomes false
    # y is [4,5] because after aggregation, the time elapsed for the aggregated set will be 0.0, the minimum

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
    trans1.set_guard(csr_matrix((0, 0)), [])

    # initial set has x0 = [0, 1], t = [0, 1], a = 1
    init_lpi = lputil.from_box([(0, 1), (0, 1), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 4.0)
    settings.stdout = HylaaSettings.STDOUT_NONE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True

    result = Core(ha, settings).run(init_list)

    # check history
    state = result.last_cur_state

    assert state.mode == m2
    assert isinstance(state.predecessor, AggregationPredecessor)
    unagg_state = state.predecessor.states[0]
    assert isinstance(unagg_state, StateSet)

    assert unagg_state.mode == m2
    assert isinstance(unagg_state.predecessor, TransitionPredecessor)
    assert unagg_state.predecessor.transition == trans1
    assert isinstance(unagg_state.predecessor.transition_lpi, LpInstance)
    prestate = unagg_state.predecessor.state
    assert isinstance(prestate, StateSet)

    assert prestate.mode == m1
    assert prestate.predecessor is None

    # check polygons in m2
    polys2 = result.mode_to_polys['m2']

    assert 4 <= len(polys2) <= 5

    assert_verts_is_box(polys2[0], [[1, 4], [0, 1]])
    assert_verts_is_box(polys2[1], [[1, 4], [1, 2]])
    assert_verts_is_box(polys2[2], [[1, 4], [2, 3]])
    assert_verts_is_box(polys2[3], [[1, 4], [3, 4]])

def test_agg_with_reset():
    'test the aggregation of states with a reset'

    # m1 dynamics: x' == 1, y' == 0, x0, y0: [0, 1], x0:[0, 1], step: 1.0
    # m1 invariant: x <= 3
    # m1 -> m2 guard: True, reset = [[0, 1, 0], [1, 0, 0]] (flip x and y and remove a)
    # m2 dynamics: x' == x+y, y' == 2x+y
    # time bound: 4
    # expect aggregation to have a sinlge bloating term (tests if aggregation directions take the reset into account)

    ha = HybridAutomaton()

    # mode one: x' = 1, y' = 0, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

    # mode two: x' = 0, y' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[1, 1], [2, 1]])

    # invariant: x <= 3.0
    m1.set_invariant([[1, 0, 0]], [3.0])

    # guard: True
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard_true()
    trans1.set_reset(np.array([[0, 1, 0], [1, 0, 0]], dtype=float))

    # initial set has x0 = [0, 1], y = [0, 1], a = 1
    init_lpi = lputil.from_box([(0, 1), (0, 1), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 4.0)
    settings.stdout = HylaaSettings.STDOUT_NONE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    result = Core(ha, settings).run(init_list)

    names = result.last_cur_state.lpi.get_names()

    assert "agg0" in names
    assert "agg1" not in names
 
    expected = ['m0_i0', 'm0_i1', 'm0_i2', 'm0_c0', 'm0_c1', 'm0_c2', # initial state variables
                'm1_i0_t0', 'm1_i1', 'm1_c0', 'm1_c1', # post reset variables
                'agg0', 'snap0', 'snap1'] # post aggregation variables
    assert names == expected

def test_agg_to_more_vars():
    'test the aggregation of states with a reset to a mode with new variables'

    ha = HybridAutomaton()

    # mode one: x' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 1], [0, 0]])

    # mode two: x' = 0, a' = 0, y' == 1 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

    # invariant: x <= 3.0
    m1.set_invariant([[1, 0]], [3.0])

    # guard: True
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard_true()

    reset_mat = [[1, 0], [0, 1], [0, 0]]
    reset_minkowski = [[0], [0], [1]]
    reset_minkowski_constraints = [[1], [-1]]
    reset_minkowski_rhs = [3, -3] # y0 == 3
    
    trans1.set_reset(reset_mat, reset_minkowski, reset_minkowski_constraints, reset_minkowski_rhs)

    # initial set has x0 = [0, 1], a = 1
    init_lpi = lputil.from_box([(0, 1), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 4.0)
    settings.stdout = HylaaSettings.STDOUT_NONE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True
    settings.plot.xdim_dir = 0
    settings.plot.ydim_dir = {'m1': 1, 'm2': 2}

    result = Core(ha, settings).run(init_list)

    names = result.last_cur_state.lpi.get_names()

    assert "agg0" in names
    assert "agg1" not in names
 
    expected = ['m0_i0', 'm0_i1', 'm0_c0', 'm0_c1', # initial state variables
                'reset0', # reset minkowsk variable
                'm1_i0_t0', 'm1_i1', 'm1_i2', 'm1_c0', 'm1_c1', 'm1_c2', # post reset variables
                'agg0', 'snap0', 'snap1', 'snap2'] # post aggregation variables
    assert names == expected

    polys = result.mode_to_polys['m1']

    # 4 steps because invariant is allowed to be false for the final step
    assert 4 <= len(polys) <= 5, "expected invariant to become false after 4/5 steps"

    assert_verts_is_box(polys[0], [[0, 1], [1, 1]])
    assert_verts_is_box(polys[1], [[1, 2], [1, 1]])
    assert_verts_is_box(polys[2], [[2, 3], [1, 1]])
    assert_verts_is_box(polys[3], [[3, 4], [1, 1]])

    polys = result.mode_to_polys['m2']

    assert_verts_is_box(polys[0], [[1, 4], [3, 3]])
    assert_verts_is_box(polys[1], [[1, 4], [4, 4]])

def test_redundant_invariants():
    'test removing of redundant invariants'

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
 
    # dynamics: x' = 1, y' = 1, a' = 0
    mode.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    # invariant: x <= 2.5
    mode.set_invariant([[1, 0, 0]], [2.5])

    # initial set has x0 = [0, 1]
    init_lpi = lputil.from_box([(0, 1), (0, 1), (1, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # settings, step size = 0.1
    settings = HylaaSettings(0.1, 5.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    result = Core(ha, settings).run(init_list)

    # check last cur_state to ensure redundant constraints were not added
    assert result.last_cur_state.lpi.get_num_rows() == 3 + 2*3 + 1 # 3 for basis matrix, 2*3 for initial constraints

def test_redundant_inv_transition():
    'test removing of redundant invariants with a transition'

    ha = HybridAutomaton()

    mode1 = ha.new_mode('mode1')
 
    # dynamics: x' = 1, y' = 1, a' = 0
    mode1.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    # invariant: x <= 2.5
    mode1.set_invariant([[1, 0, 0]], [2.5])

    mode2 = ha.new_mode('mode2')
    mode2.set_dynamics([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    ha.new_transition(mode1, mode2).set_guard([[-1, 0, 0]], [-2.5]) # x >= 2.5

    # initial set has x0 = [0, 1]
    init_lpi = lputil.from_box([(0, 1), (0, 1), (1, 1)], mode1)
    init_list = [StateSet(init_lpi, mode1)]

    # settings, step size = 0.1
    settings = HylaaSettings(0.1, 5.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    core = Core(ha, settings)
    core.setup(init_list)

    for _ in range(20):
        core.do_step()

    assert core.result.last_cur_state.lpi.get_num_rows() == 3 + 2*3 + 1 # 3 for basis matrix, 2*3 for init constraints

    assert len(core.waiting_list) > 2

    core.plotman.run_to_completion(core.do_step, core.is_finished)

def test_tt_with_invstr():
    'test time-triggered transitions combined with invariant strengthening'

    ha = HybridAutomaton()

    # mode one: x' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 1], [0, 0]])
    m1.set_invariant([[1, 0]], [2.0]) # invariant: x <= 2.0

    # mode two: x' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 1], [0, 0]])
    m2.set_invariant([[1, 1]], [4.0]) # x + a <= 4.0

    # guard: x >= 2.0
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard([[-1, 0]], [-2.0])

    # error x >= 4.0
    error = ha.new_mode('error')
    trans2 = ha.new_transition(m2, error, "to_error")
    trans2.set_guard([[-1, 0]], [-4.0])

    # initial set has x0 = [0, 1]
    init_lpi = lputil.from_box([(0, 1), (0, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 0.1
    settings = HylaaSettings(0.1, 5.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    # run setup() only and check the result
    core = Core(ha, settings)
    core.setup(init_list)

    assert trans1.time_triggered
    assert not trans2.time_triggered # not time-triggered because invariant of m2 is True

def test_inputs_reset():
    'test a system with both inputs and a reset'

    # 2-d system with one input
    # x' = x, y' = u, u \in [1, 1]
    # x0 = 1, y0 = 0
    # inv1: y <= 2.5
    
    # guard: y >= 2.5
    # reset: x := 1, y += 1 [should go from (e^3, 3.0) -> (1, 4.0)]

    # mode2:
    # x' = 2x, y' = Bu, u \in [1, 2], B = 2
    # (1, 4.0) -> (e^2, [6, 8]) -> (e^4, [8, 12])

    # mode2 -> error y >= 12

    ha = HybridAutomaton()
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[1, 0], [0, 0]])
    m1.set_inputs([[1]], [[1], [-1]], [1, -1])
    m1.set_invariant([0, 1], [2.5])

    m2 = ha.new_mode('m2')
    m2.set_dynamics([[2, 0], [0, 0]])
    m2.set_inputs([[2]], [[1], [-1]], [2, -2])

    error = ha.new_mode('error')

    t1 = ha.new_transition(m1, m2)
    t1.set_guard([[0, -1]], [-2.5]) # y >= 2.5
    reset_mat = [[0, 0], [0, 1]]
    min_mat = np.identity(2)
    min_cons = [[1], [-1], [1], [-1]]
    min_rhs = [1, -1, 10, -10]
    t1.set_reset(reset_mat, min_mat, min_cons, min_rhs)

    t2 = ha.new_transition(m2, error)
    t2.set_guard([0, -1], [-12])

    init_box = [[1, 1], [0, 0]]
    lpi = lputil.from_box(init_box, m1)

    settings = HylaaSettings(1.0, 10.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    core = Core(ha, settings)
    init_list = [StateSet(lpi, mode)]
    result = core.run(init_list)

    polys1 = result.mode_to_polys['m1']
    
    assert_verts_is_box(polys1[0], [[1, 1], [0, 0]])
    assert_verts_is_box(polys1[1], [[math.exp(1), math.exp(1)], [1, 1]])
    assert_verts_is_box(polys1[2], [[math.exp(2), math.exp(2)], [2, 2]])
    assert_verts_is_box(polys1[3], [[math.exp(3), math.exp(3)], [3, 3]])
    assert len(polys1) == 4

    # reset: x := 1, y += 1 [should go from (e^3, 3.0) -> (1, 4.0)]
    # (1, 4.0) -> (e^2, [6, 8]) -> (e^4, [8, 12])
    polys2 = result.mode_to_polys['m2']
    assert_verts_is_box(polys2[0], [[1, 1], [4, 4]])
    assert_verts_is_box(polys2[1], [[math.exp(2), math.exp(2)], [6, 8]])
    assert_verts_is_box(polys2[2], [[math.exp(4), math.exp(4)], [8, 12]])
    assert len(polys2) == 3

    # check counterexamples
    assert len(result.counterexample) == 2
    
    c1 = result.counterexample[0]
    assert c1.mode == m1
    assert c1.outgoing_transition == t1
    assert np.allclose(c1.start, [1, 0])
    assert np.allclose(c1.end, [math.exp(3), 3])
    assert len(c1.reset_minkowski_vars) == 2
    assert abs(c1.reset_minkowski_vars[0] - 1) < 1e-9
    assert abs(c1.reset_minkowski_vars[1] - 1) < 1e-9
    
    assert len(c1.inputs) == 3

    for i in c1.inputs:
        assert abs(i - 1) < 1e-9

    c2 = result.counterexample[1]
    assert c2.mode == m2
    assert c2.outgoing_transition == t2
    assert np.allclose(c2.start, [1, 4])
    assert np.allclose(c2.end, [math.exp(4), 12])
    assert len(c2.reset_minkowski_vars) == 0
    assert len(c2.inputs) == 3

    for i in c1.inputs:
        assert abs(i - 2) < 1e-9

def fail_agg_ha():
    'test aggregation with the harmonic oscillator dynamics'

    ha = HybridAutomaton('Deaggregation Example')

    m1 = ha.new_mode('green')
    m1.set_dynamics([[0, 1], [-1, 0]])

    m2 = ha.new_mode('cyan')
    m2.set_dynamics([[0, 0, 0], [0, 0, -2], [0, 0, 0]])

    t1 = ha.new_transition(m1, m2)
    t1.set_guard_true()
    reset_mat = [[1, 0], [0, 1], [0, 0]]
    t1.set_reset(reset_mat, [[0], [0], [1]], [[1], [-1]], [1, -1]) # create 3rd variable with a0 = 1

    mode = ha.modes['green']
    init_lpi = lputil.from_box([(-5, -4), (-0.5, 0.5)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    step = math.pi/4
    settings = HylaaSettings(step, 2*step)
    settings.process_urgent_guards = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.filename = "deaggregation.png"

    core = Core(ha, settings)
    core.setup(init_list)

    core.do_step() # pop
    xs, ys = zip(*core.cur_state.verts(core.plotman))
    plt.plot(xs, ys, 'k-')
    
    core.do_step() # 0
    xs, ys = zip(*core.cur_state.verts(core.plotman))
    plt.plot(xs, ys, 'k-')

    core.do_step() # 1
    xs, ys = zip(*core.cur_state.verts(core.plotman))
    plt.plot(xs, ys, 'k-')
    
    core.do_step() # 2
    assert len(core.waiting_list) > 1
    core.do_step() # pop
    assert not core.waiting_list

    lpi = core.cur_state.lpi

    xs, ys = zip(*core.cur_state.verts(core.plotman))
    plt.plot(xs, ys, 'k-')

    #plt.xlim(-6, -5)
    #plt.ylim(-0.5, 0.5)
    #plt.show()
    

    # minimize 100 * y + x, should give point (-5.5, 0)
    offset = lpi.cur_vars_offset
    cols = [offset, offset + 1]
    pt = lpi.minimize([1, 100, 0], cols)

    assert np.allclose(pt, [-5.5, 0]), "pt was {} (expected [-5.5, 0])".format(pt)
