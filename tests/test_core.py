'''
Tests for Hylaa core object. Made for use with py.test
'''

import math
import numpy as np

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.stateset import StateSet
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa import lputil, lpplot, aggstrat

from util import assert_verts_is_box

def test_ha():
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

    assert result.has_concrete_error

    ce = result.counterexample[0]

    # [-5.0, 0.6568542494923828, 0.0, 1.0] -> [4.0, 3.0710678118654737, 2.356194490192345, 1.0]

    assert ce.mode == mode
    assert np.allclose(ce.start, np.array([-5, 0.65685, 0, 1], dtype=float))
    assert np.allclose(ce.end, np.array([4, 3.07106, 2.35619, 1], dtype=float))

    # check the reachable state (should always have x <= 3.5)
    obj_list = result.plot_data.mode_to_obj_list[0][mode.name]

    for obj in obj_list:
        verts = obj[0]
        
        for vert in verts:
            x, _ = vert

            assert x <= 4.9

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

    assert not result.has_aggregated_error and not result.has_concrete_error

    # check the reachable state
    # we would expect at the end that x = [4, 5], t = pi

    obj_list = result.plot_data.mode_to_obj_list[0][mode.name]

    for vert in obj_list[0][0]:
        x, y = vert

        assert abs(y) < 1e-6, "initial poly time is wrong"
        assert abs(-5 - x) < 1e-6 or abs(-4 - x) < 1e-6

    for vert in obj_list[-1][0]:
        x, y = vert

        assert abs(math.pi - y) < 1e-6, "final poly time is wrong"
        assert abs(5 - x) < 1e-6 or abs(4 - x) < 1e-6

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
    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0][mode.name]]

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

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m1']]
    assert len(polys) == 4

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m2']]
    assert len(polys) == 3

    assert result.last_cur_state.cur_steps_since_start[0] == 5

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

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m1']]
    assert len(polys) == 3 # time 0, 1, 2

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m2']]
    assert len(polys) == 3 # times 2, 3, 4

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
    settings.stdout = HylaaSettings.STDOUT_NONE
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
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    core = Core(ha, settings)
    core.setup(init_list)

    for _ in range(20):
        core.do_step()

    assert core.result.last_cur_state.lpi.get_num_rows() == 3 + 2*3 + 1 # 3 for basis matrix, 2*3 for init constraints

    assert len(core.aggdag.waiting_list) > 2

    core.plotman.run_to_completion()

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
    # reset: x := 1, y += 2 [should go from (e^3, 3.0) -> (1, 5.0)]

    # mode2:
    # x' = 2x, y' = Bu, u \in [1, 2], B = 2
    # (1, 5.0) -> (e^2, [7, 9]) -> (e^4, [9, 13])

    # mode2 -> error y >= 13

    ha = HybridAutomaton()
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[1, 0], [0, 0]])
    m1.set_inputs([[0], [1]], [[1], [-1]], [1, -1], allow_constants=True)
    m1.set_invariant([[0, 1]], [2.5])

    m2 = ha.new_mode('m2')
    m2.set_dynamics([[2, 0], [0, 0]])
    m2.set_inputs([[0], [2]], [[1], [-1]], [2, -1])

    error = ha.new_mode('error')

    t1 = ha.new_transition(m1, m2)
    t1.set_guard([[0, -1]], [-2.5]) # y >= 2.5
    reset_mat = [[0, 0], [0, 1]]
    min_mat = np.identity(2)
    min_cons = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    min_rhs = [1, -1, 2, -2]
    t1.set_reset(reset_mat, min_mat, min_cons, min_rhs)

    t2 = ha.new_transition(m2, error)
    t2.set_guard([0, -1], [-13]) # y >= 13

    init_box = [[1, 1], [0, 0]]
    lpi = lputil.from_box(init_box, m1)

    settings = HylaaSettings(1.0, 10.0)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    core = Core(ha, settings)
    init_list = [StateSet(lpi, m1)]
    core.setup(init_list)

    core.do_step() # pop
    core.do_step() # continuous_post() to time 1

    lpi = core.result.last_cur_state.lpi

    assert lpi.get_names() == ['m0_i0', 'm0_i1', 'm0_c0', 'm0_c1', 'm0_ti0', 'm0_ti1', 'm0_I0']

    assert_verts_is_box(lpplot.get_verts(lpi), [[math.exp(1), math.exp(1)], [1, 1]])

    core.do_step() # continuous_post() to time 2
    assert_verts_is_box(lpplot.get_verts(core.result.last_cur_state.lpi), [[math.exp(2), math.exp(2)], [2, 2]])

    core.do_step() # continuous_post() to time 3
    assert_verts_is_box(lpplot.get_verts(core.result.last_cur_state.lpi), [[math.exp(3), math.exp(3)], [3, 3]])

    core.do_step() # trim to invariant
    assert core.aggdag.get_cur_state() is None
    assert len(core.aggdag.waiting_list) == 1

    core.run_to_completion()

    result = core.result

    # reset: x := 1, y += 2 [should go from (e^3, 3.0) -> (1, 5.0)]
    # (1, 5.0) -> (e^2, [7, 9]) -> (e^4, [9, 13])
    polys2 = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m2']]
    assert_verts_is_box(polys2[0], [[1, 1], [5, 5]])
    assert_verts_is_box(polys2[1], [[math.exp(2), math.exp(2)], [7, 9]])
    assert_verts_is_box(polys2[2], [[math.exp(4), math.exp(4)], [9, 13]])
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
    assert abs(c1.reset_minkowski_vars[1] - 2) < 1e-9

    assert len(c1.inputs) == 3
    for i in c1.inputs:
        assert len(i) == 1
        assert abs(i[0] - 1) < 1e-9

    c2 = result.counterexample[1]
    assert c2.mode == m2
    assert c2.outgoing_transition == t2
    assert np.allclose(c2.start, [1, 5])
    assert np.allclose(c2.end, [math.exp(4), 13])
    assert not c2.reset_minkowski_vars
    assert len(c2.inputs) == 2

    for i in c2.inputs:
        assert len(i) == 1
        assert abs(i[0] - 2) < 1e-9

def test_init_unsat():
    'initial region unsat with multiple invariant conditions'

    ha = HybridAutomaton()

    mode = ha.new_mode('A')
    mode.set_dynamics(np.identity(2))
    mode.set_invariant([[1, 0], [1, 0]], [2, 3]) # x <= 2 and x <= 3
    
    # initial set
    lpi1 = lputil.from_box([(10, 11), (0, 1)], mode)
    lpi2 = lputil.from_box([(0, 1), (0, 1)], mode)

    init_list = [StateSet(lpi1, mode), StateSet(lpi2, mode)]

    # settings
    settings = HylaaSettings(1, 5)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    
    core = Core(ha, settings)
    core.run(init_list)    # expect no exception during running

def test_over_time_range():
    'test plotting over time with aggergation (time range)'

    ha = HybridAutomaton()

    mode_a = ha.new_mode('A')
    mode_b = ha.new_mode('B')
 
    # dynamics: x' = a, a' = 0
    mode_a.set_dynamics([[0, 1], [0, 0]])
    mode_b.set_dynamics([[0, 1], [0, 0]])

    # invariant: x <= 2.5
    mode_a.set_invariant([[1, 0]], [2.5])

    trans1 = ha.new_transition(mode_a, mode_b, 'first')
    trans1.set_guard_true()

    # initial set has x0 = [0, 0]
    init_lpi = lputil.from_box([(0, 0), (1, 1)], mode_a)
    init_list = [StateSet(init_lpi, mode_a)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 4.0)
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.process_urgent_guards = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True
    settings.plot.xdim_dir = None
    settings.plot.ydim_dir = 0

    result = Core(ha, settings).run(init_list)

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0][mode_b.name]]
    # expected with aggegregation: [0, 2.5] -> [1, 3.5] -> [2, 4.5] -> [3, 5.5] -> [4, 6.5]

    # 4 steps because invariant is allowed to be false for the final step
    assert len(polys) == 5, "expected invariant to become false after 5 steps"

    for i in range(5):
        assert_verts_is_box(polys[i], [[i, i + 3.0], [i, i + 3.0]])

def test_tt_split():
    'tests time-triggered dynamics where the state is split (based on state) after some amount of time elapses'

    ha = HybridAutomaton()

    # dynamics variable order: [x0, x1, x2, x3, u, t, affine]
    pole = ha.new_mode('pole')
    a_matrix = [ \
        [0, 1, 0, 0, 0, 0, 0], \
        [0, 0, 0.7164, 0, 0.9755, 0, 0], \
        [0, 0, 0, 1, 0, 0, 0], \
        [0, 0, 0.76, 0, 0.46, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0, 0], \
        ]
    pole.set_dynamics(a_matrix)
    # 0.0 <= t & t <= 0.1
    pole.set_invariant([[0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0], ], [0, 0.1, ])

    trans = ha.new_transition(pole, pole, 'b2')
    # x3 <= 1.0229164510965 & x2 <= 2.0244571492076 & x3 > -10.0172335505486 & x2 <= 1.0329331979156 & t >= 0.1
    trans.set_guard([[0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0],
                     [-0, -0, -0, -1, -0, -0, -0],
                     [0, 0, 1, 0, 0, 0, 0], 
                     [-0, -0, -0, -0, -0, -1, -0],
                    ], [1.0229164510965, 2.0244571492076, 10.0172335505486, 1.0329331979156, -0.1, ])

    # Reset:
    # t := 0.0
    # u := 0.2
    reset_mat = [ \
        [1, 0, 0, 0, 0, 0, 0, ], \
        [0, 1, 0, 0, 0, 0, 0, ], \
        [0, 0, 1, 0, 0, 0, 0, ], \
        [0, 0, 0, 1, 0, 0, 0, ], \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ], \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ], \
        [0, 0, 0, 0, 0, 0, 1, ], \
        ]
    reset_minkowski = [ \
        [0, ], \
        [0, ], \
        [0, ], \
        [0, ], \
        [1, ], \
        [0, ], \
        [0, ], \
        ]
    minkowski_constraints = [ \
        [1, ], \
        [-1, ], \
        ]
    minkowski_rhs = [0.2, -0.2]
    trans.set_reset(reset_mat, reset_minkowski, minkowski_constraints, minkowski_rhs)

    # manually run ha.detect_tt_transitions() and check the result
    def print_none(str):
        'suppress printing'
        pass
    
    settings = HylaaSettings(0.05, 0.15)
    settings.plot.plot_mode = PlotSettings.PLOT_IMAGE
    settings.plot.xdim_dir = 0
    settings.plot.ydim_dir = 3
    settings.stdout = HylaaSettings.STDOUT_DEBUG

    init_list = []
    mode = ha.modes['pole']
    mat = [[1, 0, 0, 0, 0, 0, 0], \
        [-1, -0, -0, -0, -0, -0, -0], \
        [0, 1, 0, 0, 0, 0, 0], \
        [-0, -1, -0, -0, -0, -0, -0], \
        [0, 0, 1, 0, 0, 0, 0], \
        [-0, -0, -1, -0, -0, -0, -0], \
        [0, 0, 0, -1, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0, 0], \
        [0, 0, 0, 0, 1, 0, 0], \
        [-0, -0, -0, -0, -1, -0, -0], \
        [0, 0, 0, 0, 0, 1, 0], \
        [-0, -0, -0, -0, -0, -1, -0], \
        [0, 0, 0, 0, 0, 0, 1], \
        [-0, -0, -0, -0, -0, -0, -1], ]
        
    rhs = [0, -0, 0, -0, 0, -0, 1.3, -1.3, 0, -0, 0, -0, 1, -1, ]
    init_list.append(StateSet(lputil.from_constraints(mat, rhs, mode), mode))

    core = Core(ha, settings)
    result = core.run(init_list)    # expect no exception during running

    assert result.last_cur_state.cur_steps_since_start[0] == 3
    assert result.last_cur_state.cur_steps_since_start[1] == 3

def test_zero_dynamics():
    'test a system with zero dynamic (only should process one frame)'

    ha = HybridAutomaton()

    # with time and affine variable
    mode = ha.new_mode('mode')
    mode.set_dynamics([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # initial set
    init_lpi = lputil.from_box([(-5, -5), (0, 1), (0, 0), (1, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # settings
    settings = HylaaSettings(math.pi/4, 20*math.pi)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    
    core = Core(ha, settings)

    core.setup(init_list)
    core.do_step() # pop
    core.do_step() # propagate and remove

    assert core.aggdag.get_cur_state() is None, "cur state should be none, since mode dynamics were zero"

def test_multiple_init_states():
    'test with multiple initial states in the same mode (should NOT do aggregation)'

    ha = HybridAutomaton()

    # with time and affine variable
    mode = ha.new_mode('mode')
    mode.set_dynamics([[0, 0], [0, 0]])

    # initial set
    init_lpi = lputil.from_box([(-5, -4), (0, 1)], mode)
    init_lpi2 = lputil.from_box([(-5, -5), (2, 3)], mode)
    
    init_list = [StateSet(init_lpi, mode), StateSet(init_lpi2, mode)]

    # settings
    settings = HylaaSettings(math.pi/4, math.pi)
    settings.stdout = HylaaSettings.STDOUT_NONE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    
    core = Core(ha, settings)

    core.run(init_list)

def test_stateset_bad_init():
    'test constructing a stateset with a basis matrix that is not the identity (should raise error)'

    # this is from an issue reported by Mojtaba Zarei

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
    mode.set_dynamics([[0, 1], [-1, 0]])

    # initial set
    init_lpi = lputil.from_box([(-5, -5), (0, 1)], mode)
    init_list = [StateSet(init_lpi, mode)]

    # settings
    settings = HylaaSettings(math.pi/4, math.pi)
    settings.stdout = HylaaSettings.STDOUT_NONE
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    
    core = Core(ha, settings)
    result = core.run(init_list)

    # use last result
    stateset = result.last_cur_state
    mode = stateset.mode
    lpi = stateset.lpi

    try:
        init_states = [StateSet(lpi, mode)]
        settings = HylaaSettings(0.1, 0.1)
        core = Core(ha, settings)

        result = core.run(init_states)
        assert False, "assertion should be raised if init basis matrix is not identity"
    except RuntimeError:
        pass
        
def test_tt_09():
    'test time-triggered transition at 0.9 bug'

    # this test is from an issue reported by Mojtaba Zarei
    tt_time = 0.9
    
    ha = HybridAutomaton()

    # the test seems to be sensitive to the a_matrix... my guess is the LP is barely feasible at the tt_time
    a_matrix = np.array(
        [[6.037291088, -4.007840286, 2.870370645, 43.12729646, 10.06751155, 23.26084098, -0.001965587832, 0, 0],
         [3.896645707, -0.03417905392, -9.564966476, 15.25894014, -21.57196438, 16.60548055, 0.03473846441, 0, 0],
         [22.72995871, 14.12055097, -0.9315267908, 136.9851951, -71.66383111, 109.7143863, 0.1169799769, 0, 0],
         [-38.16694597, 3.349061908, -9.10171149, -185.1866526, 9.210877185, -165.8086527, -0.06858712649, 0, 0],
         [46.78596597, 27.7996521, 17.18120319, 285.4632424, -135.289626, 235.9427441, 0.228154713, 0, 0],
         [-8.31135303, 3.243945466, -4.523811735, -39.26067436, -9.385678542, -36.63193931, -0.0008874747046, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

    mode1 = ha.new_mode('mode')
    mode1.set_dynamics(a_matrix)

    # time-triggered invariant: t <= tt_time
    mat = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=float)
    rhs = [tt_time]

    mode1.set_invariant(mat, rhs)
    
    mode2 = ha.new_mode('mode2')
    mode2.set_dynamics(a_matrix)

    # transition, guard: x >= -2 & y > 4 & t >= tt_time

    # transition, guard: t >= 0.9
    mat = np.array([[0, 0, 0, 0, 0, 0, 0, -1, 0]], dtype=float)
    rhs = [-tt_time]
    
    t = ha.new_transition(mode1, mode2)
    t.set_guard(mat, rhs)

    # initial set
    init_box = np.array([[-0.1584, -0.1000],
                         [-0.0124, 0.0698],
                         [-0.3128, 0.0434],
                         [-0.0208, 0.0998],
                         [-0.4895, 0.1964],
                         [-0.0027, 0.0262],
                         [42.40, 42.5],
                         [0, 0], # t(0) = 0
                         [1, 1]]) # affine(0) = 1
    
    init_lpi = lputil.from_box(init_box, mode1)
    init_list = [StateSet(init_lpi, mode1)]

    # settings
    settings = HylaaSettings(0.05, 1.0)
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE #INTERACTIVE

    #settings.plot.xdim_dir = 7 #None
    #settings.plot.ydim_dir = 0
    
    core = Core(ha, settings)
    result = core.run(init_list)

    mode2_list = result.plot_data.mode_to_obj_list[0]['mode2']
    assert len(mode2_list) == 3, f"mode2_list len was {len(mode2_list)}, expected 3 (0.9, 0.95, 1.0)"

def test_unaggregation():
    'test an unaggregated discrete transition'

    ha = HybridAutomaton()

    # mode one: x' = 1, t' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    # mode two: x' = -1, t' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 0, -1], [0, 0, 1], [0, 0, 0]])

    # invariant: t <= 2.5
    m1.set_invariant([[0, 1, 0]], [2.5])

    # guard: t >= 0.5
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard([[0, -1, 0]], [-0.5])

    # error x >= 4.5
    error = ha.new_mode('error')
    trans2 = ha.new_transition(m2, error, "to_error")
    trans2.set_guard([[-1, 0, 0]], [-4.5])

    # initial set has x0 = [0, 0.2], t = [0, 0.2], a = 1
    init_lpi = lputil.from_box([(0, 0.2), (0, 0.2), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 10.0)
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.store_plot_result = True
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    settings.aggstrat = aggstrat.Unaggregated()

    result = Core(ha, settings).run(init_list)

    # expected no exception

    # m2 should be reachable
    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m2']]
    assert len(polys) > 15
