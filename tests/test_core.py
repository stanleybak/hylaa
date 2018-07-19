'''
Tests for Hylaa core object. Made for use with py.test
'''

import math
import numpy as np

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

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

def assert_verts_is_box(verts, box, tol=1e-6):
    '''check that a list of verts is almost equal to the passed-in box using assertions

    box is [[xmin, xmax], [ymin, ymax]]
    '''

    assert len(verts) == 5 and verts[0] == verts[-1]

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

    assert len(polys) == 3, "expected invariant to become false after 3 steps"

    assert_verts_is_box(polys[0], [[0, 1], [0, 1]])

    assert_verts_is_box(polys[1], [[1, 2], [1, 2]])

    assert_verts_is_box(polys[2], [[2, 2.5], [2, 3]])

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
