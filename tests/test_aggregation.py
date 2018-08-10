'''
Tests for Hylaa aggregation. Made for use with py.test
'''

import math
import random

import numpy as np

#import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet, TransitionPredecessor, AggregationPredecessor
from hylaa import lputil, lpplot
from hylaa.lpinstance import LpInstance

from util import pair_almost_in, assert_verts_is_box

def test_box_aggregate2():
    'tests box aggregation'

    mode = HybridAutomaton().new_mode('mode_name')

    lpi1 = lputil.from_box([[0, 1], [0, 1]], mode)
    lpi2 = lputil.from_box([[1, 2], [1, 2]], mode)

    agg_dirs = np.identity(2)

    # box aggregation
    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs, mode)

    verts = lpplot.get_verts(lpi)
    assert_verts_is_box(verts, [[0, 2], [0, 2]])

    # test setting basis matrix after aggregation
    lputil.set_basis_matrix(lpi, np.identity(2))

    verts = lpplot.get_verts(lpi)
    assert_verts_is_box(verts, [[0, 2], [0, 2]])

    lputil.set_basis_matrix(lpi, -1 * np.identity(2))

    verts = lpplot.get_verts(lpi)
    assert_verts_is_box(verts, [[-2, 0], [-2, 0]])

def test_rotated_aggregate():
    'tests rotated aggregation'

    mode = HybridAutomaton().new_mode('mode_name')
    lpi1 = lputil.from_box([[0, 1], [0, 1]], mode)
    lpi2 = lputil.from_box([[1, 2], [1, 2]], mode)

    sq2 = math.sqrt(2) / 2.0

    agg_dirs = np.array([[sq2, sq2], [sq2, -sq2]], dtype=float)

    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs, mode)

    assert lputil.is_point_in_lpi([0, 0], lpi)
    assert lputil.is_point_in_lpi([2, 2], lpi)
    assert lputil.is_point_in_lpi([1, 2], lpi)
    assert lputil.is_point_in_lpi([2, 1], lpi)
    assert lputil.is_point_in_lpi([0, 1], lpi)
    assert lputil.is_point_in_lpi([1, 0], lpi)

    verts = lpplot.get_verts(lpi)

    assert len(verts) == 5

    for p in [(0.5, -0.5), (-0.5, 0.5), (2.5, 1.5), (1.5, 2.5)]:
        assert pair_almost_in(p, verts)

    assert verts[0] == verts[-1]


def test_box_aggregate3():
    'tests box aggregation with 3 boxes'

    mode = HybridAutomaton().new_mode('mode_name')
    
    lpi1 = lputil.from_box([[-2, -1], [-0.5, 0.5]], mode)
    lpi2 = lpi1.clone()
    lpi3 = lpi1.clone()

    basis2 = np.array([[0, 1], [-1, 0]], dtype=float)
    lputil.set_basis_matrix(lpi2, basis2)

    basis3 = np.array([[-1, 0], [0, -1]], dtype=float)
    lputil.set_basis_matrix(lpi3, basis3)

    plot_vecs = lpplot.make_plot_vecs(256, offset=0.1) # use an offset to prevent LP dir from being aligned with axis

    # bounds for lpi1 should be [[-2, -1], [-0.5, 0.5]]
    verts = lpplot.get_verts(lpi1, plot_vecs=plot_vecs)
    assert_verts_is_box(verts, [[-2, -1], [-0.5, 0.5]])

    # bounds for lpi2 should be [[-0.5, 0.5], [1, 2]]
    verts = lpplot.get_verts(lpi2, plot_vecs=plot_vecs)
    assert_verts_is_box(verts, [[-0.5, 0.5], [1, 2]])

    # bounds for lpi3 should be [[2, 1], [-0.5, 0.5]]
    verts = lpplot.get_verts(lpi3, plot_vecs=plot_vecs)
    assert_verts_is_box(verts, [[2, 1], [-0.5, 0.5]])
 
    # box aggregation, bounds should be [[-2, 2], [-0.5, 2]]
    agg_dirs = np.identity(2)
    lpi = lputil.aggregate([lpi1, lpi2, lpi3], agg_dirs, mode)

    verts = lpplot.get_verts(lpi, plot_vecs=plot_vecs)
    assert_verts_is_box(verts, [[-2, 2], [-0.5, 2]])

def test_aggregate_on_subspace():
    '''
    test aggregation when the dynamics and sets are only on a subspace. 
    '''

    # dynamics are x' == 1, y' == 0, a' == 0
    # lpi1 is [0, 1] x [0, 1] x [1, 1]
    # lpi2 is [3, 4] x [0, 1] x [1, 1]

    # aggregation shouldn't need to introduce a variable along the y direction

    mode = HybridAutomaton().new_mode('mode_name')
    lpi1 = lputil.from_box([[0, 1], [0, 1], [1, 1]], mode)
    lpi2 = lputil.from_box([[4, 5], [0, 1], [1, 1]], mode)

    #a_csr = csr_matrix(np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float))
    sqr = math.sqrt(2) / 2
    agg_dirs = np.array([[1, 0, 0], [0, sqr, sqr], [0, sqr, -sqr]], dtype=float)

    # box aggregation
    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs, mode)

    # lpi1 corners
    for pt in [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]:
        assert lputil.is_point_in_lpi(pt, lpi)

    # lpi2 corners
    for pt in [(4, 0, 1), (4, 1, 1), (5, 0, 1), (5, 1, 1)]:
        assert lputil.is_point_in_lpi(pt, lpi)

    # make sure we have new variable names
    names = lpi.get_names()

    expected_names = ["m0_i0", "m0_i1", "m0_i2", "m0_c0", "m0_c1", "m0_c2"]

    assert names == expected_names

def test_aggregate_self():
    '''
    test aggregation on an identical set.
    '''

    mode = HybridAutomaton().new_mode('mode_name')
    lpi1 = lputil.from_box([[-2, -1], [-10, 20], [100, 200]], mode)
    lpi2 = lputil.from_box([[-2, -1], [-10, 20], [100, 200]], mode)

    agg_dirs = np.identity(3)

    # box aggregation
    lpi = lputil.aggregate([lpi1, lpi2], agg_dirs, mode)
    
    assert lpi.is_feasible()

    verts = lpplot.get_verts(lpi, xdim=0, ydim=1)
    assert_verts_is_box(verts, [[-2, -1], [-10, 20]])

    verts = lpplot.get_verts(lpi, xdim=0, ydim=2)
    assert_verts_is_box(verts, [[-2, -1], [100, 200]])

    # make sure no extra variables in lp
    names = lpi.get_names()

    expected_names = ["m0_i0", "m0_i1", "m0_i2", "m0_c0", "m0_c1", "m0_c2"]

    assert names == expected_names

    assert lpi.get_num_rows() == 3 + 3*2

def test_reorthogonalize_matrix():
    'tests the reorthgonalize_matrix function'

    mat = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0.5]], dtype=float)
    out = lputil.reorthogonalize_matrix(mat, 3)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(out, expected)

    mat = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0.5]], dtype=float)
    out = lputil.reorthogonalize_matrix(mat, 3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(out, expected)

    sqr = math.sqrt(2) / 2
    mat = np.array([[1, 1], [0, 0], [2, 2]], dtype=float)
    out = lputil.reorthogonalize_matrix(mat, 2)
    assert np.allclose(out[0], np.array([sqr, sqr], dtype=float))

    mat = np.array([[1, 1, 0, 0, 0], [0, 0, 2, 0, 0], [2, 1, 2.5, 0, 0]], dtype=float)
    out = lputil.reorthogonalize_matrix(mat, 5)

    assert out.shape == (5, 5)
    assert np.allclose(out[0], np.array([sqr, sqr, 0, 0, 0], dtype=float))
    assert np.allclose(out[1], np.array([0, 0, 1, 0, 0], dtype=float))
    assert np.allclose(out[2][2:], np.array([0, 0, 0], dtype=float))

    for a, row_a in enumerate(out):
        assert abs(np.linalg.norm(row_a) - 1.0) < 1e-6, "rows should be normalized"

        for b, row_b in enumerate(out):
            if a == b:
                continue

            assert np.dot(row_a, row_b) < 1e-6, "rows should be orthononal"

def test_aggregate3():
    'tests aggregation of 3 sets, inspired by the harmonic oscillator system'

    mode = HybridAutomaton().new_mode('mode_name')
    lpi1 = lputil.from_box([[0, 1], [0, 1]], mode)

    # middle set is a diamond
    mat = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    s = 3.5
    rhs = [6+s, -(6-s), s, s]
    lpi2 = lputil.from_constraints(mat, rhs, mode)
    
    lpi3 = lputil.from_box([[5, 6], [5, 6]], mode)

    lpi_list = [lpi1, lpi2, lpi3]
    verts = []

    for lpi in lpi_list:
        verts += lpplot.get_verts(lpi)

        #xs, ys = zip(*lpplot.get_verts(lpi))
        #plt.plot(xs, ys, 'k-')

    random.seed(0)

    for _ in range(10):
        random_mat = np.random.rand(2, 2)
        agg_dirs = lputil.reorthogonalize_matrix(random_mat, 2)

        lpi = lputil.aggregate(lpi_list, agg_dirs, mode)

        #xs, ys = zip(*lpplot.get_verts(lpi))
        #plt.plot(xs, ys, 'r--')

        for vert in verts:
            assert lputil.is_point_in_lpi(vert, lpi)

    #plt.show()
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
    settings.stdout = HylaaSettings.STDOUT_DEBUG
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
    assert isinstance(unagg_state.predecessor.premode_lpi, LpInstance)
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

    # m1 dynamics: x' == 1, y' == 0, x0: [-3, -2], y0: [0, 1], step: 1.0
    # m1 invariant: x + y <= 0
    # m1 -> m2 guard: x + y >= 0 and y <= 0.5, reset = [[0, -1, 0], [1, 0, 0]] (x' = -y, y' = x, remove a)
    # m2 dynamics: x' == 0, y' == 0
    # time bound: 4
    # expected result: last state is line (not box!) from (0, 0) to (-0.5, -0.5) 

    ha = HybridAutomaton()

    # mode one: x' = 1, y' = 0, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

    # mode two: x' = 0, y' = 1 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 0], [0, 0]])

    # invariant: x + y <= 0
    m1.set_invariant([[1, 1, 0]], [0])

    # guard: x + y == 0 & y <= 0.5
    trans1 = ha.new_transition(m1, m2, 'trans1')
    trans1.set_guard([[-1, -1, 0], [1, 1, 0], [0, 1, 0]], [0, 0, 0.5])
    #trans1.set_reset(np.identity(3)[:2])
    trans1.set_reset(np.array([[0, -1, 0], [1, 0, 0]], dtype=float))

    # initial set has x0 = [-3, -2], y = [0, 1], a = 1
    init_lpi = lputil.from_box([(-3, -2), (0, 1), (1, 1)], m1)
    init_list = [StateSet(init_lpi, m1)]

    # settings, step size = 1.0
    settings = HylaaSettings(1.0, 4.0)
    settings.stdout = HylaaSettings.STDOUT_NONE
    settings.plot.plot_mode = PlotSettings.PLOT_NONE

    settings.aggregation = HylaaSettings.AGG_BOX
    settings.aggregation_add_guard = True

    core = Core(ha, settings)
    result = core.run(init_list)

    lpi = result.last_cur_state.lpi

    # 2 basis matrix rows, 4 init constraints rows, 6 rows from guard conditions (2 from each)
    assert lpi.get_num_rows() == 2 + 4 + 6

    verts = result.last_cur_state.verts(core.plotman)
    assert len(verts) == 3
    assert verts[0] == verts[-1]
    
    assert pair_almost_in((0, 0), verts)
    assert pair_almost_in((-0.5, -0.5), verts)

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
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True
    settings.plot.xdim_dir = 0
    settings.plot.ydim_dir = {'m1': 1, 'm2': 2}

    result = Core(ha, settings).run(init_list)

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

def test_agg_ha():
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

    core = Core(ha, settings)
    core.setup(init_list)

    core.do_step() # pop
    #xs, ys = zip(*core.cur_state.verts(core.plotman))
    #plt.plot(xs, ys, 'k-')
    
    core.do_step() # 0
    #xs, ys = zip(*core.cur_state.verts(core.plotman))
    #plt.plot(xs, ys, 'k-')

    core.do_step() # 1
    #xs, ys = zip(*core.cur_state.verts(core.plotman))
    #plt.plot(xs, ys, 'k-')
    
    core.do_step() # 2
    assert len(core.waiting_list) > 1

    #for state in core.waiting_list:
    #    xs, ys = zip(*state.verts(core.plotman))
    #    plt.plot(xs, ys, 'k-')
    
    core.do_step() # pop
    assert not core.waiting_list

    lpi = core.cur_state.lpi

    #xs, ys = zip(*core.cur_state.verts(core.plotman))
    #plt.plot(xs, ys, 'r--')
    #plt.show()

    assert lputil.is_point_in_lpi((-5.5, 0, 1), lpi)

def test_agg_no_counterexample():
    'test that aggregation to error does not create a counterexample'

    # m1 dynamics: x' == 1, y' == 0, x0, y0: [0, 1], step: 1.0
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
    settings.stdout = HylaaSettings.STDOUT_DEBUG
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.plot.store_plot_result = True

    result = Core(ha, settings).run(init_list)

    assert result.counterexample is None
