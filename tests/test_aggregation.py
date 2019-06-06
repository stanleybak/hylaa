'''
Tests for Hylaa aggregation. Made for use with py.test
'''

import math
import random

import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import expm

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil, lpplot
from hylaa.aggdag import OpTransition, AggDagNode
from hylaa.aggstrat import Aggregated

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

def test_chull_ha5():
    'test convex hull aggregation of harmonic oscillator with 5 sets'

    mode = HybridAutomaton().new_mode('mode_name')

    steps = 5
    step_size = math.pi/4

    lpi_list = []

    a_mat = np.array([[0, 1], [-1, 0]], dtype=float)

    for step_num in range(steps):
        box = [[-5, -4], [-0.5, 0.5]]
        
        lpi = lputil.from_box(box, mode)

        t = step_num * step_size
        basis_mat = expm(a_mat * t)
        lputil.set_basis_matrix(lpi, basis_mat)

        lpi_list.append(lpi)
        
    verts = []

    for lpi in lpi_list:
        verts += lpplot.get_verts(lpi)

        xs, ys = zip(*lpplot.get_verts(lpi))
        #plt.plot(xs, ys, 'k-')

    lpi = lputil.aggregate_chull(lpi_list, mode)
    #xs, ys = zip(*lpplot.get_verts(lpi))
    #plt.plot(xs, ys, 'r--')

    #plt.show()

    # test if it's really convex hull
    assert lputil.is_point_in_lpi([0, 4.5], lpi)

    for vert in verts:
        assert lputil.is_point_in_lpi(vert, lpi)

def test_chull():
    'tests aggregation of a cirle of sets using convex hull'

    mode = HybridAutomaton().new_mode('mode_name')

    r = 1.0
    eps = 0.05
    num_sets = 16
    lpi_list = []

    for theta in np.linspace(0, 2*math.pi, num_sets, endpoint=False):
        y = r * math.sin(theta)
        x = r * math.cos(theta)
        
        mat = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        rhs = [x + eps, -(x - eps), y + eps, -(y - eps)]
        lpi = lputil.from_constraints(mat, rhs, mode)
    
        lpi_list.append(lpi)
        
    #verts = []

    #for lpi in lpi_list:
    #    verts += lpplot.get_verts(lpi)

    #    xs, ys = zip(*lpplot.get_verts(lpi))
    #    plt.plot(xs, ys, 'k-')

    lpi = lputil.aggregate_chull(lpi_list, mode)

    #xs, ys = zip(*lpplot.get_verts(lpi))
    #plt.plot(xs, ys, 'r--')

    #for vert in verts:
    #    assert lputil.is_point_in_lpi(vert, lpi)

    #plt.show()

    # test if it's really convex hull
    for theta in np.linspace(0, 2*math.pi, num_sets):
        y = (r + 2*eps) * math.sin(theta)
        x = (r + 2*eps) * math.cos(theta)
        
        assert not lputil.is_point_in_lpi([x, y], lpi)

        y = (r - 2*eps) * math.sin(theta)
        x = (r - 2*eps) * math.cos(theta)

        assert lputil.is_point_in_lpi([x, y], lpi)
        
def test_plain():
    'test plain aggregation of states across discrete transitions'

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

    core = Core(ha, settings)
    result = core.run(init_list)

    # check history
    state = result.last_cur_state

    assert state.mode == m2
    assert len(state.aggdag_op_list) > 1
    
    op0 = state.aggdag_op_list[0]
    op1 = state.aggdag_op_list[1]
    assert isinstance(op0, OpTransition)

    assert len(core.aggdag.roots) == 1
   
    assert op0.child_node.stateset.mode is m2
    assert op0.transition == trans1
    assert op0.parent_node == core.aggdag.roots[0]
    assert isinstance(op0.poststate, StateSet)
    assert op0.step == 1
    assert isinstance(op0.child_node, AggDagNode)
    assert op0.child_node == op1.child_node
    assert op0.child_node not in core.aggdag.roots

    assert len(op0.parent_node.stateset.aggdag_op_list) == 1
    assert op0.parent_node.stateset.aggdag_op_list[0] is None
     
    # check polygons in m2
    polys2 = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m2']]

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

    # use agg_box
    settings.aggstrat.agg_type = Aggregated.AGG_BOX

    core = Core(ha, settings)
    
    result = core.run(init_list)

    lpi = result.last_cur_state.lpi

    # 2 basis matrix rows, 4 init constraints rows, 6 rows from guard conditions (2 from each)
    assert lpi.get_num_rows() == 2 + 4 + 6

    verts = result.last_cur_state.verts(core.plotman)
    assert len(verts) == 3
    assert np.allclose(verts[0], verts[-1])
    
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

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m1']]

    # 4 steps because invariant is allowed to be false for the final step
    assert 4 <= len(polys) <= 5, "expected invariant to become false after 4/5 steps"

    assert_verts_is_box(polys[0], [[0, 1], [1, 1]])
    assert_verts_is_box(polys[1], [[1, 2], [1, 1]])
    assert_verts_is_box(polys[2], [[2, 3], [1, 1]])
    assert_verts_is_box(polys[3], [[3, 4], [1, 1]])

    polys = [obj[0] for obj in result.plot_data.mode_to_obj_list[0]['m2']]

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
    assert len(core.aggdag.waiting_list) > 1

    #for state in core.waiting_list:
    #    xs, ys = zip(*state.verts(core.plotman))
    #    plt.plot(xs, ys, 'k-')
    
    core.do_step() # pop
    assert not core.aggdag.waiting_list

    lpi = core.aggdag.get_cur_state().lpi

    # 3 constraints from basis matrix
    # 2 aggregation directions from premode arnoldi, +1 from null space
    # + 2 more aggregation directions from box (3rd is omited since it's exactly the same as null space direction)

    #print(lpi)
    #xs, ys = zip(*core.cur_state.verts(core.plotman))
    #plt.plot(xs, ys, 'r--')
    #plt.show()

    assert lpi.get_num_rows() == 3 + 2 * (5)
    assert lputil.is_point_in_lpi((-5, 2, 1), lpi)

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

    result = Core(ha, settings).run(init_list)

    assert not result.counterexample

def test_chull_lines():
    'tests aggregation of two lines in 2d using convex hull'

    mode = HybridAutomaton().new_mode('mode_name')

    center = [-5, -1, 7]
    generator = [0.5, 0.1, 1.0]
    lpi = lputil.from_zonotope(center, [generator], mode)

    t1 = math.pi / 3
    a_mat = np.array([[-0.3, 1, 0], [-1, -0.3, 0], [0, 0.1, 1.1]], dtype=float)
    bm = expm(a_mat * t1)
    lputil.set_basis_matrix(lpi, bm)

    lpi_list = [lpi.clone()]

    all_verts = []
    verts = lpplot.get_verts(lpi)
    all_verts += verts
    #xs, ys = zip(*verts)
    #plt.plot(xs, ys, 'k-')

    t2 = t1 + 0.1
    bm = expm(a_mat * t2)
    lputil.set_basis_matrix(lpi, bm)

    lpi_list.append(lpi.clone())

    verts = lpplot.get_verts(lpi)
    all_verts += verts
    #xs, ys = zip(*verts)
    #plt.plot(xs, ys, 'k-')

    chull_lpi = lputil.aggregate_chull(lpi_list, mode)

    #xs, ys = zip(*lpplot.get_verts(chull_lpi))
    #plt.plot(xs, ys, 'r--')

    #plt.show()

    for vert in all_verts:
        assert lputil.is_point_in_lpi(vert, chull_lpi)

def test_chull_drivetrain():
    'convex hull aggregation debugging from drivetrain system'

    mode = HybridAutomaton().new_mode('mode_name')

    center = [-0.0432, -11, 0, 30, 0, 30, 360, -0.0013, 30, -0.0013, 30, 0, 1]
    generator = [0.0056, 4.67, 0, 10, 0, 10, 120, 0.0006, 10, 0.0006, 10, 0, 0]

    lpi = lputil.from_zonotope(center, [generator], mode)

    # neg_angle init dynamics
    a_mat = np.array([ \
        [0, 0, 0, 0, 0, 0, 0.0833333333333333, 0, -1, 0, 0, 0, 0], \
        [13828.8888888889, -26.6666666666667, 60, 60, 0, 0, -5, -60, 0, 0, 0, 0, 116.666666666667], \
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5], \
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, -714.285714285714, -0.04, 0, 0, 0, 714.285714285714, 0, 0, 0], \
        [-2777.77777777778, 3.33333333333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -83.3333333333333], \
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
        [100, 0, 0, 0, 0, 0, 0, -1000, -0.01, 1000, 0, 0, 3], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 1000, 0, 0, 1000, 0, -2000, -0.01, 0, 0], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
        ], dtype=float)

    # plot dimensions
    xdim = 0
    ydim = 1

    step = 5.0E-2
    t1 = 0
    bm = expm(a_mat * t1)
    lputil.set_basis_matrix(lpi, bm)

    lpi_list = [lpi.clone()]

    all_verts = []
    verts = lpplot.get_verts(lpi, xdim=xdim, ydim=ydim)
    all_verts += verts
    #xs, ys = zip(*verts)
    #plt.plot(xs, ys, 'k-')

    t2 = t1 + step
    bm = expm(a_mat * t2)
    lputil.set_basis_matrix(lpi, bm)

    lpi_list.append(lpi.clone())

    verts = lpplot.get_verts(lpi, xdim=xdim, ydim=ydim)
    all_verts += verts
    #xs, ys = zip(*verts)
    #plt.plot(xs, ys, 'k-')

    chull_lpi = lputil.aggregate_chull(lpi_list, mode)

    plot_vecs = lpplot.make_plot_vecs(num_angles=256, offset=0.01)
    verts = lpplot.get_verts(chull_lpi, xdim=xdim, ydim=ydim, plot_vecs=plot_vecs)
    #xs, ys = zip(*verts)
    #plt.plot(xs, ys, 'r--')

    #plt.show()

    for vert in all_verts:
        assert lputil.is_point_in_lpi(vert, chull_lpi)

def test_chull_one_step_inputs():
    'test convex hull with one-step lpi for a system with inputs (bug where current vars was not set correctly)'

    mode = HybridAutomaton().new_mode('mode_name')

    step_size = math.pi/4

    a_mat = np.array([[0, 1], [-1, 0]], dtype=float)

    b_mat = [[1], [0]]
    b_constraints = [[1], [-1]]
    b_rhs = [0.2, 0.2]

    mode.set_dynamics(a_mat)
    mode.set_inputs(b_mat, b_constraints, b_rhs)
    mode.init_time_elapse(step_size)

    box = [[-5, -4], [0.0, 1.0]]
    lpi = lputil.from_box(box, mode)

    lpi_one_step = lpi.clone()
    bm, ie_mat = mode.time_elapse.get_basis_matrix(1)

    lputil.set_basis_matrix(lpi_one_step, bm)
    lputil.add_input_effects_matrix(lpi_one_step, ie_mat, mode)

    lpi_list = [lpi, lpi_one_step]
    chull_lpi = lputil.aggregate_chull(lpi_list, mode)

    # 2 current vars and 2 total input effect vars, so expected to be 4 from the end
    assert chull_lpi.cur_vars_offset == chull_lpi.get_num_cols() - 4, "cur_vars in wrong place"
