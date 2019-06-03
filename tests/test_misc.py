'''
tests for misc aspects of hylaa
'''

import math
import matplotlib.pyplot as plt

import numpy as np

from hylaa import symbolic, lputil, lpplot
from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.stateset import StateSet
from hylaa.settings import HylaaSettings

from util import assert_verts_equals

def test_step_slow():
    'tests slow-step with non-one step size'

    mode = HybridAutomaton().new_mode('mode_name')
    mode.set_dynamics(np.identity(2))
    mode.set_inputs([[1, 0], [0, 2]], [[1, 0], [-1, 0], [0, 1], [0, -1]], [10, -1, 10, -1])

    mode.init_time_elapse(0.5)

    # do step 1
    _, _ = mode.time_elapse.get_basis_matrix(1)

    # do step 2
    basis_mat, input_mat = mode.time_elapse.get_basis_matrix(2)

    # do step 3
    _, _ = mode.time_elapse.get_basis_matrix(3)

    # go back to step 2 (slow step) and make sure it matches
    slow_basis_mat, slow_input_mat = mode.time_elapse.get_basis_matrix(2)

    assert np.allclose(basis_mat, slow_basis_mat)
    assert np.allclose(input_mat, slow_input_mat)

def test_symbolic_amat():
    'test symbolic dynamics extraction'

    constant_dict = {'alpha': 10}

    variables = ['x', 'y']

    derivatives = ['0', 'x', '-x', 'x + y', 'x - y', '3*y + 2*x', '2*x - y', 'alpha*x', 'alpha**2*y + 1/(2*alpha) * x']
    expected = [[0, 0], [1, 0], [-1, 0], [1, 1], [1, -1], [2, 3], [2, -1], [10, 0], [0.05, 100]]

    for der, row in zip(derivatives, expected):
        ders = [der, '0']
    
        a_mat = symbolic.make_dynamics_mat(variables, ders, constant_dict)

        assert np.allclose(a_mat[0], row)

    # check with and without affine term
    variables = ['x', 'y']
    ders = ['x - alpha * alpha / 2 + 2* y ', 'y']
    a_mat = symbolic.make_dynamics_mat(variables, ders, constant_dict, has_affine_variable=True)

    print(f"a_mat:\n{a_mat}")

    expected = np.array([[1, 2, -50], [0, 1, 0], [0, 0, 0]], dtype=float)

    assert np.allclose(a_mat, expected)

    # check errors
    try:
        symbolic.make_dynamics_mat(['x', 'y'], ['x + y', 'x * 2 * y'], constant_dict)

        assert False, "expected RuntimeError (nonlinear)"
    except RuntimeError:
        pass

    try:
        symbolic.make_dynamics_mat(['x', 'y'], ['x + y', 'x + y + alpha'], constant_dict)

        assert False, "expected RuntimeError (no affine variable)"
    except RuntimeError:
        pass

    a_mat = symbolic.make_dynamics_mat(['x', 'y'], ['x + y', 'x + y + alpha'], constant_dict, has_affine_variable=True)
    expected = np.array([[1, 1, 0], [1, 1, 10], [0, 0, 0]], dtype=float)

    assert np.allclose(a_mat, expected)

def test_symbolic_condition():
    'test symbolic extraction of a condition A x <= b'

    constant_dict = {'deltap': 0.5}
    variables = ['px', 'py']
    
    orig = "px<=deltap & py<=-px*0.7 & py>=px*0.8 + 5.0"

    cond_list = orig.split('&')

    mat, rhs = symbolic.make_condition(variables, cond_list, constant_dict)
    expected_mat = np.array([[1, 0], [0.7, 1], [0.8, -1]], dtype=float)
    expected_rhs = [0.5, 0, -5]

    assert np.allclose(mat, expected_mat)
    assert np.allclose(rhs, expected_rhs)

    for cond in ["0 <= x <= 1", "0 < x", "0 >= y >= -1", "0 <= x >= 0"]: 
        try:
            symbolic.make_condition(["x", "y"], [cond], {})
            assert False, f"expected exception on condition {cond}"
        except RuntimeError:
            pass

    # try again
    cond_list = ['I >= 20']
    mat, rhs = symbolic.make_condition(['x', 'I', 'z'], cond_list, constant_dict, has_affine_variable=True)
    expected_mat = np.array([[0, -1, 0, 0]], dtype=float)
    expected_rhs = [-20]
    assert np.allclose(mat, expected_mat)
    assert np.allclose(rhs, expected_rhs)
    
def test_approx_lgg_inputs():
    'test lgg approximation model with inputs'

    # simple dynamics, x' = 1, y' = 0 + u, a' = 0, u in [0.1, 0.2]
    # step size (tau) 0.02
    # after one step, the input effect size should by tau*V \oplus beta*B
    # we'll manually assign beta to be 0.02, in order to be able to check that the constraints are correct
    # A norm is 1

    tau = 0.05

    a_matrix = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    b_mat = [[0], [1], [0]]
    b_constraints = [[1], [-1]]
    b_rhs = [0.2, -0.1]

    mode = HybridAutomaton().new_mode('mode')
    mode.set_dynamics(a_matrix)
    mode.set_inputs(b_mat, b_constraints, b_rhs)

    init_lpi = lputil.from_box([[0, 0], [0, 0], [1, 1]], mode)
    assert lputil.compute_radius_inf(init_lpi) == 1

    ss = StateSet(init_lpi, mode)
    mode.init_time_elapse(tau)
    assert_verts_equals(lpplot.get_verts(ss.lpi), [(0, 0)])

    ss.apply_approx_model(HylaaSettings.APPROX_LGG)

    assert np.linalg.norm(a_matrix, ord=np.inf) == 1.0

    v_set = lputil.from_input_constraints(mode.b_csr, mode.u_constraints_csc, mode.u_constraints_rhs, mode)
    assert lputil.compute_radius_inf(v_set) == 0.2
    alpha = (math.exp(tau) - 1 - tau) * (1 + 0.2)

    assert_verts_equals(lpplot.get_verts(ss.lpi), \
                        [(0, 0), (tau-alpha, 0.2*tau + alpha), (tau+alpha, 0.2*tau+alpha), (tau+alpha, 0.1*tau-alpha)])

    # note: c gets bloated by alpha as well
    assert (ss.lpi.minimize([0, 0, -1])[ss.lpi.cur_vars_offset + 2]) - (1 + alpha) < 1e-9
    assert (ss.lpi.minimize([0, 0, 1])[ss.lpi.cur_vars_offset + 2]) - (1 - alpha) < 1e-9

    # c is actually growing, starting at (1,1) at x=0 and going to [1-alpha, 1+alpha] at x=tau
    assert_verts_equals(lpplot.get_verts(ss.lpi, xdim=0, ydim=2), \
                    [(0, 1), (tau-alpha, 1+alpha), (tau+alpha, 1+alpha), (tau+alpha, 1-alpha), (tau-alpha, 1-alpha)])

    # ready to start    
    ss.step()

    beta = (math.exp(tau) - 1 - tau) * 0.2

    # note: c gets bloated as well! so now it's [1-epsilon, 1+epsilon], where epsilon=alpha
    # so x will grow by [tau * (1 - alpha), tau * (1 + alpha)]
    expected = [(tau + beta, -beta + tau * 0.1), \
             (tau - beta, -beta + tau * 0.1), \
             (tau - beta, beta + tau * 0.2), \
             ((tau - alpha) + tau * (1 - alpha) - beta, 2*0.2*tau + alpha + beta), \
             ((tau + alpha) + tau * (1 + alpha) + beta, 2*0.2*tau+alpha + beta), \
             ((tau + alpha) + tau * (1 + alpha) + beta, 2*0.1*tau-alpha - beta)]

    #xs, ys = zip(*expected)
    #plt.plot([x for x in xs] + [xs[0]], [y for y in ys] + [ys[0]], 'r-') # expected is red
    
    verts = lpplot.get_verts(ss.lpi)
    #xs, ys = zip(*verts)
    #plt.plot(xs, ys, 'k-+') # computed is black
    #plt.show()

    assert_verts_equals(verts, expected)

    # one more step should work without errors
    ss.step()
