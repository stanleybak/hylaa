'''
tests for misc aspects of hylaa
'''


import numpy as np

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa import symbolic

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

def test_symbloic():
    'test symbolic dynamics extraction'

    constant_dict = {'alpha': 10}

    variables = ['x', 'y']

    derivatives = ['0', 'x', '-x', 'x + y', 'x - y', '3*y + 2*x', '2*x - y', 'alpha*x', 'alpha**2*y + 1/(2*alpha) * x']
    expected = [[0, 0], [1, 0], [-1, 0], [1, 1], [1, -1], [2, 3], [2, -1], [10, 0], [0.05, 100]]

    for der, row in zip(derivatives, expected):
        ders = [der, '0']
    
        a_mat = symbolic.make_dynamics(variables, ders, constant_dict)

        assert np.allclose(a_mat[0], row)

    # check with and without affine term
    variables = ['x', 'y']
    ders = ['x - alpha * alpha / 2 + 2* y ', 'y']
    a_mat = symbolic.make_dynamics(variables, ders, constant_dict, has_affine_variable=True)

    print(f"a_mat:\n{a_mat}")

    expected = np.array([[1, 2, -50], [0, 1, 0], [0, 0, 0]], dtype=float)

    assert np.allclose(a_mat, expected)

    # check errors
    try:
        symbolic.make_dynamics(['x', 'y'], ['x + y', 'x * 2 * y'], constant_dict)

        assert False, "expected RuntimeError (nonlinear)"
    except RuntimeError:
        pass

    try:
        symbolic.make_dynamics(['x', 'y'], ['x + y', 'x + y + alpha'], constant_dict)

        assert False, "expected RuntimeError (no affine variable)"
    except RuntimeError:
        pass

    a_mat = symbolic.make_dynamics(['x', 'y'], ['x + y', 'x + y + alpha'], constant_dict, has_affine_variable=True)
    expected = np.array([[1, 1, 0], [1, 1, 10], [0, 0, 0]], dtype=float)

    assert np.allclose(a_mat, expected)
