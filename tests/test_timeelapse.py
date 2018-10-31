'''
tests for time_elapse functionality
'''


import numpy as np

from hylaa.hybrid_automaton import HybridAutomaton

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
