'''
Tests for Hylaa core object. Made for use with py.test
'''

import math
import numpy as np

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings
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
