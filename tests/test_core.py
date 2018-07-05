'''
Tests for Hylaa core object. Made for use with py.test
'''

import math
import numpy as np

from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def test_ha_line_arch18():
    'test for the harmonic oscillator example with line initial set (from ARCH 2018 paper)'

    ha = HybridAutomaton()

    # with time and affine variable
    mode = ha.new_mode('mode')
    a_matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    csr_mat = csr_matrix(np.array([[1., 0, 0, 0], [-1., 0, 0, 0]], dtype=float))
    rhs = np.array([4.0, -4.0], dtype=float)
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(csr_mat, rhs)

    # initial set
    init_lpi = lputil.from_box([(-5, -5), (0, 1), (0, 0), (1, 1)])
    init_list = [StateSet(init_lpi, mode)]

    # settings
    settings = HylaaSettings(math.pi/4, math.pi)
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    
    core = Core(ha, settings)
    result = core.run(init_list)

    assert not result.safe

    ce = result.counterexample[0]

    assert ce.mode_name == 'mode'
    assert np.allclose(ce.start, np.array([-5, -0.66, 0, 1], dtype=float))
    assert np.allclose(ce.end, np.array([4, 3.07, 2.36, 1], dtype=float))
