'''
Demonstration model for inputs and resets with Hylaa

The first mode has the harmonic oscillator dynamics with inputs (x' = y + u1, y' = -x + u2) starting from 
[-5, -4], [0, 1], with u1, u2 in [-0.5, 0.5]

Upon pi/2 time elapsing, there is a time-triggered transition which sets y' := y * -0.5.

The second mode is the harmonic oscillator in the other direction, with no inputs, x' = -y, y' = x.

This model demonstrates:
- Time-varying inputs in the dynamics (mode 1)
- Affine variable to add clock dynamics (c' == a with a(t) = 0 for all t)
- Reset upon reaching a transition
- Time-triggered transition (invariant is opposite of guard)
- Modes with different numbers of state variables (mode 1 has four, mode 2 has two)
- Plot output to an image
'''

import math

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton()

    # mode one: x' = y + u1, y' = -x + u2, c' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

    b_mat = [[1, 0], [0, 1], [0, 0], [0, 0]]
    b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_rhs = [0.5, 0.5, 0.5, 0.5]
    m1.set_inputs(b_mat, b_constraints, b_rhs)

    # mode two: x' = -y, y' = x, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, -1], [1, 0]])

    # m1 invariant: c <= pi/2
    m1.set_invariant([[0, 0, 1, 0]], [math.pi / 2])

    # guard: c >= pi/2
    trans = ha.new_transition(m1, m2)
    trans.set_guard([[0, 0, -1, 0]], [-math.pi/2])

    # Assign the reset to the transition
    # y *= -1, also the reset is what is used to change the number of system variables (m1 has four vars, m2 has two)
    
    reset_csr = [[1, 0, 0, 0], [0, -1, 0, 0]]

    # no minkowski sum terms
    trans.set_reset(reset_csr)

    return ha

def make_init(ha):
    'make the initial states'

    # initial set has x0 = [-5, -4], y = [0, 1], c = 0, a = 1
    mode = ha.modes['m1']
    init_lpi = lputil.from_box([(-5, -4), (0, 1), (0, 0), (1, 1)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(math.pi / 8, math.pi) # step size = pi/8, time bound pi
    settings.plot.plot_mode = PlotSettings.PLOT_IMAGE
    settings.stdout = HylaaSettings.STDOUT_NORMAL
    settings.plot.filename = "demo_inputs_reset.png"

    return settings

def run_hylaa():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    run_hylaa()
