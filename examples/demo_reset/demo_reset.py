'''
Demonstration model for resets with Hylaa

2-d model with x0, y0 both in [0, 0.5], step size 1.0, time bound 10.0
Dynamics in mode 1 are x'= 2, y'=1

Upon reaching x >= 9.9 (after 5 steps), a reset is taken which sets: x' := [0, 1], y' := y - 10

Dynamics in mode 2 are x' = 1, y' = 1

This model demonstrates:
- Affine variable to add clock dynamics (c' == a with a(t) = 0 for all t)
- Reset upon reaching a transition
- Reset has a minkowski sum term, in addition to a reset matrix
- Plot output to an image
'''

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton()

    # mode one: x' = 2, y' = 1, a' = 0 
    m1 = ha.new_mode('m1')
    m1.set_dynamics([[0, 0, 2], [0, 0, 1], [0, 0, 0]])

    # mode two: x' = 1, y' = 1, a' = 0 
    m2 = ha.new_mode('m2')
    m2.set_dynamics([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    # invariant: x <= 9.9
    m1.set_invariant([[1, 0, 0]], [9.9])

    # guard: x >= 9.9
    trans = ha.new_transition(m1, m2, 'transition_name')
    trans.set_guard([[-1, 0, 0]], [-9.9])

    # Assign the reset to the transition:
    #
    #    def set_reset(self, reset_csr=None, reset_minkowski_csr=None, reset_minkowski_constraints_csr=None,
    #              reset_minkowski_constraints_rhs=None):
    #    '''resets are of the form x' = Rx + My, Cy <= rhs, where y are fresh variables
    #    the reset_minowski variables can be None if no new variables are needed. If unassigned, the identity
    #    reset is assumed
    #
    #    x' are the new variables
    #    x are the old variables       
    #    reset_csr is R
    #    reset_minkowski_csr is M
    #    reset_minkowski_constraints_csr is C
    #    reset_minkowski_constraints_rhs is rhs
    #    '''

    # we want the reset to set x' := [0, 1], y' := y - 10
    
    reset_csr = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

    # two new minkowski variables, y0 = [0, 1], y1 = [-10, -10]
    minkowski_csr = [[1, 0], [0, 1], [0, 0]]
    constraints_csr = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    constraints_rhs = [1, 0, -10, 10]

    trans.set_reset(reset_csr, minkowski_csr, constraints_csr, constraints_rhs)

    return ha

def make_init(ha):
    'make the initial states'

    # initial set has x0 = [0, 0.5], y = [0, 0.5], a = 1
    mode = ha.modes['m1']
    init_lpi = lputil.from_box([(0, 0.5), (0, 0.5), (1, 1)], mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(1.0, 10.0) # step size = 1.0, time bound 10.0
    settings.plot.plot_mode = PlotSettings.PLOT_IMAGE
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.filename = "demo_reset.png"

    return settings

def run_hylaa():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    run_hylaa()
