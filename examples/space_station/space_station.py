'''
This is the International Space Station example used in the ARCH reachability tools competition in 2018.

It is a 271-dimensional continuous system, with three time-varying inputs.

This model demonstrates:
- Loading dynamics from matlab .mat files
- Error condition that is a disjunction
- Plotting with a linear combination of the system states
- Plotting over time without introducing a new variable
'''

from scipy.io import loadmat

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('iss.mat')
    a_matrix = dynamics['A']
    b_matrix = dynamics['B']

    mode.set_dynamics(a_matrix)

    # input bounds
    # 0 <= u1 <= 0.1
    # 0.8 <= u2 <= 1.0
    # 0.9 <= u3 <= 1.0
    bounds_mat = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    bounds_rhs = [0.1, 0, 1.0, -0.8, 1.0, -0.9]
    mode.set_inputs(b_matrix, bounds_mat, bounds_rhs)

    error = ha.new_mode('error')

    # the third output defines the unsafe condition
    y3 = dynamics['C'][2]

    limit = 0.0005
    #limit = 0.0007

    # Error condition: y3 * x <= -limit OR y3 >= limit
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(y3, [-limit])

    trans2 = ha.new_transition(mode, error)
    trans2.set_guard(-1 * y3, [-limit])

    return ha

def make_init(ha):
    'make the initial states'

    # initial set has every variable as [-0.0001, 0.0001]
    mode = ha.modes['mode']

    dims = mode.a_csr.shape[0]
    init_box = dims * [[-0.0001, 0.0001]]
    init_lpi = lputil.from_box(init_box, mode)
    
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(0.1, 20.0) # step size = 0.1, time bound 20.0
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.stdout = HylaaSettings.STDOUT_VERBOSE
    settings.plot.filename = "space_station.png"

    settings.plot.xdim_dir = None # x dimension will be time

    dynamics = loadmat('iss.mat')
    y3 = dynamics['C'][2]
    settings.plot.ydim_dir = y3.toarray()[0] # use y3 for the y plot direction

    return settings

def run_hylaa():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    run_hylaa()
