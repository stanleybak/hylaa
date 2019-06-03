'''
Building Example in Hylaa. This is the building example from 

ARCH-COMP19 Category Report: Continuous and Hybrid Systems with Linear Continuous Dynamics

originally from

H.-D. Tran, L. V. Nguyen, and T. T. Johnson. Large-scale linear systems from order-reduction. In
Proc. of ARCH16. 3rd International Workshop on Applied Verification for Continuous and Hybrid
Systems, 2017.

This example demonstrates:

- verification of a linear system with time-varying inputs
- doing plots over time without introducing new variables
- adding custom lines to the plot
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix

from matplotlib import collections

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha(limit):
    '''make the hybrid automaton and return it'''

    ha = HybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('build.mat')
    a_matrix = dynamics['A']
    b_matrix = csc_matrix(dynamics['B'])

    mode.set_dynamics(csr_matrix(a_matrix))

    # 0.8 <= u1 <= 1.0
    u_mat = [[1.0], [-1.0]]
    u_rhs = [1.0, -0.8]

    mode.set_inputs(b_matrix, u_mat, u_rhs)

    error = ha.new_mode('error')

    y1 = dynamics['C'][0]
    mat = csr_matrix(y1, dtype=float)

    trans1 = ha.new_transition(mode, error)
    rhs = np.array([-limit], dtype=float) # safe
    trans1.set_guard(mat, rhs) # y3 >= limit

    return ha

def make_init(ha):
    '''returns list of initial states'''

    bounds_list = []

    dims = list(ha.modes.values())[0].a_csr.shape[0]

    for dim in range(dims):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim == 25:
            lb = -0.0001
            ub = 0.0001
        else:
            lb = ub = 0

        bounds_list.append((lb, ub))

    mode = ha.modes['mode']
    init_lpi = lputil.from_box(bounds_list, mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list

def define_settings(ha, limit):
    'get the hylaa settings object'

    step = 0.0025
    max_time = 1.0
    settings = HylaaSettings(step, max_time)

    #settings.interval_guard_optimization = False
    #settings.time_elapse.scipy_sim.max_step = 0.001

    #settings.time_elapse.scipy_sim.rtol = 1e-9
    #settings.time_elapse.scipy_sim.atol = 1e-12

    settings.stdout = stdout = HylaaSettings.STDOUT_VERBOSE

    plot_settings = settings.plot

    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE

    #plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    #plot_settings.filename = 'building.mp4'
        
    plot_settings.xdim_dir = None
    plot_settings.ydim_dir = ha.transitions[0].guard_csr[0].toarray()[0]

    plot_settings.label.y_label = '$y_{1}$'
    plot_settings.label.x_label = 'Time'
    plot_settings.label.title = 'Building (Uncertain Inputs)'
    #plot_settings.label.axes_limits = (0.4, 0.6, -0.0002, -0.0001)
    plot_settings.plot_size = (12, 8)
    plot_settings.label.big(size=36)

    settings.stop_on_concrete_error = False
    settings.make_counterexample = False

    line = [(0.0, -limit), (max_time, -limit)]
    lc = collections.LineCollection([line], animated=True, colors=('red'), linewidths=(1), linestyle='dashed')
    plot_settings.extra_collections = [lc]

    return settings

def run_hylaa():
    'Runs hylaa with the given settings'

    #limit = 0.004 # reachable
    limit = 0.005 # unreachable

    ha = define_ha(limit)
    
    tuples = []
    tuples.append((HylaaSettings.APPROX_NONE, "approx_none.png"))
    tuples.append((HylaaSettings.APPROX_CHULL, "approx_chull.png"))
    #tuples.append((HylaaSettings.APPROX_LGG, "approx_lgg.png"))

    for model, filename in tuples:
        settings = define_settings(ha, limit)
        settings.approx_model, settings.plot.filename = model, filename

        init_states = make_init(ha)
        print(f"\nMaking {filename}...")
        Core(ha, settings).run(init_states)

if __name__ == '__main__':
    run_hylaa()
