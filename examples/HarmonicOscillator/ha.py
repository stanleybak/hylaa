'''
Harmonic Oscillator (with time) Example in Hylaa-Continuous
'''

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, add_time_var
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton'''

    ha = LinearHybridAutomaton()

    a_matrix = np.array([[0, 1], [-1, 0]], dtype=float)
    a_matrix = csc_matrix(a_matrix, dtype=float)
    a_matrix = add_time_var(a_matrix)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix)

    # error if x >= 5
    error = ha.new_mode('error')
    trans = ha.new_transition(mode, error)
    guard_matrix = np.array([-1, 0, 0, 0], dtype=float)
    trans.set_guard(csr_matrix(guard_matrix, dtype=float), np.array([-5], dtype=float))

    return ha

def define_init(ha, hylaa_settings):
    '''returns a star'''

    values = []
    indices = []
    indptr = []

    constraint_rhs = []

    time_var = ha.dims - 2
    affine_var = ha.dims - 1

    for dim in xrange(ha.dims):
        if dim == 0:
            lb = -6
            ub = -5
        elif dim == 1:
            lb = 0
            ub = 1
        elif dim < time_var:
            lb = ub = 0
        elif dim == time_var:
            lb = ub = 0 # time variable
        elif dim == affine_var:
            lb = ub = 1 # affine variable
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        # upper bound
        values.append(1)
        indices.append(dim)
        indptr.append(2*dim)
        constraint_rhs.append(ub)

        # lower bound
        values.append(-1)
        indices.append(dim)
        indptr.append(2*dim+1)
        constraint_rhs.append(-lb)

    indptr.append(len(values))
    constraint_matrix = csr_matrix((values, indices, indptr), shape=(2*ha.dims, ha.dims), dtype=float)
    constraint_rhs = np.array(constraint_rhs, dtype=float)

    return Star(hylaa_settings, constraint_matrix, constraint_rhs, ha.modes['mode'])

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    # save a video file instead
    # plot_settings.make_video("vid.mp4", frames=220, fps=40)

    plot_settings.num_angles = 128
    plot_settings.max_shown_polys = 2048
    plot_settings.label.y_label = 'y'
    plot_settings.label.x_label = 'x'
    plot_settings.label.title = ''
    plot_settings.plot_size = (8, 8)
    #plot_settings.label.big(size=40)

    settings = HylaaSettings(step=0.1, max_time=6.0, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.MATRIX_EXP

    return settings

def run_hylaa(hylaa_settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha = define_ha()
    init = define_init(ha, hylaa_settings)

    engine = HylaaEngine(ha, hylaa_settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(define_settings())
