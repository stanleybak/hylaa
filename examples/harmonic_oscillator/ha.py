'''
Harmonic Oscillator (with time) Example in Hylaa-Continuous
'''
import math

import numpy as np
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, make_constraint_matrix, make_seperated_constraints
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton'''

    ha = LinearHybridAutomaton()

    # with time and affine variable
    a_matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
    a_matrix = csr_matrix(a_matrix, dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')
    dims = a_matrix.shape[0]

    # x1 >= 4.0 & x1 <= 4.0
    mat = csr_matrix(([-1, 1], [0, 0], [0, 1, 2]), dtype=float, shape=(2, dims))
    rhs = np.array([-4.0, 4.0], dtype=float)
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)

    return ha

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    rv = None
    bounds_list = [] # bounds on each dimension

    for dim in xrange(ha.dims):
        if dim == 0: # x == -5
            lb = -5
            ub = -5
        elif dim == 1: # y in [0, 1]
            lb = 0
            ub = 1
        elif dim == 2: # t == 0
            lb = 0
            ub = 0
        elif dim == 3: # a == 1
            lb = 1
            ub = 1
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    if not hylaa_settings.simulation.seperate_constant_vars:
        init_mat, init_rhs = make_constraint_matrix(bounds_list)
        rv = Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs)
    else:
        init_mat, init_rhs, variable_dim_list, fixed_dim_tuples = make_seperated_constraints(bounds_list)

        rv = Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs, \
                  var_list=variable_dim_list, fixed_tuples=fixed_dim_tuples)

    return rv

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_INTERACTIVE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    # save a video file instead
    #plot_settings.make_video("vid.mp4", frames=20, fps=5)

    plot_settings.num_angles = 128
    plot_settings.max_shown_polys = 2048
    plot_settings.label.y_label = 'y'
    plot_settings.label.x_label = 'x'
    plot_settings.label.title = ''
    plot_settings.plot_size = (8, 8)
    #plot_settings.label.big(size=40)

    settings = HylaaSettings(step=math.pi/4, max_time=math.pi, plot_settings=plot_settings)
    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    #settings.simulation.sim_mode = SimulationSettings.MATRIX_EXP

    settings.simulation.guard_mode = SimulationSettings.GUARD_FULL_LP
    settings.simulation.sim_mode = SimulationSettings.KRYLOV
    settings.simulation.seperate_constant_vars = False
    settings.simulation.pipeline_arnoldi_expm = False

    settings.simulation.check_answer = True

    return settings

def run_hylaa(hylaa_settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha = define_ha()
    init = make_init_star(ha, hylaa_settings)

    engine = HylaaEngine(ha, hylaa_settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(define_settings())
