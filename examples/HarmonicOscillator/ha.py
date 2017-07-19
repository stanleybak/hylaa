'''
Harmonic Oscillator (with time) Example in Hylaa-Continuous
'''
import math

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

    b_matrix = np.array([[1, 0], [0, 1]], dtype=float)
    b_matrix = csc_matrix(b_matrix, dtype=float)

    a_matrix, b_matrix = add_time_var(a_matrix, b_matrix)

    mode = ha.new_mode('mode')

    mode.set_dynamics(a_matrix, b_matrix)

    # error if x >= 7.5
    error = ha.new_mode('error')
    trans = ha.new_transition(mode, error)
    guard_matrix = np.array([-1, 0, 0, 0], dtype=float)
    trans.set_guard(csr_matrix(guard_matrix, dtype=float), np.array([-7.5], dtype=float))

    return ha

def make_init_constraints(ha):
    '''return (init_mat, init_rhs)'''

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

    init_mat = csr_matrix((values, indices, indptr), shape=(2*ha.dims, ha.dims), dtype=float)
    init_rhs = np.array(constraint_rhs, dtype=float)

    return (init_mat, init_rhs)

def make_input_constraints(ha):
    '''return (input_mat, input_rhs)'''

    values = []
    indices = []
    indptr = []

    constraint_rhs = []

    for i in xrange(ha.inputs):
        if i == 0:
            lb = -0.5
            ub = 0.5
        elif i == 1:
            lb = -0.5
            ub = 0.5
        else:
            raise RuntimeError('Unknown input: {}'.format(i))

        # upper bound
        values.append(1)
        indices.append(i)
        indptr.append(2*i)
        constraint_rhs.append(ub)

        # lower bound
        values.append(-1)
        indices.append(i)
        indptr.append(2*i+1)
        constraint_rhs.append(-lb)

    indptr.append(len(values))

    input_mat = csr_matrix((values, indices, indptr), shape=(2*ha.inputs, ha.inputs), dtype=float)
    input_rhs = np.array(constraint_rhs, dtype=float)

    return (input_mat, input_rhs)

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    init_mat, init_rhs = make_init_constraints(ha)
    input_mat, input_rhs = make_input_constraints(ha)

    return Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs, input_mat, input_rhs)
    #return Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs)

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
    settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    #settings.simulation.sim_mode = SimulationSettings.MATRIX_EXP

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
