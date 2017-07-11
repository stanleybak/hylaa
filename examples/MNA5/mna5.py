'''
MNA5 Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, add_time_var
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('MNA_5.mat')

    a_matrix = dynamics['A']
    b_matrix = dynamics['B']

    a_matrix, b_matrix = add_time_var(a_matrix, b_matrix)
    mode.set_dynamics(a_matrix, b_matrix)

    error = ha.new_mode('error')

    dims = a_matrix.shape[0]

    # x1 >= 0.2
    mat = csr_matrix(([-1], [0], [0, 1]), dtype=float, shape=(1, dims))
    rhs = np.array([-0.2], dtype=float)
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)

    # x2 >= 0.15
    mat = csr_matrix(([-1], [1], [0, 1]), dtype=float, shape=(1, dims))
    rhs = np.array([-0.15], dtype=float)
    trans2 = ha.new_transition(mode, error)
    trans2.set_guard(mat, rhs)

    return ha

def make_init_constraints(ha):
    '''returns a Star'''

    values = []
    indices = []
    indptr = []
    constraint_rhs = []

    n = ha.dims

    time_var = n - 2
    affine_var = n - 1

    for dim in xrange(n):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim < time_var:
            lb = ub = 0
        elif dim == time_var:
            lb = ub = 0 # time variable
        elif dim == affine_var:
            lb = ub = 1 # affine variable

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
        if i < 5:
            ub = lb = 0.1
        elif i < 9:
            ub = lb = 0.2
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

def define_settings(ha):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    plot_settings.xdim_dir = (ha.dims - 2)
    plot_settings.ydim_dir = ha.transitions[0].guard_matrix[0]

    # save a video file instead
    # plot_settings.make_video("vid.mp4", frames=220, fps=40)

    plot_settings.num_angles = 3
    plot_settings.max_shown_polys = 2048
    plot_settings.label.y_label = '$y_{1}$'
    plot_settings.label.x_label = 'Time'
    plot_settings.label.title = ''
    #plot_settings.label.axes_limits = (0, 1, -0.007, 0.006)
    plot_settings.plot_size = (12, 10)
    plot_settings.label.big(size=40)

    settings = HylaaSettings(step=0.05, max_time=20.0, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.EXP_MULT

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha = define_ha()
    settings = define_settings(ha)
    init = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
