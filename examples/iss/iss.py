'''
International Space Station Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('iss.mat')
    a_matrix = dynamics['A']
    b_matrix = dynamics['B']

    # append b_matrix to a_matrix, by adding a column to A for each input (the extra row is all zeros)
    print b_matrix.shape
    exit()

    a_matrix, b_matrix = add_time_var(a_matrix, b_matrix)

    mode.set_dynamics(a_matrix, b_matrix)

    error = ha.new_mode('error')

    # add two more variables due to the time term
    guard_matrix = csr_matrix(add_zero_cols(dynamics['C'][2], 2)) # extract y3

    #trans1 = ha.new_transition(mode, error)
    #trans1.set_guard(guard_matrix, np.array([-0.0005], dtype=float)) # y3 <= -0.0005

    trans2 = ha.new_transition(mode, error)
    trans2.set_guard(-guard_matrix, np.array([-0.0005], dtype=float)) # y3 >= 0.0005

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
        if dim < time_var:
            lb = -0.0001
            ub = 0.0001
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
            lb = 0.0
            ub = 0.1
        elif i == 1:
            lb = 0.8
            ub = 1.0
        elif i == 2:
            lb = 0.9
            ub = 1.0
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
    plot_settings.label.y_label = '$y_{3}$'
    plot_settings.label.x_label = 'Time'
    plot_settings.label.title = ''
    #plot_settings.label.axes_limits = (0, 1, -0.007, 0.006)
    plot_settings.plot_size = (12, 10)
    plot_settings.label.big(size=40)

    settings = HylaaSettings(step=0.005, max_time=20.0, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    settings.simulation.guard_mode = SimulationSettings.GUARD_DECOMPOSED

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
