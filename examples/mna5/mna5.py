'''
MNA5 Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, TimeElapseSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('MNA_5.mat')

    a_matrix = dynamics['A']
    col_ptr = [n for n in a_matrix.indptr]
    rows = [n for n in a_matrix.indices]
    data = [n for n in a_matrix.data]

    b_matrix = dynamics['B']
    num_inputs = b_matrix.shape[1]

    for u in xrange(num_inputs):
        rows += [n for n in b_matrix[:, u].indices]
        data += [n for n in b_matrix[:, u].data]
        col_ptr.append(len(data))

    combined_mat = csc_matrix((data, rows, col_ptr), \
    shape=(a_matrix.shape[0] + num_inputs, a_matrix.shape[1] + num_inputs))

    mode.set_dynamics(csr_matrix(combined_mat))

    error = ha.new_mode('error')
    dims = combined_mat.shape[0]

    # error condition x1 >= 0.2 and x2 >= 0.15
    output_space = csr_matrix(([1., 1.], [0, 1], [0, 1, 2]), shape=(2, dims))
    mode.set_output_space(output_space)

    # x1 >= 0.2
    mat = csr_matrix(([-1], [0], [0, 1]), dtype=float, shape=(1, 2))
    rhs = np.array([-0.2], dtype=float) # safe
    #rhs = np.array([-0.1], dtype=float) # unsafe
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)

    # x2 >= 0.15
    mat = csr_matrix(([-1], [1], [0, 1]), dtype=float, shape=(1, 2))
    rhs = np.array([-0.15], dtype=float)
    trans2 = ha.new_transition(mode, error)
    trans2.set_guard(mat, rhs)

    return ha

def make_init_star(ha, hylaa_settings):
    '''returns a Star'''

    n = ha.dims
    bounds_list = [] # bounds on each dimension
    input_start_dim = 10913

    for dim in xrange(n):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim < input_start_dim:
            lb = ub = 0
        elif dim >= input_start_dim and dim < input_start_dim + 5:
            # first 5 inputs
            lb = ub = 0.1
        elif dim >= input_start_dim + 5 and dim < input_start_dim + 9:
            # second 4 inputs
            lb = ub = 0.2
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    init_space, init_mat, init_mat_rhs, init_range_tuples = bounds_list_to_init(bounds_list)

    return Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_mat_rhs, \
                init_range_tuples=init_range_tuples)

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE
    #plot_settings.plot_mode = PlotSettings.PLOT_FULL
    plot_settings.xdim_dir = None
    plot_settings.ydim_dir = 0
    plot_settings.max_shown_polys = None

    settings = HylaaSettings(step=0.001, max_time=20.0, plot_settings=plot_settings)
    settings.time_elapse.method = TimeElapseSettings.KRYLOV


    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha = define_ha()
    settings = define_settings()
    init = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
