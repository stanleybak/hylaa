'''
MNA5 Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton
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

    # x1 >= 0.2
    mat = csr_matrix(([-1], [0], [0, 1]), dtype=float, shape=(1, dims))
    #rhs = np.array([-0.2], dtype=float) # safe
    rhs = np.array([-0.1], dtype=float) # unsafe
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)

    # x2 >= 0.15
    mat = csr_matrix(([-1], [1], [0, 1]), dtype=float, shape=(1, dims))
    rhs = np.array([-0.15], dtype=float)
    trans2 = ha.new_transition(mode, error)
    trans2.set_guard(mat, rhs)

    return ha

def make_init_constraints(ha):
    '''returns a tuple: (Star, fixed_dim_tuple_list)'''

    values = []
    indices = []
    indptr = []
    constraint_rhs = []
    fixed_dim_tuples = []

    n = ha.dims
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

        if lb == ub:
            fixed_dim_tuples.append((dim, lb))
        else:
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

    return (init_mat, init_rhs, fixed_dim_tuples)

def make_init_star(ha, hylaa_settings):
    '''returns a tuple: a star and a list of fixed dimensions'''

    init_mat, init_rhs, fixed_dim_list = make_init_constraints(ha)

    return Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs), fixed_dim_list

def define_settings(ha):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    settings = HylaaSettings(step=0.1, max_time=2.5, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.KRYLOV

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha = define_ha()
    settings = define_settings(ha)
    init, fixed_dim_list = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init, fixed_dim_list)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
