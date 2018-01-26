'''
International Space Station Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('iss.mat')
    a_matrix = dynamics['A']

    # a is a csc_matrix
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

    y3 = dynamics['C'][2]
    col_ptr = [n for n in y3.indptr] + num_inputs * [y3.data.shape[0]]
    y3 = csc_matrix((y3.data, y3.indices, col_ptr), shape=(1, y3.shape[1] + num_inputs))
    output_space = csr_matrix(y3)

    #limit = 0.0005
    limit = 0.00017
    trans1 = ha.new_transition(mode, error)
    mat = csr_matrix(([1], [0], [0, 1]), dtype=float, shape=(1, 1))
    rhs = np.array([-limit], dtype=float) # safe
    trans1.set_guard(output_space, mat, rhs) # y3 <= -limit

    trans2 = ha.new_transition(mode, error)
    mat = csr_matrix(([-1], [0], [0, 1]), dtype=float, shape=(1, 1))
    rhs = np.array([-limit], dtype=float) # safe
    trans2.set_guard(output_space, mat, rhs) # y3 >= limit

    return ha

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    bounds_list = []

    for dim in xrange(ha.dims):
        if dim == 270: # input 1
            lb = 0
            ub = 0.1
        elif dim == 271: # input 2
            lb = 0.8
            ub = 1.0
        elif dim == 272: # input 3
            lb = 0.9
            ub = 1.0
        elif dim < 270:
            lb = -0.0001
            ub = 0.0001
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    init_space, init_mat, init_mat_rhs = bounds_list_to_init(bounds_list)

    return Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_mat_rhs)

def define_settings(_):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    max_time = 20.0 # 20.0
    step_size = 0.001
    settings = HylaaSettings(step=step_size, max_time=max_time, plot_settings=plot_settings)
    #settings.simulation.guard_mode = SimulationSettings.GUARD_DECOMPOSED

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    settings.simulation.sim_mode = SimulationSettings.KRYLOV
    #settings.simulation.check_answer = True

    #settings.simulation.krylov_check_all_rel_error = True
    #settings.simulation.krylov_rel_error = 1e-6
    #settings.simulation.krylov_transpose = True
    settings.simulation.krylov_stdout = True

    #settings.skip_step_times = False

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    #print "!!! in iss.py run_hylaa(). Check if early break in arnoldi loop actually helps performance on this example"

    ha = define_ha()
    settings = define_settings(ha)
    init = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
