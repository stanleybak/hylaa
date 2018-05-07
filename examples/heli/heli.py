'''
SpaceEx Helicopter Benchmark

Scalable Version that allows replicating the dynamics
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.io import loadmat

from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, TimeElapseSettings
from hylaa.star import Star

def define_ha(stdout=True, use_safe=True, num_heli=1, full_io_space=False):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    if stdout:
        print "making a matrix"

    mode = ha.new_mode('mode')
    a_matrix = heli_a_matrix(num_heli)
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    if stdout:
        print "made heli_a_matrix ({} dims)... making output constriants with safe={}".format(
            a_matrix.shape[0], use_safe)

    error_threshold = 0.45 if use_safe else 0.4 # 0.45 = safe, 0.4 = unsafe

    if not full_io_space:
        #error: average of all helicopter's x8 >= 0.45
        data = [1.] * num_heli
        indices = [7 + 28*h for h in xrange(num_heli)]
        indptrs = [0, len(data)]

        output_space = csr_matrix((data, indices, indptrs), shape=(1, 28 * num_heli), dtype=float)

        mat = csr_matrix(([-1.0], [0], [0, 1]), shape=(1, 1))
        rhs = np.array([-error_threshold * num_heli], dtype=float)
    else:
        # full i/o space
        output_space = csr_matrix(np.identity(28 * num_heli))

        data = [-1.] * num_heli
        indices = [7 + 28 * h for h in xrange(num_heli)]
        indptrs = [0, len(data)]

        mat = csr_matrix((data, indices, indptrs), shape=(1, 28 * num_heli), dtype=float)
        rhs = np.array([-error_threshold * num_heli], dtype=float)

    mode.set_output_space(output_space)
    trans = ha.new_transition(mode, error)
    trans.set_guard(mat, rhs)

    return ha

def heli_a_matrix(num_heli):
    'make the helicopter a matrix'

    mat = csr_matrix(loadmat('heli28.mat')['A']) # single helicopter dynamics
    dims_per_heli = mat.shape[0]

    dims = num_heli * dims_per_heli

    mat_data = [n for n in mat.data]
    mat_indptr = [n for n in mat.indptr[:-1]] # exclude the last one
    data = []
    indices = []
    indptr = []

    for instance in xrange(num_heli):
        col_offset = instance * dims_per_heli

        indices += [col_offset + n for n in mat.indices]
        indptr += [len(data) + n for n in mat_indptr]
        data += mat_data

    indptr.append(len(data)) # add the last one

    return csr_matrix((data, indices, indptr), shape=(dims, dims), dtype=float)

def make_init_star(ha, hylaa_settings, full_io_space=False):
    '''returns a star'''

    bounds_list = []
    n = ha.modes.values()[0].a_matrix_csr.shape[0]

    for h in xrange(n):
        ub = lb = 0.0

        if h % 28 >= 0 and h % 28 <= 7:
            ub = 0.1
            lb = -0.1

        bounds_list.append((lb, ub))

    init_space, init_mat, init_mat_rhs, init_range_tuples = bounds_list_to_init(bounds_list, full_space=full_io_space)

    return Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_mat_rhs, \
                init_range_tuples=init_range_tuples)

def define_settings(_):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE

    plot_settings.xdim_dir = None
    plot_settings.ydim_dir = 7

    settings = HylaaSettings(step=0.1, max_time=30.0, plot_settings=plot_settings)

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    settings.time_elapse.method = TimeElapseSettings.KRYLOV
    #settings.simulation.krylov_use_gpu = True
    #settings.simulation.krylov_profiling = True
    #settings.simulation.check_answer = True

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT

    #settings.simulation.check_answer = True
    #settings.simulation.guard_mode = SimulationSettings.GUARD_FULL_LP

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha = define_ha()
    settings = define_settings(ha)

    print "Defining initial states..."
    init = make_init_star(ha, settings)

    print "Making HylaaEngine..."
    engine = HylaaEngine(ha, settings)

    print "Starting computation..."
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
