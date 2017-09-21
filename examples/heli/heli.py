'''
SpaceEx Helicopter Benchmark

Scalable Version that allows replicating the dynamics
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.io import loadmat

from hylaa.hybrid_automaton import LinearHybridAutomaton, make_constraint_matrix, make_seperated_constraints
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    #num_heli = 1 # 30 dims
    #num_heli = 10 # 300 dims
    #num_heli = 100 # 3000 dims
    #num_heli = 1000 # 30k dimensions
    #num_heli = 10000 # 300k dimensions
    #num_heli = 33334 # 1 million dimensions -> expected GPU memory @ 32 arnoldi iterations = 0.25 GB
    #num_heli = 333334 # 10 million dimensions -> expected GPU memory = 2.5 GB -> 801 seconds (LP time: 80%)
    num_heli = 2 * 333334 # 20 million dimensions -> expected GPU memory = 5.0 GB => 1649.47 sec (LP time: 80%)

    num_key_dirs = 10

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    a_matrix = heli_a_matrix(num_heli)
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    # error = x8 >= 0.45 for the first key_dir matrices
    num_key_dirs = min(num_heli, num_key_dirs)
    data = []
    cols = []
    indptrs = [0]

    for instance in xrange(num_key_dirs):
        offset = instance * 30

        data.append(-1.0)
        cols.append(offset + 8)
        indptrs.append(len(data))

    guard_matrix = csr_matrix((data, cols, indptrs), shape=(num_key_dirs, 30 * num_heli), dtype=float)
    trans = ha.new_transition(mode, error)
    trans.set_guard(guard_matrix, np.array([-0.45] * num_key_dirs, dtype=float)) # x8 >= 0.45 = safe, 0.4 = unsafe

    return ha

def heli_a_matrix(num_heli):
    'make the helicopter a matrix'

    dims = 30 * num_heli
    rv = lil_matrix((dims, dims), dtype=float)

    mat = loadmat('heli.mat')['A']

    for instance in xrange(num_heli):
        offset = instance * 30
        rv[offset:offset+30, offset:offset+30] = mat

    return csr_matrix(rv)

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    rv = None
    bounds_list = []

    affine_dim = 29

    for dim in xrange(ha.dims):
        if dim % 30 == affine_dim:
            lb = ub = 1.0
        elif dim % 30 >= 1 and dim % 30 <= 8:
            lb = -0.1
            ub = 0.1
        else:
            lb = ub = 0.0

        bounds_list.append((lb, ub))

    print "Finishing with bounds... making star"

    if not hylaa_settings.simulation.krylov_seperate_constant_vars or \
            hylaa_settings.simulation.sim_mode != SimulationSettings.KRYLOV:
        init_mat, init_rhs = make_constraint_matrix(bounds_list)
        rv = Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs)
    else:
        init_mat, init_rhs, variable_dim_list, fixed_dim_tuples = make_seperated_constraints(bounds_list)

        rv = Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs, \
                  var_lists=[variable_dim_list], fixed_tuples=fixed_dim_tuples)

    return rv

def define_settings(_):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    #plot_settings.xdim_dir = 0
    #plot_settings.ydim_dir = 8


    settings = HylaaSettings(step=0.1, max_time=30.0, plot_settings=plot_settings)

    settings.simulation.krylov_multithreaded_arnoldi_expm = True
    settings.simulation.krylov_multithreaded_rel_error = False

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    settings.simulation.sim_mode = SimulationSettings.KRYLOV
    settings.simulation.krylov_use_gpu = True
    settings.simulation.krylov_profiling = True
    #settings.simulation.check_answer = True

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT

    #settings.simulation.check_answer = True
    settings.simulation.guard_mode = SimulationSettings.GUARD_FULL_LP

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    print "Creating automaton..."
    ha = define_ha()
    settings = define_settings(ha)

    print "Defining initial states..."
    init = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)

    print "Starting computation..."
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
