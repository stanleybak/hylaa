'''
SpaceEx Helicopter Benchmark

Scalable Version that allows replicating the dynamics
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.io import loadmat

from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    num_heli = 1 # 30 dims
    #num_heli = 10 # 300 dims
    #num_heli = 100 # 3000 dims
    #num_heli = 1000 # 30k dimensions
    #num_heli = 10000 # 300k dimensions
    #num_heli = 33334 # 1 million dimensions -> expected GPU memory @ 32 arnoldi iterations = 0.25 GB
    #num_heli = 333334 # 10 million dimensions -> expected GPU memory = 2.5 GB -> 801 seconds (LP time: 80%)
    #num_heli = 2 * 333334 # 20 million dimensions -> expected GPU memory = 5.0 GB => 1649.47 sec (LP time: 80%)

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    a_matrix = heli_a_matrix(num_heli)
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    print "made heli_a_matrix... making output constriants"

    combined_error_condition = True
    error_threshold = 0.45 # 0.45 = safe, 0.4 = unsafe

    if combined_error_condition:
        #error: average of all helicopter's x8 >= 0.45
        data = [1. for _ in xrange(num_heli)]
        indices = [8 + 30*h for h in xrange(num_heli)]
        indptrs = [0, len(data)]

        output_space = csr_matrix((data, indices, indptrs), shape=(1, 30 * num_heli), dtype=float)

        mat = csr_matrix(([-1.0], [0], [0, 1]), shape=(1, 1))
        rhs = np.array([-error_threshold * num_heli], dtype=float)
    else:
        # error: x8 >= 0.45 for each helicopter
        data = [1.] * num_heli
        indices = [8 + 30*h for h in xrange(num_heli)]
        indptrs = xrange(num_heli + 1)

        output_space = csr_matrix((data, indices, indptrs), shape=(num_heli, 30 * num_heli), dtype=float)

        # -x8 <= -0.45 -> x8 >= 0.45
        data = [-1.] * num_heli
        indices = xrange(num_heli)
        indptr = xrange(num_heli + 1)
        mat = csr_matrix((data, indices, indptr), shape=(num_heli, num_heli))

        rhs = np.array([-error_threshold] * num_heli, dtype=float)

    trans = ha.new_transition(mode, error)
    trans.set_guard(output_space, mat, rhs)

    return ha

def heli_a_matrix(num_heli):
    'make the helicopter a matrix'

    dims = 30 * num_heli
    rv = lil_matrix((dims, dims), dtype=float)

    mat = loadmat('heli.mat')['A']

    for instance in xrange(num_heli):
        if instance > 0 and instance % 1000 == 0:
            print "making a matrix {} / {}".format(instance, num_heli)

        offset = instance * 30
        rv[offset:offset+30, offset:offset+30] = mat

    if num_heli > 1000:
        print "converting a_mat to csr_matrix..."

    return csr_matrix(rv)

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    # dim0 = time (initially 0)
    # dim1-28 = heli dynamics
    # dim29 = affine dimension (initially 1)

    # vec1 is <0, 1, 0, 0> with the constraint that 0 <= vec1 <= 1
    # vec2 is <-5, 0, 0, 1> with the constraint that vec2 == 1

    num_heli = ha.dims / 30

    if True:
        bounds_list = []

        for h in xrange(ha.dims):
            ub = lb = 0.0

            if h % 30 == 29: # affine term
                ub = lb = 1.0
            elif h % 30 >= 1 and h % 30 <= 8:
                ub = 0.1
                lb = -0.1

            bounds_list.append((lb, ub))

        init_space, mat, init_rhs = bounds_list_to_init(bounds_list)
    else: # old way

        data = []
        indices = []
        indptrs = [0]

        # each column of init space (csc_matrix) is a vector of the initial space

        # first vector is all the fixed terms, with the constraint that vec1 == 1
        for h in xrange(num_heli):
            data.append(1.0)
            indices.append(30 * h + 29)

        indptrs.append(len(data))

        rhs_row = []

        mat_data = []
        mat_indices = []
        mat_indptrs = [0]

        # initial (affine) row
        rhs_row.append(1)
        rhs_row.append(-1)

        # and eight dims for each helicopter
        for h in xrange(num_heli):
            for i in xrange(1, 9):
                data.append(1.0)
                indices.append(30 * h + i)
                indptrs.append(len(data))

                rhs_row.append(0.1)
                rhs_row.append(0.1)

        for c in xrange(1 + 8 * num_heli):
            mat_data.append(1.)
            mat_indptrs.append(len(mat_data))
            mat_data.append(-1.)
            mat_indptrs.append(len(mat_data))

            mat_indices.append(c)
            mat_indices.append(c)

        mat_width = 8 * num_heli + 1
        mat_height = 2 + 8 * num_heli * 2
        mat = csr_matrix((mat_data, mat_indices, mat_indptrs), dtype=float, shape=(mat_height, mat_width))

        init_space = csc_matrix((data, indices, indptrs), shape=(30*num_heli, 8 * num_heli + 1))
        init_rhs = np.array(rhs_row, dtype=float)

    return Star(hylaa_settings, ha.modes['mode'], init_space, mat, init_rhs)

def define_settings(_):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_FULL

    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 8

    settings = HylaaSettings(step=0.1, max_time=30.0, plot_settings=plot_settings)

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    settings.simulation.sim_mode = SimulationSettings.KRYLOV
    #settings.simulation.krylov_use_gpu = True
    #settings.simulation.krylov_profiling = True
    #settings.simulation.check_answer = True

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT

    settings.simulation.krylov_transpose = True
    settings.simulation.check_answer = True
    #settings.simulation.guard_mode = SimulationSettings.GUARD_FULL_LP

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    print "Creating automaton..."
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
