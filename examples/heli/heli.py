'''
SpaceEx Helicopter Benchmark

Scalable Version that allows replicating the dynamics
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.io import loadmat

from hylaa.hybrid_automaton import LinearHybridAutomaton
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    #num_heli = 1 # 30 dims
    #num_heli = 10 # 300 dims
    #num_heli = 100 # 3000 dims
    #num_heli = 1000 # 30k dimensions
    #num_heli = 10000 # 300k dimensions
    num_heli = 33334 # 1 million dimensions -> expected GPU memory @ 32 arnoldi iterations = 0.25 GB
    #num_heli = 333334 # 10 million dimensions -> expected GPU memory = 2.5 GB -> 801 seconds (LP time: 80%)
    #num_heli = 2 * 333334 # 20 million dimensions -> expected GPU memory = 5.0 GB => 1649.47 sec (LP time: 80%)

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    a_matrix = heli_a_matrix(num_heli)
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    print "made heli_a_matrix... making output constriants"

    # x1 >= 4.0 & x1 <= 4.0
    # error: x8 >= 0.45  (safe) or 0.4 (unsafe)
    data = [1.] * num_heli
    indices = [8 + 30*h for h in xrange(num_heli)]
    indptrs = xrange(num_heli + 1)

    output_space = csr_matrix((data, indices, indptrs), shape=(num_heli, 30 * num_heli), dtype=float)

    threshold = 0.45
    # -x8 <= -0.45 -> x8 >= 0.45
    mat = np.zeros((num_heli, num_heli), dtype=float)

    for h in xrange(num_heli):
        mat[h, h] = -1

    rhs = np.array([-threshold] * num_heli, dtype=float)

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

    mat_width = 8 * num_heli + 1
    mat_height = 2 + 8 * num_heli * 2

    print "mat space = {} MB".format(8 * mat_width * mat_height / 1024. / 1024.)
    mat = np.zeros((mat_height, mat_width), dtype=float)
    mat[0, 0] = 1
    rhs_row.append(1)

    mat[1, 0] = -1
    rhs_row.append(-1)

    # next is each initial vector, with the constraints -0.1 <= vec <= 0.1, for each helicopter's x1-x8
    for h in xrange(num_heli):
        for i in xrange(1, 9):
            data.append(1.0)
            indices.append(30 * h + i)
            indptrs.append(len(data))

            mat[2 + 8*h + i - 1, 0] = 1
            rhs_row.append(0.1)

            mat[2 + 8*h + i, 0] = -1
            rhs_row.append(0.1)

    init_space = csc_matrix((data, indices, indptrs), shape=(30*num_heli, 8 * num_heli + 1))
    init_rhs = np.array(rhs_row, dtype=float)

    rv = Star(hylaa_settings, ha.modes['mode'], init_space, mat, init_rhs)

    return rv

def define_settings(_):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 8

    settings = HylaaSettings(step=0.1, max_time=30.0, plot_settings=plot_settings)

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT
    settings.simulation.sim_mode = SimulationSettings.KRYLOV
    #settings.simulation.krylov_use_gpu = True
    #settings.simulation.krylov_profiling = True
    #settings.simulation.check_answer = True

    #settings.simulation.sim_mode = SimulationSettings.EXP_MULT

    #settings.simulation.check_answer = True
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
