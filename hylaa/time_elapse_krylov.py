'''
Time Elapse for the Krylov method using CPU or GPU
'''

import numpy as np
from scipy.sparse.linalg import expm

from hylaa.gpu_krylov_sim import GpuKrylovSim
from hylaa.containers import SimulationSettings

def make_cur_time_elapse_mat_list(time_elapser):
    'get the cur_time_elapse matrix at every step'

    settings = time_elapser.settings
    key_dir_mat = time_elapser.key_dir_mat
    dims = time_elapser.dims
    step_time = settings.step
    a_matrix = time_elapser.a_matrix
    krylov_iter = min(time_elapser.settings.simulation.krylov_dimension, a_matrix.shape[0])

    # Choose using CPU (host_memory) or GPU (device_memory)

    use_gpu = settings.simulation.sim_mode is SimulationSettings.KRYLOV_GPU
    GpuKrylovSim.set_use_gpu(use_gpu)

    rv = []
    rv.append(np.array(key_dir_mat.todense(), dtype=float)) # step zero

    GpuKrylovSim.load_matrix(a_matrix.tocsr()) # load a_matrix into device memory
    GpuKrylovSim.load_keyDirSparseMatrix(key_dir_mat) # load key direction matrix into device memory
    cusp_h_tuples, _ = GpuKrylovSim.arnoldi_parallel(dims, krylov_iter)

    for step in xrange(1, time_elapser.settings.num_steps + 1):
        exp_h_tuples = []

        for i in xrange(0, dims):
            exp_h_tuples.append(expm(step * step_time * cusp_h_tuples[i, :, :]))

        # TODO: get rid of time_elapser.key_dir_mat.shape[0] and dims in this call
        cur_step_result = GpuKrylovSim.getKeySimResult_parallel(time_elapser.key_dir_mat.shape[0], dims, \
            krylov_iter, exp_h_tuples)

        print "step {}, type = {}, result = {}".format(step, type(cur_step_result), repr(cur_step_result))

        rv.append(cur_step_result)

    return rv
