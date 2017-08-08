'''
Time Elapse for the Krylov method using CPU or GPU
'''

import math

import numpy as np
from scipy.sparse.linalg import expm

from hylaa.gpu_krylov_sim import GpuKrylovSim
from hylaa.containers import SimulationSettings
from hylaa.timerutil import Timers

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

    # add zeros
    for step in xrange(0, time_elapser.settings.num_steps):
        rv.append(np.zeros(key_dir_mat.shape, dtype=float))

    GpuKrylovSim.load_matrix(a_matrix.tocsr()) # load a_matrix into device memory
    GpuKrylovSim.load_keyDirSparseMatrix(key_dir_mat) # load key direction matrix into device memory

    # do a certain number of vectors at a time (stride)
    # GPU stores 2 copies of V matrix during computation
    single_v_size_bytes = 2 * dims * krylov_iter * 8.0
    stride = settings.simulation.gpu_mem_vmatrix_mb * 1024.0 * 1024.0 / single_v_size_bytes
    stride = max(1, int(math.floor(stride)))

    for start_vec in xrange(0, dims, stride):
        end_vec = min(start_vec + stride, dims)

        num_vec = end_vec - start_vec

        Timers.tic('krylov arnoldi_parallel')
        cusp_h_tuples = GpuKrylovSim.arnoldi_parallel(start_vec, end_vec - 1, krylov_iter)
        Timers.toc('krylov arnoldi_parallel')

        for step in xrange(1, time_elapser.settings.num_steps + 1):
            expm_column_list = []

            Timers.tic('krylov expm (first step)')
            mat = step * step_time * cusp_h_tuples[i, :, :]
            exp = expm(mat)
            cur_col = exp[:, 0]
            expm_column_list.append(cur_col)
            Timers.toc('krylov expm (first step)')

            Timers.tic('krylov expm (other steps)')

            for i in xrange(1, num_vec):
                cur_col = np.dot(exp, cur_col)
                expm_column_list.append(cur_col)

            Timers.toc('krylov expm (other steps)')


            # TODO: get rid of time_elapser.key_dir_mat.shape[0] in this call
            # getKeySimResult_parallel(dirMatrix_numRows, numInitVec, numIter, expHt_tuples):
            Timers.tic('krylov getKeySimResult_parallel')

            cur_step_result = GpuKrylovSim.getKeySimResult_parallel(time_elapser.key_dir_mat.shape[0], num_vec, \
                krylov_iter, expm_column_list)
            Timers.toc('krylov getKeySimResult_parallel')

            mat = rv[step]
            mat[:, start_vec:end_vec] = cur_step_result

    return rv
