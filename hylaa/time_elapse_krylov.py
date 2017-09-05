'''
Time Elapse for the Krylov method using CPU or GPU
'''

import time

import numpy as np
from scipy.sparse.linalg import expm

from hylaa.krylov_interface import KrylovInterface
from hylaa.timerutil import Timers

def format_secs(sec):
    'convert seconds (float) to a human-readable string'

    rv = ""

    if sec < 60:
        rv = "{:.2f} secs".format(sec)
    elif sec < 60 * 60:
        rv = "{:.2f} mins".format(sec / 60.0)
    elif sec < 60 * 60 * 48:
        rv = "{:.2f} hours".format(sec / 60.0 / 60.0)
    else:
        rv = "{:.2f} days".format(sec / 60.0 / 60.0 / 24.0)

    return rv

def make_cur_time_elapse_mat_list(time_elapser):
    '''
    Get the cur_time_elapse matrix at every step.

    This is called one time, and returns a list, element N is the time_elapse_mat at step N
    '''

    settings = time_elapser.settings
    key_dir_mat = time_elapser.key_dir_mat
    dims = time_elapser.dims
    step_time = settings.step
    a_matrix = time_elapser.a_matrix

    # todo: compute the number of krylov iterations...
    krylov_dimension = 15

    print "TODO: compute arnoldi iter based on error after matrix exp and multiplication"
    arnoldi_iter = min(krylov_dimension, a_matrix.shape[0])

    if settings.simulation.krylov_use_gpu:
        if settings.print_output:
            print "Initializing GPU (use 'sudo nvidia-smi -pm 1' if slow)"

        Timers.tic("initialize gpu")
        KrylovInterface.set_use_gpu(True)
        Timers.toc("initialize gpu")

        if settings.print_output:
            print "Initialized\n"

    KrylovInterface.set_use_profiling(settings.simulation.krylov_profiling)

    rv = []
    rv.append(np.array(key_dir_mat.todense(), dtype=float)) # step zero

    # add zeros (allocate storage for result)
    for step in xrange(0, time_elapser.settings.num_steps):
        rv.append(np.zeros(key_dir_mat.shape, dtype=float))

    Timers.tic("krylov preallocate and load dynamics")

    # make sure we can allocated a single initial vector (stride = 1)
    KrylovInterface.preallocate_memory(arnoldi_iter, 1, a_matrix.shape[0], key_dir_mat.shape[0], error_on_fail=True)
    KrylovInterface.load_a_matrix(a_matrix) # load a_matrix into device memory
    KrylovInterface.load_key_dir_matrix(key_dir_mat) # load key direction matrix into device memory

    # try to allocate multiple initial vectors (larger stride)
    stride = 32

    while stride >= 1:
        error_on_fail = (stride == 1)

        print "Trying to allocate stride = {}".format(stride)
        success = KrylovInterface.preallocate_memory(arnoldi_iter, stride, a_matrix.shape[0],
                                                     key_dir_mat.shape[0], error_on_fail=error_on_fail)

        if success:
            break

        # memory didn't fit on device... try a smaller number of parallel initial vectors
        stride = stride / 2

    Timers.toc("krylov preallocate and load dynamics")

    if settings.print_output:
        if stride < dims:
            print "\nUsing stride of {} (systems has {} dims)".format(stride, dims)
        else:
            print "\nComputing arnoldi for all {} dims in parallel".format(dims)

    start = last_print = time.time()

    for start_vec in xrange(0, dims, stride):
        end_vec = min(start_vec + stride, dims)

        num_vec = end_vec - start_vec

        if settings.print_output:
            now = time.time()

            if now - last_print > 1.0: # print every second
                last_print = now

                frac = float(start_vec) / dims

                if frac > 1e-9:
                    elapsed_sec = now - start
                    total_sec = elapsed_sec / frac
                    eta_sec = total_sec - elapsed_sec

                    eta = format_secs(eta_sec)

                    print "Arnoldi Parallel {} / {} ({:.2f}%, ETA: {})".format(start_vec, dims, 100.0 * frac, eta)


        Timers.tic('krylov arnoldi_parallel')
        h_list, pv_list = KrylovInterface.arnoldi_parallel(start_vec)
        Timers.toc('krylov arnoldi_parallel')

        for index in xrange(num_vec):
            dim = start_vec + index
            h_mat = h_list[index]
            pv_mat = pv_list[index]

            h_mat = h_mat[:-1, :]
            pv_mat = pv_mat[:, :-1]

            #print "h_matrix trimmed shape = {}".format(h_mat.shape)
            #print "pv_matrix trimmed shape = {}".format(pv_mat.shape)

            Timers.tic('krylov expm (first step)')
            matrix_exp = expm(h_mat * step_time)
            Timers.toc('krylov expm (first step)')

            # save the first step's corresponding column
            cur_col = matrix_exp[:, 0]

            Timers.tic('multiply by pv_mat')
            rv[1][:, dim] = np.dot(pv_mat, cur_col)
            Timers.toc('multiply by pv_mat')

            Timers.tic('krylov expm (other steps)')
            for step in xrange(2, time_elapser.settings.num_steps + 1):
                cur_col = np.dot(matrix_exp, cur_col)

                Timers.tic('multiply by pv_mat')
                rv[step][:, dim] = np.dot(pv_mat, cur_col)
                Timers.toc('multiply by pv_mat')

            Timers.toc('krylov expm (other steps)')

    if settings.simulation.krylov_profiling:
        KrylovInterface.print_profiling_data()

    if settings.print_output:
        elapsed = format_secs(time.time() - start)
        print "Krylov Computation Total Time: {}\n".format(elapsed)

    return rv
