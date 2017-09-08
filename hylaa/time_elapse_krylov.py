'''
Time Elapse for the Krylov method using CPU or GPU
'''

import time
import sys

from multiprocessing import Pool

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

def reallocate_memory(arnoldi_iter, dims, key_dirs, start_stride):
    'try to allocate multiple initial vectors (larger stride). returns the new stride'
    Timers.tic("reallocate memory")

    stride = start_stride

    while stride >= 1:
        error_on_fail = (stride == 1)

        success = KrylovInterface.preallocate_memory(arnoldi_iter, stride, dims, key_dirs, error_on_fail=error_on_fail)

        if success:
            break

        # memory didn't fit on device... try a smaller number of parallel initial vectors
        stride = stride / 2

    Timers.toc("reallocate memory")

    return stride

def get_krylov_result(arg_tuple):
    '''
    Compute the krylov result and return a list of tuples (step, dim, col_vec)
    '''

    step_size, num_steps, dim, h_mat, pv_mat = arg_tuple

    rv = []

    h_mat = h_mat[:-1, :].copy()
    pv_mat = pv_mat[:, :-1].copy()

    matrix_exp = expm(h_mat * step_size)
    cur_col = matrix_exp[:, 0]
    cur_result = np.dot(pv_mat, cur_col)
    rv.append((1, dim, cur_result))

    for s in xrange(2, num_steps + 1):
        cur_col = np.dot(matrix_exp, cur_col)
        cur_result = np.dot(pv_mat, cur_col)
        rv.append((s, dim, cur_result))

    return rv

def init_krylov(time_elapser, arnoldi_iter):
    '''
    initialize krylov interface for the computation

    returns a list of empty matrices to be filled in by the subsequent computation
    '''

    settings = time_elapser.settings
    key_dir_mat = time_elapser.key_dir_mat
    a_matrix = time_elapser.a_matrix

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
    for _ in xrange(0, time_elapser.settings.num_steps):
        rv.append(np.zeros(key_dir_mat.shape, dtype=float))

    Timers.tic("krylov preallocate and load dynamics")

    # make sure we can allocated a single initial vector (stride = 1, arnoldi_iter = 2)
    KrylovInterface.preallocate_memory(arnoldi_iter, 1, a_matrix.shape[0], key_dir_mat.shape[0], error_on_fail=True)
    KrylovInterface.load_a_matrix(a_matrix) # load a_matrix into device memory
    KrylovInterface.load_key_dir_matrix(key_dir_mat) # load key direction matrix into device memory

    Timers.toc("krylov preallocate and load dynamics")

    return rv

def relative_error(correct, estimate):
    'compute the relative error between the correct value and an estimate'

    rel_error = 0
    norm = np.linalg.norm(correct)

    if norm > 1e-9:
        diff = correct - estimate
        err = np.linalg.norm(diff)
        rel_error = err / norm

    return rel_error

def get_rel_error(settings, h_mat, pv_mat, arnoldi_iter=None):
    '''
    Get the relative error given the h and pv matrices, for the given number of arnoldi_iterations.
    If arnoldi_iter is None, then use the full passed-in matrices.
    '''

    # use less arnoldi iterations than what's in the matrices
    if arnoldi_iter is not None:
        h_mat = h_mat[:arnoldi_iter, :arnoldi_iter].copy()
        pv_mat = pv_mat[:, :arnoldi_iter].copy()

    small_h_mat = h_mat[:-1, :-1].copy()
    small_pv_mat = pv_mat[:, :-1].copy()

    matrix_exp = expm(h_mat * settings.step)
    cur_col = matrix_exp[:, 0]

    # for accuracy check
    small_matrix_exp = expm(small_h_mat * settings.step)
    small_col = small_matrix_exp[:, 0]

    for _ in xrange(2, settings.num_steps + 1):
        cur_col = np.dot(matrix_exp, cur_col)

        # for accuracy check
        small_col = np.dot(small_matrix_exp, small_col)

    cur_result = np.dot(pv_mat, cur_col)
    small_result = np.dot(small_pv_mat, small_col)

    rel_error = relative_error(cur_result, small_result)

    return rel_error

def get_max_rel_error(settings, dim_list, limit):
    '''
    Get the maximum relative error of the passed-in number of arnoldi iterations.

    returns max_rel_error, h_mat_list, pv_mat_list
    '''

    max_rel_error = 0
    h_list = []
    pv_list = []

    for dim in dim_list:
        Timers.tic('krylov arnoldi_parallel')
        h_mats, pv_mats = KrylovInterface.arnoldi_parallel(dim)
        Timers.toc('krylov arnoldi_parallel')

        h_mat = h_mats[0]
        pv_mat = pv_mats[0]

        h_mat = h_mat[:-1, :].copy()
        h_list.append(h_mat)

        pv_mat = pv_mat[:, :-1].copy()
        pv_list.append(pv_mat)

        rel_error = get_rel_error(settings, h_mat, pv_mat)

        max_rel_error = max(max_rel_error, rel_error)

        if max_rel_error > limit:
            break

    return max_rel_error, h_list, pv_list

def find_iter_with_accuracy(settings, h_list, pv_list):
    '''
    Find the exact number of arnoldi iterations that has the desired accuracy for all of the
    passed in h_mat lists, for the complete reach time.
    '''

    error_limit = settings.simulation.krylov_rel_error_expm_h
    max_iter = h_list[0].shape[0]
    desired_rel_error = settings.simulation.krylov_rel_error_expm_h

    # binary search the number of arnoldi iterations until we go below the accuracy threshold
    top = max_iter
    bottom = max_iter / 2

    while True:
        if top == bottom + 1:
            break

        aiter = (top + bottom) / 2
        max_rel_error = 0

        if settings.print_output:
            print "{}".format(aiter),
            sys.stdout.flush()

        for h_mat, pv_mat in zip(h_list, pv_list):
            rel_error = get_rel_error(settings, h_mat, pv_mat, arnoldi_iter=aiter)
            max_rel_error = max(max_rel_error, rel_error)

            if max_rel_error > error_limit:
                break

        if max_rel_error < desired_rel_error:
            top = aiter
        else:
            bottom = aiter

    return top

def choose_arnoldi_iter(settings, dims, key_dirs, arnoldi_iter):
    '''
    Choose the number of arnoldi iterations based on the desired error. This is done through a
    sampling of a number of vectors and using cauchy error.
    '''

    error_limit = settings.simulation.krylov_rel_error_expm_h

    # sample accuracy in a small number of dimensions
    samples = min(dims, settings.simulation.krylov_rel_error_samples)

    dim_list = [int(d) for d in np.linspace(0, dims-1, num=samples)]

    if settings.print_output:
        print "{}".format(arnoldi_iter),
        sys.stdout.flush()

    cur_rel_error, h_list, pv_list = get_max_rel_error(settings, dim_list, error_limit)

    while cur_rel_error > error_limit:
        arnoldi_iter *= 2

        reallocate_memory(arnoldi_iter, dims, key_dirs, start_stride=1)

        if settings.print_output:
            print "{}".format(arnoldi_iter),
            sys.stdout.flush()

        cur_rel_error, h_list, pv_list = get_max_rel_error(settings, dim_list, error_limit)

    # ok, at this point the error is below the threshold
    # find the exact value of iterations that makes this work for all the samples
    arnoldi_iter = find_iter_with_accuracy(settings, h_list, pv_list)

    return arnoldi_iter

def make_cur_time_elapse_mat_list(time_elapser):
    '''
    Get the cur_time_elapse matrix at every step.

    This is called one time, and returns a list, element N is the time_elapse_mat at step N
    '''

    settings = time_elapser.settings
    dims = time_elapser.a_matrix.shape[0]
    key_dirs = time_elapser.key_dir_mat.shape[0]

    arnoldi_iter = min(2, dims)

    rv = init_krylov(time_elapser, arnoldi_iter)

    if settings.print_output:
        print "Determining number of arnoldi iterations for desired accuracy...",
        sys.stdout.flush()

    Timers.tic("choose arnoldi iterations")
    arnoldi_iter = choose_arnoldi_iter(settings, dims, key_dirs, arnoldi_iter)
    Timers.toc("choose arnoldi iterations")

    pool = Pool()

    if settings.print_output:
        print "\nUsing {} Arnoldi Iterations".format(arnoldi_iter)

    # re-allocate with correct number of arnoldi iterations and larger stride
    stride = reallocate_memory(arnoldi_iter, dims, key_dirs, start_stride=settings.simulation.krylov_max_stride)

    start = last_print = time.time()
    start_vec = 0
    pool_res = None

    for start_vec in xrange(0, dims, stride):
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

        # update result from the previous iteration (skipped on first iteration)
        if pool_res is not None:
            Timers.tic('krylov wait for expm results')
            list_of_results = pool_res.get()
            Timers.toc('krylov wait for expm results')

            Timers.tic('krylov update result list')
            for krylov_result in list_of_results:
                for step, dim, col_vec in krylov_result:
                    rv[step][:, dim] = col_vec

            Timers.toc('krylov update result list')

        # push the computation of the (small) matrix exponential to another thread
        Timers.tic('krylov send expm to another thread')
        args = [(settings.step, settings.num_steps, start_vec + i, h_list[i], pv_list[i]) for i in xrange(len(pv_list))]
        pool_res = pool.map_async(get_krylov_result, args)
        Timers.toc('krylov send expm to another thread')

    # loop ended, update results from the last iteration
    Timers.tic('krylov wait for expm results')
    list_of_results = pool_res.get()
    Timers.toc('krylov wait for expm results')

    Timers.tic('krylov update result list')
    for krylov_result in list_of_results:
        for step, dim, col_vec in krylov_result:
            rv[step][:, dim] = col_vec

    Timers.toc('krylov update result list')

    if settings.simulation.krylov_profiling:
        KrylovInterface.print_profiling_data()

    pool.close()
    pool.join()

    if settings.print_output:
        elapsed = format_secs(time.time() - start)
        print "Krylov Computation Total Time: {}\n".format(elapsed)

    return rv
