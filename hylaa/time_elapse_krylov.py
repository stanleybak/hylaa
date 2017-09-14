'''
Time Elapse for the Krylov method using CPU or GPU
'''

import math
import time
import sys

from multiprocessing import Pool

import numpy as np
from scipy.sparse.linalg import expm

from hylaa.krylov_interface import KrylovInterface
from hylaa.timerutil import Timers

def compress_fixed(mat, fixed_tuples):
    'compress the fixed variables in the time_elapse matrix'

    # we need to create a new dense matrix based on the variable reordering
    num_vars = mat.shape[1] - len(fixed_tuples) + 1
    rv = np.zeros((mat.shape[0], num_vars), dtype=float)

    compressed_var_index = 0
    uncompressed_dim = 0

    for dim in xrange(mat.shape[1]):
        col = mat[:, dim]

        if compressed_var_index >= len(fixed_tuples) or dim < fixed_tuples[compressed_var_index][0]:
            # uncompressed dim
            rv[:, uncompressed_dim] = col
            uncompressed_dim += 1
        else:
            # compressed dim
            fixed_val = fixed_tuples[compressed_var_index][1]
            rv[:, -1] += col * fixed_val
            compressed_var_index += 1

    return rv

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

def get_krylov_result(arg_tuple):
    '''
    Compute the krylov result and return a list of tuples (step, dim, col_vec)
    '''

    num_steps, time_step, dim, h_mat, pv_mat = arg_tuple

    rv = []

    h_mat = h_mat[:-1, :].copy()
    pv_mat = pv_mat[:, :-1].copy()

    matrix_exp = expm(h_mat * time_step) # loaded a-mat already takes step-time into account
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

    Timers.tic('initilaizing step zero from key dir mat')
    dense_key_dir_mat = np.array(key_dir_mat.todense(), dtype=float)

    if settings.simulation.seperate_constant_vars:
        dense_key_dir_mat = compress_fixed(dense_key_dir_mat, time_elapser.fixed_tuples)

    rv.append(dense_key_dir_mat) # step zero

    Timers.toc('initilaizing step zero from key dir mat')

    # add zeros (allocate storage for result)
    for _ in xrange(0, time_elapser.settings.num_steps):
        rv.append(np.zeros(rv[0].shape, dtype=float))

    Timers.tic("krylov preallocate and load dynamics")

    # make sure we can allocate with arnoldi_iter = 2
    KrylovInterface.preallocate_memory(arnoldi_iter, a_matrix.shape[0], key_dir_mat.shape[0], error_on_fail=True)
    KrylovInterface.load_a_matrix(a_matrix) # load a_matrix into device memory
    KrylovInterface.load_key_dir_matrix(key_dir_mat) # load key direction matrix into device memory

    Timers.toc("krylov preallocate and load dynamics")

    return rv

def relative_error(correct, estimate):
    'compute the relative error between the correct value and an estimate'

    rel_error = 1e16 # large error is returned if it can't be computed due to numerical issues

    try:
        norm = np.linalg.norm(correct)

        if not math.isinf(norm) and not math.isnan(norm):
            if norm < 1e-9:
                rel_error = 0
            else:
                diff = correct - estimate
                abs_error = np.linalg.norm(diff)

                if not math.isinf(abs_error) and not math.isnan(abs_error):
                    rel_error = abs_error / norm
    except FloatingPointError:
        pass

    assert not math.isinf(rel_error) and not math.isnan(rel_error)

    return rel_error

def get_rel_error(settings, h_mat, pv_mat, arnoldi_iter=None, return_projected_sims=False):
    '''
    Get the relative error given the h and pv matrices, for the given number of arnoldi_iterations.
    If arnoldi_iter is None, then use the full passed-in matrices.

    This compares the error at all time steps.

    If return_projected_sims is True, then a tuple is returned where the second element is list of the
    sim points at each time step.
    '''

    projected_sims = None

    if return_projected_sims:
        projected_sims = []

    # use less arnoldi iterations than what's in the matrices
    if arnoldi_iter is not None:
        h_mat = h_mat[:arnoldi_iter, :arnoldi_iter].copy()
        pv_mat = pv_mat[:, :arnoldi_iter].copy()

    small_h_mat = h_mat[:-1, :-1].copy()
    small_pv_mat = pv_mat[:, :-1].copy()

    matrix_exp = expm(settings.step * h_mat) # step time is already included in loaded a_mat
    cur_col = matrix_exp[:, 0]

    # for accuracy check
    small_matrix_exp = expm(settings.step * small_h_mat) # step time is already included in loaded a_mat
    small_col = small_matrix_exp[:, 0]

    # do the comparison at the first step
    cur_result = np.dot(pv_mat, cur_col)
    small_result = np.dot(small_pv_mat, small_col)
    rel_error = relative_error(cur_result, small_result)

    if return_projected_sims:
        projected_sims.append(cur_result)

    for _ in xrange(2, settings.num_steps + 1):
        cur_col = np.dot(matrix_exp, cur_col)
        small_col = np.dot(small_matrix_exp, small_col)

        # maybe we want to check relative error in the middle as well
        cur_result = np.dot(pv_mat, cur_col)
        small_result = np.dot(small_pv_mat, small_col)
        rel_error = max(rel_error, relative_error(cur_result, small_result))

        if return_projected_sims:
            projected_sims.append(cur_result)

    return rel_error if not return_projected_sims else (rel_error, projected_sims)

def get_max_rel_error(settings, dim_list, limit):
    '''
    Get the maximum relative error of the passed-in number of arnoldi iterations.

    returns max_rel_error, h_mat_list, pv_mat_list
    '''

    max_rel_error = 0
    h_list = []
    pv_list = []

    for dim in dim_list:
        Timers.tic('krylov arnoldi_unit()')
        h_mat, pv_mat = KrylovInterface.arnoldi_unit(dim)
        Timers.toc('krylov arnoldi_unit()')

        h_mat = h_mat[:-1, :].copy()
        h_list.append(h_mat)

        pv_mat = pv_mat[:, :-1].copy()
        pv_list.append(pv_mat)

        rel_error = get_rel_error(settings, h_mat, pv_mat)
        max_rel_error = max(max_rel_error, rel_error)

        if max_rel_error > limit:
            break

    return max_rel_error, h_list, pv_list

def find_iter_with_accuracy(settings, h_list, pv_list, desired_rel_error):
    '''
    Find the exact number of arnoldi iterations that has the desired accuracy for all of the
    passed in h_mat lists, for the complete reach time.
    '''

    max_iter = h_list[0].shape[0]

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

            if max_rel_error > desired_rel_error:
                break

        if max_rel_error < desired_rel_error:
            top = aiter
        else:
            bottom = aiter

    return top

def krylov_sim_fixed_terms(time_elapser):
    '''simulate the fixed effect vector up to the desired relative error'''

    settings = time_elapser.settings
    dims = time_elapser.a_matrix.shape[0]
    key_dirs = time_elapser.key_dir_mat.shape[0]
    error_limit = settings.simulation.krylov_rel_error

    init_vec = time_elapser.fixed_init_vec
    arnoldi_iter = min(dims, 2)

    if settings.print_output:
        print "Computing fixed term using krylov simulation with iterations: ",
        print "{}".format(arnoldi_iter),
        sys.stdout.flush()

    KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

    h_mat, pv_mat = KrylovInterface.arnoldi_vec(init_vec)
    h_mat = h_mat[:-1, :].copy()
    pv_mat = pv_mat[:, :-1].copy()

    cur_rel_error, simulation = get_rel_error(settings, h_mat, pv_mat, arnoldi_iter, True)

    while cur_rel_error > error_limit:
        assert arnoldi_iter <= dims, "arnoldi_iter > dims and still not converging. Try reducing time" + \
            " step and bound, or increasing acceptable error."

        arnoldi_iter = min(dims + 1, arnoldi_iter * 2)

        KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

        if settings.print_output:
            print "{}".format(arnoldi_iter),
            sys.stdout.flush()

        h_mat, pv_mat = KrylovInterface.arnoldi_vec(init_vec)
        h_mat = h_mat[:-1, :].copy()
        pv_mat = pv_mat[:, :-1].copy()

        cur_rel_error, simulation = get_rel_error(settings, h_mat, pv_mat, arnoldi_iter, True)

    return simulation

def choose_arnoldi_iter(time_elapser):
    '''
    Choose the number of arnoldi iterations based on the desired error. This is done through a
    sampling of a number of vectors and using cauchy error.
    '''

    settings = time_elapser.settings
    dims = time_elapser.a_matrix.shape[0]
    key_dirs = time_elapser.key_dir_mat.shape[0]
    arnoldi_iter = min(2, dims)

    if settings.print_output:
        print "Determining number of arnoldi iterations for desired accuracy...",
        sys.stdout.flush()

    if dims <= 2:
        arnoldi_iter = dims
    else:
        error_limit = settings.simulation.krylov_rel_error

        # sample accuracy in a small number of dimensions
        samples = min(dims, settings.simulation.krylov_rel_error_samples)

        if settings.simulation.seperate_constant_vars:
            # only sample in variable dimensions
            var_list = time_elapser.var_list

            dim_list = [var_list[int(d)] for d in np.linspace(0, len(var_list)-1, num=samples)]
        else:
            dim_list = [int(d) for d in np.linspace(0, dims-1, num=samples)]

        # optimization: evaluate the last dimension first (since it's often the highest error one)
        dim_list = [dim_list[-1]] + dim_list[0:-1]

        # eliminate duplicates
        dim_list = list(set(dim_list))

        if settings.print_output:
            print "{}".format(arnoldi_iter),
            sys.stdout.flush()

        cur_rel_error, h_list, pv_list = get_max_rel_error(settings, dim_list, error_limit)

        while cur_rel_error > error_limit:
            assert arnoldi_iter <= dims, "arnoldi_iter > dims and still not converging. Try reducing time" + \
                " step and bound, or increasing acceptable error."

            arnoldi_iter = min(dims + 1, arnoldi_iter * 2)

            KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

            if settings.print_output:
                print "{}".format(arnoldi_iter),
                sys.stdout.flush()

            cur_rel_error, h_list, pv_list = get_max_rel_error(settings, dim_list, error_limit)

        # ok, at this point the error is below the threshold
        # find the exact value of iterations that makes this work for all the samples
        arnoldi_iter = find_iter_with_accuracy(settings, h_list, pv_list, error_limit)

    if settings.print_output:
        print "\nUsing {} Arnoldi Iterations".format(arnoldi_iter)

    return arnoldi_iter

def assign_fixed_terms(time_elapser, rv):
    'assign the fixed effect terms using a krylov simulation'

    Timers.tic("krylov sim fixed terms")
    fixed_sim = krylov_sim_fixed_terms(time_elapser)
    assert len(fixed_sim) == len(rv) - 1
    Timers.toc("krylov sim fixed terms")

    # assign results from fixed_sim
    Timers.tic('krylov update result list')

    for i in xrange(len(fixed_sim)):
        rv[i+1][:, -1] = fixed_sim[i]

    Timers.toc('krylov update result list')

def make_cur_time_elapse_mat_list(time_elapser):
    '''
    Get the cur_time_elapse matrix at every step.

    This is called one time, and returns a list, element N is the time_elapse_mat at step N
    '''

    # numpy raise errors on floating point errors (these should be caught and handled explicitly)
    np.seterr(all='warn', over='raise')

    settings = time_elapser.settings
    dims = time_elapser.a_matrix.shape[0]
    key_dirs = time_elapser.key_dir_mat.shape[0]

    if settings.simulation.pipeline_arnoldi_expm:
        pool = Pool()

    rv = init_krylov(time_elapser, 2)

    if settings.simulation.seperate_constant_vars and not settings.simulation.expm_mult_fixed_terms:
        Timers.tic("krylov assign fixed terms")
        start = time.time()
        assign_fixed_terms(time_elapser, rv)
        diff = time.time() - start
        print "krylov fix sim time: {:.2f}ms".format(diff * 1000)
        Timers.toc("krylov assign fixed terms")

    Timers.tic("choose arnoldi iterations")
    arnoldi_iter = choose_arnoldi_iter(time_elapser)
    Timers.toc("choose arnoldi iterations")

    #print "debug fixed arnoldi_iter=37"
    #arnoldi_iter = 37

    # re-allocate with correct number of arnoldi iterations
    KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

    start = last_print = time.time()
    start_vec = 0
    pool_res = None

    if settings.simulation.seperate_constant_vars:
        variable_dim_list = time_elapser.var_list

        # compute the constant_variable effect
    else:
        variable_dim_list = xrange(0, dims)

    store_dim_index = 0

    for start_vec in variable_dim_list:

        print "start_vec = {}".format(start_vec)

        #if start_vec == 64:
        #    print "debug break at 64"
        #    break

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

        Timers.tic('krylov arnoldi_unit()')
        h_mat, pv_mat = KrylovInterface.arnoldi_unit(start_vec)
        Timers.toc('krylov arnoldi_unit()')

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

        ### compute matrix exp ###
        args = [(settings.num_steps, settings.step, store_dim_index, h_mat, pv_mat)]
        store_dim_index += 1

        if settings.simulation.pipeline_arnoldi_expm:
            # push the computation to another thread
            Timers.tic('krylov send expm to another thread')
            pool_res = pool.map_async(get_krylov_result, args)
            Timers.toc('krylov send expm to another thread')
        else:
            # do matrix exp right away
            list_of_results = [get_krylov_result(a) for a in args]

            Timers.tic('krylov update result list')
            for krylov_result in list_of_results:
                for step, dim, col_vec in krylov_result:
                    rv[step][:, dim] = col_vec
            Timers.toc('krylov update result list')

    # loop ended, update results from the last iteration (if pipelining was used)
    if settings.simulation.pipeline_arnoldi_expm:
        Timers.tic('krylov wait for expm results')
        list_of_results = pool_res.get()
        Timers.toc('krylov wait for expm results')

        Timers.tic('krylov update result list')
        for krylov_result in list_of_results:
            for step, dim, col_vec in krylov_result:
                rv[step][:, dim] = col_vec

        Timers.toc('krylov update result list')
        pool.close()
        pool.join()

    if settings.simulation.krylov_profiling:
        KrylovInterface.print_profiling_data()

    if settings.print_output:
        elapsed = format_secs(time.time() - start)
        print "Krylov Computation Total Time: {}\n".format(elapsed)

    #print "debug exit"
    #exit(1)

    return rv
