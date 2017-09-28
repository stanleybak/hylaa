'''
Time Elapse for the Krylov method using CPU or GPU
'''

import math
import time
import sys

from multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
from scipy.integrate import odeint

from hylaa.krylov_interface import KrylovInterface
from hylaa.timerutil import Timers

def projected_odeint_sim(arg):
    '''
    simulate a given dense a-matrix with the provided initial vector, for a certain number of steps,
    returning the projected result at each step (excluding step zero)

    arg is tuple (pv_matrix, a_matrix, start_vec, settings)
    '''

    pv_matrix, a_matrix, start_vec, settings = arg

    assert isinstance(a_matrix, np.ndarray)
    assert isinstance(pv_matrix, np.ndarray)

    step = settings.step
    num_steps = settings.num_steps
    sim_tol = settings.simulation.krylov_odeint_simtol

    der_func = lambda state, _: np.dot(a_matrix, state)
    a_transpose = a_matrix.transpose().copy()
    jac_func = lambda dummy_state, dummy_t: a_transpose

    times = np.linspace(0, step * num_steps, num=num_steps+1)

    result = odeint(der_func, start_vec, times, Dfun=jac_func, col_deriv=True, atol=sim_tol, rtol=sim_tol, \
            mxstep=int(1e8)) # mxstep = maximum number of internal steps

    rv = []

    for i in xrange(1, len(result)): # skip step zero
        rv.append(np.dot(pv_matrix, result[i]))

    return rv

def compress_fixed(key_dir_mat, fixed_tuples):
    'compress the fixed variables in the time_elapse matrix'

    assert isinstance(key_dir_mat, csr_matrix)

    csc_mat = csc_matrix(key_dir_mat)

    # we need to create a new dense matrix based on the variable reordering
    num_vars = csc_mat.shape[1] - len(fixed_tuples) + 1
    rv = np.zeros((csc_mat.shape[0], num_vars), dtype=float)

    compressed_var_index = 0
    uncompressed_dim = 0

    for dim in xrange(csc_mat.shape[1]):
        if compressed_var_index >= len(fixed_tuples) or dim < fixed_tuples[compressed_var_index][0]:
            # uncompressed dim

            for index in xrange(csc_mat.indptr[dim], csc_mat.indptr[dim+1]):
                row = csc_mat.indices[index]
                n = csc_mat.data[index]
                rv[row, uncompressed_dim] = n

            uncompressed_dim += 1
        else:
            # compressed dim
            fixed_val = fixed_tuples[compressed_var_index][1]

            for index in xrange(csc_mat.indptr[dim], csc_mat.indptr[dim+1]):
                row = csc_mat.indices[index]
                n = csc_mat.data[index]
                rv[row, uncompressed_dim] += fixed_val * n

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
    Compute the krylov result.

    Expected arguments (a single tuple):
    (num_steps, time_step, dim, h_mat, pv_mat, compute_rel_error)

    Returns a tuple:
    first element: list of tuples (step, dim, col_vec)
    second element: None or the relative error, if compute_rel_error was True
    '''

    settings, dim, h_mat, pv_mat = arg_tuple

    num_steps = settings.num_steps
    time_step = settings.step
    compute_rel_error = settings.simulation.krylov_check_all_rel_error

    rv = []
    rel_error = None

    h_mat = h_mat[:-1, :].copy()
    pv_mat = pv_mat[:, :-1].copy()

    if compute_rel_error is not None:
        small_h_mat = h_mat[:-1, :-1].copy()
        small_pv_mat = pv_mat[:, :-1].copy()

    if settings.simulation.krylov_use_odeint:
        start_vec = np.array([1.0 if d == 0 else 0.0 for d in xrange(h_mat.shape[0])], dtype=float)
        small_start_vec = start_vec[:-1].copy()

        arg = pv_mat, h_mat, start_vec, settings
        sim = projected_odeint_sim(arg)

        if compute_rel_error:
            arg = smal_pv_mat, small_h_mat, small_start_vec, settings
            small_sim = odeint_sim(arg)

        for step in xrange(len(sim)):
            cur_result = sim[step]

            rv.append((1 + step, dim, cur_result))

            if compute_rel_error:
                small_result = small_sim[step]
                rel_error = max(rel_error, relative_error(cur_result, small_result))
    else:
        # use matrix exp
        matrix_exp = np.array(expm(h_mat * time_step), dtype=float)
        cur_col = matrix_exp[:, 0]

        cur_result = np.dot(pv_mat, cur_col)
        rv.append((1, dim, cur_result))

        if compute_rel_error is not None:
            small_matrix_exp = np.array(expm(small_h_mat * time_step), dtype=float)
            small_col = small_matrix_exp[:, 0]

            small_result = np.dot(small_pv_mat, small_col)
            rel_error = relative_error(cur_result, small_result)

        for s in xrange(2, num_steps + 1):
            cur_col = np.dot(matrix_exp, cur_col)
            cur_result = np.dot(pv_mat, cur_col)
            rv.append((s, dim, cur_result))

            if compute_rel_error is not None:
                small_col = np.dot(small_matrix_exp, small_col)
                small_result = np.dot(small_pv_mat, small_col)
                rel_error = max(rel_error, relative_error(cur_result, small_result))

    return rv, rel_error

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

    if settings.simulation.krylov_seperate_constant_vars:
        dense_key_dir_mat = compress_fixed(key_dir_mat, time_elapser.fixed_tuples)
    else:
        dense_key_dir_mat = np.array(key_dir_mat.todense(), dtype=float)

    rv.append(dense_key_dir_mat) # step zero

    Timers.toc('initilaizing step zero from key dir mat')

    # add zeros (allocate storage for result)
    for _ in xrange(0, time_elapser.settings.num_steps):
        rv.append(np.zeros(rv[0].shape, dtype=float))

    Timers.tic("krylov preallocate and load dynamics")

    print "maybe add a check here if we're out of memory... zeros seems to do funny things"
    print "!! steps = {}".format(time_elapser.settings.num_steps)
    print "!! Total memory: {}".format(rv[0].nbytes * time_elapser.settings.num_steps / 1024.0**3)

    # make sure we can allocate with arnoldi_iter = 2
    KrylovInterface.preallocate_memory(arnoldi_iter, a_matrix.shape[0], key_dir_mat.shape[0], error_on_fail=True)
    KrylovInterface.load_a_matrix(a_matrix) # load a_matrix into device memory
    KrylovInterface.load_key_dir_matrix(key_dir_mat) # load key direction matrix into device memory

    Timers.toc("krylov preallocate and load dynamics")

    return rv

def relative_error(correct, estimate):
    'compute the relative error between the correct value and an estimate'

    rel_error = 1.0e16 # large error is returned if it can't be computed due to numerical issues

    try:
        norm = np.linalg.norm(correct)

        if not math.isinf(norm) and not math.isnan(norm):
            if norm < 1e-13: # if norm is small, return absolute error
                rel_error = norm
            else:
                diff = correct - estimate
                abs_error = np.linalg.norm(diff)

                if not math.isinf(abs_error) and not math.isnan(abs_error):
                    rel_error = abs_error / norm
    except FloatingPointError:
        pass

    assert not math.isinf(rel_error) and not math.isnan(rel_error)

    return rel_error

def get_rel_error(settings, h_mat, pv_mat, pool, arnoldi_iter=None, return_projected_sims=False, limit=None):
    '''
    Get the relative error given the h and pv matrices, for the given number of arnoldi_iterations.
    If arnoldi_iter is None, then use the full passed-in matrices.

    This compares the error at all time steps.

    If return_projected_sims is True, then a tuple is returned where the second element is list of the
    sim points at each time step.

    if limit is not None, this will break as soon as the relative error exceeds the limit
    '''

    projected_sims = []

    # use less arnoldi iterations than what's in the matrices
    if arnoldi_iter is not None:
        h_mat = h_mat[:arnoldi_iter, :arnoldi_iter].copy()
        pv_mat = pv_mat[:, :arnoldi_iter].copy()

    small_h_mat = h_mat[:-1, :-1].copy()
    small_pv_mat = pv_mat[:, :-1].copy()

    if settings.simulation.krylov_use_odeint:
        start_vec = np.array([1.0 if d == 0 else 0.0 for d in xrange(h_mat.shape[0])], dtype=float)
        small_start_vec = start_vec[:-1].copy()

        args = [(pv_mat, h_mat, start_vec, settings), (small_pv_mat, small_h_mat, small_start_vec, settings)]

        Timers.tic('get_rel_error odeint')
        if pool is not None:
            # use multithreaded (cuts time in half)
            sim, small_sim = pool.map(projected_odeint_sim, args)
        else:
            sim, small_sim = [projected_odeint_sim(a) for a in args]
        Timers.toc('get_rel_error odeint')

        rel_error = 0

        for step in xrange(len(sim)):
            cur_result = sim[step]
            small_result = small_sim[step]

            projected_sims.append(cur_result)
            rel_error = max(rel_error, relative_error(cur_result, small_result))

            if limit is not None and rel_error > limit:
                break

    else:
        Timers.tic('get_rel_error expm')
        matrix_exp = expm(settings.step * h_mat)
        cur_col = matrix_exp[:, 0]
        Timers.toc('get_rel_error expm')

        # for accuracy check
        Timers.tic('get_rel_error expm')
        small_matrix_exp = expm(settings.step * small_h_mat) # step time is already included in loaded a_mat
        small_col = small_matrix_exp[:, 0]
        Timers.toc('get_rel_error expm')

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

            projected_sims.append(cur_result)

            if limit is not None and rel_error > limit:
                break

    return rel_error if not return_projected_sims else (rel_error, projected_sims)

def get_max_rel_error(settings, dim_list, limit, pool):
    '''
    Get the maximum relative error of the passed-in number of arnoldi iterations.

    returns max_rel_error, h_mat_list, pv_mat_list
    '''

    max_rel_error = 0
    h_list = []
    pv_list = []
    all_sims = []

    for dim in dim_list:
        Timers.tic('krylov arnoldi_unit()')
        h_mat, pv_mat = KrylovInterface.arnoldi_unit(dim)
        Timers.toc('krylov arnoldi_unit()')

        h_mat = h_mat[:-1, :].copy()
        h_list.append(h_mat)

        pv_mat = pv_mat[:, :-1].copy()
        pv_list.append(pv_mat)

        rel_error, projected_sim = get_rel_error(settings, h_mat, pv_mat, pool, return_projected_sims=True, limit=limit)
        all_sims.append(projected_sim)

        max_rel_error = max(max_rel_error, rel_error)

        if max_rel_error > limit:
            break

    # if the max_rel error is small and simulations are near zero... we probably don't have enough arnoldi_iter
    if max_rel_error < 1e-13 and settings.simulation.krylov_reject_zero_rel_error:
        all_small = True

        for sim in all_sims:
            for state in sim:
                if np.linalg.norm(state) > 1e-13:
                    all_small = False
                    break

        # use a large relative error to force more arnoldi iterations
        if all_small:
            max_rel_error = 1e16

    return max_rel_error, h_list, pv_list

def find_iter_with_accuracy(settings, h_list, pv_list, desired_rel_error, pool):
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
            rel_error = get_rel_error(settings, h_mat, pv_mat, pool, arnoldi_iter=aiter, limit=desired_rel_error)
            max_rel_error = max(max_rel_error, rel_error)

            if max_rel_error > desired_rel_error:
                break

        if max_rel_error < desired_rel_error:
            top = aiter
        else:
            bottom = aiter

    return top

def should_compute_vec(vec, dims_to_compute):
    'check if the given vec has any effect on the key directions (using dims_to_compute list)'

    has_effect = False

    for dim in xrange(len(dims_to_compute)):
        needs_computation = dims_to_compute[dim]

        if needs_computation and vec[dim, 0] != 0:
            has_effect = True
            break

    return has_effect

def krylov_sim_fixed_terms(time_elapser, dims_to_compute, pool):
    '''
    simulate the fixed effect vector up to the desired relative error.

    This returns the simulation at each step, or None if no fixed terms or no fixed term effect
    '''

    settings = time_elapser.settings
    dims = time_elapser.a_matrix.shape[0]
    key_dirs = time_elapser.key_dir_mat.shape[0]
    error_limit = settings.simulation.krylov_rel_error
    init_vec = time_elapser.fixed_init_vec
    arnoldi_iter = 0

    if np.linalg.norm(init_vec) < 1e-9:
        simulation = None

        if settings.print_output:
            print "Skipping fixed-term simulation (no fixed variables in system)"
    elif not should_compute_vec(init_vec, dims_to_compute):
        simulation = None

        if settings.print_output:
            print "Skipping fixed-term simulation (no effect on key directions)"
    elif settings.simulation.krylov_force_arnoldi_iter is not None:
        arnoldi_iter = settings.simulation.krylov_force_arnoldi_iter

        KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

        h_mat, pv_mat = KrylovInterface.arnoldi_vec(init_vec)
        h_mat = h_mat[:-1, :].copy()
        pv_mat = pv_mat[:, :-1].copy()

        print "getting simulation with fixed krylov iter"

        (_, simulation) = get_rel_error(settings, h_mat, pv_mat, pool, arnoldi_iter, return_projected_sims=True)

        print "returning sim with len {}".format(len(simulation))
    else:
        arnoldi_iter = min(dims, 4)

        if settings.print_output:
            print "Computing fixed terms using krylov simulation with iterations: ",
            print "{}".format(arnoldi_iter),
            sys.stdout.flush()

        KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

        h_mat, pv_mat = KrylovInterface.arnoldi_vec(init_vec)
        h_mat = h_mat[:-1, :].copy()
        pv_mat = pv_mat[:, :-1].copy()

        cur_rel_error, simulation = get_rel_error(settings, h_mat, pv_mat, pool, arnoldi_iter, \
            return_projected_sims=True, limit=error_limit)

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

            cur_rel_error, simulation = get_rel_error(settings, h_mat, pv_mat, pool, arnoldi_iter, \
                                                      return_projected_sims=True, limit=error_limit)

        if settings.print_output:
            print "\n"

    time_elapser.arnoldi_iter.append(arnoldi_iter) # record stats

    return simulation

def print_rel_error_at_each_step(settings, pool, h_list, pv_list):
    '''
    a profiling function. If this is used, output a file with the relative error for every number of
    arnoldi iteartions, and then quit.
    '''

    filename = settings.simulation.krylov_print_rel_error_filename

    print "Printing relative errors to file: {}".format(filename)

    max_iter = h_list[0].shape[0]

    with open(filename, 'w') as f:

        for aiter in xrange(2, max_iter):
            max_rel_error = 0.0

            for h_mat, pv_mat in zip(h_list, pv_list):
                rel_error = get_rel_error(settings, h_mat, pv_mat, pool, arnoldi_iter=aiter)
                max_rel_error = max(max_rel_error, rel_error)

            line = "{}\t{:.20f}\n".format(aiter, max_rel_error)
            print line,
            f.write(line)

    print "print_rel_error_at_each_step data written to {}, exiting".format(filename)

def choose_arnoldi_iter(time_elapser, var_list, dims_to_compute, pool):
    '''
    Choose the number of arnoldi iterations based on the desired error. This is done through a
    sampling of a number of vectors and using cauchy error.
    '''

    settings = time_elapser.settings
    dims = time_elapser.a_matrix.shape[0]
    key_dirs = time_elapser.key_dir_mat.shape[0]

    if settings.print_output:
        print "Determining number of arnoldi iterations for desired accuracy...",
        sys.stdout.flush()

    # filter out variables with no effect on key directions
    var_list = remove_no_effect_variables(var_list, dims_to_compute)

    if len(var_list) == 0:
        if settings.print_output:
            print "var_list contains no variables with an effect on key directions"

        arnoldi_iter = 1
    elif dims <= 4:
        if settings.print_output:
            print "System dims < 4, using krylov_iter == dims"

        arnoldi_iter = dims
    else:
        error_limit = settings.simulation.krylov_rel_error

        # sample accuracy in a small number of dimensions
        samples = min(dims, settings.simulation.krylov_rel_error_samples)

        # only sample in variable dimensions
        dim_list = [var_list[int(d)] for d in np.linspace(0, len(var_list)-1, num=samples)]

        # eliminate duplicates
        dim_list = list(set(dim_list))

        # optimization: evaluate the last dimension first (since it's often the highest error one)
        dim_list = [dim_list[-1]] + dim_list[0:-1]

        arnoldi_iter = 2
        cur_rel_error = 2 * error_limit # force the loop to start

        while cur_rel_error > error_limit:
            arnoldi_iter = arnoldi_iter * 2

            if arnoldi_iter > dims:
                arnoldi_iter = dims

            KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

            if settings.print_output:
                print "{}".format(arnoldi_iter),
                sys.stdout.flush()

            cur_rel_error, h_list, pv_list = get_max_rel_error(settings, dim_list, error_limit, pool)

            # print "!cur_rel_error = {}".format(cur_rel_error)

            if arnoldi_iter == dims:
                break

        if settings.simulation.krylov_print_rel_error_filename is not None:
            print_rel_error_at_each_step(settings, pool, h_list, pv_list)

        # ok, at this point the error is below the threshold
        # find the exact value of iterations that makes this work for all the samples
        arnoldi_iter = find_iter_with_accuracy(settings, h_list, pv_list, error_limit, pool)

    if settings.print_output:
        print "\nUsing {} Arnoldi Iterations".format(arnoldi_iter)

    if settings.simulation.krylov_print_rel_error_filename is not None:
        print "Exiting because settings.simulation.krylov_print_rel_error_filename was set"
        exit(0)

    return arnoldi_iter

def assign_fixed_terms(time_elapser, rv, dims_to_compute, pool):
    'assign the fixed effect terms using a krylov simulation'

    if time_elapser.settings.simulation.krylov_seperate_constant_vars:
        Timers.tic("krylov sim fixed terms")
        fixed_sim = krylov_sim_fixed_terms(time_elapser, dims_to_compute, pool)
        Timers.toc("krylov sim fixed terms")

        if fixed_sim is not None:
            # assign results from fixed_sim
            assert len(fixed_sim) == len(rv) - 1
            Timers.tic('update result list')

            for i in xrange(len(fixed_sim)):
                rv[i+1][:, -1] = fixed_sim[i]

            Timers.toc('update result list')
    else:
        time_elapser.arnoldi_iter.append(0) # no fixed-term iterations

def update_result_list(list_of_results, settings, rv):
    'populate rv based on the list of results'

    Timers.tic('update result list')

    for krylov_result in list_of_results:
        result_list, rel_error = krylov_result

        if rel_error is not None:
            assert rel_error < settings.simulation.krylov_check_all_rel_error, \
                "Got rel error {} > {} (max) in dimension {}".format(rel_error, \
                settings.simulation.krylov_rel_error, result_list[0][1])

        for (step, lp_var, col_vec) in result_list:
            rv[step][:, lp_var] = col_vec

    Timers.toc('update result list')

def get_dims_to_compute(a_matrix, key_dir_mat):
    '''
    get the dimensions that have an impact on the key directions

    returns a list of booleans, one for each dimension
    '''

    assert isinstance(a_matrix, csr_matrix)
    assert isinstance(key_dir_mat, csr_matrix)

    dims = a_matrix.shape[0]
    assert key_dir_mat.shape[1] == dims

    # we essentially perform a breadth-first search, starting with any non-zeros in the key direction matrix
    marked = [False] * dims
    pending = []

    # inititalize with nonzeros in key direction matrix
    for row in key_dir_mat:
        for col, val in zip(row.indices, row.data):
            if val != 0: # it shouldn't be zero, but who knows

                if marked[col] is False:
                    marked[col] = True
                    pending.append(col)

    # loop until fixpoint
    while len(pending) > 0:
        row_num = pending.pop()
        row_start = a_matrix.indptr[row_num]
        row_end = a_matrix.indptr[row_num+1]

        for index in xrange(row_start, row_end):
            col = a_matrix.indices[index]
            val = a_matrix.data[index]

            if val != 0: # it shouldn't be zero, but who knows
                if marked[col] is False:
                    marked[col] = True
                    pending.append(col)

    return marked

def remove_no_effect_variables(var_list, dims_to_compute):
    'given a list of dimensions, remove the ones with no effect on the key directions (using dims_to_compute)'

    rv = []

    for dim in var_list:
        if dims_to_compute[dim]:
            rv.append(dim)

    return rv

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

    pool = arnoldi_expm_pool = rel_error_pool = None

    if settings.simulation.krylov_multithreaded_arnoldi_expm or settings.simulation.krylov_multithreaded_rel_error:
        pool = Pool()

        if settings.simulation.krylov_multithreaded_arnoldi_expm:
            arnoldi_expm_pool = pool

        if settings.simulation.krylov_multithreaded_rel_error:
            rel_error_pool = pool

    rv = init_krylov(time_elapser, 2)

    if settings.print_output:
        print "Computing dimensions with effect on key directions..."

    Timers.tic("get_dims_to_compute()")
    dims_to_compute = get_dims_to_compute(time_elapser.a_matrix, time_elapser.key_dir_mat)
    Timers.toc("get_dims_to_compute()")

    if settings.simulation.krylov_force_arnoldi_iter and settings.print_output:
        print "Using fixed arnoldi_iter from settings: {}".format(settings.simulation.krylov_force_arnoldi_iter)

    Timers.tic("krylov assign fixed terms")
    assign_fixed_terms(time_elapser, rv, dims_to_compute, rel_error_pool)
    Timers.toc("krylov assign fixed terms")

    if settings.simulation.krylov_seperate_constant_vars:
        variable_dim_sublists = time_elapser.var_lists
    else:
        variable_dim_sublists = [xrange(0, dims)]

    num_vars = sum([len(sublist) for sublist in variable_dim_sublists])
    completed_vars = 0

    pool_res = None

    # print "Num_Vars (i) = {}".format(num_vars)

    for var_sublist in variable_dim_sublists:
        if settings.simulation.krylov_force_arnoldi_iter is not None:
            arnoldi_iter = settings.simulation.krylov_force_arnoldi_iter
        else:
            Timers.tic("choose arnoldi iterations")
            arnoldi_iter = choose_arnoldi_iter(time_elapser, var_sublist, dims_to_compute, rel_error_pool)
            Timers.toc("choose arnoldi iterations")
            time_elapser.arnoldi_iter.append(arnoldi_iter) # record stats

        # re-allocate with correct number of arnoldi iterations
        KrylovInterface.preallocate_memory(arnoldi_iter, dims, key_dirs, error_on_fail=True)

        start = last_print = time.time()

        for dim_index in xrange(len(var_sublist)):
            dim = var_sublist[dim_index]

            # no effect on key directions
            if dims_to_compute[dim] is False:
                completed_vars += 1
                continue

            #if dim == 64:
            #    print "debug break at 64"
            #    break

            if settings.print_output:
                now = time.time()

                if now - last_print > 1.0: # print every second
                    last_print = now
                    frac = float(completed_vars) / num_vars

                    if frac > 1e-9:
                        elapsed_sec = now - start
                        total_sec = elapsed_sec / frac
                        eta_sec = total_sec - elapsed_sec
                        eta = format_secs(eta_sec)

                        print "Arnoldi {} / {} ({:.2f}%, ETA: {})".format(
                            completed_vars, num_vars, 100.0 * frac, eta)

            Timers.tic('krylov arnoldi_unit()')
            h_mat, pv_mat = KrylovInterface.arnoldi_unit(dim)
            Timers.toc('krylov arnoldi_unit()')

            # update result from the previous iteration (skipped on first iteration)
            if pool_res is not None:
                Timers.tic('krylov wait for expm(H) results')
                list_of_results = pool_res.get()
                Timers.toc('krylov wait for expm(H) results')

                update_result_list(list_of_results, settings, rv)

            ### compute matrix exp ###
            lp_var = time_elapser.dim_to_lp_var[dim]
            args = [(settings, lp_var, h_mat, pv_mat)]
            completed_vars += 1

            if arnoldi_expm_pool is not None:
                # push the computation to another thread
                Timers.tic('krylov send expm(H) to another thread')
                pool_res = arnoldi_expm_pool.map_async(get_krylov_result, args)
                Timers.toc('krylov send expm(H) to another thread')
            else:
                # do matrix exp right away
                Timers.tic('krylov expm(H)')
                list_of_results = [get_krylov_result(a) for a in args]
                Timers.toc('krylov expm(H)')

                update_result_list(list_of_results, settings, rv)

    # loop ended, update results from the last iteration (if krylov_multithreaded was used)
    if pool_res is not None:
        Timers.tic('krylov wait for expm(H) results')
        list_of_results = pool_res.get()
        Timers.toc('krylov wait for expm(H) results')

        update_result_list(list_of_results, settings, rv)

    if pool is not None:
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
