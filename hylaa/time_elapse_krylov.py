'''
Time Elapse for the Krylov method using CPU or GPU
'''

import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm
from scipy.integrate import odeint

from hylaa.timerutil import Timers
from hylaa.krylov_python import get_free_memory_mb, python_arnoldi, python_lanczos
from hylaa.settings import HylaaSettings

def projected_odeint_sim(arg):
    '''
    simulate a given dense a-matrix with the provided initial vector, for a certain number of steps,
    returning the projected result at each step (excluding step zero)

    arg is tuple (pv_matrix, a_matrix, start_vec, settings)
    '''

    pv_matrix, a_matrix, start_vec, settings = arg

    assert a_matrix.shape[1] > 0
    assert isinstance(pv_matrix, np.ndarray)
    assert isinstance(start_vec, np.ndarray)
    assert isinstance(settings, HylaaSettings)

    step = settings.step
    num_steps = settings.num_steps
    sim_tol = settings.simulation.krylov_odeint_simtol

    if isinstance(a_matrix, np.ndarray):
        # was arnoldi iteration, a_matrix (H) is a dense matrix
        der_func = lambda state, _: np.dot(a_matrix, state)
        a_transpose = a_matrix.transpose().copy()
        jac_func = lambda dummy_state, dummy_t: a_transpose
    else:
        # was lanczos iteration, a_matrix (H) is a sparse matrix
        assert isinstance(a_matrix, csr_matrix)
        der_func = lambda state, _: (a_matrix * state)
        jac_func = None

    times = np.linspace(0, step * num_steps, num=num_steps+1)

    result = odeint(der_func, start_vec, times, Dfun=jac_func, col_deriv=True, atol=sim_tol, rtol=sim_tol, \
            mxstep=int(1e8)) # mxstep = maximum number of internal steps

    rv = []

    for i in xrange(1, len(result)): # skip step zero
        rv.append(np.dot(pv_matrix, result[i]))

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
            arg = small_pv_mat, small_h_mat, small_start_vec, settings
            small_sim = projected_odeint_sim(arg)

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

def check_available_memory_basis(stdout, s, k, i):
    'check if enough memory is available to store the basis matrix'

    required_mb = (s * k * i * 8) / 1024.0 / 1024.0
    available_mb = get_free_memory_mb()

    if stdout:
        print "Basis Matrix Required GB = {:.3f} (+1), available GB = {:.3f} (s = {}, k = {}, i+1 = {})".format(
            required_mb / 1024.0, available_mb / 1024.0, s, k, i)

    if required_mb + 1024 > available_mb: # add 1024 mb since we want 1 GB free for other things
        raise MemoryError("Not enogh memory for storing the basis matrices.")

def init_krylov(time_elapser):
    '''
    initialize krylov interface for the computation

    returns a list of empty matrices to be filled in by the subsequent computation
    '''

    settings = time_elapser.settings
    key_dir_mat = time_elapser.key_dir_mat
    init_space_csc = time_elapser.init_space_csc

    # check available memory before computing
    i = time_elapser.init_space_csc.shape[0]

    check_available_memory_basis(settings.print_output, time_elapser.settings.num_steps, key_dir_mat.shape[0], i)

    time_elapser.stats['arnoldi_iter'] = []
    time_elapser.stats['arnoldi_mem_start'] = get_free_memory_mb()

    rv = []

    # initialize step zero
    step_zero_mat = (key_dir_mat * init_space_csc).toarray()
    rv.append(step_zero_mat)

    # add zeros (allocate storage for result)
    for _ in xrange(0, time_elapser.settings.num_steps):
        rv.append(np.zeros(rv[0].shape, dtype=float))

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

def get_rel_error(settings, h_mat, pv_mat, arnoldi_iter=None, return_sim=False, limit=None):
    '''
    Get the relative error given the h and pv matrices, for the given number of arnoldi_iterations.
    If arnoldi_iter is None, then use the full passed-in matrices.

    This compares the error at all time steps.

    If return_sim is True, then a tuple is returned where the second element is list of the
    sim points at each time step.

    if limit is not None, this will break as soon as the relative error exceeds the limit
    '''

    assert h_mat.shape[0] > 1

    sim = None

    # use less arnoldi iterations than what's in the matrices
    if arnoldi_iter is not None:
        h_mat = h_mat[:arnoldi_iter, :arnoldi_iter].copy()
        pv_mat = pv_mat[:, :arnoldi_iter].copy()

    if limit is not None:
        small_h_mat = h_mat[:-1, :-1].copy()
        small_pv_mat = pv_mat[:, :-1].copy()

    if settings.simulation.krylov_use_odeint:
        Timers.tic('get_rel_error odeint')
        start_vec = np.array([1.0 if d == 0 else 0.0 for d in xrange(h_mat.shape[0])], dtype=float)
        rel_error = 0

        if limit is None:
            sim = projected_odeint_sim((pv_mat, h_mat, start_vec, settings))
        else:
            small_start_vec = start_vec[:-1].copy()
            args = [(pv_mat, h_mat, start_vec, settings), (small_pv_mat, small_h_mat, small_start_vec, settings)]
            sim, small_sim = [projected_odeint_sim(a) for a in args]

            for step in xrange(len(sim)):
                cur_result = sim[step]
                small_result = small_sim[step]

                rel_error = max(rel_error, relative_error(cur_result, small_result))

                if rel_error > limit:
                    break

        Timers.toc('get_rel_error odeint')
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

        if return_sim:
            sim = [cur_result]

        for step in xrange(2, settings.num_steps + 1):
            cur_col = np.dot(matrix_exp, cur_col)
            small_col = np.dot(small_matrix_exp, small_col)

            # maybe we want to check relative error in the middle as well
            cur_result = np.dot(pv_mat, cur_col)
            small_result = np.dot(small_pv_mat, small_col)
            rel_error = max(rel_error, relative_error(cur_result, small_result))

            if return_sim:
                sim.append(cur_result)

            if limit is not None and rel_error > limit:
                if settings.print_output:
                    print "Relative Error {} exceeded limit {} at step {}".format(rel_error, limit, step)

                break

    return rel_error if not return_sim else (rel_error, sim)

def print_rel_error_at_each_step(settings, h_list, pv_list):
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
                rel_error = get_rel_error(settings, h_mat, pv_mat, arnoldi_iter=aiter)
                max_rel_error = max(max_rel_error, rel_error)

            line = "{}\t{:.20f}\n".format(aiter, max_rel_error)
            print line,
            f.write(line)

    print "print_rel_error_at_each_step data written to {}, exiting".format(filename)

def arnoldi_sim_with_max_rel_error(time_elapser, init_vec_csr, iterations, rel_error_limit):
    '''
    Run an arnoldi simulation with a fixed number of iterations and a target max relative error.
    If rel_error_limit is None, just run the whole simulation.

    returns a 2-tuple (a, b) with:
    a: projected simulation at each step, or None if the error limit is exceeded.
    b: the number of arnoldi iterations actually used
    '''

    settings = time_elapser.settings
    a_mat = time_elapser.a_matrix
    key_dir_mat = time_elapser.key_dir_mat
    stdout = settings.simulation.krylov_stdout

    if settings.simulation.krylov_use_lanczos:
        Timers.tic('lanczos')
        pv_mat, h_mat = python_lanczos(a_mat, init_vec_csr, iterations, key_dir_mat, print_status=stdout)
        Timers.toc('lanczos')
    else:
        Timers.tic('arnoldi')
        pv_mat, h_mat = python_arnoldi(a_mat, init_vec_csr, iterations, key_dir_mat, print_status=stdout)
        Timers.toc('arnoldi')

    if h_mat.shape[0] < iterations:
        rel_error_limit = None
        iterations = h_mat.shape[0]

        if stdout:
            print "Arnoldi terminated early. Simulating without relative error limit."

    h_mat = h_mat[:-1, :].copy()
    pv_mat = pv_mat[:, :-1].copy()

    rel_error, projected_sim = get_rel_error(settings, h_mat, pv_mat, return_sim=True, limit=rel_error_limit)

    if rel_error_limit is None or rel_error < rel_error_limit:
        if stdout and rel_error_limit is not None:
            print "Relative error {} was below threshold: {}".format(rel_error, rel_error_limit)

        rv = projected_sim
    else:
        rv = None

    return rv, iterations

# projected_simulation = arnoldi_projected_simulation(time_elapser, init_vec)
def arnoldi_sim_autotune(time_elapser, init_vec_csr):
    '''
    Perform a projected simulation from a given initial vector. This auto-tunes the number
    of arnoldi iterations based on the relative error.

    returns the projected simulation at each step.
    '''

    settings = time_elapser.settings
    stdout = settings.simulation.krylov_stdout
    n = time_elapser.a_matrix.shape[0]

    error_limit = settings.simulation.krylov_rel_error

    arnoldi_iter = 2
    sim = None

    while True:
        arnoldi_iter = arnoldi_iter * 2

        if arnoldi_iter >= n:
            arnoldi_iter = n
            error_limit = None # do not target any relative error in this case

            if stdout:
                print "Arnoldi iter ({}) reached system dimension; skipping relative error".format(arnoldi_iter)

        if stdout:
            print "Trying {} arnoldi iterations...".format(arnoldi_iter)

        sim, arnoldi_iter = arnoldi_sim_with_max_rel_error(time_elapser, init_vec_csr, arnoldi_iter, error_limit)

        if sim is not None:
            break

    if stdout:
        print "Simulation was accurate enough with {} arnoldi iterations...".format(arnoldi_iter)

    time_elapser.stats['arnoldi_iter'].append(arnoldi_iter)

    #if settings.simulation.krylov_print_rel_error_filename is not None:
    #    print_rel_error_at_each_step(settings, h_list, pv_list)
    #    print "Exiting because settings.simulation.krylov_print_rel_error_filename was set"
    #    exit(0)

    return sim

def assign_from_sim(rv, sim, index):
    'assign a simulation to the result object'

    assert len(sim) == len(rv) - 1, "Got sim of length {}, expected {}".format(len(sim), len(rv) - 1)
    Timers.tic('update result list')

    for i in xrange(len(sim)):
        rv[i+1][:, index] = sim[i]

    Timers.toc('update result list')

def update_result_list(list_of_results, settings, rv):
    'populate rv based on the list of results'

    Timers.tic('update result list')

    for krylov_result in list_of_results:
        result_list, rel_error = krylov_result

        if rel_error is not None and settings.simulation.krylov_check_all_rel_error is not None:
            assert rel_error < settings.simulation.krylov_check_all_rel_error, \
                "Got rel error {} > {} (max) in dimension {}".format(rel_error, \
                settings.simulation.krylov_rel_error, result_list[0][1])

        for (step, lp_var, col_vec) in result_list:
            rv[step][:, lp_var] = col_vec

    Timers.toc('update result list')

def make_cur_basis_mat_list(time_elapser):
    '''
    Main work function. This returns the basis matrix at every step.

    This is called one time, and returns a list, element N is the basis matrix at step N
    '''

    # numpy raise errors on floating point errors (these should be caught and handled explicitly)
    np.seterr(all='warn', over='raise')

    settings = time_elapser.settings
    init_space_csc = time_elapser.init_space_csc

    rv = init_krylov(time_elapser)

    start = last_print = time.time()

    for init_index in xrange(init_space_csc.shape[1]):
        # this seems unnecessary... check if it hurts performance
        Timers.tic('debug convert csc init to csr')
        init_vec_csr = csr_matrix(init_space_csc[:, init_index].transpose())
        Timers.toc('debug convert csc init to csr')

        sim = arnoldi_sim_autotune(time_elapser, init_vec_csr)

        assign_from_sim(rv, sim, init_index)

        if settings.print_output:
            now = time.time()

            if now - last_print > 1.0: # print every second
                last_print = now
                frac = float(init_index) / init_space_csc.shape[1]

                if frac > 1e-9:
                    elapsed_sec = now - start
                    total_sec = elapsed_sec / frac
                    eta_sec = total_sec - elapsed_sec
                    eta = format_secs(eta_sec)

                    print "Arnoldi {} / {} ({:.2f}%, ETA: {})".format(
                        init_index, init_space_csc.shape[1], 100.0 * frac, eta)

    if settings.print_output:
        elapsed = format_secs(time.time() - start)
        print "Krylov Computation Total Time: {}\n".format(elapsed)

    return rv
