'''
Time Elapse for the Krylov method using CPU or GPU
'''

import math
import time

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import odeint, simps
from scipy.sparse.linalg import eigs

import psutil

from hylaa.timerutil import Timers
from hylaa.settings import HylaaSettings
from hylaa.util import Freezable
from hylaa.krylov_python import KrylovIteration

def is_symmetric(mat):
    'is the passed-in square matrix symmetric?'

    Timers.tic('is_symmetric')

    start = time.time()
    rv = (mat != mat.T).nnz == 0
    print "is_symmetric time = {}".format(time.time() - start)

    Timers.toc('is_symmetric')

    return rv

class TimeElapseKrylov(Freezable):
    'container object for time elapse krylov method'

    def __init__(self, time_elapser):
        self.time_elapser = time_elapser
        self.settings = time_elapser.settings

        self.cur_basis_mat_list = None
        self.use_transpose = not time_elapser.use_init_space

        # key_dir_mat is the output projection matrix in arnoldi
        if self.use_transpose:
            key_dir_mat = csr_matrix(time_elapser.init_space_csc.transpose())
        else:
            key_dir_mat = time_elapser.output_space_csr

        a_matrix = time_elapser.a_matrix
        use_lanczos = is_symmetric(a_matrix)

        # for error bound computation
        self.nu_a = self.compute_nu_a(a_matrix, use_lanczos)
        print ". computing_nu_a = {}".format(self.nu_a)

        if self.use_transpose and not use_lanczos:
            # we need to compute with the transpose of the a matrix
            a_matrix = csr_matrix(a_matrix.transpose())
        else:
            a_matrix = a_matrix

        # modify in place
        a_matrix *= -1

        self.krylov_iteration = KrylovIteration(time_elapser.settings, a_matrix, use_lanczos, key_dir_mat)

        time_elapser.a_matrix = None # this makes sure it's not used since we modified it in place

        # -- performance statistics --
        # arnoldi_iter -> list of the number of arnoldi iterations for each initial vector
        #
        self.stats = {} # performance statistics, map name -> value

        self.freeze_attrs()

    def step(self):
        'krylov-based step function'

        if self.cur_basis_mat_list is None:
            if self.settings.print_output:
                print "Using Krylov method to make basis matrices"

            self.cur_basis_mat_list = self.make_cur_basis_mat_list()

        self.time_elapser.cur_basis_mat = self.cur_basis_mat_list[self.time_elapser.next_step].copy()

    def compute_nu_a(self, a_matrix, use_lanczos):
        '''initialize the krylov error bounds'''

        Timers.tic('compute_nu_a')

        mat = a_matrix

        if not use_lanczos: # not symmetric already
            Timers.tic('create_symmetric')
            mat = (mat + mat) / 2.0
            Timers.toc('create_symmetric')

        # minimum eigenvalue with -A is negative maximum eigenvalue of with A
        if self.settings.time_elapse.krylov.use_lanczos_eigenvalues:
            raise RuntimeError('unimplemented (use_lanczos_eigenvalues=True)')
        else:
            if mat.shape[0] > 1e6:
                print "Warning: krylov.use_lanczos_eigenvalues=False for high-dimensional system... may be slow"

            print "n = {}, norm = {}".format(mat.shape[0], np.linalg.norm(mat.toarray()))
            eig = -eigs(mat, k=1, which='LR', return_eigenvectors=False)[0].real

        Timers.toc('compute_nu_a')

        assert isinstance(eig, float)

        return eig

    def make_cur_basis_mat_list(self):
        '''
        Main work function. This returns the basis matrix at every step.

        This is called one time, and returns a list, element N is the basis matrix at step N
        '''

        # numpy raise errors overflow errors, ignore underflow
        np.seterr(all='warn', over='raise', under='ignore')

        settings = self.settings

        rv = self.init_krylov()

        if self.use_transpose:
            kry_init_space = self.time_elapser.output_space_csr
        else:
            kry_init_space = csr_matrix(self.time_elapser.init_space_csc.transpose())

        start = last_print = time.time()
        num_kry_init_vecs = kry_init_space.shape[0]

        if settings.print_output:
            print "Simulating from {} initial vector(s)".format(num_kry_init_vecs)

        for init_index in xrange(num_kry_init_vecs):
            sim = arnoldi_sim_autotune(self.time_elapser, kry_init_space[init_index])

            assign_from_sim(rv, sim, init_index, settings, self.use_transpose)

            if settings.print_output:
                now = time.time()

                if now - last_print > 1.0: # print every second
                    last_print = now
                    frac = float(init_index) / num_kry_init_vecs

                    if frac > 1e-9:
                        elapsed_sec = now - start
                        total_sec = elapsed_sec / frac
                        eta_sec = total_sec - elapsed_sec
                        eta = format_secs(eta_sec)

                        print "Arnoldi {} / {} ({:.2f}%, ETA: {})".format(
                            init_index, num_kry_init_vecs, 100.0 * frac, eta)

        if settings.print_output:
            elapsed = format_secs(time.time() - start)
            print "Krylov Simulation Total Time: {}\n".format(elapsed)

        # restore numpy error
        np.seterr(all='warn')

        return rv

    def init_krylov(self):
        '''
        initialize krylov interface for the computation

        returns a list of empty matrices to be filled in by the subsequent computation
        '''

        settings = self.settings
        key_dir_mat = self.time_elapser.output_space_csr
        init_space_csc = self.time_elapser.init_space_csc

        # check available memory before computing
        #i = time_elapser.init_space_csc.shape[1]
        #check_available_memory_basis(settings.print_output, time_elapser.settings.num_steps, key_dir_mat.shape[0], i)

        self.stats['arnoldi_iter'] = []
        #time_elapser.stats['arnoldi_mem_start'] = get_free_memory_mb()

        rv = []

        # initialize step zero
        step_zero_mat = (key_dir_mat * csr_matrix(init_space_csc)).toarray()

        rv.append(step_zero_mat)

        if settings.print_output:
            print "Basis matrix shape: {}".format(step_zero_mat.shape)

        # check if there's enough memory (otherwise python likes to consume everything before freezing or crashing)
        available_mb = psutil.virtual_memory().available / 1024.0 / 1024.0

        needed_mb = self.time_elapser.settings.num_steps * rv[0].shape[0] * rv[0].shape[1] * 8 / 1024.0 / 1024.0

        # keep a reserve of 1024 mb for other things
        if needed_mb + 1024 > available_mb:
            raise RuntimeError("Memory to store basis matrices ({:.1f}MB) exceeds available memory ({:.1f}MB)".format(\
                needed_mb + 1024, available_mb))

        # add zeros (allocate storage for result)
        for _ in xrange(self.settings.num_steps):
            rv.append(np.zeros(rv[0].shape, dtype=float))

        return rv

def odeint_sim(arg):
    '''
    simulate a given dense a-matrix with the provided initial vector, for a certain number of steps,
    returning the result at each step

    arg is tuple (a_matrix, start_vec, settings)
    '''

    a_matrix, start_vec, settings = arg

    assert a_matrix.shape[1] > 0
    assert isinstance(start_vec, np.ndarray)
    assert isinstance(settings, HylaaSettings)

    step = settings.step
    num_steps = settings.num_steps
    sim_tol = settings.time_elapse.krylov.odeint_simtol

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

    Timers.tic('odeint')
    result = odeint(der_func, start_vec, times, Dfun=jac_func, col_deriv=True, atol=sim_tol, rtol=sim_tol, \
            mxstep=int(1e8)) # mxstep = maximum number of internal steps
    Timers.toc('odeint')

    return result

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

def projected_sim(settings, h_mat, pv_mat):
    '''
    Simulate the system defined by h_mat, projected the result with pv_mat at each step

    returns a list of states at each time point
    '''

    assert h_mat.shape[0] > 1

    sim = None
    use_rel_error = settings.time_elapse.krylov.use_rel_error
    error = 0

    # use less arnoldi iterations than what's in the matrices
    if arnoldi_iter is not None:
        h_mat = h_mat[:arnoldi_iter, :arnoldi_iter].copy()
        pv_mat = pv_mat[:, :arnoldi_iter].copy()

    if limit is not None or arnoldi_iter is not None:
        small_h_mat = h_mat[:-1, :-1].copy()
        small_pv_mat = pv_mat[:, :-1].copy()

    if settings.time_elapse.krylov.use_odeint:
        Timers.tic('get_error odeint')
        start_vec = np.array([1.0 if d == 0 else 0.0 for d in xrange(h_mat.shape[0])], dtype=float)

        if limit is None:
            full_sim = odeint_sim((h_mat, start_vec, settings))

            sim = np.zeros((full_sim.shape[0] - 1, pv_mat.shape[0]), dtype=float)

            for i in xrange(1, full_sim.shape[0]): # skip step zero
                sim[i-1] = np.dot(pv_mat, full_sim[i])
        else:
            small_start_vec = start_vec[:-1].copy()
            args = [(h_mat, start_vec, settings), (small_h_mat, small_start_vec, settings)]

            full_sim, small_full_sim = [odeint_sim(a) for a in args]

            if np.all(abs(full_sim[1]) < 1e-9): # was compare with new zeros vec
                if settings.print_output:
                    print "First step of simulation was almost all zeros... increasing num iterations"

                error = limit + 1

            # sample last / middle / first before going through the whole thing
            steps = full_sim.shape[0]

            for step in [steps-1, steps / 2, 1]:
                cur_result = np.dot(pv_mat, full_sim[step])
                small_result = np.dot(small_pv_mat, small_full_sim[step])

                error = max(error, compute_error(cur_result, small_result, use_rel_error))

                if error > limit:
                    if settings.print_output:
                        print "Sim error with {} krylov iterations ({}) above limit ({})".format(
                            h_mat.shape[0], error, limit)
                    break

            if error < limit: # go through each step
                Timers.tic('krylov multiply by PV')

                if settings.print_output:
                    print "Sim error with {} iter at sampled times was low enough, checking all steps...".format(
                        h_mat.shape[0])

                sim = np.dot(full_sim[1:], pv_mat.T)

                for step in xrange(0, sim.shape[0]):
                    cur_result = sim[step]
                    small_result = np.dot(small_pv_mat, small_full_sim[step + 1])

                    error = max(error, compute_error(cur_result, small_result, use_rel_error))

                    if error > limit:
                        if settings.print_output:
                            print "Simulation error at step {} exceeds threshold: {} (limit: {})".format(
                                step, error, limit)

                        sim = None
                        break

                if settings.print_output and error < limit:
                    print "Simulation error was low enough at all steps: {} (limit: {})".format(error, limit)

                Timers.toc('krylov multiply by PV')

        Timers.toc('get_error odeint')
    else:
        Timers.tic('get_error expm')
        matrix_exp = expm(settings.step * h_mat)
        cur_col = matrix_exp[:, 0]
        Timers.toc('get_error expm')

        # for accuracy check
        Timers.tic('get_error expm')
        small_matrix_exp = expm(settings.step * small_h_mat) # step time is already included in loaded a_mat
        small_col = small_matrix_exp[:, 0]
        Timers.toc('get_error expm')

        # do the comparison at the first step
        cur_result = np.dot(pv_mat, cur_col)
        small_result = np.dot(small_pv_mat, small_col)
        error = max(error, compute_error(cur_result, small_result, use_rel_error))

        if return_sim:
            sim = [cur_result]

        for step in xrange(2, settings.num_steps + 1):
            cur_col = np.dot(matrix_exp, cur_col)
            small_col = np.dot(small_matrix_exp, small_col)

            # maybe we want to check error in the middle as well
            cur_result = np.dot(pv_mat, cur_col)
            small_result = np.dot(small_pv_mat, small_col)
            error = max(error, compute_error(cur_result, small_result, use_rel_error))

            if return_sim:
                sim.append(cur_result)

            if limit is not None and error > limit:
                if settings.print_output:
                    print "Error {} exceeded limit {} at step {}".format(error, limit, step)

                break

    return error if not return_sim else (error, sim)

def get_a_posterori_error(settings, h_mat, nu, error_limit):
    '''
    Compute the a posterori error bound

    from Theorem 3.1 in "Error Bounds for the Krylov Subspace Methods for Computations of Matrix Exponentials",
    by Hao Wang and Qiang Ye

    Err[T] <= H[k+1, k] * integral_{0}^{T} |h(t)| * g(t) dt
    where: h(t) = e_k^T * exp(-t * H_k) * e_1
        g(t) = exp((t - T) * nu(A))
        nu(A) = lamda_{min} (A + A^T)/2
    '''

    Timers.tic('a_posterori_error')

    max_time = settings.num_steps * settings.step
    k = h_mat.shape[1]
    assert h_mat.shape[0] == k + 1

    e_1 = np.array([[1.0] if n == 0 else [0] for n in xrange(k)], dtype=float)
    e_k_t = np.array([[1.0 if n == k-1 else 0 for n in xrange(k)]], dtype=float)

    h_mat_square = h_mat[:-1, :].toarray()

    def g(t):
        'g(t) = exp((t - T) * nu(A))'

        rv = math.exp((t - max_time) * nu)

        return rv

    def h(t):
        'h(t) = e_k^T * exp(-t * H_k) * e_1'

        exp_mat = expm(-t * h_mat_square)

        dotted = np.dot(exp_mat, e_1)

        rv = np.dot(e_k_t, dotted)[0, 0]

        assert isinstance(rv, float), "h's rv was not a scalar, instead it was: {} ({})".format(rv, type(rv))

        return rv

    def int_func(t):
        '|h(t)| * g(t)'

        rv = abs(h(t)) * g(t)

        assert isinstance(rv, float), "int_func's rv was not a scalar, instead it was: {} ({})".format(rv, type(rv))

        return rv


    samples = settings.time_elapse.krylov.integral_samples
    assert samples > 4

    print "start compute forsimps"

    print "TODO: optimize loop to compute in backwards order and exit upon excessive error"
    x_list = []
    y_list = []
    for i in xrange(samples):
        x = i * max_time / (samples-1.)
        y = int_func(x)

        x_list.append(x)
        y_list.append(y)

    print "end compute for simps"

    rv = h_mat[k, k-1] * simps(y_list, x_list, even='avg')

    Timers.toc('a_posterori_error')
    print "error bound for {} iterations: {}".format(k, rv)

    return rv

def arnoldi_sim_with_max_error(time_elapser, kry_init_vec_csr, iterations, error_limit):
    '''
    Run an arnoldi simulation with a fixed number of iterations and a target max error.
    If error_limit is None, just run the whole simulation

    returns a 2-tuple (a, b) with:
    a: projected simulation at each step, or None if the error limit is exceeded.
    b: the number of arnoldi iterations actually used
    '''

    settings = time_elapser.settings
    stdout = settings.time_elapse.krylov.stdout
    error = None
    obj = time_elapser.time_elapse_obj

    pv_mat, h_mat = obj.krylov_iteration.run_iteration(kry_init_vec_csr, iterations)

    if stdout:
        print "Finished Arnoldi/Lanczos... checking error"

    if h_mat.shape[0] <= iterations:
        error = 0
        iterations = h_mat.shape[0]

        if stdout:
            print "Arnoldi terminated early (after {} iterations). Simulating without error limit.".format(iterations)
    elif error_limit is not None:
        error = get_a_posterori_error(settings, h_mat, obj.nu_a, error_limit)

    if error_limit is None or error < error_limit:
        if stdout and error_limit is not None:
            print "Error {} was below threshold: {}".format(error, error_limit)

        h_mat = h_mat[:-1, :].copy()
        pv_mat = pv_mat[:, :-1].copy()

        print "implement projected_sim based off of get_error"
        #error, projected_sim = get_error(settings, h_mat, pv_mat, return_sim=True, limit=error_limit)
        rv = projected_sim(settings, h_mat, pv_mat)
    else:
        rv = None

    return rv, iterations

def arnoldi_sim_autotune(time_elapser, kry_init_vec_csr):
    '''
    Perform a projected simulation from a given initial vector. This auto-tunes the number
    of arnoldi iterations based on the error.

    returns the projected simulation at each step.
    '''

    settings = time_elapser.settings
    stdout = settings.time_elapse.krylov.stdout
    n = time_elapser.dims

    error_limit = settings.time_elapse.krylov.target_error

    arnoldi_iter = 4
    sim = None

    while True:
        if arnoldi_iter >= n:
            arnoldi_iter = n
            error_limit = None # do not target any error in this case

            if stdout:
                print "Arnoldi iter ({}) reached system dimension; skipping error".format(arnoldi_iter)

        if stdout:
            print "Trying {} iterations...".format(arnoldi_iter)

        sim, arnoldi_iter = arnoldi_sim_with_max_error(time_elapser, kry_init_vec_csr, arnoldi_iter, error_limit)

        if sim is not None:
            break
        else:
            arnoldi_iter = int(arnoldi_iter * 1.5) # increase the number of iterations and try again

    # update max used memory
    #prev_mem = time_elapser.stats['min_free_memory']
    #time_elapser.stats['min_free_memory'] = min(prev_mem, get_free_memory_mb('update_mem'))

    time_elapser.time_elapse_obj.krylov_iteration.reset() # done with the current start vector, free memory

    if stdout:
        print "Simulation was accurate enough with {} arnoldi iterations...".format(arnoldi_iter)

    time_elapser.time_elapse_obj.stats['arnoldi_iter'].append(arnoldi_iter)

    return sim

def assign_from_sim(rv, sim, index, settings, use_transpose):
    'assign a simulation to the result object'

    assert len(sim) == len(rv) - 1, "Got sim of length {}, expected {}".format(len(sim), len(rv) - 1)
    Timers.tic('update result list')

    for i in xrange(len(sim)):

        if settings.time_elapse.krylov.add_ones_key_dir:
            piece = sim[i][:-1]
        else:
            piece = sim[i][:]

        if use_transpose:
            rv[i+1][index, :piece.shape[0]] = piece
        else:
            rv[i+1][:piece.shape[0], index] = piece

    Timers.toc('update result list')
