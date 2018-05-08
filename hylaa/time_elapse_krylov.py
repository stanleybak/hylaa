'''
Time Elapse for the Krylov method using CPU or GPU
'''

import math
import time

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, csc_matrix
from scipy.integrate import odeint, simps
from scipy.sparse.linalg import eigsh

import psutil

from hylaa.timerutil import Timers
from hylaa.settings import HylaaSettings
from hylaa.util import Freezable
from hylaa.krylov_python import KrylovIteration

def is_symmetric(mat):
    'is the passed-in square matrix symmetric?'

    Timers.tic('is_symmetric')

    rv = (mat != mat.T).nnz == 0

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

        kryset = self.settings.time_elapse.krylov
        assert not (kryset.force_arnoldi and kryset.force_lanczos)
        if kryset.force_arnoldi:
            use_lanczos = False
        elif kryset.force_lanczos:
            use_lanczos = True
        else:
            use_lanczos = is_symmetric(a_matrix)

        # for error bound computation
        if not kryset.skip_error_bound:
            self.nu_a = self.compute_nu_a(a_matrix, use_lanczos)

        if self.use_transpose and not use_lanczos:
            # we need to compute with the transpose of the a matrix
            a_matrix = csr_matrix(a_matrix.transpose())
        else:
            a_matrix = a_matrix

        # modify in place, use -A with the krylov iteration
        a_matrix *= -1

        self.krylov_iteration = KrylovIteration(time_elapser.settings, a_matrix, use_lanczos, key_dir_mat)

        time_elapser.a_matrix = None # this makes sure it's not used since we modified it in place

        # -- performance statistics --
        # arnoldi_iter -> first index = initial vector, second index = multistep index, value = # of arnoldi iterations
        #
        self.stats = {} # performance statistics, map name -> value

        self.freeze_attrs()

    def find_max_eig_lanczos(self, mat, tol):
        'find the maximum eigenvalue of the passed-in matrix using lanczos ritz values'

        use_lanczos = False
        kry_iter = KrylovIteration(self.settings, mat, use_lanczos, None)
        n = mat.shape[0]
        rtol = 1e-4

        init = csr_matrix(np.random.rand(n)) # random initial vector
        last_eig = None

        for iterations in xrange(10, max(n+1, 11)):
            _, h_mat = kry_iter.run_iteration(init, iterations)

            h_mat = h_mat[:-1, :].copy()
            assert h_mat.shape[0] == h_mat.shape[1] # not sure how this works on early termination

            eig = eigsh(h_mat, k=1, which='LA', tol=tol, return_eigenvectors=False)[0].real

            # lanczos converged
            if h_mat.shape[0] < iterations:
                break

            # relative error termination condition
            if last_eig is not None:
                mag = max(rtol, abs(eig))
                rel_error = abs(last_eig - eig) / mag

                if self.settings.time_elapse.krylov.stdout:
                    print "Iter {}, eig = {}, rel_error = {}".format(iterations, eig, rel_error)

                if rel_error < rtol:
                    break

            last_eig = eig

        kry_iter.reset() # explicitly free memory

        return eig

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
        tol = 1e-5

        if not use_lanczos: # not symmetric already
            Timers.tic('create_symmetric')
            mat = (mat + mat) / 2.0
            Timers.toc('create_symmetric')

        # minimum eigenvalue with -A is negative maximum eigenvalue of with A
        if self.settings.time_elapse.krylov.use_lanczos_eigenvalues:
            eig = -self.find_max_eig_lanczos(mat, tol)
        else:
            if mat.shape[0] >= 1e5:
                print "Warning: krylov.use_lanczos_eigenvalues=False for high-dimensional system. May be slow."

            eig = -eigsh(mat, k=1, which='LA', tol=tol, return_eigenvectors=False)[0].real

        Timers.toc('compute_nu_a')

        assert isinstance(eig, float)

        if self.settings.time_elapse.krylov.stdout:
            print "Computed nu(A): {}".format(eig)

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

    Timers.tic("projected_sim")
    assert h_mat.shape[0] > 1

    sim = []
    cur_col = None

    Timers.tic('h-mat expm')
    one_step_mat = csc_matrix(-1 * settings.step * h_mat) # use negative time step
    matrix_exp = expm(one_step_mat).toarray()
    cur_col = matrix_exp[:, 0]
    Timers.toc('h-mat expm')

    sim.append(np.dot(pv_mat, cur_col))

    for _ in xrange(2, settings.num_steps + 1):
        cur_col = np.dot(matrix_exp, cur_col)
        sim.append(np.dot(pv_mat, cur_col))

    Timers.toc('projected_sim')

    return sim

def compute_h_list_sim(settings, h_mat_square, k, h_list_time_step, samples):
    '''compute error bound's h_list using simulation method'''

    Timers.tic('h_list sim')
    h_list = []

    e_1 = np.array([1.0 if n == 0 else 0 for n in xrange(k)], dtype=float)

    # simulate the system h_mat_square, starting at e1, getting the value of the last index (k-1) at each step
    der_func = lambda _, state, mat=h_mat_square: np.dot(mat, state)

    kryset = settings.time_elapse.krylov
    max_step = kryset.max_step
    atol, rtol = kryset.atol, kryset.rtol
    t_bound = -1 * settings.step * settings.num_steps
    sim_obj = kryset.ode_class(der_func, 0, e_1, t_bound, max_step, atol, rtol)

    cur_time = 0 # current time in h_list
    assert t_bound < 0
    assert h_list_time_step > 0

    while sim_obj.status == 'running' and sim_obj.t > t_bound:
        dense_out = None

        Timers.tic('dense_output')
        while sim_obj.t < cur_time:
            if dense_out is None:
                dense_out = sim_obj.dense_output()

            h_list.append(dense_out(cur_time)[k-1])
            cur_time -= h_list_time_step
        Timers.toc('dense_output')

        Timers.tic('step')
        sim_obj.step()
        Timers.toc('step')

    assert sim_obj.status == 'finished', 'RK45 failed. Status was: {}'.format(sim_obj.status)
    assert sim_obj.t == t_bound # should be exact

    # append the last value
    h_list.append(sim_obj.y[k-1])

    Timers.toc('h_list sim')

    return h_list

def compute_h_list_expmult(h_mat_square, k, h_list_time_step, samples):
    '''compute error bound's h_list using expm + mult method'''

    h_list = []

    e_1 = np.array([[1.0] if n == 0 else [0] for n in xrange(k)], dtype=float)

    Timers.tic('h(t) expm')
    one_step_expm = expm(-h_list_time_step * h_mat_square)
    Timers.toc('h(t) expm')
    cur_vec = e_1
    h_list.append(cur_vec[k-1][0])

    Timers.tic('h(t) mult')
    for _ in xrange(1, samples):
        cur_vec = np.dot(one_step_expm, cur_vec)

        h_list.append(cur_vec[k-1][0])

    Timers.toc('h(t) mult')

    return h_list

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

    assert settings.time_elapse.krylov.integral_samples > 4
    samples = settings.time_elapse.krylov.integral_samples

    max_time = settings.num_steps * settings.step
    k = h_mat.shape[1]
    assert h_mat.shape[0] == k + 1

    h_mat_square = h_mat[:-1, :]

    if not isinstance(h_mat_square, np.ndarray): # for arnoldi it's ndarray, for lanczos it's csr_matrix
        h_mat_square = h_mat_square.toarray()

    h_list_time_step = max_time / (samples-1.)

    # compute h_list for each step
    if settings.time_elapse.krylov.ode_class is None:
        h_list = compute_h_list_expmult(h_mat_square, k, h_list_time_step, samples)
    else:
        h_list = compute_h_list_sim(settings, h_mat_square, k, h_list_time_step, samples)

    def g(t):
        'g(t) = exp((t - T) * nu(A))'

        rv = math.exp((t - max_time) * nu)

        return rv

    def h(step):
        'h(t) = e_k^T * exp(-t * H_k) * e_1'

        return h_list[step]

    def int_func(t, step):
        '|h(t)| * g(t)'

        rv = abs(h(step)) * g(t)

        assert isinstance(rv, float), "int_func's rv was not a scalar, instead it was: {} ({})".format(rv, type(rv))

        return rv

    if nu * -max_time > 200: # would overflow exp() in computation of g()
        rv = np.inf

        if settings.time_elapse.krylov.stdout:
            print "Warning: excessive value of nu * -max_time in a priori error bound (was -nu large?)"
    else:
        x_list = []
        y_list = []

        for i in xrange(samples):
            x = i * max_time / (samples-1.)
            y = int_func(x, i)

            x_list.append(x)
            y_list.append(y)

        rv = h_mat[k, k-1] * simps(y_list, x_list, even='avg')

    Timers.toc('a_posterori_error')

    return rv

def arnoldi_sim_with_max_error(time_elapser, kry_init_vec_csr, iterations, error_limit):
    '''
    Run an arnoldi simulation with a fixed number of iterations and a target max error.
    If error_limit is None, just run the whole simulation

    returns a 2-tuple (a, b) with:
    a: projected simulation at each step, or None if the error limit was exceeded.
    b: the number of arnoldi iterations actually used
    '''

    settings = time_elapser.settings
    stdout = settings.time_elapse.krylov.stdout
    error = None
    obj = time_elapser.time_elapse_obj

    pv_mat, h_mat = obj.krylov_iteration.run_iteration(kry_init_vec_csr, iterations)

    if h_mat.shape[0] <= iterations:
        error = 0
        iterations = h_mat.shape[0]

        if settings.print_output:
            print "Arnoldi terminated early (after {} iterations). Simulating without error limit.".format(iterations)
    elif error_limit is not None:
        if settings.time_elapse.krylov.skip_error_bound:
            error = np.inf
        else:
            error = get_a_posterori_error(settings, h_mat, obj.nu_a, error_limit)

    if stdout:
        print "{} iterations had a posterori error {}".format(iterations, error)

    if error_limit is None or error < error_limit:
        if settings.print_output and error_limit is not None:
            print "Krylov error {} was below threshold ({}) with {} iterations".format(error, error_limit, iterations)

        h_mat = h_mat[:-1, :].copy()
        pv_mat = pv_mat[:, :-1].copy()

        rv = projected_sim(settings, h_mat, pv_mat)
    else:
        rv = None

    return rv, iterations

def arnoldi_sim_autotune(time_elapser, kry_init_vec_csr):
    '''
    Perform a projected simulation from a given initial vector. This auto-tunes the number
    of arnoldi iterations based on the desired error.

    returns the projected simulation at each step.
    '''

    settings = time_elapser.settings
    n = time_elapser.dims

    target_error = settings.time_elapse.krylov.target_error
    arnoldi_iter = 4

    while True:
        error_limit = target_error if arnoldi_iter < n else None

        rv, num_iter = arnoldi_sim_with_max_error(time_elapser, kry_init_vec_csr, arnoldi_iter, error_limit)

        if rv is None:
            arnoldi_iter = int(math.ceil(1.1 * arnoldi_iter))
        else:
            break

    time_elapser.time_elapse_obj.krylov_iteration.reset()
    time_elapser.time_elapse_obj.stats['arnoldi_iter'].append(num_iter)

    return rv

def assign_from_sim(rv, sim, index, settings, use_transpose):
    'assign a simulation to the result object'

    assert len(sim) == len(rv) - 1, "Got sim of length {}, expected {}".format(len(sim), len(rv) - 1)
    Timers.tic('update result list')

    for i in xrange(len(sim)):
        piece = sim[i][:]

        if use_transpose:
            rv[i+1][index, :piece.shape[0]] = piece
        else:
            rv[i+1][:piece.shape[0], index] = piece

    Timers.toc('update result list')
