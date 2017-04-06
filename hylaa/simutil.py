'''
Simulation Utility functions for Hylaa
This is a version which performs simulations at fixed time steps.

Stanley Bak
September 2016
'''

import os
import time
import multiprocessing

from scipy.integrate import odeint
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm as sparse_expm
from scipy.linalg import expm as dense_expm

import numpy as np

from hylaa.containers import SimulationSettings
from hylaa.util import Freezable
from hylaa.timerutil import Timers
import hylaa.openblas as openblas

class DyData(Freezable):
    'a container for dynamics-related data in a serializable (picklable) format'

    def __init__(self, a_matrix, b_vector, sparse):
        '''a' = ax + b_vec, b_vec can be null'''

        assert isinstance(a_matrix, csr_matrix)

        if b_vector is not None:
            assert isinstance(b_vector, csr_matrix)

        self.sparse = sparse

        self.sparse_a_matrix = a_matrix
        self.sparse_b_vector = b_vector
        self.affine = b_vector is not None
        self.num_dims = a_matrix.shape[0]

        # constructed from sparse matrix only after make_*_func() is called
        self.dense_a_matrix = None
        self.dense_b_vector = None

        # sometimes used with dense matrix
        self.jac_func = None
        self.max_upper = None
        self.max_lower = None

        self.freeze_attrs()

    def make_dense_matrices(self):
        '''
        make the dense versions of a_matrix and b_vector, if needed
        '''

        if self.dense_a_matrix is None:
            self.dense_a_matrix = self.sparse_a_matrix.toarray()

            if self.affine:
                self.dense_b_vector = self.sparse_b_vector.toarray()
                self.dense_b_vector.shape = (self.num_dims,)

    def make_der_func(self):
        '''
        get the function which returns the derivative, for use in ODEINT
        '''

        if self.sparse:
            if self.affine:
                def der_func(state, _):
                    'affine derivative function'

                    rv = np.array(self.sparse_a_matrix * state + self.sparse_b_vector)
                    rv.shape = (self.num_dims,)

                    return rv
            else:
                def der_func(state, _):
                    'linear derivative function'

                    rv = np.array(self.sparse_a_matrix * state)
                    rv.shape = (self.num_dims,)

                    return rv

        else: # use dense matrix
            self.make_dense_matrices()

            if self.affine:
                der_func = lambda state, _: np.add(np.dot(self.dense_a_matrix, state), self.dense_b_vector)
            else:
                der_func = lambda state, _: np.dot(self.dense_a_matrix, state)

        return der_func

    def make_jac_func(self):
        '''get the function which returns the jacobian, for use in ODEINT.

        This returns a tuple, (jac_func, max_upper, max_lower) where the second two are params for banded jacobians
        if self.sparse, returns (None, None, None)
        '''

        if self.sparse:
            rv = None, None, None
        else:
            if self.jac_func is None:
                self.make_dense_matrices()

                a_matrix = self.dense_a_matrix

                banded_jac, max_upper, max_lower = self.make_banded_jacobian()

                if len(banded_jac) * len(banded_jac[0]) < a_matrix.shape[0] * a_matrix.shape[1]:
                    # use banded jacobian
                    banded_jac_transpose = np.transpose(np.array(banded_jac, dtype=a_matrix.dtype)).copy()
                    self.jac_func = lambda dummy_state, dummy_t: banded_jac_transpose
                    self.max_upper = max_upper
                    self.max_lower = max_lower
                else:
                    # use non-banded jacobian
                    a_transpose = a_matrix.transpose().copy()
                    self.jac_func = lambda dummy_state, dummy_t: a_transpose

            rv = self.jac_func, self.max_upper, self.max_lower

        return rv

    def make_banded_jacobian(self):
        '''returns a banded jacobian list (in odeint's format), along with mu and ml parameters'''

        assert not self.sparse

        if self.dense_a_matrix is None:
            self.dense_a_matrix = self.sparse_a_matrix.toarray()

        matrix = self.dense_a_matrix

        # first find the values of mu and ml
        dims = matrix.shape[0]
        assert dims == matrix.shape[1]
        mu = 0
        ml = 0

        for row in xrange(dims):
            for col in xrange(dims):
                if matrix[row][col] != 0:
                    if col > row:
                        dif = col - row
                        mu = max(mu, dif)
                    else:
                        dif = row - col
                        ml = max(ml, dif)

        banded = []

        for yoffset in xrange(-mu, ml+1):
            row = []

            for diag in xrange(dims):
                x_index = diag
                y_index = diag + yoffset

                if y_index < 0 or y_index >= dims:
                    row.append(0.0)
                else:
                    row.append(matrix[y_index][x_index])

            banded.append(row)

        return (banded, mu, ml)

class SimulationBundle(Freezable):
    'a simulation bundle of basis vectors in a fixed set of dynamics (single mode)'

    def __init__(self, a_mat, b_vec, settings):
        '''x' = Ax + b  (b_vector here is the constant part of the dynamics, NOT the B input-effect matrix in 'BU') '''

        if isinstance(a_mat, list):
            a_mat = np.array(a_mat, dtype=float)

        if isinstance(b_vec, list):
            b_vec = np.array(b_vec, dtype=float)

        assert isinstance(a_mat, np.ndarray)
        assert isinstance(b_vec, np.ndarray)

        assert isinstance(settings, SimulationSettings)
        assert settings.step > 0

        self.settings = settings
        self.num_dims = b_vec.shape[0]
        assert self.num_dims > 0
        assert a_mat.shape[0] == a_mat.shape[1], "expected A matrix to be a square"
        assert len(b_vec.shape) == 1, "expected passed-in b_vector to be a single row: {}".format(b_vec)
        assert a_mat.shape[0] == b_vec.shape[0], "A matrix and b vector sizes should match"

        self.dy_data = DyData(csr_matrix(a_mat), csr_matrix(b_vec), settings.sparse)

        # initialize simulation result variables
        self.origin_sim = None
        self.vec_values = None
        self.step_offset = None

        # itemsize is bytes per float
        if self.settings.sim_mode == SimulationSettings.SIMULATION:
            mb_per_step = np.dtype(float).itemsize * self.num_dims * self.num_dims / 1024.0 / 1024.0
            self.max_steps_in_mem = max(1, int(settings.sim_in_memory_mb / mb_per_step) - 1)
        elif self.settings.sim_mode == SimulationSettings.MATRIX_EXP:
            self.max_steps_in_mem = 1

            if self.num_dims < 150:
                # dense expm is fast if dims < 150
                self.matrix_exp = dense_expm(a_mat * settings.step).transpose().copy()
            else:
                # sparse expm
                self.matrix_exp = sparse_expm(csc_matrix(a_mat * settings.step)).transpose().toarray()

        self.freeze_attrs()

    def simulate_origin(self, start, steps, include_step_zero=False):
        '''simulate the origin, from the last point in self.origin_sim, for a certain number of steps'''

        Timers.tic("simulation")
        result = raw_sim_one(start, steps, self.dy_data, self.settings, include_step_zero=include_step_zero)
        Timers.toc("simulation")

        return result

    def simulate_vecs(self, start_list, steps, include_step_zero=False):
        '''simulate from each of the vector points, from the last state in self.vec_values'''

        Timers.tic("simulation")

        # create the linear (nonaffine) dynamics
        linear_dy = DyData(self.dy_data.sparse_a_matrix, None, self.settings.sparse)

        sim_start_time = time.time()
        args = []

        for dim in xrange(self.num_dims):
            args.append([sim_start_time, start_list[dim], steps, include_step_zero, linear_dy, self.settings])

        if self.settings.stdout:
            SHARED_NEXT_PRINT.value = sim_start_time + self.settings.print_interval_secs
            SHARED_COMPLETED_SIMS.value = 0

            mb_per_step = np.dtype(float).itemsize * self.num_dims * self.num_dims / 1024.0 / 1024.0
            print "Simulating {} steps (~{:.2f} GB in memory)...".format(
                steps, steps * mb_per_step / 1024.0)

        result = self.parallel_sim(args)

        if self.settings.stdout:
            print "Total Simulation Time: {:.2f} secs".format(time.time() - sim_start_time)

        Timers.toc("simulation")

        # need to convert the results to a list at each time step before returning
        transpose_start = time.time()

        rv = []

        for step in xrange(result[0].shape[0]):
            single_step_result = np.empty((self.num_dims, self.num_dims))

            for dim in xrange(self.num_dims):
                single_step_result[dim] = result[dim][step]

            rv.append(single_step_result)

        # was like 11.5 seconds for PDE transpose time
        if self.settings.stdout:
            print "Transpose time: {:.2f} secs".format(time.time() - transpose_start)

        return rv

    def parallel_sim(self, args_list):
        '''actually call the parallel simulation function

        args_list is a list of tuples, each one is an arg to pool_sim_func
        '''

        num_threads = self.settings.threads if self.settings.threads is not None else multiprocessing.cpu_count()

        # unless threads is manually chosen, don't use pool for small dimensional systems
        if self.settings.threads is None and self.num_dims < 5:
            num_threads = 1

        if num_threads > 1:
            os.environ['OMP_NUM_THREADS'] = '1'

            # prevents using multiple threads within each thread
            openblas.set_num_threads(1)

            pool = multiprocessing.Pool(self.settings.threads)
            result = pool.map(pool_sim_func, args_list)
            pool.close()

            # restore state
            openblas.set_num_threads(num_threads)
        else:
            result = [pool_sim_func(a) for a in args_list]

        return result

    def presimulate(self, desired_step):
        '''
        as an optimization, run simulations up to some bound in preperation for many consecutive calls to
        get_vecs_origin_at_step(). This may get truncated due to memory limits.
        '''

        Timers.tic("sim + overhead")

        # if there are currently no simulations, or if the offset != 0
        if self.settings.sim_mode == SimulationSettings.SIMULATION and self.step_offset != 0:
            self.step_offset = 0

            # try to ensure step [0, desired_step] is in memory
            if desired_step >= self.max_steps_in_mem:
                desired_step = self.max_steps_in_mem - 1

            # presimulate origin
            start = np.zeros((self.num_dims))
            self.origin_sim = self.simulate_origin(start, desired_step, include_step_zero=True)

            assert len(self.origin_sim) == 1 + desired_step

            # presimulate vec_values
            start_list = np.identity(self.num_dims)
            self.vec_values = self.simulate_vecs(start_list, desired_step, include_step_zero=True)

            assert len(self.vec_values) == 1 + desired_step
            assert len(self.vec_values) == len(self.origin_sim)

        Timers.toc("sim + overhead")

    def get_vecs_origin_at_step(self, step, max_steps):
        '''
        get the exact state of the basis vectors and origin simulation
        at a specific, absolute step number (multiply by self.size to get time)

        max_step is the maximum number of steps we'd ever want (optimization so we don't waste time simulating too far)

        returns a tuple (list_of_basis_vecs, origin_sim_list)
        '''

        assert step <= max_steps

        Timers.tic("sim + overhead")

        if step == 0 or self.step_offset is None or step - self.step_offset < 0:
            if self.step_offset != 0:
                # reset origin sim and, if needed, vec_values
                self.origin_sim = [np.zeros((self.num_dims,))] # index is step (may be offset)
                self.vec_values = [np.identity(self.num_dims)] # index is step (may be offset)
                self.step_offset = 0

        rel_step = step - self.step_offset
        assert rel_step >= 0, 'relative step < 0?'

        # if we need to compute more steps
        while rel_step >= len(self.origin_sim):
            self.step_offset += len(self.origin_sim)
            rel_step = step - self.step_offset

            # double the simulation length each time
            num_new_steps = 2 * len(self.origin_sim)

            # but don't simulate past max_steps
            if self.step_offset + num_new_steps > max_steps + 1:
                num_new_steps = max_steps + 1 - self.step_offset

            # and obey desired memory limits
            if num_new_steps > self.max_steps_in_mem:
                num_new_steps = self.max_steps_in_mem

            # always advance by at least one step
            num_new_steps = max(1, num_new_steps)

            # advance origin
            start = self.origin_sim[-1].copy()
            self.origin_sim = None
            self.origin_sim = self.simulate_origin(start, num_new_steps)

            # also advance vec_values
            start_list = self.vec_values[-1].copy()
            self.vec_values = None

            if self.settings.sim_mode == SimulationSettings.SIMULATION:
                self.vec_values = self.simulate_vecs(start_list, num_new_steps)
            elif self.settings.sim_mode == SimulationSettings.MATRIX_EXP:
                self.vec_values = self.matrix_exp_vecs(start_list, num_new_steps)

        Timers.toc("sim + overhead")

        return (self.vec_values[rel_step], self.origin_sim[rel_step])

    def matrix_exp_vecs(self, start_list, num_steps):
        'use the one-step matrix exp strategy to get the next value of the basis vectors and origin simulation'

        assert num_steps == 1, "sim_mode == MATRIX_EXP so expected to advance 1 step but got {}".format(num_steps)

        # vec_values[-1] is the previous step's matrix
        cur_matrix = start_list

        # ensure you're using openblas for top performance of np.dot, see the readme
        result = np.dot(cur_matrix, self.matrix_exp)

        return [result]

    def compute_gbt(self, b_matrix):
        '''
        compute the transpose of G(A, h) * B using simulations

        Simulates from the origin for one step, using a fixed u1, u2, ...
        '''

        Timers.tic("input-effect simulation")
        assert b_matrix.shape[0] == self.num_dims

        num_inputs = b_matrix.shape[1]
        sim_start_time = time.time()

        args = []
        origin = np.zeros((self.num_dims), dtype=float)

        for dim in xrange(num_inputs):
            b_col = csr_matrix(b_matrix[:, dim])
            input_dy_data = DyData(self.dy_data.sparse_a_matrix, b_col, self.settings.sparse)

            args.append([sim_start_time, origin, 1, False, input_dy_data, self.settings])

        if self.settings.stdout:
            SHARED_NEXT_PRINT.value = sim_start_time + self.settings.print_interval_secs
            SHARED_COMPLETED_SIMS.value = 0

        result = self.parallel_sim(args)

        if self.settings.stdout:
            print "Total Input Simulation Time: {:.2f} secs".format(time.time() - sim_start_time)

        rv = np.zeros((num_inputs, self.num_dims), dtype=float)

        for dim in xrange(num_inputs):
            rv[dim, :] = result[dim][0]

        Timers.toc("input-effect simulation")

        return rv

    def sim_until_inv_violated(self, pt, inv_list, max_steps):
        '''
        simulate a point until the invariant is violated or max_steps is reached

        returns the list of points (each one is an ndarray)
        '''

        assert isinstance(pt, np.ndarray)

        num_steps = 1
        rv = [pt]

        inv_violated = False

        while len(rv) - 1 < max_steps:
            if len(rv) - 1 + num_steps > max_steps:
                num_steps = max_steps - len(rv) + 1

            start = rv[-1]
            new_states = raw_sim_one(start, num_steps, self.dy_data, self.settings)

            # for every new state, check if it violates the invariant
            for state in new_states:
                for inv in inv_list:
                    if np.dot(inv.vector, state) > inv.value:
                        inv_violated = True
                        break

                if inv_violated:
                    break
                else:
                    rv.append(state)

            if inv_violated:
                break
            else:
                num_steps *= 2

        return rv

# shared time variable used for occasional printing across processes
SHARED_NEXT_PRINT = multiprocessing.Value('d')
SHARED_COMPLETED_SIMS = multiprocessing.Value('i')

def pool_sim_func(args):
    'perform a single simulation possibly as part of parallel solving with multiprocessing.Pool'

    sim_start_time, start_point, steps, include_step_zero, dy_data, settings = args
    num_dims = start_point.shape[0]

    rv = raw_sim_one(start_point, steps, dy_data, settings, include_step_zero=include_step_zero)

    # print progress occasionally
    if settings.stdout:
        with SHARED_COMPLETED_SIMS.get_lock():
            SHARED_COMPLETED_SIMS.value += 1

        with SHARED_NEXT_PRINT.get_lock():
            now = time.time()

            if now > SHARED_NEXT_PRINT.value:
                SHARED_NEXT_PRINT.value = now + settings.print_interval_secs

                elapsed_time = now - sim_start_time
                percent = 100.0 * (SHARED_COMPLETED_SIMS.value) / (num_dims)
                total_time = elapsed_time / (percent / 100.0)

                print "{}/{} simulations ({:.1f}%); {:.1f}s (elapsed) / {:.1f}s (estimate)".format(
                    SHARED_COMPLETED_SIMS.value, num_dims, percent, elapsed_time, total_time)

    return rv

def raw_sim_one(start, steps, dy_data, settings, include_step_zero=False):
    '''
    simulate from a single point at the given times

    return an nparray of states at those times, possibly excluding time zero
    '''

    start.shape = (dy_data.num_dims,)

    times = np.linspace(0, settings.step * steps, num=steps+1)

    der_func = dy_data.make_der_func()
    jac_func, max_upper, max_lower = dy_data.make_jac_func()
    sim_tol = settings.sim_tol

    result = odeint(der_func, start, times, Dfun=jac_func, col_deriv=True, \
                    mxstep=int(1e8), mu=max_upper, ml=max_lower, \
                    atol=sim_tol, rtol=sim_tol)

    if not include_step_zero:
        result = result[1:]

    return result
