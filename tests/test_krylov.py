'''
Unit tests for Hylass's gpu_interface.py
Stanley Bak
August 2017
'''

import unittest
import random
import math
import time

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix, dia_matrix
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.integrate import odeint

from hylaa.settings import HylaaSettings, TimeElapseSettings
from hylaa.krylov_python import KrylovIteration
from hylaa.time_elapse import TimeElapser
from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init

from krypy.utils import arnoldi as krypy_arnoldi # krypy is used for testing

def make_settings():
    'make a hylaa settings object'

    h = HylaaSettings(0.1, 1.0)

    return h

def random_five_diag_sym_matrix(dims, print_progress=False):
    '''make a random symmetric csr_matrix 5-diagonal matrix

    there are 5 elements per row, for row index n we have:
    q_{n-2} p_{n-1} d_n p_n q_n
    '''

    q_n = None
    p_n = None
    q_n_minus_2 = None
    q_n_minus_1 = None
    p_n_minus_1 = None

    start = last_print = time.time()

    data_len = dims * 5 - 6

    if data_len >= 2**31:
        index_dtype = np.dtype('int64')
    else:
        index_dtype = np.dtype('int32')

    if print_progress:
        print "index_dtype is {}".format(index_dtype)

    data = np.zeros((data_len,), dtype=np.dtype('float64'))
    indices = np.zeros((data_len,), dtype=index_dtype)
    data_index = 0

    indptrs = np.zeros((dims+1,), dtype=index_dtype)
    indptr_index = 1 # zero element already in place at index 0

    if print_progress:
        total_bytes = data.nbytes + indices.nbytes + indptrs.nbytes
        print "allocate csr_matrix data time {:.1f}s, memory = {:.3f} GB".format(time.time() - start, \
            float(total_bytes) / 1024. / 1024. / 1024.)

    for row in xrange(dims):
        if print_progress and row > 0 and row % 100000 == 0 and time.time() - last_print > 1.0:
            last_print = time.time()
            elapsed = last_print - start

            eta = elapsed / (row / float(dims)) - elapsed
            print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s, ETA: {:.1f}m".format(row, dims, 100.0 * row / dims, \
                elapsed, eta / 60.0)


        if row > 1:
            data[data_index] = q_n_minus_2
            indices[data_index] = row-2
            data_index += 1

        if row > 0:
            data[data_index] = p_n_minus_1
            indices[data_index] = row-1
            data_index += 1

        d_n = random.random()
        data[data_index] = d_n
        indices[data_index] = row
        data_index += 1

        if row + 1 < dims:
            p_n = random.random()
            data[data_index] = p_n
            indices[data_index] = row+1
            data_index += 1

        if row + 2 < dims:
            q_n = random.random()
            data[data_index] = q_n
            indices[data_index] = row+2
            data_index += 1

        # update
        p_n_minus_1 = p_n
        q_n_minus_2 = q_n_minus_1
        q_n_minus_1 = q_n

        indptrs[indptr_index] = data_index
        indptr_index += 1

    if print_progress:
        elapsed = last_print - start
        print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s".format(dims, dims, 100.0, elapsed)

    start = time.time()

    rv = csr_matrix((data, indices, indptrs), shape=(dims, dims))

    if print_progress:
        print "making csr_matrix time {:.1f}s".format(time.time() - start)

    assert rv.data.base is data
    assert rv.indices.base is indices

    return rv

def random_sparse_matrix(dims, entries_per_row, symmetric=False, random_cols=True, print_progress=False):
    'make a random sparse matrix with the given number of entries per row'

    row_inds = []
    cols = []
    vals = []

    start = last_print = time.time()

    for row in xrange(dims):
        row_inds.append(len(vals))

        if print_progress and row % 10000 == 0 and time.time() - last_print > 1.0:
            last_print = time.time()
            elapsed = last_print - start
            print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s".format(row, dims, 100.0 * row / dims, elapsed)

        for entry_index in xrange(entries_per_row):

            if random_cols:
                r = random.random() * dims
                col = int(math.floor(r))
                cols.append(col)
            else:
                cols.append(entry_index)

            vals.append(random.random())

    row_inds.append(len(vals))

    if print_progress:
        elapsed = last_print - start
        print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s".format(dims, dims, 100.0, elapsed)

    start = time.time()

    rv = csr_matrix((vals, cols, row_inds), shape=(dims, dims), dtype=float)

    if print_progress:
        print "making csr_matrix time {:.1f}s".format(time.time() - start)

    if symmetric:
        start = time.time()

        rv = rv + rv.T

        if print_progress:
            print "transpose add time {:.1f}s".format(time.time() - start)

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

def load_pde():
    'loads the pde dynamics a matrix, returns (LinearAutomatonMode, init_matrix_csc)'

    ha = LinearHybridAutomaton()
    mode = ha.new_mode('mode')

    dynamics = loadmat('pde.mat')

    a_matrix = csc_matrix(dynamics['A'])
    col_ptr = [n for n in a_matrix.indptr]
    rows = [n for n in a_matrix.indices]
    data = [n for n in a_matrix.data]

    b_matrix = csc_matrix(dynamics['B'])
    num_inputs = b_matrix.shape[1]

    for u in xrange(num_inputs):
        rows += [n for n in b_matrix[:, u].indices]
        data += [n for n in b_matrix[:, u].data]
        col_ptr.append(len(data))

    combined_mat = csc_matrix((data, rows, col_ptr), \
        shape=(a_matrix.shape[0] + num_inputs, a_matrix.shape[1] + num_inputs))

    a_matrix = csr_matrix(combined_mat)
    mode.set_dynamics(a_matrix)

    c_matrix = csr_matrix(dynamics['C'])
    output_space = csr_matrix((c_matrix.data, c_matrix.indices, c_matrix.indptr), shape=(c_matrix.shape[0], \
                               c_matrix.shape[1] + num_inputs))

    mode.set_output_space(output_space)


    # making initial space
    n = a_matrix.shape[0]
    bounds_list = [] # bounds on each dimension

    for dim in xrange(n):
        if dim < 64:
            lb = 0
            ub = 0
        elif dim < 80:
            lb = 0.001
            ub = 0.0015
        elif dim < 84:
            lb = -0.002
            ub = -0.0015
        elif dim == n-1: # u1
            lb = 0.5
            ub = 1.0
        else:
            raise RuntimeError('unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    init_space, _, _, _ = bounds_list_to_init(bounds_list)

    return mode, init_space

def heat3d_dia(samples):
    'fast dia_matrix construction for heat3d dynamics'

    diffusity_const = 0.01
    heat_exchange_const = 0.5

    samples_sq = samples**2
    dims = samples**3
    step = 1.0/(samples + 1)

    a = diffusity_const * 1.0 / step**2
    d = -2.0 * (a + a + a)

    data = np.zeros((7, dims), dtype=float)
    offsets = np.array([-samples_sq, -samples, -1, 0, 1, samples, samples_sq], dtype=float)

    # element with z = -1
    data[0, :-samples_sq] = a

    # element with y = -1
    for s in xrange(samples):
        start = s*samples_sq
        end = (s+1)*(samples_sq) - samples
        data[1, start:end] = a

    # element with x = -1
    for s in xrange(samples_sq):
        start = s*samples
        end = (s+1)*(samples) - 1
        data[2, start:end] = a

    #### diagonal element ####
    data[3, :] = d # (prefill)

    # adjust when z = 0 or z = samples-1
    data[3, :samples_sq] += a
    data[3, -samples_sq:] += a

    # adjust when y = 0 or y = samples-1
    for z in xrange(samples):
        z_offset = z * samples_sq

        data[3, z_offset:z_offset + samples] += a
        data[3, z_offset+samples_sq-samples:z_offset+samples_sq] += a

    # adjust when x = 0 (and add diffusion term when x = samples-1)
    for z in xrange(samples):
        for y in xrange(samples):
            offset = z * samples_sq + y * samples

            data[3, offset] += a

            data[3, offset + samples - 1] += a/(1+heat_exchange_const*step)

    #### end diagnal element ####
    # element with x = +1
    for s in xrange(samples_sq):
        start = 1 + s * samples
        end = (s+1) * samples
        data[4, start:end] = a

    # element with y = +1
    for s in xrange(samples):
        start = s*samples_sq+samples
        end = (s+1)*(samples_sq)
        data[5, start:end] = a

    # element with z = +1
    data[6, samples_sq:] = a

    rv = dia_matrix((data, offsets), shape=(dims, dims))
    assert np.may_share_memory(rv.data, data) # make sure we didn't copy memory

    return rv

def load_heat(samples):
    'loads the pde dynamics a matrix, returns (LinearAutomatonMode, init_matrix_csc)'

    assert samples >= 10 and samples % 10 == 0, "init region isn't evenly divided by discretization"

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')

    a_matrix = heat3d_dia(samples)

    n = a_matrix.shape[0]

    assert isinstance(a_matrix, dia_matrix)
    mode.set_dynamics(a_matrix)

    # set output space
    center_1d = int(math.floor(samples/2.0))
    center_dim = center_1d * samples**2 + center_1d * samples + center_1d
    output_space = csr_matrix(np.array([[1.0 if d == center_dim else 0.0 for d in xrange(n)]], dtype=float))
    
    mode.set_output_space(output_space)

    # create init space
    data = []
    inds = []
    indptrs = [0]

    for z in xrange(samples / 10 + 1):
        zoffset = z * samples * samples

        for y in xrange(2 * samples / 10 + 1):
            yoffset = y * samples

            for x in xrange(4 * samples / 10 + 1):
                dim = x + yoffset + zoffset

                data.append(1)
                inds.append(dim)

    indptrs.append(len(data))
    init_space = csc_matrix((data, inds, indptrs), dtype=float, shape=(n, 1))

    return mode, init_space

class TestKrylov(unittest.TestCase):
    'Unit tests for hylaa.krylov'

    def setUp(self):
        'test setup'

        np.set_printoptions(suppress=True)
        random.seed(1)

        # testing code for vector comparison
        #for x in xrange(dims):
        #    print "{}, res1={}, res2={}".format(x, res1[x], res2[x])
        #    self.assertAlmostEqual(res1[x], res2[x], places=3, msg='result[{}] differs'.format(x))

    def test_arnoldi(self):
        'compare the krypy implementation against the python version'

        dims = 10
        iterations = 5
        a_matrix = random_sparse_matrix(dims, entries_per_row=3)
        key_dir_mat = csr_matrix(np.identity(dims))

        init1_dense = np.array([[1.] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        init2_dense = np.array([[2.] if d == 1 else [.4] if d == dims-1 else [0.0] for d in xrange(dims)], dtype=float)

        for init_dense in [init1_dense, init2_dense]:
            init_sparse = csr_matrix(init_dense.T)

            # for krypy, we must manually normalize and rescale
            norm = np.linalg.norm(init_dense)
            init_dense /= norm
            krypy_v, krypy_h = krypy_arnoldi(a_matrix, init_dense, maxiter=iterations)
            krypy_v *= norm

            # using python
            use_transpose = False
            ksim = KrylovIteration(make_settings(add_ones_row=False), a_matrix, use_transpose, key_dir_mat)
            python_pv, python_h = ksim.run_iteration(init_sparse, iterations)

            self.assertEquals(python_pv.shape, krypy_v.shape)
            self.assertEquals(python_h.shape, krypy_h.shape)

            self.assertTrue(np.allclose(python_h, krypy_h), "Python h matrix incorrect")
            self.assertTrue(np.allclose(python_pv, krypy_v), "Python v matrix incorrect")

    def test_lanczos(self):
        'compare the krypy implementation against the python version'

        dims = 5
        iterations = 3
        a_matrix = random_sparse_matrix(dims, entries_per_row=2, symmetric=True)

        key_dir_mat = csr_matrix(np.identity(dims))

        init1_dense = np.array([[1.] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        init2_dense = np.array([[2.] if d == 1 else [.4] if d == dims-1 else [0.0] for d in xrange(dims)], dtype=float)

        for init_dense in [init1_dense, init2_dense]:
            init_sparse = csr_matrix(init_dense.T)

            # for krypy, manually scale based on initial vec
            norm = np.linalg.norm(init_dense)
            init_dense /= norm
            krypy_v, krypy_h = krypy_arnoldi(a_matrix, init_dense, maxiter=iterations, ortho='lanczos')
            krypy_v *= norm

            # using python
            ksim = KrylovIteration(make_settings(False), a_matrix, False, key_dir_mat)
            ksim.lanczos = True # force usin lanczos iteration
            python_pv, python_h = ksim.run_iteration(init_sparse, iterations)
            python_h = python_h.toarray()

            self.assertEquals(python_pv.shape, krypy_v.shape)
            self.assertEquals(python_h.shape, krypy_h.shape)

            self.assertTrue(np.allclose(python_h, krypy_h), "Python h matrix incorrect")
            self.assertTrue(np.allclose(python_pv, krypy_v), "Python v matrix incorrect")

    def test_lanczos_sim(self):
        'compare simulation vs python_lanczos'

        dims = 1000
        iterations = 50
        sim_time = 0.1

        e1_sparse = csr_matrix(([1.0], [0], [0, 1]), shape=(1, dims))
        e1_dense = np.array([1.0 if d == 0 else 0.0 for d in xrange(iterations)], dtype=float)

        a_matrix_sparse = random_sparse_matrix(dims, entries_per_row=50, symmetric=True)

        # two key directions
        key_dir_mat = csr_matrix([[1.0 for _ in xrange(dims)], [1.0 if i == 0 else 0.0 for i in xrange(dims)]])

        # using python lanczos
        ksim = KrylovIteration(make_settings(False), a_matrix_sparse, False, key_dir_mat)
        ksim.lanczos = True # force usin lanczos iteration
        python_pv, python_h = ksim.run_iteration(e1_sparse, iterations)

        python_pv = python_pv[:, :iterations]
        python_h = python_h[:iterations, :iterations]

        python_result = np.dot(python_pv, expm_multiply(python_h * sim_time, e1_dense))

        # using odeint
        a_matrix = a_matrix_sparse.toarray()
        der_func = lambda state, _: np.dot(a_matrix, state)
        a_transpose = a_matrix.transpose().copy()
        jac_func = lambda dummy_state, dummy_t: a_transpose

        times = np.linspace(0, sim_time)
        start_vec = np.array([1.0 if d == 0 else 0.0 for d in xrange(dims)], dtype=float)
        odeint_result = odeint(der_func, start_vec, times, Dfun=jac_func, col_deriv=True, mxstep=int(1e8))[-1]
        proj_odeint_result = key_dir_mat * odeint_result

        self.assertTrue(np.allclose(python_result, proj_odeint_result, atol=1e-3), "python result incorrect")

    def test_lanczos_reinit(self):
        'compare simulation vs python_lanczos'

        dims = 1000
        iterations = 50
        sim_time = 0.1

        e1_sparse = csr_matrix(([1.0], [0], [0, 1]), shape=(1, dims))
        e1_dense = np.array([1.0 if d == 0 else 0.0 for d in xrange(iterations)], dtype=float)

        a_matrix_sparse = random_sparse_matrix(dims, entries_per_row=50, symmetric=True)

        # two key directions
        key_dir_mat = csr_matrix([[1.0 for _ in xrange(dims)], [1.0 if i == 0 else 0.0 for i in xrange(dims)]])

        # using python lanczos
        ksim = KrylovIteration(make_settings(False), a_matrix_sparse, False, key_dir_mat)
        ksim.lanczos = True # force usin lanczos iteration
        python_pv, python_h = ksim.run_iteration(e1_sparse, iterations)

        python_pv = python_pv[:, :iterations]
        python_h = python_h[:iterations, :iterations]

        python_result = np.dot(python_pv, expm_multiply(python_h * sim_time, e1_dense))

        ###########
        # continue for more iterations
        reinit_pv, reinit_h = ksim.run_iteration(e1_sparse, 2*iterations)

        reinit_pv = reinit_pv[:, :iterations]
        reinit_h = reinit_h[:iterations, :iterations]

        reinit_result = np.dot(reinit_pv, expm_multiply(reinit_h * sim_time, e1_dense))
        ##########
        # reset and restart
        ksim.reset()

        reset_pv, reset_h = ksim.run_iteration(e1_sparse, iterations)

        reset_pv = reset_pv[:, :iterations]
        reset_h = reset_h[:iterations, :iterations]

        reset_result = np.dot(reset_pv, expm_multiply(reset_h * sim_time, e1_dense))

        # using odeint
        a_matrix = a_matrix_sparse.toarray()
        der_func = lambda state, _: np.dot(a_matrix, state)
        a_transpose = a_matrix.transpose().copy()
        jac_func = lambda dummy_state, dummy_t: a_transpose

        times = np.linspace(0, sim_time)
        start_vec = np.array([1.0 if d == 0 else 0.0 for d in xrange(dims)], dtype=float)
        odeint_result = odeint(der_func, start_vec, times, Dfun=jac_func, col_deriv=True, mxstep=int(1e8))[-1]
        proj_odeint_result = key_dir_mat * odeint_result

        self.assertTrue(np.allclose(python_result, proj_odeint_result, atol=1e-3), "python result incorrect")
        self.assertTrue(np.allclose(reinit_result, proj_odeint_result, atol=1e-3), "reinit result incorrect")
        self.assertTrue(np.allclose(reset_result, proj_odeint_result, atol=1e-3), "reset result incorrect")

    def test_arnoldi_one_norm(self):
        'test arnoldi computaiton with one norm setting'

        dims = 10
        iterations = 5
        a_matrix = random_sparse_matrix(dims, entries_per_row=3)
        key_dir_mat = csr_matrix(np.identity(dims))

        data = key_dir_mat.data
        indices = key_dir_mat.indices
        indptr = key_dir_mat.indptr

        data = np.concatenate((data, [1.0 for _ in xrange(dims)]))
        indices = np.concatenate((indices, [i for i in xrange(dims)]))
        indptr = np.concatenate((indptr, [len(data)]))
        key_dir_mat_with_one_norm = csr_matrix((data, indices, indptr), shape=(dims + 1, dims))

        init_dense = np.array([[2.] if d == 1 else [.4] if d == dims-1 else [0.0] for d in xrange(dims)], dtype=float)
        #init_dense = np.array([[1.] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)

        init_sparse = csr_matrix(init_dense.T)

        ksim = KrylovIteration(make_settings(True), a_matrix, False, key_dir_mat)
        python_pv_auto, python_h_auto = ksim.run_iteration(init_sparse, iterations)

        # using python
        ksim = KrylovIteration(make_settings(False), a_matrix, False, key_dir_mat_with_one_norm)
        python_pv, python_h = ksim.run_iteration(init_sparse, iterations)

        self.assertTrue(np.allclose(python_h, python_h_auto), "H matrices don't match")
        self.assertTrue(np.allclose(python_pv, python_pv_auto), "PV matrices don't match")

    def test_lanczos_one_norm(self):
        'test lanczos computation with one norm setting'

        dims = 10
        iterations = 5
        a_matrix = random_sparse_matrix(dims, entries_per_row=2, symmetric=True)
        key_dir_mat = csr_matrix(np.identity(dims))

        data = key_dir_mat.data
        indices = key_dir_mat.indices
        indptr = key_dir_mat.indptr

        data = np.concatenate((data, [1.0 for _ in xrange(dims)]))
        indices = np.concatenate((indices, [i for i in xrange(dims)]))
        indptr = np.concatenate((indptr, [len(data)]))
        key_dir_mat_with_one_norm = csr_matrix((data, indices, indptr), shape=(dims + 1, dims))

        init_dense = np.array([[2.] if d == 1 else [.4] if d == dims-1 else [0.0] for d in xrange(dims)], dtype=float)
        #init_dense = np.array([[1.] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)

        init_sparse = csr_matrix(init_dense.T)

        ksim = KrylovIteration(make_settings(True), a_matrix, False, key_dir_mat)
        self.assertTrue(ksim.lanczos)
        python_pv_auto, python_h_auto = ksim.run_iteration(init_sparse, iterations)

        # using python
        ksim = KrylovIteration(make_settings(False), a_matrix, False, key_dir_mat_with_one_norm)
        ksim.lanczos = True
        python_pv, python_h = ksim.run_iteration(init_sparse, iterations)

        self.assertEqual(python_h.shape, python_h_auto.shape)
        self.assertEqual(python_pv.shape, python_pv_auto.shape)

        self.assertTrue(np.allclose(python_h.toarray(), python_h_auto.toarray()), "H matrices don't match")
        self.assertTrue(np.allclose(python_pv, python_pv_auto), "PV matrices don't match")

    def test_tuning(self):
        'test tuning method with krylov simulation'

        mode, init_space_csc = load_heat(40) #load_pde()

        settings = make_settings()
        settings.time_elapse.method = TimeElapseSettings.KRYLOV
        settings.time_elapse.check_answer = True
        k = settings.time_elapse.krylov

        k.stdout = True
        k.ode_class = None

        print "making timeelapser"
        te = TimeElapser(mode, settings, init_space_csc)

        print "calling step"
        te.step()

        self.assertLess(te.time_elapse_obj.stats['arnoldi_iter'][0], 80)

    def test_eigenvalues(self):
        '''performance testing for computing eigenvalues of large matrices'''

        mode, init_space_csc = load_heat(20) #load_pde()
        print "Finished Making ODEs...\n"

        mat = mode.a_matrix_csr
        k=1

        start = time.time()
        vals_LA = eigsh(mat, k=k, return_eigenvectors=False, which='LA')
        print "eigsh result = {}".format(vals_LA[0].real)
        print "LA computation time: {:.2} sec".format(time.time() - start)

if __name__ == '__main__':
    unittest.main()
