'''
Unit tests for Hylass's gpu_interface.py
Stanley Bak
August 2017
'''
import array

import unittest
import random
import math
import time

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from scipy.integrate import odeint

from hylaa.krylov_python import python_arnoldi, python_lanczos
from hylaa.krylov_interface import KrylovInterface

from hylaa.containers import HylaaSettings

from krypy.utils import arnoldi as krypy_arnoldi # krypy is used for testing

def get_projected_simulation(settings, dim, use_mult=False):
    '''
    Get the projected simulation using the current settings.
    '''

    h_mat, pv_mat = KrylovInterface.arnoldi_unit(dim)

    h_mat = h_mat[:-1, :].copy()
    pv_mat = pv_mat[:, :-1].copy()

    time_mult = settings.step if use_mult else settings.step * settings.num_steps

    matrix_exp = expm(h_mat * time_mult)
    cur_col = matrix_exp[:, 0]

    if use_mult:
        for _ in xrange(2, settings.num_steps + 1):
            cur_col = np.dot(matrix_exp, cur_col)

    cur_result = np.dot(pv_mat, cur_col)
    cur_result.shape = (pv_mat.shape[0], 1)

    return cur_result

def make_spring_mass_matrix(num_dims):
    '''get the A matrix corresponding to the dynamics for the spring mass system'''

    # construct as a csr_matrix
    values = []
    indices = []
    indptr = []

    assert num_dims % 2 == 0
    num_masses = num_dims / 2

    for mass in xrange(num_masses):
        dim = 2*mass

        indptr.append(len(values))

        if dim - 1 >= 0:
            indices.append(dim-1)
            values.append(1.0)

        indices.append(dim+1)
        values.append(-2.0)

        if dim + 3 < num_dims:
            indices.append(dim + 3)
            values.append(1.0)

        indptr.append(len(values))

        indices.append(dim)
        values.append(1.0)

    indptr.append(len(values))

    return csr_matrix(csc_matrix((values, indices, indptr), shape=(num_dims, num_dims), dtype=float))

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

    data = np.zeros((data_len,), dtype=np.dtype('float64'))
    indices = np.zeros((data_len,), dtype=index_dtype)
    data_index = 0

    indptrs = np.zeros((dims+1,), dtype=index_dtype)
    indptr_index = 1 # zero element already in place at index 0

    if print_progress:
        print "allocate csr_matrix data time {:.1f}s".format(time.time() - start)

    for row in xrange(dims):
        if print_progress and row > 0 and row % 100000 == 0 and time.time() - last_print > 1.0:
            last_print = time.time()
            elapsed = last_print - start

            eta = elapsed / (row / float(dims)) - elapsed
            print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s, ETA: {:.1f}min".format(row, dims, 100.0 * row / dims,
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

class TestKrylov(unittest.TestCase):
    'Unit tests for hylaa.krylov'

    def setUp(self):
        'test setup'

        KrylovInterface.reset()
        np.set_printoptions(suppress=True)
        random.seed(1)

        # testing code for vector comparison
        #for x in xrange(dims):
        #    print "{}, res1={}, res2={}".format(x, res1[x], res2[x])
        #    self.assertAlmostEqual(res1[x], res2[x], places=3, msg='result[{}] differs'.format(x))

    def test_arnoldi(self):
        'compare the krypy implementation with the python and cusp implementations'

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
            python_pv, python_h = python_arnoldi(a_matrix, init_sparse, iterations, key_dir_mat)

            self.assertTrue(np.allclose(python_h, krypy_h), "Python h matrix incorrect")
            self.assertTrue(np.allclose(python_pv, krypy_v), "Python v matrix incorrect")

            # using cusp
            KrylovInterface.preallocate_memory(iterations, dims, key_dir_mat.shape[0], False, True)
            KrylovInterface.load_a_matrix(a_matrix)
            KrylovInterface.load_key_dir_matrix(key_dir_mat)
            cusp_pv, cusp_h = KrylovInterface.arnoldi(init_sparse)

            self.assertTrue(np.allclose(cusp_h, krypy_h), "Cusp h matrix incorrect")
            self.assertTrue(np.allclose(cusp_pv, krypy_v), "Cusp v matrix incorrect")

    def test_lanczos(self):
        'compare the krypy implementation with the python and cusp implementations'

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
            python_pv, python_h = python_lanczos(a_matrix, init_sparse, iterations, key_dir_mat, compat=True)
            python_h = python_h.toarray()

            self.assertTrue(np.allclose(python_h, krypy_h), "Python h matrix incorrect")
            self.assertTrue(np.allclose(python_pv, krypy_v), "Python v matrix incorrect")

            # using cusp
            KrylovInterface.preallocate_memory(iterations, dims, key_dir_mat.shape[0], True, False)
            KrylovInterface.load_a_matrix(a_matrix)
            KrylovInterface.load_key_dir_matrix(key_dir_mat)

            cusp_pv, cusp_h = KrylovInterface.lanczos(init_sparse)
            cusp_h = cusp_h.toarray()

            self.assertTrue(np.allclose(cusp_h, krypy_h), "Cusp h matrix incorrect")
            self.assertTrue(np.allclose(cusp_pv, krypy_v), "Cusp v matrix incorrect")

    def test_lanczos_sim(self):
        'compare simulation vs python_lanczos vs cusp_lanczos'

        dims = 1000
        iterations = 50
        sim_time = 0.1

        e1_sparse = csr_matrix(([1.0], [0], [0, 1]), shape=(1, dims))
        e1_dense = np.array([1.0 if d == 0 else 0.0 for d in xrange(iterations)], dtype=float)

        a_matrix_sparse = random_sparse_matrix(dims, entries_per_row=50, symmetric=True)

        # two key directions
        key_dir_mat = csr_matrix([[1.0 for _ in xrange(dims)], [1.0 if i == 0 else 0.0 for i in xrange(dims)]])

        # using python lanczos
        python_pv, python_h = python_lanczos(a_matrix_sparse, e1_sparse, iterations, key_dir_mat)

        python_pv = python_pv[:, :iterations]
        python_h = python_h[:iterations, :iterations]

        python_result = np.dot(python_pv, expm_multiply(python_h * sim_time, e1_dense))

        # using cusp lanczos
        KrylovInterface.preallocate_memory(iterations, dims, key_dir_mat.shape[0], True, True)
        KrylovInterface.load_a_matrix(a_matrix_sparse)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        cusp_pv, cusp_h = KrylovInterface.lanczos(e1_sparse)

        cusp_pv = cusp_pv[:, :iterations]
        cusp_h = cusp_h[:iterations, :iterations]

        cusp_result = np.dot(cusp_pv, expm_multiply(cusp_h * sim_time, e1_dense))

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
        self.assertTrue(np.allclose(cusp_result, proj_odeint_result, atol=1e-3), "cusp result incorrect")

    def test_lanczos_profile(self):
        'test my implementation of lanczos with a large system'

        # laptop, 2e7, allocate 24.5 secs, lanczos ~3.6 secs
        # desktop, 2e7, allocate 24.2 secs, lanczos ~2.9 secs
        # 1e8, allocate 122 secs, lanczos ~15 secs

        dims = int(1e9)
        iterations = 10

        a_matrix = random_five_diag_sym_matrix(dims, True)
        k_mat = csr_matrix(([1.0], [0], [0, 1]), shape=(1, dims)) # first coordinate
        e1_sparse = csr_matrix(([1.0], [0], [0, 1]), shape=(1, dims))

        # using cusp
        start = time.time()
        KrylovInterface.set_use_profiling(True)
        KrylovInterface.set_print_output(True)
        KrylovInterface.preallocate_memory(iterations, dims, k_mat.shape[0], True, True)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(k_mat)
        iter_start = time.time()
        cusp_pv, cusp_h = KrylovInterface.lanczos(e1_sparse)

        KrylovInterface.print_profiling_data()

        print "cusp lanczos time = {}".format(time.time() - iter_start)
        print "cusp total time = {}\n".format(time.time() - start)

        # using python
        start = time.time()
        python_pv, python_h = python_lanczos(a_matrix, e1_sparse, iterations, k_mat, compat=True, profile=True)
        print "python lanczos time = {}\n".format(time.time() - start)

        print "python pv shape = {}".format(python_pv.shape)

        self.assertEqual(cusp_h.shape, python_h.shape)
        self.assertEqual(cusp_pv.shape, python_pv.shape)

    def test_arnoldi_vec(self):
        'test arnoldi simulation with a passed in initial vector'

        dims = 5
        iterations = 5 # with iterations = dims, answer should be exact
        key_dirs = 2

        init_vec = np.array([[1 + 2.0 * d] for d in xrange(dims)], dtype=float)
        a_matrix = random_sparse_matrix(dims, entries_per_row=3)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # do direct expm
        real_vec = expm_multiply(csc_matrix(a_matrix), init_vec)
        real_answer = key_dir_mat * real_vec
        real_answer.shape = (key_dirs,)

        # do cusp
        KrylovInterface.preallocate_memory(iterations, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(csr_matrix(key_dir_mat))

        result_h, result_pv = KrylovInterface.arnoldi_vec(init_vec)

        h_mat = result_h[:-1, :].copy()
        pv_mat = result_pv[:, :-1].copy()

        exp = expm(h_mat)[:, 0]
        cusp_answer = np.dot(pv_mat, exp)

        self.assertEqual(len(real_answer), len(cusp_answer), "real and cusp answer don't have same length")
        self.assertTrue(np.allclose(real_answer, cusp_answer))

    def test_arnoldi_offset(self):
        'compare the python implementation with the cusp implementation with a single initial vector (2nd column)'

        #KrylovInterface.set_use_profiling(True)

        for gpu in [False, True]:
            if gpu and not KrylovInterface.has_gpu():
                continue

            KrylovInterface.set_use_gpu(gpu)

            dims = 5
            iterations = 2
            key_dirs = 2

            a_matrix = random_sparse_matrix(dims, entries_per_row=2)

            key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

            # using python
            init_vec = np.array([[1.0] if d == 1 else [0.0] for d in xrange(dims)], dtype=float)
            v_mat_testing, h_mat_testing = python_arnoldi(a_matrix, init_vec, iterations)

            projected_v_mat_testing = key_dir_mat * v_mat_testing

            # using cusp

            KrylovInterface.preallocate_memory(iterations, dims, key_dirs)
            KrylovInterface.load_a_matrix(a_matrix)
            KrylovInterface.load_key_dir_matrix(key_dir_mat)

            result_h, result_pv = KrylovInterface.arnoldi_unit(1)

            self.assertTrue(np.allclose(result_h, h_mat_testing), "Incorrect h matrix use_gpu = {}".format(gpu))
            self.assertTrue(np.allclose(result_pv, projected_v_mat_testing), \
                "Incorrect projected v matrix, use_gpu = {}".format(gpu))

    def test_iss(self):
        'test the cusp implementation using the iss model'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        iterations = 10

        a_matrix = csr_matrix(loadmat('iss.mat')['A'])
        dims = a_matrix.shape[0]

        dir1 = np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float)
        dir2 = np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float)
        key_dir_mat = csr_matrix([dir1, dir2], dtype=float)

        #key_dir_mat = csr_matrix(np.identity(dims))

        # use initial dimensions 100 and 101 and 102

        # using python
        init_vec1 = np.array([[1.0] if d == 100 else [0.0] for d in xrange(dims)], dtype=float)
        init_vec2 = np.array([[1.0] if d == 101 else [0.0] for d in xrange(dims)], dtype=float)
        init_vec3 = np.array([[1.0] if d == 102 else [0.0] for d in xrange(dims)], dtype=float)

        v_mat_testing1, h_mat_testing1 = python_arnoldi(a_matrix, init_vec1, iterations)
        projected_v_mat_testing1 = key_dir_mat * v_mat_testing1

        v_mat_testing2, h_mat_testing2 = python_arnoldi(a_matrix, init_vec2, iterations)
        projected_v_mat_testing2 = key_dir_mat * v_mat_testing2

        v_mat_testing3, h_mat_testing3 = python_arnoldi(a_matrix, init_vec3, iterations)
        projected_v_mat_testing3 = key_dir_mat * v_mat_testing3

        # using cusp
        KrylovInterface.preallocate_memory(iterations, dims, key_dir_mat.shape[0])
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h1, result_pv1 = KrylovInterface.arnoldi_unit(100)
        result_h2, result_pv2 = KrylovInterface.arnoldi_unit(101)
        result_h3, result_pv3 = KrylovInterface.arnoldi_unit(102)

        self.assertTrue(np.allclose(result_h1, h_mat_testing1), "Correct h matrix init vec 100")
        self.assertTrue(np.allclose(result_pv1, projected_v_mat_testing1), "Correct projV matrix for init vec 100")

        self.assertTrue(np.allclose(result_h2, h_mat_testing2), "Correct h matrix init vec 101")
        self.assertTrue(np.allclose(result_pv2, projected_v_mat_testing2), "Correct projV matrix for init vec 101")

        self.assertTrue(np.allclose(result_h3, h_mat_testing3), "Correct h matrix init vec 102")
        self.assertTrue(np.allclose(result_pv3, projected_v_mat_testing3), "Correct projV matrix for init vec 102")

    def test_time_large_random(self):
        'compare the cusp implementation gpu vs cpu (if a gpu is detected) on a large example'

        # this test is manually enabled, since it can take a long time
        test_enabled = False

        if test_enabled:
            print "running cpu / gpu timing comparison on large random matrix"

            dims = 10 * 1000 * 1000
            iterations = 10

            print "making random matrix..."
            a_matrix = random_sparse_matrix(dims, entries_per_row=6, random_cols=False, print_progress=True)
            print "done"

            dir1 = np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float)
            dir2 = np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float)
            dir_list = [dir1, dir2]
            #dir1 = np.array([float(n) if n == 0 else 0.0 for n in xrange(dims)], dtype=float)
            #dir_list = [dir1]
            key_dir_mat = csr_matrix(dir_list)

            result_h_list = []
            result_pv_list = []

            for use_gpu in [False, True]:
                if use_gpu and not KrylovInterface.has_gpu():
                    break

                print "\n---------------\n"
                print "running with use_gpu = {}".format(use_gpu)

                KrylovInterface.set_use_gpu(use_gpu)
                KrylovInterface.set_use_profiling(True)

                KrylovInterface.preallocate_memory(iterations, dims, len(dir_list))

                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_unit(0)

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            if len(result_h_list) == 2:
                self.assertTrue(np.allclose(result_h_list[0], result_h_list[1]), "h-mat mismatch")
                self.assertTrue(np.allclose(result_pv_list[0], result_pv_list[1]), "mismatch projV")

    def test_large_spring(self):
        'compare the cusp implementation gpu vs cpu (if a gpu is detected) on a large spring example'

        # this test is manually enabled, since it can take a long time
        test_enabled = False

        if test_enabled:
            print "running cpu / gpu timing comparison on large random matrix"

            dims = 1 * 1000
            iterations = 10

            print "making spring matrix..."
            a_matrix = make_spring_mass_matrix(dims)
            print "done"

            dir1 = np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float)
            dir2 = np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float)
            dir_list = [dir1, dir2]
            #dir1 = np.array([1.0 if n == 0 else 0.0 for n in xrange(dims)], dtype=float)
            #dir_list = [dir1]
            key_dir_mat = csr_matrix(dir_list)

            result_h_list = []
            result_pv_list = []

            for use_gpu in [False, True]:
                if use_gpu and not KrylovInterface.has_gpu():
                    break

                print "\n---------------\n"
                print "running with use_gpu = {}".format(use_gpu)

                KrylovInterface.set_use_gpu(use_gpu)
                KrylovInterface.set_use_profiling(True)

                KrylovInterface.preallocate_memory(iterations, dims, len(dir_list))

                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_unit(0)

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            if len(result_h_list) == 2:
                self.assertTrue(np.allclose(result_h_list[0], result_h_list[1]), "h-mat mismatch")
                self.assertTrue(np.allclose(result_pv_list[0], result_pv_list[1]), "mismatch projV")

    def test_krylov_spring_accuracy(self):
        'test the if the krylov method is accurate enough'

        step = 0.01
        max_time = step * 100
        settings = HylaaSettings(step, max_time)

        cur_dim = 20

        a_matrix = make_spring_mass_matrix(10000)
        a_mat_csc = csc_matrix(a_matrix)
        dims = a_matrix.shape[0]

        dir_list = []
        dir_list.append(np.array([float(1.0) for _ in xrange(dims)], dtype=float))
        dir_list.append(np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float))
        dir_list.append(np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float))
        key_dirs = csr_matrix(dir_list)

        KrylovInterface.preallocate_memory(2, a_matrix.shape[0], key_dirs.shape[0], error_on_fail=True)
        KrylovInterface.load_a_matrix(a_matrix) # load a_matrix into device memory
        KrylovInterface.load_key_dir_matrix(key_dirs) # load key direction matrix into device memory

        b_vec = np.array([[1.0] if d == cur_dim else [0.0] for d in xrange(dims)])

        total_time = settings.step * settings.num_steps
        real_answer = expm_multiply(a_mat_csc * total_time, b_vec)
        real_proj = np.dot(np.array(key_dirs.todense()), real_answer)

        a_iter = 40
        # test reallocating with more arnoldi iterations
        KrylovInterface.preallocate_memory(a_iter, dims, key_dirs.shape[0], error_on_fail=True)

        cur_sim = get_projected_simulation(settings, cur_dim, use_mult=True)

        abs_error = np.linalg.norm(cur_sim - real_proj)
        rel_error = relative_error(real_proj, cur_sim)

        self.assertTrue(abs_error < 1e-6)
        self.assertTrue(rel_error < 1e-6)

    def test_iss_inputs(self):
        'test with iss example with forcing inputs'

        tol = 1e-9
        dims = 273
        iterations = 260
        compare_time = 20.0

        initial_vec_index = dims-3
        init_vec = np.array([[1.0] if d == initial_vec_index else [0.0] for d in xrange(dims)], dtype=float)

        dynamics = loadmat('iss.mat')
        raw_a_matrix = dynamics['A']

        # raw_a_matrix is a csc_matrix
        col_ptr = [n for n in raw_a_matrix.indptr]
        rows = [n for n in raw_a_matrix.indices]
        data = [n for n in raw_a_matrix.data]

        b_matrix = dynamics['B']

        for u in xrange(3):
            rows += [n for n in b_matrix[:, u].indices]
            data += [n for n in b_matrix[:, u].data]
            col_ptr.append(len(data))

        a_matrix_csc = csc_matrix((data, rows, col_ptr), shape=(raw_a_matrix.shape[0] + 3, raw_a_matrix.shape[1] + 3))
        a_matrix = csr_matrix(a_matrix_csc)
        self.assertEqual(dims, a_matrix.shape[0])

        ############
        y3 = dynamics['C'][2]
        col_ptr = [n for n in y3.indptr] + 3 * [y3.data.shape[0]]
        key_dir_mat = csc_matrix((y3.data, y3.indices, col_ptr), shape=(1, dims))
        key_dir_mat = csr_matrix(key_dir_mat)

        #key_dir_mat = csr_matrix(np.identity(dims, dtype=float))
        # key dir is identity (no projection needed)

        # real answer
        real_answer = expm_multiply(a_matrix_csc * compare_time, init_vec)
        real_answer.shape = (dims,)
        real_answer = key_dir_mat * real_answer

        # python comparison
        python_v, python_h = python_arnoldi(a_matrix, init_vec, iterations)
        h_mat = python_h[:-1, :].copy()
        v_mat = python_v[:, :-1].copy()
        exp = expm(h_mat * compare_time)[:, 0]
        python_answer = np.dot(v_mat, exp)
        python_answer = key_dir_mat * python_answer

        for d in xrange(real_answer.shape[0]):
            self.assertLess(abs(real_answer[d] - python_answer[d]), tol, \
                "Mismatch in dimension {}, {} (real) vs {} (python)".format(d, real_answer[d], python_answer[d]))

        # cusp comparison
        KrylovInterface.preallocate_memory(iterations, dims, key_dir_mat.shape[0])
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        result_h, result_pv = KrylovInterface.arnoldi_unit(initial_vec_index)
        h_mat = result_h[:-1, :].copy()
        pv_mat = result_pv[:, :-1].copy()

        exp = expm(h_mat * compare_time)[:, 0]
        cusp_answer = np.dot(pv_mat, exp)

        #print "real_answer = {}".format(real_answer)
        #print "cusp answer = {}".format(cusp_answer)

        for d in xrange(real_answer.shape[0]):
            self.assertLess(abs(real_answer[d] - cusp_answer[d]), tol, \
                "Mismatch in dimension {}, {} (real) vs {} (cusp)".format(d, real_answer[d], cusp_answer[d]))

if __name__ == '__main__':
    unittest.main()
