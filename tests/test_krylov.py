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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import odeint

from hylaa.settings import HylaaSettings
from hylaa.krylov_python import KrylovIterator

from krypy.utils import arnoldi as krypy_arnoldi # krypy is used for testing

def make_settings(lanczos=False):
    'make a hylaa settings object'

    h = HylaaSettings(0.1, 1.0)
    h.simulation.krylov_lanczos = lanczos

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
            ksim = KrylovIterator(make_settings(), a_matrix, key_dir_mat)
            python_pv, python_h = ksim.run_iteration(init_sparse, iterations)

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
            ksim = KrylovIterator(make_settings(True), a_matrix, key_dir_mat)
            python_pv, python_h = ksim.run_iteration(init_sparse, iterations)
            python_h = python_h.toarray()

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
        ksim = KrylovIterator(make_settings(True), a_matrix_sparse, key_dir_mat)
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
        ksim = KrylovIterator(make_settings(True), a_matrix_sparse, key_dir_mat)
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

        ksim = KrylovIterator(make_settings(), a_matrix, key_dir_mat, add_one_norm=True)
        python_pv_auto, python_h_auto = ksim.run_iteration(init_sparse, iterations)

        # using python
        ksim = KrylovIterator(make_settings(), a_matrix, key_dir_mat_with_one_norm, add_one_norm=False)
        python_pv, python_h = ksim.run_iteration(init_sparse, iterations)

        self.assertTrue(np.allclose(python_h, python_h_auto), "H matrices don't match")
        self.assertTrue(np.allclose(python_pv, python_pv_auto), "PV matrices don't match")

    def test_lanczos_one_norm(self):
        'test lanczos computaiton with one norm setting'

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

        #init_dense = np.array([[2.] if d == 1 else [.4] if d == dims-1 else [0.0] for d in xrange(dims)], dtype=float)
        init_dense = np.array([[1.] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)

        init_sparse = csr_matrix(init_dense.T)

        ksim = KrylovIterator(make_settings(True), a_matrix, key_dir_mat, add_one_norm=True)
        python_pv_auto, python_h_auto = ksim.run_iteration(init_sparse, iterations)

        # using python
        ksim = KrylovIterator(make_settings(), a_matrix, key_dir_mat_with_one_norm, add_one_norm=False)
        python_pv, python_h = ksim.run_iteration(init_sparse, iterations)

        self.assertTrue(np.allclose(python_h, python_h_auto.toarray()), "H matrices don't match")
        self.assertTrue(np.allclose(python_pv, python_pv_auto), "PV matrices don't match")

if __name__ == '__main__':
    unittest.main()
