'''
A set of performance measurements for various linear algebra operations. These tests print output.
You can run them one-by-one using: python test_performance.py TestPerformance.test_norm
'''

import time
import unittest
import multiprocessing
from multiprocessing.pool import ThreadPool

import numpy as np
from scipy.sparse import csr_matrix

from test_krylov import random_five_diag_sym_matrix

# used for the multi-threadded cases
global_thread_pool = ThreadPool(multiprocessing.cpu_count())

def pmult(a_mat, vec, force_parallel=False):
    'parallel dot product'

    global global_thread_pool
    assert isinstance(a_mat, csr_matrix)
    dims = a_mat.shape[0]

    # will change back after
    preshape = vec.shape
    vec.shape = (dims, 1)

    if a_mat.shape[0] < 150 and not force_parallel:
        # single-threaded is fast if dims < 150
        rv = a_mat * vec
    else:
        # multi-threaded version

        def mult_func((sub_mat, vec)):
            'multiplication of matrices, for parallel map'

            return sub_mat * vec

        split = multiprocessing.cpu_count()
        args = []

        for i in xrange(split):
            start_row = i * dims / split
            end_row = (i+1) * dims / split

            if start_row == end_row:
                continue

            # I think this way would make a copy of the data, which we want to avoid
            #sub_a_mat = a_mat[start_row:end_row, :]

            # construct a sub csr_matrix (no copy... hopefully)
            start = a_mat.indptr[start_row]
            end = a_mat.indptr[end_row]

            # I think sub_ind_ptr we have to copy
            sub_ind_ptr = a_mat.indptr[start_row:end_row+1].copy()

            for j in xrange(sub_ind_ptr.shape[0]):
                sub_ind_ptr[j] -= start
            
            sub_a_mat_tuple = (a_mat.data[start:end], a_mat.indices[start:end], sub_ind_ptr)
            sub_a_mat = csr_matrix(sub_a_mat_tuple, dtype=a_mat.dtype, shape=(end_row-start_row, a_mat.shape[1]))

            print "share mem? {}".format(np.may_share_memory(sub_a_mat.data, a_mat.data))

            args.append((sub_a_mat, vec))

        result = global_thread_pool.map(mult_func, args)

        rv = np.concatenate(result)

    rv.shape = (dims,)
    vec.shape = preshape

    return rv

def paxpy(a, alpha, b, force_parallel=False):
    'parallel version of: a += alpha * b'

    global global_thread_pool
    assert len(a.shape) == 1
    assert len(b.shape) == 1
    assert a.shape == b.shape

    if a.shape[0] < 150 and not force_parallel:
        # single-threaded is fast if dims < 150
        a += alpha * b
    else:
        # multi-threaded version

        def axpy_func((a, alpha, b)):
            'axpy function, for parallel map'

            a += alpha * b

        size = a.shape[0]
        split = multiprocessing.cpu_count()
        args = []

        for i in xrange(split):
            start_index = i * size / split
            end_index = (i+1) * size / split

            if start_index == end_index:
                continue

            args.append((a[start_index:end_index], alpha, b[start_index:end_index]))

        global_thread_pool.map(axpy_func, args)

def pdot(a, b, force_parallel=False):
    'parallel dot product'

    global global_thread_pool
    assert len(a.shape) == 1
    assert len(b.shape) == 1
    assert a.shape == b.shape

    if a.shape[0] < 150 and not force_parallel:
        # single-threaded is fast if dims < 150
        rv = np.dot(a, b)
    else:
        # multi-threaded version

        def mult_func((num1, num2)):
            'multiplication of matrices, for parallel map'

            return np.dot(num1, num2)

        size = a.shape[0]
        split = multiprocessing.cpu_count()
        args = []

        for i in xrange(split):
            start_index = i * size / split
            end_index = (i+1) * size / split

            if start_index == end_index:
                continue

            args.append((a[start_index:end_index], b[start_index:end_index]))

        result = global_thread_pool.map(mult_func, args)

        rv = sum(result)

    return rv

class TestPerformance(unittest.TestCase):
    'Performance unit tests'

    def test_norm(self):
        '''compare numpy's norm and norm(2)'''

        dims = int(1e8)
        vec = np.random.random_sample((1, dims))

        start = time.time()
        norm = np.linalg.norm(vec)
        print "norm(vec) time: {:.2}s".format(time.time() - start)

        start = time.time()
        norm2 = np.linalg.norm(vec, 2)
        print "norm(vec, 2) time: {:.2}s".format(time.time() - start)
        
        self.assertTrue(np.allclose(norm, norm2))

    def test_pmult(self):
        'test parallel csr matrix mult'

        dims = int(1e6) # 1e7, par = 0.4, serial = 0.1

        a_matrix = random_five_diag_sym_matrix(dims, True)

        start = time.time()
        vec = np.random.random_sample((dims,))
        print "allocated vector in {}s".format(time.time() - start)

        start = time.time()
        res_parallel = pmult(a_matrix, vec, force_parallel=True)
        print "parallel time = {}s".format(time.time() - start)

        start = time.time()
        vec.shape = (dims, 1)
        res_serial = a_matrix * vec
        res_serial.shape = (dims,)
        print "serial time = {}s".format(time.time() - start)

        self.assertTrue(np.allclose(res_parallel, res_serial))

    def test_paxpy(self):
        'test parallel axpy function'

        dims = int(1e8)

        start = time.time()
        a = np.random.random_sample((dims,))
        print "allocated first vector in {}s".format(time.time() - start)

        start = time.time()
        par_a = a.copy()
        print "copied first vector in {}s".format(time.time() - start)

        start = time.time()
        b = np.random.random_sample((dims,))
        print "allocated second vector in {}s".format(time.time() - start)

        start = time.time()
        paxpy(par_a, -0.5, b, force_parallel=True)
        print "parallel time = {}s".format(time.time() - start)

        start = time.time()
        a += -0.5 * b
        print "serial time = {}s".format(time.time() - start)

        self.assertTrue(np.allclose(par_a, a))

    def test_pdot(self):
        'test parallel dot function'

        dims = int(1e8)

        start = time.time()
        
        #a = np.zeros((dims,), dtype=float)
        #for i in xrange(dims):
        #    a[i] = random.random()
        a = np.random.random_sample((dims,))
        print "allocated first vector in {}s".format(time.time() - start)

        start = time.time()
        b = np.random.random_sample((dims,))
        print "allocated second vector in {}s".format(time.time() - start)

        start = time.time()
        res_parallel = pdot(a, b, force_parallel=True)
        print "parallel time = {}s".format(time.time() - start)

        start = time.time()
        res_serial = np.dot(a, b)
        print "serial time = {}s".format(time.time() - start)

        self.assertTrue(np.allclose(res_parallel, res_serial))
