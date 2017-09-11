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
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm, expm_multiply

from hylaa.krylov_interface import KrylovInterface
from hylaa.containers import HylaaSettings

#from my_krypy_utils import arnoldi as krypy_arnoldi

def krypy_arnoldi(A, v, maxiter):
    'krypy arnoldi'

    numpy = np

    def inner(X, Y, ip_B=None):
        '''inner product
        '''

        return numpy.dot(X.T, Y)

    def norm(x, y=None, ip_B=None):
        '''Compute norm (Euclidean and non-Euclidean).
        :param x: a 2-dimensional ``numpy.array``.
        :param y: a 2-dimensional ``numpy.array``.
        :param ip_B: see :py:meth:`inner`.
        Compute :math:`\sqrt{\langle x,y\rangle}` where the inner product is
        defined via ``ip_B``.
        '''

        return numpy.linalg.norm(x)


    V = numpy.zeros((v.shape[0], maxiter+1), dtype=float)
    H = numpy.zeros((maxiter+1, maxiter), dtype=float)

    V[:, [0]] = v / numpy.linalg.norm(v)

    for k in xrange(maxiter):
        N = V.shape[0]

        # the matrix-vector multiplication
        Av = A * V[:, [k]]

        # determine vectors for orthogonalization
        start = 0

        # orthogonalize
        for j in range(start, k+1):
            alpha = inner(V[:, [j]], Av)[0, 0]
            H[j, k] += alpha
            Av -= alpha * V[:, [j]]

        norm_val = norm(Av)
        H[k+1, k] = norm_val

        if norm_val > 1e-9:
            V[:, [k+1]] = Av / norm_val


    return V, H


def get_projected_simulation(settings, dim, use_mult=False):
    '''
    Get the projected simulation using the current settings.
    '''

    h_list, pv_list = KrylovInterface.arnoldi_parallel(dim)

    h_mat = h_list[0]
    pv_mat = pv_list[0]

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

def random_sparse_matrix(dims, entries_per_row, random_cols=True, print_progress=False):
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

    return rv

def arnoldi(mat, vec, iterations):
    'arnoldi for a single initial vector'

    dims = mat.shape[0]
    assert vec.shape == (dims, 1), "vec.shape was {}, expected (1, {})".format(vec.shape, dims)

    vec = vec.transpose()

    v_mat, h_mat = arnoldi_parallel(mat.T, vec, iterations)

    v_mat.shape = (iterations + 1, dims)
    h_mat.shape = (iterations, iterations + 1)

    return v_mat.transpose(), h_mat.transpose()


def arnoldi_parallel(mat_transpose, vecs, iterations):
    'arnoldi with split multiple initial vecs'

    num_init = vecs.shape[0]
    size = vecs.shape[1]

    prev_v = np.zeros((num_init, (iterations + 1) * size))
    h_mat = np.zeros((num_init, (iterations + 1) * (iterations)))

    for c in xrange(num_init):
        vec = vecs[c, :]
        vec = vec / np.linalg.norm(vec)
        prev_v[c, 0:size] = vec

    # use a-transpose
    #a_transpose = mat.T.copy()

    for cur_it in xrange(1, iterations + 1):

        # do all the multiplications up front
        for cur_vec in xrange(num_init):
            #vec = np.dot(prev_v[cur_vec, (cur_it-1)*size:cur_it*size], a_transpose)

            vec = prev_v[cur_vec, (cur_it-1)*size:cur_it*size] * mat_transpose

            prev_v[cur_vec, cur_it*size:(cur_it+1)*size] = vec

        for cur_vec in xrange(num_init):
            vec = prev_v[cur_vec, cur_it*size:(cur_it+1)*size]

            prev_mat = prev_v[cur_vec, 0:cur_it*size]
            prev_mat.shape = (cur_it, size)

            dots = h_mat[cur_vec][(iterations + 1) * (cur_it-1):(iterations + 1) * (cur_it-1) + cur_it]
            dots[:] = np.dot(prev_mat, vec.T)

            sub_vecs = np.dot(np.diag(dots), prev_mat)

            for c in xrange(cur_it):
                vec -= sub_vecs[c]

            norm = np.linalg.norm(vec)
            h_mat[cur_vec][cur_it + (iterations+1) * (cur_it-1)] = norm

            if norm >= 1e-6:
                vec = vec / norm
                prev_v[cur_vec, cur_it*size:(cur_it+1)*size] = vec

    return prev_v, h_mat

class TestKrylovInterface(unittest.TestCase):
    'Unit tests for krylov utilities'

    def setUp(self):
        'test setup'

        random.seed(1)
        KrylovInterface.reset()

    def test_arnoldi_single(self):
        'compare the python implementation with the cusp implementation with a single initial vector'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 1

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)

        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec = np.array([[1.0] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        v_mat_testing, h_mat_testing = arnoldi(a_matrix, init_vec, iterations)

        projected_v_mat_testing = key_dir_mat * v_mat_testing

        # using cusp

        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing), "Correct projected v matrix")

    def test_arnoldi_offset(self):
        'compare the python implementation with the cusp implementation with a single initial vector (2nd column)'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 1

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)

        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec = np.array([[1.0] if d == 1 else [0.0] for d in xrange(dims)], dtype=float)
        v_mat_testing, h_mat_testing = arnoldi(a_matrix, init_vec, iterations)

        projected_v_mat_testing = key_dir_mat * v_mat_testing

        # using cusp

        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_parallel(1)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing), "Correct projected v matrix")

    def test_arnoldi_off_end(self):
        'test arnodli when doing 2 parallel vectors at a time when straddling the end (only 1 should be done)'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec4 = np.array([[1.0] if d == 4 else [0.0] for d in xrange(dims)], dtype=float)

        v_mat_testing4, h_mat_testing4 = arnoldi(a_matrix, init_vec4, iterations)
        projected_v_mat_testing4 = key_dir_mat * v_mat_testing4

        # using cusp
        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_parallel(4)

        self.assertEqual(len(result_h), len(result_pv))
        self.assertEqual(len(result_h), 1)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing4), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing4), "Correct projV matrix")

    def test_arnoldi_double(self):
        'compare the cusp implementation with a two initial vectors versus the python implementation'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec1 = np.array([[1.0] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        init_vec2 = np.array([[1.0] if d == 1 else [0.0] for d in xrange(dims)], dtype=float)

        v_mat_testing1, h_mat_testing1 = arnoldi(a_matrix, init_vec1, iterations)
        projected_v_mat_testing1 = key_dir_mat * v_mat_testing1

        v_mat_testing2, h_mat_testing2 = arnoldi(a_matrix, init_vec2, iterations)
        projected_v_mat_testing2 = key_dir_mat * v_mat_testing2

        # using cusp
        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing1), "Correct h matrix init vec 1")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing1), "Correct projV matrix for init vec 1")

        self.assertTrue(np.allclose(result_h[1], h_mat_testing2), "Correct h matrix init vec 2")
        self.assertTrue(np.allclose(result_pv[1], projected_v_mat_testing2), "Correct projV matrix for init vec 2")

    def test_arnoldi_double_gpu(self):
        'compare the gpu cusp implementation with a two initial vectors versus the python implementation'

        if KrylovInterface.has_gpu():
            #KrylovInterface.set_use_profiling(True)
            KrylovInterface.set_use_gpu(False)

            dims = 5
            iterations = 2
            key_dirs = 2
            num_parallel = 2

            a_matrix = random_sparse_matrix(dims, entries_per_row=2)
            key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

            # using python
            init_vec1 = np.array([[1.0] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
            init_vec2 = np.array([[1.0] if d == 1 else [0.0] for d in xrange(dims)], dtype=float)

            v_mat_testing1, h_mat_testing1 = arnoldi(a_matrix, init_vec1, iterations)
            projected_v_mat_testing1 = key_dir_mat * v_mat_testing1

            v_mat_testing2, h_mat_testing2 = arnoldi(a_matrix, init_vec2, iterations)
            projected_v_mat_testing2 = key_dir_mat * v_mat_testing2

            # using cusp
            KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
            KrylovInterface.load_a_matrix(a_matrix)
            KrylovInterface.load_key_dir_matrix(key_dir_mat)

            result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

            self.assertTrue(np.allclose(result_h[0], h_mat_testing1), "Correct h matrix init vec 1")
            self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing1), "Correct projV matrix for init vec 1")

            self.assertTrue(np.allclose(result_h[1], h_mat_testing2), "Correct h matrix init vec 2")
            self.assertTrue(np.allclose(result_pv[1], projected_v_mat_testing2), "Correct projV matrix for init vec 2")

    def test_iss(self):
        'test the cusp implementation using the iss model'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        iterations = 10
        num_parallel = 3

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

        v_mat_testing1, h_mat_testing1 = arnoldi(a_matrix, init_vec1, iterations)
        projected_v_mat_testing1 = key_dir_mat * v_mat_testing1

        v_mat_testing2, h_mat_testing2 = arnoldi(a_matrix, init_vec2, iterations)
        projected_v_mat_testing2 = key_dir_mat * v_mat_testing2

        v_mat_testing3, h_mat_testing3 = arnoldi(a_matrix, init_vec3, iterations)
        projected_v_mat_testing3 = key_dir_mat * v_mat_testing3

        # using cusp
        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dir_mat.shape[0])
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_parallel(100)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing1), "Correct h matrix init vec 100")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing1), "Correct projV matrix for init vec 100")

        self.assertTrue(np.allclose(result_h[1], h_mat_testing2), "Correct h matrix init vec 101")
        self.assertTrue(np.allclose(result_pv[1], projected_v_mat_testing2), "Correct projV matrix for init vec 101")

        self.assertTrue(np.allclose(result_h[2], h_mat_testing3), "Correct h matrix init vec 102")
        self.assertTrue(np.allclose(result_pv[2], projected_v_mat_testing3), "Correct projV matrix for init vec 102")

    def test_compare_gpu_cpu(self):
        'compare the cusp implementation gpu vs cpu (if a gpu is detected)'

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        if KrylovInterface.has_gpu():
            result_h_list = []
            result_pv_list = []

            for use_gpu in [False, True]:
                KrylovInterface.set_use_gpu(use_gpu)
                #print "\n---------------\n"
                #KrylovInterface.set_use_profiling(True)

                KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_parallel(2) # offset by 2 just because

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            for i in [0, 1]:
                self.assertTrue(np.allclose(result_h_list[0][i], result_h_list[1][i]), "bad h-matrix i={}".format(i))
                self.assertTrue(np.allclose(result_pv_list[0][i], result_pv_list[1][i]), "bad projV i={}".format(i))

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

                num_parallel = 2 if use_gpu else 1
                num_parallel_preallocate = num_parallel

                KrylovInterface.preallocate_memory(iterations, num_parallel_preallocate, dims, len(dir_list))


                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            if len(result_h_list) == 2:
                #diff = result_h_list[0][0] - result_h_list[1][0]
                #print "norm difference = {}".format(np.linalg.norm(diff))
                self.assertTrue(np.allclose(result_h_list[0][0], result_h_list[1][0]), "h-mat mismatch")
                self.assertTrue(np.allclose(result_pv_list[0][0], result_pv_list[1][0]), "mismatch projV")

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

            #dir1 = np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float)
            #dir2 = np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float)
            #dir_list = [dir1, dir2]
            dir1 = np.array([1.0 if n == 0 else 0.0 for n in xrange(dims)], dtype=float)
            dir_list = [dir1]
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

                num_parallel = 1 if use_gpu else 1
                num_parallel_preallocate = num_parallel

                KrylovInterface.preallocate_memory(iterations, num_parallel_preallocate, dims, len(dir_list))


                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            if len(result_h_list) == 2:
                #diff = result_h_list[0][0] - result_h_list[1][0]
                #print "norm difference = {}".format(np.linalg.norm(diff))
                self.assertTrue(np.allclose(result_h_list[0][0], result_h_list[1][0]), "h-mat mismatch")
                self.assertTrue(np.allclose(result_pv_list[0][0], result_pv_list[1][0]), "mismatch projV")

    def test_krylov_spring_accuracy(self):
        'test the if the krylov method is accurate enough'

        def relative_error(correct, estimate):
            'compute the relative error between the correct value and an estimate'

            rel_error = 0
            norm = np.linalg.norm(correct)

            if norm > 1e-9:
                diff = correct - estimate
                err = np.linalg.norm(diff)
                rel_error = err / norm

            return rel_error

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

        KrylovInterface.preallocate_memory(2, 1, a_matrix.shape[0], key_dirs.shape[0], error_on_fail=True)
        KrylovInterface.load_a_matrix(a_matrix) # load a_matrix into device memory
        KrylovInterface.load_key_dir_matrix(key_dirs) # load key direction matrix into device memory

        b_vec = np.array([[1.0] if d == cur_dim else [0.0] for d in xrange(dims)])

        total_time = settings.step * settings.num_steps
        real_answer = expm_multiply(a_mat_csc * total_time, b_vec)
        real_proj = np.dot(np.array(key_dirs.todense()), real_answer)

        a_iter = 40
        KrylovInterface.preallocate_memory(a_iter, 1, dims, key_dirs.shape[0], error_on_fail=True)

        cur_sim = get_projected_simulation(settings, cur_dim, use_mult=True)

        abs_error = np.linalg.norm(cur_sim - real_proj)
        rel_error = relative_error(real_proj, cur_sim)

        self.assertTrue(abs_error < 1e-6)
        self.assertTrue(rel_error < 1e-6)

    def test_iss_inputs(self):
        'test with iss example with forcing inputs'

        compare_time = 1.0

        dims = 273
        iterations = 96 ####### 96 works okay!
        initial_vec_index = dims-1
        init_vec = np.array([[1.0] if d == initial_vec_index else [0.0] for d in xrange(dims)], dtype=float)

        dynamics = loadmat('iss.mat')
        raw_a_matrix = dynamics['A']

        raw_a_matrix = compare_time * raw_a_matrix

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

        print "a_mat norm = {}".format(np.linalg.norm(a_matrix.todense()))

        ############

        #key_dir_mat = csr_matrix(np.identity(dims, dtype=float))
        #dir1 = np.array([1.0 if d == 0 else 0.0 for d in xrange(dims)], dtype=float)
        #key_dir_mat = csr_matrix([dir1])
        key_dir_mat = csr_matrix(np.identity(dims))

        v_mat_testing, h_mat_testing = arnoldi(a_matrix, init_vec, iterations)
        projected_v_mat_testing = key_dir_mat * v_mat_testing

        # using krypy
        k_v, k_h = krypy_arnoldi(a_matrix, init_vec, maxiter=iterations)

        # using cusp
        KrylovInterface.preallocate_memory(iterations, 1, dims, key_dir_mat.shape[0])
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_parallel(initial_vec_index)

        #self.compare_matrices("h-matrix", "krypy", k_h, "cusp", result_h[0])
        #print result_h - h_mat_testing

        # check vs real answer with total_time = 1.0
        real_answer = expm_multiply(a_matrix_csc * (1.0 / compare_time), init_vec)
        real_proj = np.dot(np.array(key_dir_mat.todense()), real_answer)
        real_proj.shape = (dims,)


        #h_mat = result_h[0][:-1, :].copy()
        #pv_mat = result_pv[0][:, :-1].copy()

        # here, substituted for krypy result

        if k_h.shape[0] == k_h.shape[1]:
            h_mat = k_h
            pv_mat = k_v
        else:
            h_mat = k_h[:-1, :].copy()
            pv_mat = k_v[:, :-1].copy()

        h_mat = h_mat * (1.0 / compare_time)
        #h_mat = h_mat * (1.0 / compare_time)
        print "h_mat shape = {}".format(h_mat.shape)

        exp = expm(h_mat)[:, 0]
        print "h_mat exp = {}".format(exp)

        krylov_proj = np.dot(pv_mat, exp)

        for d in xrange(dims):
            self.assertLess(abs(real_proj[d] - krylov_proj[d]), 1e-4, \
                "Mismatch in dimension {}, {} (real) vs {} (krylov)".format(d, real_proj[d], krylov_proj[d]))

    def compare_matrices(self, compare_name, mat1_name, mat1, mat2_name, mat2, tol=1e-6):
        'compare two matrices with labels, printing errors if found'

        max_dif = 0
        max_x = -1
        max_y = -1

        self.assertEqual(mat1.shape[0], mat2.shape[0], "matrix number of rows mismatch {} ({}) vs {} ({})".format(
            mat1.shape[0], mat1_name, mat2.shape[0], mat2_name))

        self.assertEqual(mat1.shape[1], mat2.shape[1], "matrix number of rows mismatch {} ({}) vs {} ({})".format(
            mat1.shape[1], mat1_name, mat2.shape[1], mat2_name))

        for y in xrange(mat1.shape[0]):
            for x in xrange(mat1.shape[1]):
                dif = abs(mat1[y, x] - mat2[y, x])

                if dif > max_dif:
                    max_dif = dif
                    max_x = x
                    max_y = y

        self.assertLess(max_dif, tol, "{} mismatch in index ({}, {}), dif = {}, {} ({}) vs {} ({})".format(\
            compare_name, max_y, max_x, max_dif, mat1[max_y, max_x], mat1_name, mat2[max_y, max_x], mat2_name))

if __name__ == '__main__':
    unittest.main()
