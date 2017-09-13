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

def python_arnoldi(mat, vec, iterations):
    'arnoldi for a single initial vector'

    dims = mat.shape[0]
    assert vec.shape == (dims, 1), "vec.shape was {}, expected (1, {})".format(vec.shape, dims)

    vec = vec.transpose()

    v_mat, h_mat = python_arnoldi_par(mat.T, vec, iterations)

    v_mat.shape = (iterations + 1, dims)
    h_mat.shape = (iterations, iterations + 1)

    return v_mat.transpose(), h_mat.transpose()


def python_arnoldi_par(mat_transpose, vecs, iterations):
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
        for init_index in xrange(num_init):
            #vec = np.dot(prev_v[cur_vec, (cur_it-1)*size:cur_it*size], a_transpose)

            vec = prev_v[init_index, (cur_it-1)*size:cur_it*size] * mat_transpose

            prev_v[init_index, cur_it*size:(cur_it+1)*size] = vec

        for init_index in xrange(num_init):
            cur_vec = prev_v[init_index, cur_it*size:(cur_it+1)*size]

            for c in xrange(cur_it):
                prev_vec = prev_v[init_index, c*size:(c+1)*size]

                dot_val = np.dot(prev_vec, cur_vec)
                h_mat[init_index][(iterations + 1) * (cur_it-1) + c] = dot_val

                sub_vec = prev_v[init_index, c*size:(c+1)*size]
                cur_vec -= sub_vec * dot_val

            norm = np.linalg.norm(cur_vec)
            h_mat[init_index][cur_it + (iterations+1) * (cur_it-1)] = norm

            if norm >= 1e-6:
                cur_vec = cur_vec / norm
                prev_v[init_index, cur_it*size:(cur_it+1)*size] = cur_vec

    return prev_v, h_mat

def relative_error(correct, estimate):
    'compute the relative error between the correct value and an estimate'

    rel_error = 0
    norm = np.linalg.norm(correct)

    if norm > 1e-9:
        diff = correct - estimate
        err = np.linalg.norm(diff)
        rel_error = err / norm

    return rel_error

class TestKrylovInterface(unittest.TestCase):
    'Unit tests for krylov utilities'

    def setUp(self):
        'test setup'

        random.seed(1)
        KrylovInterface.reset()
        KrylovInterface.set_use_gpu(False)

    def test_arnoldi_single(self):
        'compare the python implementation with the cusp implementation with a single initial vector'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)

        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec = np.array([[1.0] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        v_mat_testing, h_mat_testing = python_arnoldi(a_matrix, init_vec, iterations)

        projected_v_mat_testing = key_dir_mat * v_mat_testing

        # using cusp

        KrylovInterface.preallocate_memory(iterations, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)

        result_h, result_pv = KrylovInterface.arnoldi_unit(0)

        self.assertTrue(np.allclose(result_h, h_mat_testing), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv, projected_v_mat_testing), "Correct projected v matrix")

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
        test_enabled = True

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
        test_enabled = True

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

        dims = 273
        iterations = 90
        initial_vec_index = dims-1
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

        key_dir_mat = csr_matrix(np.identity(dims, dtype=float))
        # key dir is identity (no projection needed)

        # real answer
        real_answer = expm_multiply(a_matrix_csc, init_vec)
        real_answer.shape = (dims,)

        # python comparison
        python_v, python_h = python_arnoldi(a_matrix, init_vec, iterations)
        h_mat = python_h[:-1, :].copy()
        v_mat = python_v[:, :-1].copy()
        exp = expm(h_mat)[:, 0]
        python_answer = np.dot(v_mat, exp)

        for d in xrange(dims):
            self.assertLess(abs(real_answer[d] - python_answer[d]), 1e-4, \
                "Mismatch in dimension {}, {} (real) vs {} (python)".format(d, real_answer[d], python_answer[d]))

        # cusp comparison
        KrylovInterface.preallocate_memory(iterations, dims, key_dir_mat.shape[0])
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        result_h, result_v = KrylovInterface.arnoldi_unit(initial_vec_index)
        h_mat = result_h[:-1, :].copy()
        v_mat = result_v[:, :-1].copy()

        exp = expm(h_mat)[:, 0]
        cusp_answer = np.dot(v_mat, exp)

        for d in xrange(dims):
            self.assertLess(abs(real_answer[d] - cusp_answer[d]), 1e-4, \
                "Mismatch in dimension {}, {} (real) vs {} (cusp)".format(d, real_answer[d], cusp_answer[d]))

if __name__ == '__main__':
    unittest.main()
