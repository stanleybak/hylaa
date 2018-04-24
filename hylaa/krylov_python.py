'''
Dung Tran & Stanley Bak
August 2017

Simulating a linear system x' = Ax using krylov supspace methods (arnoldi and lanczos)
'''

import math
import time
import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.sparse import csr_matrix, csc_matrix, dia_matrix
from scipy.sparse.linalg import norm as sparse_norm

from hylaa.timerutil import Timers
from hylaa.util import Freezable, get_script_path


def normalize_sparse(vec):
    'normalize a sparse vector (passed in as a 1xn csr_matrix), and return a tuple: scaled_vec, original_norm'

    assert isinstance(vec, csr_matrix) and vec.shape[0] == 1

    norm = sparse_norm(vec)

    assert not math.isinf(norm) and not math.isnan(norm) and norm > 1e-9, \
        "bad initial vec norm in normalize_sparse: {}".format(norm)

    # divide in place
    rv = vec / norm

    return rv, norm

def ones_dot(vec):
    'dot the vector with a row of ones'

    total = 0

    for i in xrange(0, len(vec)):
        total += vec[i]

    return total

def is_symmetric(mat):
    'is the passed-in square matrix symmetric?'

    Timers.tic('is_symmetric')

    start = time.time()
    rv = (mat != mat.T).nnz == 0
    print "is_symmetric time = {}".format(time.time() - start)

    Timers.toc('is_symmetric')

    return rv

class KrylovIteration(Freezable):
    'Krylov Iteration container class (Arnoldi or Lanczos iteration)'

    def __init__(self, hylaa_settings, a_matrix, use_transpose, key_dir_mat):
        assert a_matrix.shape[0] == a_matrix.shape[1], "a_mat should be square"
        assert key_dir_mat.shape[1] == a_matrix.shape[0], "key_dir_mat width should equal number of dims"
        assert not isinstance(a_matrix, np.ndarray), "a_matrix should be a sparse matrix"
        assert isinstance(key_dir_mat, csr_matrix), "key_dir_mat should be a csr_matrix"

        self.kry_settings = hylaa_settings.time_elapse.krylov
        self.lanczos = is_symmetric(a_matrix)
        self.print_status = self.kry_settings.stdout and a_matrix.shape[0] >= int(1e6)

        if use_transpose and not self.lanczos:
            # we need to compute with the transpose of the a matrix
            self.a_matrix = csr_matrix(a_matrix.transpose())
        else:
            self.a_matrix = a_matrix

        self.key_dir_mat = key_dir_mat
        self.tol = 1e-9 # tolerance for termination checking during arnoldi / lanczos iteration

        # from reset
        self.init_norm = None
        self.pv_mat = None
        self.h_data = None
        self.h_inds = None
        self.h_indptrs = None
        self.cur_vec = None
        self.prev_vec = None
        self.prev_prev_vec = None
        self.prev_norm = None
        self.v_mat = None
        self.h_mat = None

        self.elapsed = 0
        self.reinit = True
        self.cur_it = None

        self._init_fast_mult()

        self.freeze_attrs()

    def _init_fast_mult(self):
        'intialize fast parallel multiplication in C++ for diag_matrix'

        lib_path = os.path.join(get_script_path(__file__), 'fast_mult', 'fast_mult.so')
        lib = ctypes.CDLL(lib_path)

        self.dia_fast_mult = lib.diaFastMult
        self.dia_fast_mult.restype = None

        #double* result, double* vec, int matW, int matH, double* data, int* offsets, int numOffsets, int numSplit
        self.dia_fast_mult.argtypes = \
            [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
             ctypes.c_int, ctypes.c_int,\
             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

    def mult(self, mat, vec):
        'fast matrix vector multiplication'

        dims = mat.shape[0]

        if isinstance(mat, dia_matrix):
            rv = np.empty((dims,), dtype=float)
            cpus = multiprocessing.cpu_count()

            self.dia_fast_mult(rv, vec, dims, dims, mat.data, mat.offsets, len(mat.offsets), cpus)
        else:
            rv = mat * vec

        return rv

    def reset(self):
        'free memory from earlier runs'

        self.init_norm = None

        self.pv_mat = None
        self.h_data = None
        self.h_inds = None
        self.h_indptrs = None

        self.cur_vec = None
        self.prev_vec = None
        self.prev_prev_vec = None
        self.prev_norm = None
        self.v_mat = None
        self.h_mat = None

        self.elapsed = 0
        self.reinit = True
        self.cur_it = None

    def _realloc(self, init_vec, iterations):
        'allocate (or re-allocate) h, v, and pv storage'

        dims = self.a_matrix.shape[0]
        key_dirs = self.key_dir_mat.shape[0]

        #if self.add_ones_key_dir:
        #    key_dirs += 1

        if self.reinit:
            scaled_vec, self.init_norm = normalize_sparse(init_vec)

            self.cur_it = 1

            if self.lanczos:

                self.pv_mat = np.zeros((iterations + 1, key_dirs), dtype=float)
                self.h_data = []
                self.h_inds = []
                self.h_indptrs = [0]

                self.cur_vec = scaled_vec.toarray()
                self.prev_vec = None
                self.prev_prev_vec = None
                self.prev_norm = None
                self.prev_norm = None

                self.cur_vec.shape = (self.cur_vec.shape[1],)

                # sparse assignment of initial vector
                #if self.add_ones_key_dir:
                #    self.pv_mat[0, :-1] = (self.key_dir_mat * scaled_vec.T).toarray()[:, 0]
                #    self.pv_mat[0, -1] = ones_dot(self.cur_vec)
                #else:
                #    self.pv_mat[0, :] = (self.key_dir_mat * scaled_vec.T).toarray()[:, 0]
                self.pv_mat[0, :] = (self.key_dir_mat * scaled_vec.T).toarray()[:, 0]
            else:
                # arnoldi

                self.v_mat = np.zeros((iterations + 1, dims), dtype=float)
                self.h_mat = np.zeros((iterations + 1, iterations), dtype=float)

                # sparse assignment of initial vector
                for i in xrange(len(scaled_vec.data)):
                    self.v_mat[0, scaled_vec.indices[i]] = scaled_vec.data[i]

            self.reinit = False # next time don't reinitialize
        else:
            # continue the computation (allocate more memory)

            if self.lanczos:
                new_pv_mat = np.zeros((iterations + 1, key_dirs), dtype=float)
                new_pv_mat[:self.pv_mat.shape[0], :self.pv_mat.shape[1]] = self.pv_mat
                self.pv_mat = new_pv_mat
            else:
                # arnoldi
                new_v_mat = np.zeros((iterations + 1, dims), dtype=float)
                new_h_mat = np.zeros((iterations + 1, iterations), dtype=float)

                # copy from old
                new_v_mat[:self.v_mat.shape[0], :self.v_mat.shape[1]] = self.v_mat
                new_h_mat[:self.h_mat.shape[0], :self.h_mat.shape[1]] = self.h_mat

                # replace
                self.v_mat = new_v_mat
                self.h_mat = new_h_mat

    def run_iteration(self, init_vec, iterations):
        '''run arnoldi or lanczos'''

        assert isinstance(init_vec, csr_matrix), "init_vec should be csr_matrix"
        assert init_vec.shape[0] == 1

        self._realloc(init_vec, iterations)
        rv = None

        if self.lanczos:
            rv = self._lanczos(iterations)
        else:
            rv = self._arnoldi(iterations)

        return rv

    def _arnoldi(self, iterations):
        '''run the arnoldi algorithm

        this returns pv_mat, h_mat
        '''

        Timers.tic('arnoldi')

        start = time.time()

        while self.cur_it < iterations + 1:
            if self.print_status:
                elapsed = time.time() - start + self.elapsed

                # we expect quadratic scalability for arnoldi
                frac = self.cur_it * self.cur_it / float(iterations * iterations)
                eta = elapsed / frac - elapsed

                print "arnoldi iteration {} / {}, Elapsed: {:.2f}m, ETA: {:.2f}m".format(self.cur_it-1, iterations, \
                    elapsed / 60.0, eta / 60.0)

            Timers.tic('arnoldi mult')
            cur_vec = self.mult(self.a_matrix, self.v_mat[self.cur_it - 1])
            Timers.toc('arnoldi mult')

            for c in xrange(self.cur_it):
                prev_vec = self.v_mat[c]

                Timers.tic('arnoldi dot')
                dot_val = np.dot(prev_vec, cur_vec)
                Timers.toc('arnoldi dot')

                self.h_mat[c, self.cur_it - 1] = dot_val

                Timers.tic('arnoldi axpy')
                cur_vec -= prev_vec * dot_val
                Timers.toc('arnoldi axpy')

            Timers.tic('arnoldi norm')
            norm = np.linalg.norm(cur_vec, 2)
            Timers.toc('arnoldi norm')

            assert not math.isinf(norm) and not math.isnan(norm), "vector norm was infinite in arnoldi"

            self.h_mat[self.cur_it, self.cur_it-1] = norm

            if norm >= self.tol:
                Timers.tic('arnoldi norm div')
                cur_vec = cur_vec / norm
                Timers.toc('arnoldi norm div')

                self.v_mat[self.cur_it] = cur_vec
            elif self.cur_it > 1:
                #cur_vec *= 0
                #print "break! norm {} <= tol {}".format(norm, self.tol)
                self.v_mat = self.v_mat[:self.cur_it+1, :]
                self.h_mat = self.h_mat[:self.cur_it+1, :self.cur_it]
                break

            self.cur_it += 1

        self.elapsed += time.time() - start

        #if self.add_ones_key_dir:
        #    pv_mat = np.zeros((self.key_dir_mat.shape[0] + 1, self.v_mat.shape[0]), dtype=float)
        #    pv_mat[:-1] = self.key_dir_mat * self.v_mat.transpose()

        #    for i in xrange(self.v_mat.shape[0]):
        #        pv_mat[-1, i] = ones_dot(self.v_mat[i])
        #else:
        #    pv_mat = self.key_dir_mat * self.v_mat.transpose()
        pv_mat = self.key_dir_mat * self.v_mat.transpose()

        pv_mat *= self.init_norm

        Timers.toc('arnoldi')

        return pv_mat, self.h_mat

    def _lanczos(self, iterations):
        '''run the lanczos algorithm, tailored to very large sparse systems

        This will project each of the v vectors using the key directions matrix, to make pv_mat, a k x n matrix

        further, h_mat is returned as a csr_matrix

        this returns pv_mat, h_mat
        '''

        Timers.tic('lanczos')

        start = time.time()

        while self.cur_it < iterations + 1:
            if self.print_status:
                elapsed = time.time() - start + self.elapsed

                eta = elapsed / (self.cur_it / float(iterations)) - elapsed

                print "lanczos iteration {} / {}, Elapsed: {:.2f}m, ETA: {:.2f}m".format(self.cur_it-1, iterations, \
                    elapsed / 60.0, eta / 60.0)

            # three-term recurrance relation
            self.prev_prev_vec = self.prev_vec
            self.prev_vec = self.cur_vec

            Timers.tic('lanczos mult')
            self.cur_vec = self.mult(self.a_matrix, self.prev_vec)
            Timers.toc('lanczos mult')

            if self.prev_prev_vec is not None:
                dot_val = self.prev_norm # reuse norm from previous iteration
                self.h_data.append(dot_val)
                self.h_inds.append(self.cur_it-2)

                Timers.tic('lanczos axpy')
                self.cur_vec -= self.prev_prev_vec * dot_val
                Timers.toc('lanczos axpy')

            Timers.tic('lanczos dot')
            dot_val = np.dot(self.prev_vec, self.cur_vec)
            Timers.toc('lanczos dot')

            self.h_data.append(dot_val)
            self.h_inds.append(self.cur_it-1)

            Timers.tic('lanczos axpy')
            self.cur_vec -= self.prev_vec * dot_val
            Timers.toc('lanczos axpy')

            Timers.tic('lanczos norm')
            self.prev_norm = norm = np.linalg.norm(self.cur_vec)
            Timers.toc('lanczos norm')

            assert not math.isinf(norm) and not math.isnan(norm), "vector norm was infinite in lanczos"

            if norm >= self.tol:
                self.h_data.append(norm)
                self.h_inds.append(self.cur_it)
                self.h_indptrs.append(len(self.h_data))

                Timers.tic('lanczos norm div')
                self.cur_vec /= norm
                Timers.toc('lanczos norm div')

                #if self.add_ones_key_dir:
                #    self.pv_mat[self.cur_it, :-1] = (self.key_dir_mat * self.cur_vec)
                #    self.pv_mat[self.cur_it, -1] = ones_dot(self.cur_vec)
                #else:
                #    self.pv_mat[self.cur_it, :] = (self.key_dir_mat * self.cur_vec)
                self.pv_mat[self.cur_it, :] = (self.key_dir_mat * self.cur_vec)

            elif self.cur_it > 1:
                # break early
                iterations = self.cur_it - 1
                self.pv_mat = self.pv_mat[:iterations + 1, :]
                break

            self.cur_it += 1

        self.elapsed += time.time() - start

        # h is easier to construct as a csc matrix, but we want to use it as a csr_matrix
        h_csc = csc_matrix((self.h_data, self.h_inds, self.h_indptrs), shape=(iterations + 1, iterations))
        h_csr = csr_matrix(h_csc)
        rv_pv = (self.pv_mat * self.init_norm).transpose()

        Timers.toc('lanczos')

        return rv_pv, h_csr
