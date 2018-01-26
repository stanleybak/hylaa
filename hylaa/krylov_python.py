'''
Dung Tran & Stanley Bak
August 2017

Simulating a linear system x' = Ax using krylov supspace methods (arnoldi and lanczos)
'''

import math
import time
import os

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import norm as sparse_norm

from hylaa.timerutil import Timers
from hylaa.util import Freezable

def get_free_memory_mb_deprecated():
    'get the amount of free memory available'

    # one-liner to get free memory from:
    # https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
    _, _, available_mb = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

    return available_mb

def normalize_sparse(vec):
    'normalize a sparse vector (passed in as a 1xn csr_matrix), and return a tuple: scaled_vec, original_norm'

    assert isinstance(vec, csr_matrix) and vec.shape[0] == 1

    norm = sparse_norm(vec)

    assert not math.isinf(norm) and not math.isnan(norm) and norm > 1e-9, \
        "bad initial vec norm in normalize_sparse: {}".format(norm)

    # divide in place
    rv = vec / norm

    return rv, norm

#def check_available_memory_arnoldi(stdout, a, n):
#    'check if enough memory is available to store the V and H matrix'

#    required_mb = (((a+1) * n) + (a*(a+1))) * 8 / 1024.0 / 1024.0
#    available_mb = get_free_memory_mb()

#    if stdout:
#        print "Arnoldi Required GB = {:.3f} (+1), available GB = {:.3f} (a = {}, n = {})".format(
#            required_mb / 1024.0, available_mb / 1024.0, a, n)

#    if required_mb + 1024 > available_mb: # add 1024 mb since we want 1 GB free for other things
#        raise MemoryError("Not enogh memory for arnoldi computation.")

class KrylovIterator(Freezable):
    'Krylov Iterator container class'

    def __init__(self, hylaa_settings, a_matrix, key_dir_mat, add_ones_key_dir=False):
        assert a_matrix.shape[0] == a_matrix.shape[1], "a_mat should be square"
        assert key_dir_mat.shape[1] == a_matrix.shape[0], "key_dir_mat width should equal number of dims"
        assert isinstance(a_matrix, csr_matrix), "a_matrix should be a csr_matrix"
        assert isinstance(key_dir_mat, csr_matrix), "key_dir_mat should be a csr_matrix"

        self.settings = hylaa_settings
        self.lanczos = self.settings.simulation.krylov_lanczos
        self.print_status = self.settings.simulation.krylov_stdout# and a_matrix.shape[0] >= int(1e6)
        self.add_ones_key_dir = add_ones_key_dir

        if self.settings.simulation.krylov_transpose and not self.lanczos:
            # we need to compute with the transpose of the a matrix
            self.a_matrix = csr_matrix(a_matrix.transpose())
        else:
            self.a_matrix = a_matrix

        self.key_dir_mat = key_dir_mat
        self.tol = 1e-9

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

        self.freeze_attrs()

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

        if self.add_ones_key_dir:
            key_dirs += 1

        if self.reinit:
            scaled_vec, self.init_norm = normalize_sparse(init_vec)

            self.cur_it = 1

            #check_available_memory_arnoldi(print_status, iterations, dims)
            if self.lanczos:

                self.pv_mat = np.zeros((iterations + 1, key_dirs))
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
                if self.add_ones_key_dir:
                    self.pv_mat[0, :-1] = (self.key_dir_mat * scaled_vec.T).toarray()[:, 0]
                    self.pv_mat[0, -1] = sum(self.cur_vec)
                else:
                    self.pv_mat[0, :] = (self.key_dir_mat * scaled_vec.T).toarray()[:, 0]
            else:
                # arnoldi

                self.v_mat = np.zeros((iterations + 1, dims))
                self.h_mat = np.zeros((iterations + 1, iterations))

                # sparse assignment of initial vector
                for i in xrange(len(scaled_vec.data)):
                    self.v_mat[0, scaled_vec.indices[i]] = scaled_vec.data[i]

            self.reinit = False # next time don't reinitialize
        else:
            # continue the computation (allocate more memory)

            if self.lanczos:
                new_pv_mat = np.zeros((iterations + 1, key_dirs))
                new_pv_mat[:self.pv_mat.shape[0], :self.pv_mat.shape[1]] = self.pv_mat
                self.pv_mat = new_pv_mat
            else:
                # arnoldi
                new_v_mat = np.zeros((iterations + 1, dims))
                new_h_mat = np.zeros((iterations + 1, iterations))

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

        start = time.time() - self.elapsed

        while self.cur_it < iterations + 1:
            if self.print_status:
                elapsed = time.time() - start

                # we expect quadratic scalability for arnoldi
                frac = self.cur_it * self.cur_it / float(iterations * iterations)
                eta = elapsed / frac - elapsed

                print "arnoldi iteration {} / {}, Elapsed: {:.2f}m, ETA: {:.2f}m".format(self.cur_it-1, iterations, \
                    elapsed / 60.0, eta / 60.0)

            cur_vec = self.a_matrix * self.v_mat[self.cur_it - 1]

            for c in xrange(self.cur_it):
                prev_vec = self.v_mat[c]

                dot_val = np.dot(prev_vec, cur_vec)
                self.h_mat[c, self.cur_it - 1] = dot_val

                cur_vec -= prev_vec * dot_val

            norm = np.linalg.norm(cur_vec, 2)

            self.h_mat[self.cur_it, self.cur_it-1] = norm

            if norm >= self.tol:
                cur_vec = cur_vec / norm
                self.v_mat[self.cur_it] = cur_vec
            elif self.cur_it > 1:
                #print "break! norm {} <= tol {}".format(norm, self.tol)
                self.v_mat = self.v_mat[:self.cur_it+1, :]
                self.h_mat = self.h_mat[:self.cur_it+1, :self.cur_it]
                break

            self.cur_it += 1

        self.elapsed += time.time() - start

        if self.add_ones_key_dir:
            pv_mat = np.zeros((self.key_dir_mat.shape[0] + 1, self.v_mat.shape[0]), dtype=float)
            pv_mat[:-1] = self.key_dir_mat * self.v_mat.transpose()

            for i in xrange(self.v_mat.shape[0]):
                pv_mat[-1, i] = sum(self.v_mat[i])
        else:
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

        start = time.time() - self.elapsed

        while self.cur_it < iterations + 1:
            if self.print_status:
                elapsed = time.time() - start

                eta = elapsed / (self.cur_it / float(iterations)) - elapsed

                print "lanczos iteration {} / {}, Elapsed: {:.2f}m, ETA: {:.2f}m".format(self.cur_it-1, iterations, \
                    elapsed / 60.0, eta / 60.0)

            # three-term recurrance relation
            self.prev_prev_vec = self.prev_vec
            self.prev_vec = self.cur_vec

            self.cur_vec = self.a_matrix * self.prev_vec

            if self.prev_prev_vec is not None:
                dot_val = self.prev_norm # reuse norm from previous iteration
                self.h_data.append(dot_val)
                self.h_inds.append(self.cur_it-2)

                self.cur_vec -= self.prev_prev_vec * dot_val

            dot_val = np.dot(self.prev_vec, self.cur_vec)

            self.h_data.append(dot_val)
            self.h_inds.append(self.cur_it-1)

            self.cur_vec -= self.prev_vec * dot_val
            self.prev_norm = norm = np.linalg.norm(self.cur_vec)

            self.h_data.append(norm)
            self.h_inds.append(self.cur_it)
            self.h_indptrs.append(len(self.h_data))

            if norm >= self.tol:
                self.cur_vec = self.cur_vec / norm

                if self.add_ones_key_dir:
                    self.pv_mat[self.cur_it, :-1] = (self.key_dir_mat * self.cur_vec)
                    self.pv_mat[self.cur_it, -1] = sum(self.cur_vec)
                else:
                    self.pv_mat[self.cur_it, :] = (self.key_dir_mat * self.cur_vec)

            elif self.cur_it > 1:
                # ugggg figure out what to do here... probably safe to ignore
                #v_mat = v_mat[:cur_it+1, :]
                #h_mat = h_mat[:cur_it+1, :cur_it]
                break

            self.cur_it += 1

        self.elapsed += time.time() - start

        # h is easier to construct as a csc matrix, but we want to use it as a csr_matrix
        h_csc = csc_matrix((self.h_data, self.h_inds, self.h_indptrs), shape=(iterations + 1, iterations))
        h_csr = csr_matrix(h_csc)
        rv_pv = (self.pv_mat * self.init_norm).transpose()

        Timers.toc('lanczos')

        return rv_pv, h_csr
