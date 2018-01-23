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

def get_free_memory_mb():
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

def check_available_memory_arnoldi(stdout, a, n):
    'check if enough memory is available to store the V and H matrix'

    required_mb = (((a+1) * n) + (a*(a+1))) * 8 / 1024.0 / 1024.0
    available_mb = get_free_memory_mb()

    if stdout:
        print "Arnoldi Required GB = {:.3f} (+1), available GB = {:.3f} (a = {}, n = {})".format(
            required_mb / 1024.0, available_mb / 1024.0, a, n)

    if required_mb + 1024 > available_mb: # add 1024 mb since we want 1 GB free for other things
        raise MemoryError("Not enogh memory for arnoldi computation.")

def python_arnoldi(a_mat, init_vec, iterations, key_dir_mat, tol=1e-9, print_status=False):
    '''run the arnoldi algorithm

    this returns pv_mat, h_mat
    '''

    Timers.tic('python_arnoldi')

    assert isinstance(key_dir_mat, csr_matrix), "key_dir_mat should be a csr_matrix"
    assert isinstance(a_mat, csr_matrix), "a_mat should be a csr_matrix"
    assert isinstance(init_vec, csr_matrix), "init_vec should be csr_matrix"
    assert init_vec.shape[0] == 1

    scaled_vec, init_norm = normalize_sparse(init_vec)

    dims = a_mat.shape[0]

    check_available_memory_arnoldi(print_status, iterations, dims)

    v_mat = np.zeros((iterations + 1, dims))
    h_mat = np.zeros((iterations + 1, iterations))

    # sparse assignment of initial vector
    for i in xrange(len(scaled_vec.data)):
        v_mat[0, scaled_vec.indices[i]] = scaled_vec.data[i]

    start = time.time()

    for cur_it in xrange(1, iterations + 1):
        if print_status:
            elapsed = time.time() - start

            # we expect quadratic scalability for arnoldi
            frac = cur_it * cur_it / float(iterations * iterations)
            eta = elapsed / frac - elapsed

            print "arnoldi iteration {} / {}, Elapsed: {:.2f}m, ETA: {:.2f}m".format(cur_it, iterations, \
                elapsed / 60.0, eta / 60.0)

        cur_vec = a_mat * v_mat[cur_it - 1]

        for c in xrange(cur_it):
            prev_vec = v_mat[c]

            dot_val = np.dot(prev_vec, cur_vec)
            h_mat[c, cur_it - 1] = dot_val

            cur_vec -= prev_vec * dot_val

        norm = np.linalg.norm(cur_vec, 2)

        h_mat[cur_it, cur_it-1] = norm

        if norm >= tol:
            cur_vec = cur_vec / norm
            v_mat[cur_it] = cur_vec
        else:
            v_mat = v_mat[:cur_it+1, :]
            h_mat = h_mat[:cur_it+1, :cur_it]
            break

    pv_mat = key_dir_mat * v_mat.transpose()

    pv_mat *= init_norm

    Timers.toc('python_arnoldi')

    return pv_mat, h_mat

def python_lanczos(a_mat, init_vec, iterations, key_dir_mat, tol=1e-9, print_status=False):
    '''run the lanczos algorithm, tailored to very large sparse systems

    This will project each of the v vectors using the key directions matrix, to make pv_mat, a k x n matrix

    further, h_mat is returned as a csr_matrix

    this returns pv_mat, h_mat
    '''

    Timers.tic('python_lanczos')

    assert isinstance(a_mat, csr_matrix), "a_mat should be a csr_matrix"
    assert isinstance(key_dir_mat, csr_matrix), "key_dir_mat should be a csr_matrix"
    assert isinstance(init_vec, csr_matrix), "init_vec should be csr_matrix"
    assert init_vec.shape[0] == 1
    assert a_mat.shape[0] == a_mat.shape[1], "a_mat should be square"
    assert key_dir_mat.shape[1] == a_mat.shape[0], "key_dir_mat width should equal number of dims"

    key_dirs = key_dir_mat.shape[0]

    pv_mat = np.zeros((iterations + 1, key_dirs))
    h_data = []
    h_inds = []
    h_indptrs = [0]

    scaled_vec, init_norm = normalize_sparse(init_vec)

    cur_vec = scaled_vec.toarray()
    prev_vec = None
    prev_prev_vec = None
    prev_norm = None

    # sparse assignment of initial vector
    pv_mat[0, :] = (key_dir_mat * scaled_vec.T).toarray()[:, 0]

    start = time.time()

    for cur_it in xrange(1, iterations + 1):
        if print_status:
            elapsed = time.time() - start

            eta = elapsed / (cur_it / float(iterations)) - elapsed

            print "lanczos iteration {} / {}, Elapsed: {:.2f}m, ETA: {:.2f}m".format(cur_it, iterations, \
                elapsed / 60.0, eta / 60.0)

        # three-term recurrance relation
        prev_prev_vec = prev_vec
        prev_vec = cur_vec

        cur_vec = (a_mat * prev_vec.T).T

        if prev_prev_vec is not None:
            dot_val = prev_norm # reuse norm from previous iteration
            h_data.append(dot_val)
            h_inds.append(cur_it-2)

            cur_vec -= prev_prev_vec * dot_val

        dot_val = np.dot(prev_vec[0], cur_vec[0])

        h_data.append(dot_val)
        h_inds.append(cur_it-1)

        cur_vec -= prev_vec * dot_val

        prev_norm = norm = np.linalg.norm(cur_vec)

        h_data.append(norm)
        h_inds.append(cur_it)
        h_indptrs.append(len(h_data))

        if norm >= tol:
            cur_vec = cur_vec / norm

            pv_mat[cur_it, :] = (key_dir_mat * cur_vec[0])
        else:
            break

    # scale back
    pv_mat *= init_norm

    # h is easier to construct as a csc matrix, but we want to use it as a csr_matrix
    h_csc = csc_matrix((h_data, h_inds, h_indptrs), shape=(iterations + 1, iterations))
    h_csr = csr_matrix(h_csc)
    rv_pv = pv_mat.transpose()

    Timers.toc('python_lanczos')

    return rv_pv, h_csr
