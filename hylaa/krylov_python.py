'''
Dung Tran & Stanley Bak
August 2017

Simulating a linear system x' = Ax using krylov supspace methods (arnoldi and lanczos)
'''

import math
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import norm as sparse_norm

from hylaa.timerutil import Timers

def normalize_sparse(vec):
    'normalize a sparse vector (passed in as a 1xn csr_matrix), and return a tuple: scaled_vec, original_norm'

    assert isinstance(vec, csr_matrix) and vec.shape[0] == 1

    norm = sparse_norm(vec)

    assert not math.isinf(norm) and not math.isnan(norm) and norm > 1e-9, \
        "bad initial vec norm in normalize_sparse: {}".format(norm)

    # divide in place
    rv = vec / norm

    return rv, norm

def python_arnoldi(a_mat, init_vec, iterations, key_dir_mat, tol=1e-9):
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

    v_mat = np.zeros((iterations + 1, dims))
    h_mat = np.zeros((iterations + 1, iterations))

    # sparse assignment of initial vector
    for i in xrange(len(scaled_vec.data)):
        v_mat[0, scaled_vec.indices[i]] = scaled_vec.data[i]

    for cur_it in xrange(1, iterations + 1):
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

    pv_mat = key_dir_mat * v_mat.transpose()

    pv_mat *= init_norm

    Timers.toc('python_arnoldi')

    return pv_mat, h_mat

def python_lanczos(a_mat, init_vec, iterations, key_dir_mat, tol=1e-9, compat=False, profile=False):
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

    dims = a_mat.shape[0]
    key_dirs = key_dir_mat.shape[0]

    pv_mat = np.zeros((iterations + 1, key_dirs))
    h_data = []
    h_inds = []
    h_indptrs = [0]

    scaled_vec, init_norm = normalize_sparse(init_vec)

    # sparse assignment of initial vector
    pv_mat[0, :] = (key_dir_mat * scaled_vec.T).toarray()[:, 0]

    cur_vec = scaled_vec.toarray()
    prev_vec = None
    prev_prev_vec = None
    prev_norm = None

    # profiling
    nonzeros = a_mat.getnnz()

    if profile:
        print "lanczos a_mat nonzeros: {}".format(nonzeros)

    dot_ops = 0
    dot_secs = 0.0

    sub_ops = 0
    sub_secs = 0.0

    mult_ops = 0
    mult_secs = 0.0

    norm_ops = 0
    norm_secs = 0.0

    for cur_it in xrange(1, iterations + 1):
        if profile:
            print "iteration {} / {}".format(cur_it, iterations)

        # three-term recurrance relation
        prev_prev_vec = prev_vec
        prev_vec = cur_vec

        if profile:
            start = time.time()

        cur_vec = (a_mat * prev_vec.T).T

        if profile:
            mult_secs += time.time() - start
            mult_ops += 2 * nonzeros

        if prev_prev_vec is not None:
            dot_val = prev_norm # reuse norm from previous iteration
            h_data.append(dot_val)
            h_inds.append(cur_it-2)

            if profile:
                start = time.time()

            cur_vec -= prev_prev_vec * dot_val

            if profile:
                sub_secs += time.time() - start
                sub_ops += 2 * dims

        if profile:
            start = time.time()

        if compat:
            dot_val = np.dot(prev_vec[0].T.conj(), cur_vec[0])
        else:
            dot_val = np.dot(prev_vec[0], cur_vec[0])

        if profile:
            dot_secs += time.time() - start
            dot_ops += 2 * dims

        h_data.append(dot_val)
        h_inds.append(cur_it-1)

        if profile:
            start = time.time()

        cur_vec -= prev_vec * dot_val

        if profile:
            sub_secs += time.time() - start
            sub_ops += 2 * dims

        if profile:
            start = time.time()

        if compat:
            ip = np.dot(cur_vec.conj(), cur_vec.T)
            prev_norm = norm = np.sqrt(np.linalg.norm(ip, 2))
        else:
            prev_norm = norm = np.linalg.norm(cur_vec, 2)

        if profile:
            norm_secs += time.time() - start
            norm_ops += 2 * dims

        h_data.append(norm)
        h_inds.append(cur_it)
        h_indptrs.append(len(h_data))

        if norm >= tol:
            cur_vec = cur_vec / norm

            pv_mat[cur_it, :] = (key_dir_mat * cur_vec[0])

    if profile:
        names = ['dot', 'sub', 'mult', 'norm']
        ops = [dot_ops, sub_ops, mult_ops, norm_ops]
        secs = [dot_secs, sub_secs, mult_secs, norm_secs]

        for i in xrange(len(names)):
            mflops = ops[i] / secs[i] / 1e6

            print "{}: {} MOPS in {} ms ({} MFLOPS)".format(names[i], ops[i] / 1e6, secs[i] * 1000, mflops)

    # scale back
    pv_mat *= init_norm

    # h is easier to construct as a csc matrix, but we want to use it as a csr_matrix
    h_csc = csc_matrix((h_data, h_inds, h_indptrs), shape=(iterations + 1, iterations))
    h_csr = csr_matrix(h_csc)
    rv_pv = pv_mat.transpose()

    Timers.toc('python_lanczos')

    return rv_pv, h_csr
