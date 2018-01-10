'''
Dung Tran & Stanley Bak
August 2017

Simulating a linear system x' = Ax using krylov supspace methods (arnoldi and lanczos)
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import norm as sparse_norm

from hylaa.timerutil import Timers

def arnoldi(a_mat, init_vec, iterations, tol=1e-9):
    '''run the arnoldi algorithm

    this returns v_mat.T, h_mat
    '''

    Timers.tic('arnoldi')

    assert isinstance(a_mat, csr_matrix), "a_mat should be a csr_matrix"
    assert isinstance(init_vec, csr_matrix), "init_vec should be csr_matrix"
    assert init_vec.shape[0] == 1
    assert abs(sparse_norm(init_vec) - 1.0) < tol, "init vec norm should be 1.0"

    dims = a_mat.shape[0]

    v_mat = np.zeros((iterations + 1, dims))
    h_mat = np.zeros((iterations + 1, iterations))

    # sparse assignment of initial vector
    for i in xrange(len(init_vec.data)):
        v_mat[0, init_vec.indices[i]] = init_vec.data[i]

    for cur_it in xrange(1, iterations + 1):
        cur_vec = a_mat * v_mat[cur_it - 1]

        for c in xrange(cur_it):
            prev_vec = v_mat[c]

            dot_val = np.dot(prev_vec, cur_vec)
            h_mat[c, cur_it - 1] = dot_val

            cur_vec -= prev_vec * dot_val

        norm = np.linalg.norm(cur_vec)
        h_mat[cur_it, cur_it-1] = norm

        if norm >= tol:
            cur_vec = cur_vec / norm
            v_mat[cur_it] = cur_vec

    Timers.toc('arnoldi')

    return v_mat, h_mat

def lanczos(a_mat, init_vec, iterations, k_mat, tol=1e-9):
    '''run the lanczos algorithm, tailored to very large sparse systems

    This will project each of the v vectors using the key directions matrix, to make pv_mat, a k x n matrix

    further, h_mat is returned as a csc_matrix

    this returns pv_mat.T, h_mat
    '''

    Timers.tic('lanczos')

    assert isinstance(a_mat, csr_matrix), "a_mat should be a csr_matrix"
    assert isinstance(k_mat, csr_matrix), "k_mat should be a csr_matrix"
    assert isinstance(init_vec, csr_matrix), "init_vec should be csr_matrix"
    assert init_vec.shape[0] == 1
    assert abs(sparse_norm(init_vec) - 1.0) < tol, "init vec norm should be 1.0"
    assert a_mat.shape[0] == a_mat.shape[1], "a_mat should be square"
    assert k_mat.shape[0] == a_mat.shape[0], "k_mat height should equal number of dims"

    key_dirs = k_mat.shape[1]

    pv_mat = np.zeros((iterations + 1, key_dirs))
    h_data = []
    h_inds = []
    h_indptrs = [0]

    # sparse assignment of initial vector
    pv_mat[0, :] = (k_mat * init_vec.T).toarray()[:, 0]

    cur_vec = init_vec.toarray()
    prev_vec = None
    prev_prev_vec = None

    for cur_it in xrange(1, iterations + 1):

        # three-term recurrance relation
        prev_prev_vec = prev_vec
        prev_vec = cur_vec
        cur_vec = (a_mat * prev_vec.T).T

        if prev_prev_vec is not None:
            dot_val = np.dot(prev_prev_vec[0], cur_vec[0])
            h_data.append(dot_val)
            h_inds.append(cur_it-2)
            cur_vec -= prev_prev_vec * dot_val

        dot_val = np.dot(prev_vec[0], cur_vec[0])
        h_data.append(dot_val)
        h_inds.append(cur_it-1)
        cur_vec -= prev_vec * dot_val

        print "ARE WE DOING TOO MUCH WORK HERE? ISNT HE H MATRIX SYMMETRIC SO WE DONT NEED SO MANY DOT PRODUCTS??"

        norm = np.linalg.norm(cur_vec)
        h_data.append(norm)
        h_inds.append(cur_it)
        h_indptrs.append(len(h_data))

        if norm >= tol:
            cur_vec = cur_vec / norm

            pv_mat[cur_it, :] = cur_vec[0]

    Timers.toc('lanczos')

    return pv_mat, csc_matrix((h_data, h_inds, h_indptrs), shape=(iterations + 1, iterations))
