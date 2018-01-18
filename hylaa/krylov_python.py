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
import multiprocessing
from multiprocessing.pool import ThreadPool

print "Note: krylov_python using global thread_pool"
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

def python_lanczos(a_mat, init_vec, iterations, key_dir_mat, tol=1e-9, profile=False):
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

    cur_vec = scaled_vec.toarray()
    prev_vec = None
    prev_prev_vec = None
    prev_norm = None

    # profiling
    nonzeros = a_mat.getnnz()

    if profile:
        print "lanczos a_mat nonzeros: {}".format(nonzeros)

    dot_secs = 0.0
    axpy_secs = 0.0
    mult_secs = 0.0
    norm_secs = 0.0
    proj_secs = 0.0

    start = time.time()
    # sparse assignment of initial vector
    pv_mat[0, :] = (key_dir_mat * scaled_vec.T).toarray()[:, 0]
    proj_secs += time.time() - start

    for cur_it in xrange(1, iterations + 1):
        if profile:
            print "iteration {} / {}".format(cur_it, iterations)

        # three-term recurrance relation
        prev_prev_vec = prev_vec
        prev_vec = cur_vec

        start = time.time()
        cur_vec = (a_mat * prev_vec.T).T
        mult_secs += time.time() - start

        if prev_prev_vec is not None:
            dot_val = prev_norm # reuse norm from previous iteration
            h_data.append(dot_val)
            h_inds.append(cur_it-2)

            start = time.time()
            #cur_vec -= prev_prev_vec * dot_val
            paxpy(cur_vec[0], -1 * dot_val, prev_prev_vec[0])
            axpy_secs += time.time() - start

        start = time.time()
        #dot_val = np.dot(prev_vec[0], cur_vec[0])
        dot_val = pdot(prev_vec[0], cur_vec[0])
        dot_secs += time.time() - start

        h_data.append(dot_val)
        h_inds.append(cur_it-1)

        start = time.time()
        #cur_vec -= prev_vec * dot_val
        paxpy(cur_vec[0], -1 * dot_val, prev_vec[0])
        axpy_secs += time.time() - start

        start = time.time()
        prev_norm = norm = np.linalg.norm(cur_vec)
        norm_secs += time.time() - start

        h_data.append(norm)
        h_inds.append(cur_it)
        h_indptrs.append(len(h_data))

        if norm >= tol:
            cur_vec = cur_vec / norm

            start = time.time()
            pv_mat[cur_it, :] = (key_dir_mat * cur_vec[0])
            proj_secs = time.time() - start
        else:
            break

    if profile:
        names = ['dot', 'axpy', 'mult', 'norm', 'proj']
        secs = [dot_secs, axpy_secs, mult_secs, norm_secs, proj_secs]
        sum_secs = 0.0

        for i in xrange(len(names)):
            print "{}: {} ms".format(names[i], secs[i] * 1000)
            sum_secs += secs[i]

        print "Profiled Total seconds: {}s".format(sum_secs)

    # scale back
    pv_mat *= init_norm

    # h is easier to construct as a csc matrix, but we want to use it as a csr_matrix
    h_csc = csc_matrix((h_data, h_inds, h_indptrs), shape=(iterations + 1, iterations))
    h_csr = csr_matrix(h_csc)
    rv_pv = pv_mat.transpose()

    Timers.toc('python_lanczos')

    return rv_pv, h_csr
