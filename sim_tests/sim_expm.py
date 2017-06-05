'''
Stanley Bak
Simulate Dense expm
'''

import time

import numpy as np

from scipy.sparse.linalg import expm as sparse_expm, expm_multiply as sparse_expm_multiply
from scipy.linalg import expm as dense_expm

def sim_dense_expm(sparse_a_matrix, init_vec, step):
    'use dense expm'

    dense_a_matrix = sparse_a_matrix.toarray()

    start = time.time()
    result = np.dot(dense_expm(dense_a_matrix * step), init_vec)
    runtime = time.time() - start

    return runtime, result

def sim_sparse_expm(sparse_a_matrix, init_vec, step):
    'use sparse expm'

    start = time.time()
    result = sparse_expm(sparse_a_matrix * step) * init_vec
    runtime = time.time() - start

    return runtime, result

def sim_expm_mult(sparse_a_matrix, init_vec, step):
    'use expm_multiply'

    start = time.time()
    result = sparse_expm_multiply(sparse_a_matrix * step, init_vec)
    runtime = time.time() - start

    return runtime, result
