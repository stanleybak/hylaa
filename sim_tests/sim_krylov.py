'''
Simulation using Krylov subspace approximation method
'''

import time
import numpy as np

# sparse_a_matrix is a scipy csc_matrix
# init_vec is a np.array
# step is a time
# returns a tuple, (runtime, result), where result is an np.array, and runtime is in seconds
def sim_krylov_sparse(sparse_a_matrix, init_vec, step):
    'compute e^(A * step) * init'

    num_dims = sparse_a_matrix.shape[0]
    start = time.time()

    # compute result = e^{A*step} * init_vec
    result = np.zeros((num_dims,), dtype=float)


    runtime = time.time() - start
    return runtime, result
