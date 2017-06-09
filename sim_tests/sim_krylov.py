'''
Simulation using Krylov subspace approximation method
'''

import time
import numpy as np
from krypy.utils import arnoldi
import scipy 


# sparse_a_matrix is a scipy csc_matrix
# init_vec is a np.array
# step is a time
# kry_dim is krylov supspace dimension
# returns a tuple, (runtime, result), where result is an np.array, and runtime is in seconds

def sim_krylov_sparse(sparse_a_matrix, init_vec, step, dim=8):
    'compute e^(A * step) * init'

    num_dims = sparse_a_matrix.shape[0]
    start = time.time()
    
    # compute result = e^{A*step} * init_vec

    result = np.zeros((num_dims,), dtype=float)

    x0 = np.asmatrix(init_vec).transpose()

    
    m = dim # krylov supspace dimension
    arn_start = time.time()
    V, H = arnoldi(sparse_a_matrix,x0,m)
    print "arnoldi time = {}ms".format(1000 * (time.time() - arn_start))
    
    Vm = V[:,0:m]
    Hm = H[0:m,:]
    beta = np.linalg.norm(x0)
    Hms = step*Hm
    Vmm = beta*Vm
    In = np.eye(m)
    e1 =In[:,0]

    P = scipy.linalg.expm(Hms)
    VmP = np.dot(Vmm,P)
    result = np.dot(VmP,e1)
    
    runtime = time.time() - start
    return runtime, result
