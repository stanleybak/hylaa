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
# numStep is the number of step use in simulation
# kry_dim is krylov supspace dimension
# returns a tuple, (runtime, result), where result is an np.array, and runtime is in seconds
def sim_krylov_sparse(sparse_a_matrix, init_vec, step, numStep, kry_dim):
    
    num_dims = sparse_a_matrix.shape[0]
    start = time.time()
    
    # compute result = e^{A*step} * init_vec

    result = np.zeros((num_dims,numStep), dtype=float)
    
    V, H = arnoldi(sparse_a_matrix,init_vec,kry_dim)
    Vm = V[:,0:kry_dim]
    Hm = H[0:kry_dim,:]
    beta = np.linalg.norm(init_vec)
    Hms = step*Hm
    Vmm = beta*Vm
    In = np.eye(kry_dim)
    e1 =In[:,0]

    for i in range(0,numStep):
        P = scipy.linalg.expm(i*Hms)
        VmP = np.dot(Vmm,P)
        result[:,i] = np.dot(VmP,e1)
    
    runtime = time.time() - start
    return runtime, result
