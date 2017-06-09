import scipy.io as sio
import matplotlib.pyplot as plt
import scipy 
import numpy as np
from krypy.utils import arnoldi
from sim_krylov2 import sim_krylov_sparse



def make_random_init_vector(num_dims):
    'make a random initial vector'

    rv = np.random.random((num_dims,))

    return rv

mat_contents = sio.loadmat('building.mat')

A = mat_contents['A']

# implement simulation using Krylov supspace method

m = 10

# create initial state x0
n = A.shape[1] # dimension of the system

x0 = np.zeros((n,1))
for i in range(0,n):
    if (i<10):
        x0[i] = 0.002
    if (i==24):
        x0[i] = -0.0001

V, H = arnoldi(A,x0,m)

Vm = V[:,0:m]
Hm = H[0:m,:]

# set up time step for simulation, number of step 
timeStep = 0.01
numStep = 1000

beta = np.linalg.norm(x0)
Hms = timeStep*Hm
Vmm = beta*Vm
In = np.eye(m)
e1 =In[:,0]
X = np.zeros((n,numStep))

for i in range(0,numStep):
    P = scipy.linalg.expm(i*Hms)
    VmP = np.dot(Vmm,P)
    X[:,i] = np.dot(VmP,e1)


runtime, result = sim_krylov_sparse(A,x0,timeStep,numStep,m)

# plot the result
t = np.arange(0,timeStep*numStep,timeStep)
for i in range(0,n):
    plt.plot(t,X[i,:])
plt.show()


