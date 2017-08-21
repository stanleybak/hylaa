"""This module contains methods for producing ODEs from PDEs"""
from __future__ import division
import copy
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np

class HeatEquationOneDimension(object):
    """Generate ODEs from 1-d diffusion heat equation"""

    # consider 1-d diffusion problem
    # boundary conditions (BCs): u(0,t) = u0 in range [a, b]; u(len_x, t) = uN in range [c, d]
    # initial condition (IC): u(x,0) = 0
    # no heat source

    # todo: generate ODEs of 1-d diffusion equation with heat source: u_t = a*u_xx + u(x,t)
    # where heat source position is given

    def __init__(self, diffusity_const, len_x):
        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0 # diffusity constant
        self.len_x = len_x if len_x > 0 else 0 # length along x-axis
        if self.diffusity_const == 0 or self.len_x == 0:
            raise ValueError('inappropriate parameters')

    def get_odes(self, num_mesh_point):
        'Generate linear state space model dot(x) = Ax + Bu'

        # method returns matrix A and B in the form of csr sparse matrices

        if num_mesh_point <= 0: # number of mesh point between x = 0 and x = len_x
            raise ValueError('num_mesh_point <= 0')
        else:
            disc_step = float((self.len_x)/(num_mesh_point +1)) # discretization step
            alpha = self.diffusity_const/disc_step**2

            print "\ndiscretization step: {}".format(disc_step)
            print "\nnumber of mesh point: {}".format(num_mesh_point)
            print "\nlength of grid: {}".format(self.len_x)
            print "\nalpha: {}".format(alpha)

            data_start = np.array([-2, 1])
            data_end = np.array([1, -2])
            data_middle = np.array([1, -2, 1])

            # fill data
            data = copy.copy(data_start)
            for i in xrange(0, num_mesh_point):

                if i == num_mesh_point - 1 and i > 0:
                    data = np.append(data, data_end)
                elif i > 0:
                    data = np.append(data, data_middle)

            data = alpha*data

            # fill index pointer
            indptr = np.zeros((num_mesh_point + 1,), dtype=np.int64)
            indptr[0] = 0
            for i in xrange(1, num_mesh_point + 1):
                if i == 1 or i == num_mesh_point:
                    indptr[i] = indptr[i-1] + 2
                else:
                    indptr[i] = indptr[i-1] + 3

            # number of nonzeros 
            if num_mesh_point <= 2:
                nnz = 2*num_mesh_point
            else:
                nnz = 4 + (num_mesh_point-2)*3

            # fill indices
            indices = np.zeros((nnz,), dtype = np.int64) # indices of csr matrix
            for i in xrange(0, num_mesh_point):
                j = indptr[i+1] - 1
                if j < nnz - 1:
                    indices[j] = i+1
                else:
                    indices[j] = i
                while j > indptr[i]:
                    j = j -1
                    indices[j] = indices[j+1] - 1

            matrix_a = csr_matrix((data, indices, indptr), shape=(num_mesh_point, num_mesh_point))

            z = np.zeros((num_mesh_point, 2))
            z[0, 0] = alpha
            z[num_mesh_point-1, 1] = alpha

            matrix_b = sparse.csr_matrix(z)

            return matrix_a, matrix_b

#class HeatEquationTwoDimensions(object):
#    """Generate ODEs from 2-d Heat equation"""

#    def __init__(self, diffusity_const, len_x, len_y):

def test():
    'test'
    he = HeatEquationOneDimension(0.1, 1)
    matrix_a, matrix_b = he.get_odes(10)
    print "\nmatrix_a:\n{}".format(matrix_a.toarray())
    print "\nmatrix_b:\n{}".format(matrix_b.toarray())

if __name__ == '__main__':
    test()
