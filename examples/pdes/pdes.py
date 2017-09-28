"""This module contains methods for producing ODEs from PDEs"""
from __future__ import division

import time
import math
from scipy import sparse
from scipy.integrate import odeint
import numpy as np


class HeatOneDimension(object):
    """Generate ODEs from 1-d diffusion heat equation"""

    # This benchmark is from the book: "Partial differential
    # equations for scientists and engineers"
    # S. J. Farlow, Courier Corporation, 1993, Page 39,

    def __init__(self, diffusity_const, thermal_cond, heat_exchange_coeff, len_x):

        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0  # diffusity constant
        self.len_x = len_x if len_x > 0 else 0    # length along x-axis
        self.thermal_cond = thermal_cond if thermal_cond > 0 else 0  # thermal conductivity constant
        self.heat_exchange_coeff = heat_exchange_coeff if heat_exchange_coeff > 0 else 0  # heat exchange constant
        if self.diffusity_const == 0 or self.len_x == 0 or self.thermal_cond == 0 or self.heat_exchange_coeff == 0:
            raise ValueError('inappropriate parameters')

        self.heat_lost_const = self.heat_exchange_coeff / self.thermal_cond

    def get_odes(self, num_x):
        'Generate linear state space model dot(x) = Ax + Bu'

        'obtain linear model of the benchmark'
        assert isinstance(num_x, int), "number of mesh point should be an integer"

        if num_x <= 0:
            raise ValueError('number of mesh points should be larger than zero')

        disc_step_x = self.len_x / (num_x + 1)  # dicrezation step along x axis
        print "\ndiscretization step along x-axis is: {} cm".format(disc_step_x)

        # changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient
        matrix_a = sparse.lil_matrix((num_x, num_x))
        matrix_b = sparse.lil_matrix((num_x, 1))

        a = 1 / disc_step_x**2
        k = self.heat_lost_const

        a = 1
        k = 1

        # fill matrix_a

        for i in xrange(0, num_x):
            matrix_a[i, i] = -2 * a  # filling diagonal

            # fill along x - axis
            if (i - 1 >= 0):
                matrix_a[i, i - 1] = a
            else:
                matrix_a[i, i] = matrix_a[i, i] + a
            if (i + 1 <= num_x - 1):
                matrix_a[i, i + 1] = a
            else:
                matrix_a[i, i] = matrix_a[i, i] + a / (1 + disc_step_x * k)
                matrix_b[i, 0] = k / (disc_step_x * (1 + disc_step_x * k))

        return self.diffusity_const * (matrix_a.tocsr()), self.diffusity_const * (matrix_b.tocsr())


class HeatTwoDimension1(object):
    """Generate ODEs from 2-dimensional heat-flow problem inside a copper plate"""
    # This benchmark is from the book: "Partial differential equations for scientists and engineers
    # S. J. Farlow, Courier Corporation, 1993, Page 40,

    def __init__(self, diffusity_const, heat_exchange_coeff, thermal_cond, len_x, len_y):
        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0  # diffusity constant
        self.heat_exchange_coeff = heat_exchange_coeff if heat_exchange_coeff > 0 else 0  # heat exchange coefficient
        self.thermal_cond = thermal_cond if thermal_cond > 0 else 0  # thermal conductivity
        self.len_x = len_x if len_x > 0 else 0  # length x
        self.len_y = len_y if len_y > 0 else 0  # length y

        if self.diffusity_const == 0 or self.heat_exchange_coeff == 0 or \
                self.thermal_cond == 0 or  self.len_x == 0 or self.len_y == 0:
            raise ValueError("inappropriate parameters")
        self.heat_lost_const = self.heat_exchange_coeff / self.thermal_cond

    def get_odes(self, num_x, num_y):
        'obtain linear model of the benchmark'
        assert isinstance(num_x, int), "number of mesh point should be an integer"
        assert isinstance(num_y, int), "number of messh point should be an integer"

        if num_x <= 0 or num_y <= 0:
            raise ValueError('number of mesh points should be larger than zero')

        disc_step_x = self.len_x / (num_x + 1)  # dicrezation step along x axis
        print "\ndiscretization step along x-axis is: {} cm".format(disc_step_x)
        disc_step_y = self.len_y / (num_y + 1)  # discrezation step along y axis
        print "\ndiscretization step along y-axis is: {} cm\n".format(disc_step_y)

        num_var = num_x * num_y  # number of discrezation state variables
        # changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient
        matrix_a = sparse.lil_matrix((num_var, num_var))
        matrix_b = sparse.lil_matrix((num_var, 3))

        a = 1 / disc_step_x**2
        b = 1 / disc_step_y**2
        c = -2 * (a + b)
        k = self.heat_lost_const
        step_x = disc_step_x

        # fill matrix_a

        for i in xrange(0, num_var):
            matrix_a[i, i] = c    # filling diagonal
            x_pos = i % num_x     # x-position corresponding to i-th state variable
            y_pos = int((i - x_pos) / num_x)
            print "the {}th variable is the temperature at the mesh point ({},{})".format(i, x_pos, y_pos)
            # fill along x - axis
            if x_pos - 1 >= 0:
                matrix_a[i, i - 1] = a
            else:
                matrix_a[i, i] = matrix_a[i, i] + a
                matrix_b[i, 0] = math.sqrt(a)
            if x_pos + 1 <= num_x - 1:
                matrix_a[i, i + 1] = a
            else:
                # fill diffusion term
                matrix_a[i, i] = matrix_a[i, i] + a / (1 + k * step_x)
                matrix_b[i, 2] = a * (k * step_x) / (1 + k * step_x)
            # fill along y-axis
            if y_pos - 1 >= 0:
                matrix_a[i, (y_pos - 1) * num_x + x_pos] = b
            else:
                matrix_b[i, 1] = b

            if y_pos + 1 <= num_y - 1:
                matrix_a[i, (y_pos + 1) * num_x + x_pos] = b
            else:
                matrix_a[i, i] = matrix_a[i, i] + b

        return self.diffusity_const * (matrix_a.tocsr()), self.diffusity_const * (matrix_b.tocsr())


class HeatTwoDimension2(object):
    """Generate ODEs from 2-d Heat equation"""

    # We consider the 2-d heat diffusion equation on a rectangle metal plate with the form:
    # u_t = alpha*(u_xx + u_yy)
    # we assume the following boundary conditions (BCs):
    # BC1:u(x, len_y, t) = 0 (the top), BC2: u(0,y,t) = 0 (the left),
    # BC3_1: u(x,0,t) = f(t), 0 <= x <= (1/k)len_x (the bottom 1)
    # BC3_2: u(x,0,t) = 0, (1/k) len_x < x <= len_x (the bottom 2)

    # In the right side of the metal plate, the heat is exchanged with the environment
    # The boundary condition for the right hand side is: BC4:  u_x(len_x,y,t) = -k*[u(len_x,y,t) - g(t)]
    # k is the heat_lost_constant and g(t) is environment temperature, c1 <= g(t) <= c2
    # The BC4 shows that the metal plate lost its heat linearly to the environment

    # we assume the following initial condition (IC): u(x,y,0) = 0

    # This is a generation of ZhiHan benchmark in his thesis
    # "Formal verification of hybrid systems using model order reduction and decomposition"
    # 2005, page 68

    def __init__(self, diffusity_const, heat_exchange_coeff, thermal_cond, len_x, len_y, has_heat_source, heat_source_pos):
        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0  # diffusity constant
        self.heat_exchange_coeff = heat_exchange_coeff if heat_exchange_coeff > 0 else 0  # heat exchange coefficient
        self.thermal_cond = thermal_cond if thermal_cond > 0 else 0  # thermal conductivity
        self.len_x = len_x if len_x > 0 else 0  # length x
        self.len_y = len_y if len_y > 0 else 0  # length y

        if (self.diffusity_const == 0 or self.heat_exchange_coeff == 0 or
           self.thermal_cond == 0 or self.len_x == 0 or self.len_y == 0):
            raise ValueError("inappropriate parameters")
        self.heat_lost_const = self.heat_exchange_coeff / self.thermal_cond

        self.has_heat_source = has_heat_source
        assert isinstance(heat_source_pos, np.ndarray), "heat source pos is not an ndarray"
        if heat_source_pos.shape != (2,):
            raise ValueError("heat source position is not 2x1 array")
        if heat_source_pos[0] < 0 or heat_source_pos[1] > self.len_x:
            raise ValueError("Heat source position value error")
        self.heat_source_pos = heat_source_pos  # an array to indicate the position of heat source
        # heat_source_pos = ([[x_start, x_end])

    def get_odes(self, num_x, num_y):
        'obtain the linear model of 2-d heat equation'

        assert isinstance(num_x, int), "number of mesh point should be an integer"
        assert isinstance(num_y, int), "number of messh point should be an integer"

        if num_x <= 0 or num_y <= 0:
            raise ValueError('number of mesh points should be larger than zero')

        disc_step_x = self.len_x / (num_x + 1)  # dicrezation step along x axis
        print "\ndiscretization step along x-axis:{}".format(disc_step_x)
        disc_step_y = self.len_y / (num_y + 1)  # discrezation step along y axis
        print "\ndiscretization step along y-axis:{}".format(disc_step_y)

        if self.has_heat_source:
            heat_start_pos_x = int(math.floor(self.heat_source_pos[0] / disc_step_x)) - 1
            heat_end_pos_x = int(math.floor(self.heat_source_pos[1] / disc_step_x)) - 1

            print "\nheat source is from point {} to point {} on x-axis\n".format(heat_start_pos_x, heat_end_pos_x)

        # we use explicit semi- finite-difference method to obtain the
        # linear model of heat equation

        num_var = num_x * num_y  # number of discrezation variables
        # changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient
        matrix_a = sparse.lil_matrix((num_var, num_var))
        matrix_b = sparse.lil_matrix((num_var, 2))
        a = 1 / disc_step_x**2
        b = 1 / disc_step_y**2
        c = -2 * (a + b)
        k = self.heat_lost_const
        step_x = disc_step_x

        # fill matrix_a

        for i in xrange(0, num_var):
            matrix_a[i, i] = c  # filling diagonal
            x_pos = i % num_x  # x-position corresponding to i-th state variable
            y_pos = int((i - x_pos) / num_x)  # y-position corresponding to i-th variable
            print "the {}-th state variable is the temperature at the point ({},{})".format(i, x_pos, y_pos)

            # fill along x - axis
            if x_pos - 1 >= 0:
                matrix_a[i, i - 1] = a
            else:
                matrix_a[i, i] = matrix_a[i, i] + a

            if x_pos + 1 <= num_x - 1:
                matrix_a[i, i + 1] = a
            else:
                # fill diffusion term
                matrix_a[i, i] = matrix_a[i, i] + a / (1 + k * step_x)
                matrix_b[i, 1] = a * (k * step_x) / (1 + k * step_x)

            # fill along y-axis
            if y_pos - 1 >= 0:
                matrix_a[i, (y_pos - 1) * num_x + x_pos] = b
            else:

                if self.has_heat_source and x_pos >= heat_start_pos_x and x_pos <= heat_end_pos_x:
                    matrix_b[i, 0] = b
                else:
                    matrix_a[i, i] = matrix_a[i, i] + b

            if y_pos + 1 <= num_y - 1:
                matrix_a[i, (y_pos + 1) * num_x + x_pos] = b
            else:
                matrix_a[i, i] = matrix_a[i, i] + b

        return self.diffusity_const * (matrix_a.tocsr()), self.diffusity_const * (matrix_b.tocsr())


class HeatThreeDimension(object):
    '3-dimensional heat equation'

    def __init__(self, diffusity_const, heat_exchange_const, len_x, len_y, len_z, heat_source_pos):

        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0
        self.heat_exchange_const = heat_exchange_const if heat_exchange_const > 0 else 0
        self.len_x = len_x if len_x > 0 else 0
        self.len_y = len_y if len_y > 0 else 0
        self.len_z = len_z if len_z > 0 else 0

        if (self.diffusity_const == 0 or self.heat_exchange_const == 0 or self.len_x == 0 or
                self.len_y == 0 or self.len_z == 0):
            raise ValueError("inappropriate parameters")

        assert isinstance(heat_source_pos, np.ndarray)
        if heat_source_pos.shape != (2, 2):
            raise ValueError("heat source position should be 2 x 2 array")
        if (heat_source_pos[0, 0] < 0 or heat_source_pos[0, 1] > self.len_x or heat_source_pos[1, 0] < 0 or
                heat_source_pos[1, 1] > self.len_y):
            raise ValueError("heat source position should be inside the 3-d object")

        self.heat_source_pos = heat_source_pos

    def get_odes(self, num_x, num_y, num_z):

        assert isinstance(num_x, int)
        assert isinstance(num_y, int)
        assert isinstance(num_z, int)

        if num_x <= 0 or num_y <= 0 or num_z <= 0:
            raise ValueError("number of mesh points should be large than zero")

        step_x = self.len_x / (num_x + 1)
        step_y = self.len_y / (num_y + 1)
        step_z = self.len_z / (num_z + 1)

        print"\nstep_x = {}, step_y = {}, step_z = {}".format(step_x, step_y, step_z)
        print"\n------------------"

        a = 1 / step_x**2
        b = 1 / step_y**2
        c = 1 / step_z**2
        d = -2 * (a + b + c)

        heat_start_pos_x = int(math.ceil(self.heat_source_pos[0, 0] / step_x)) - 1
        heat_stop_pos_x = int(math.floor(self.heat_source_pos[0, 1] / step_x)) - 1

        heat_start_pos_y = int(math.ceil(self.heat_source_pos[1, 0] / step_y)) - 1
        heat_stop_pos_y = int(math.floor(self.heat_source_pos[1, 1] / step_y)) - 1

        print"\nheat source is from pos_x = {} to pos_x = {}".format(heat_start_pos_x, heat_stop_pos_x)
        print"\nheat source is from pos_y = {} to pos_y = {}".format(heat_start_pos_y, heat_stop_pos_y)
        print"\n------------------"
        num_var = num_x * num_y * num_z

        matrix_a = sparse.lil_matrix((num_var, num_var))
        matrix_b = sparse.lil_matrix((num_var, 1))

        for i in xrange(0, num_var):
            z_pos = int(math.floor(i / num_x / num_y))
            y_pos = int(math.floor((i - z_pos * num_x * num_y) / num_x))
            x_pos = i - z_pos * num_x * num_y - y_pos * num_x

            print"\nposition of the {}-th variable is: ({},{},{})".format(i, x_pos, y_pos, z_pos)

            matrix_a[i, i] = d  # fill the diagonal

            if x_pos - 1 >= 0:
                matrix_a[i, i - 1] = matrix_a[i, i - 1] + a
            if x_pos + 1 <= num_x - 1:
                matrix_a[i, i + 1] = matrix_a[i, i + 1] + a

            if y_pos - 1 >= 0:
                ind = z_pos * num_x * num_y + (y_pos - 1) * num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + b
            if y_pos + 1 <= num_y - 1:
                ind = z_pos * num_x * num_y + (y_pos + 1) * num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + b

            if z_pos - 1 >= 0:
                ind = (z_pos - 1) * num_x * num_y + y_pos * num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + c
            if z_pos + 1 <= num_z - 1:
                ind = (z_pos + 1) * num_x * num_y + y_pos * num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + c

            # boundary conditions

            if x_pos == 0:   # u(0, j, k) = u(1, j, k): the left face
                matrix_a[i, i] = matrix_a[i, i] + a
            if y_pos == num_y - 1:  # u(i, num_y, k) = u(i, num_y - 1, k): the back face
                matrix_a[i, i] = matrix_a[i, i] + b
            if y_pos == 0:    # u(i, 0, k) = u(i, 1, k): the front face
                matrix_a[i, i] = matrix_a[i, i] + b
            if z_pos == num_z - 1:    # u(i, j, num_z) = u(i, j, num_z - 1): the top face
                matrix_a[i, i] = matrix_a[i, i] + c

            # heat source
            if z_pos == 0:
                if (x_pos >= heat_start_pos_x and x_pos <= heat_stop_pos_x and
                        y_pos >= heat_start_pos_y and y_pos <= heat_stop_pos_y):
                    matrix_b[i, 0] = c
                else:
                    matrix_a[i, i] = matrix_a[i, i] + c

            # diffusion
            if x_pos == num_x - 1:
                matrix_a[i, i] = matrix_a[i, i] + a / (1 + self.heat_exchange_const * step_x)

        return self.diffusity_const * (matrix_a.tocsr()), self.diffusity_const * (matrix_b.tocsr())


def sim_odeint_sparse(sparse_a_matrix, init_vec, input_vec, step, num_steps):
    'use odeint and keep the A matrix sparse'

    num_dims = sparse_a_matrix.shape[0]
    times = np.linspace(0, step, num_steps)

    def der_func(state, _):
        'linear derivative function'

        rv = np.array(sparse_a_matrix * state) + input_vec
        rv.shape = (num_dims,)

        return rv

    start = time.time()
    result = odeint(der_func, init_vec, times, rtol=1e-13, atol=1e-13)
    runtime = time.time() - start

    return runtime, result
