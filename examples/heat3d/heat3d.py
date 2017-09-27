'''
3D Heat Equation Based on Tran's model
'''

import math

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, make_constraint_matrix, make_seperated_constraints
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha(samples_per_side):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')

    # parameters
    diffusity_const = 0.01
    heat_exchange_const = 0.5

    len_x = 1.0
    len_y = 1.0
    len_z = 1.0

    heat_source_pos = np.array([[0.0, 0.4], [0.0, 0.2]])
    he = HeatThreeDimension(diffusity_const, heat_exchange_const, len_x, len_y, len_z, heat_source_pos)

    print "Making {}x{}x{} ({} dims) 3d Heat Plate ODEs...".format(samples_per_side, samples_per_side, \
                                                                   samples_per_side, samples_per_side**3)
    a_matrix, b_matrix = he.get_odes(samples_per_side, samples_per_side, samples_per_side)
    print "Finished Making ODEs"

    a_matrix *= 500.0
    print "Matrix F-norm: {}".format( \
        sp.sparse.linalg.norm(a_matrix))
    #sym_a = (a_matrix + a_matrix.T) / 2.0
    #print "computed symmetrix part"
    #print "eig: {}".format(sp.sparse.linalg.eigs(sym_a, k=1, return_eigenvectors=False))
    print "largest SVD: {}".format(sp.sparse.linalg.svds(a_matrix, k=1, return_singular_vectors=False))
    exit(1)

    assert isinstance(a_matrix, csc_matrix)

    dims = a_matrix.shape[0]
    col_ptr = [n for n in a_matrix.indptr]
    rows = [n for n in a_matrix.indices]
    data = [n for n in a_matrix.data]

    b_matrix = b_matrix.transpose()[0, :]

    # add single input effects column
    for u in [0]:
        rows += [n for n in b_matrix[u, :].indices]
        data += [n for n in b_matrix[u, :].data]
        col_ptr.append(len(data))

    # nothing is a derivative of time, add a column of zeros
    col_ptr.append(len(data))

    # time variables is a derivative of the affine variable
    rows.append(dims + 1) # +0 = inputs, +1 = time var, +2 = affine var
    data.append(1.0) # t' = a
    col_ptr.append(len(data))

    combined_mat = csc_matrix((data, rows, col_ptr), \
                              shape=(a_matrix.shape[0] + 3, a_matrix.shape[1] + 3))

    mode.set_dynamics(csr_matrix(combined_mat))

    error = ha.new_mode('error')
    dims = combined_mat.shape[0]

    center_x = int(math.floor(samples_per_side/2.0))
    center_y = int(math.floor(samples_per_side/2.0))
    center_z = int(math.floor(samples_per_side/2.0))

    center_dim = center_z * samples_per_side * samples_per_side + center_y * samples_per_side + center_x

    # x_center >= 0.9
    mat = csr_matrix(([-1], [center_dim], [0, 1]), dtype=float, shape=(1, dims))
    #rhs = np.array([-0.5], dtype=float) # 0.5 = safe
    rhs = np.array([-0.4], dtype=float) # 0.4 = unsafe
    #trans1 = ha.new_transition(mode, error)
    #trans1.set_guard(mat, rhs)

    return ha

class HeatThreeDimension(object):
    '3-dimensional heat equation'

    def __init__(self, diffusity_const, heat_exchange_const, len_x, len_y, len_z, heat_source_pos):

        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0
        self.heat_exchange_const = heat_exchange_const if heat_exchange_const > 0 else 0
        self.len_x = len_x if len_x > 0 else 0
        self.len_y = len_y if len_y > 0 else 0
        self.len_z = len_z if len_z > 0 else 0

        if self.diffusity_const == 0 or self.heat_exchange_const == 0 or self.len_x == 0 \
          or self.len_y == 0 or self.len_z == 0:
            raise ValueError("inappropriate parameters")

        assert isinstance(heat_source_pos, np.ndarray)
        if heat_source_pos.shape != (2, 2):
            raise ValueError("heat source position should be 2 x 2 array")
        if heat_source_pos[0, 0] < 0 or heat_source_pos[0, 1] > self.len_x or heat_source_pos[1, 0] < 0 \
          or heat_source_pos[1, 1] > self.len_y:
            raise ValueError("heat source position should be inside the 3-d object")

        self.heat_source_pos = heat_source_pos

    def get_odes(self, num_x, num_y, num_z):
        'get the a and b matrix for the system discretized to the given level'

        assert num_x > 0 and num_y > 0 and num_z > 0, "number of mesh points should be large than zero"

        step_x = self.len_x/(num_x + 1)
        step_y = self.len_y/(num_y + 1)
        step_z = self.len_z/(num_z + 1)

        a = 1/step_x**2
        b = 1/step_y**2
        c = 1/step_z**2
        d = -2*(a + b + c)

        heat_start_pos_x = int(math.ceil(self.heat_source_pos[0, 0]/step_x)) - 1
        heat_stop_pos_x = int(math.floor(self.heat_source_pos[0, 1]/step_x)) - 1

        heat_start_pos_y = int(math.ceil(self.heat_source_pos[1, 0]/step_y)) - 1
        heat_stop_pos_y = int(math.floor(self.heat_source_pos[1, 1]/step_y)) - 1

        num_var = num_x*num_y*num_z

        matrix_a = lil_matrix((num_var, num_var))
        matrix_b = lil_matrix((num_var, 1))

        for i in xrange(0, num_var):
            z_pos = int(math.floor(i/num_x/num_y))
            y_pos = int(math.floor((i - z_pos*num_x*num_y)/num_x))
            x_pos = i - z_pos*num_x*num_y - y_pos*num_x

            matrix_a[i, i] = d # fill the diagonal

            if x_pos - 1 >= 0:
                matrix_a[i, i-1] = matrix_a[i, i-1] + a
            if x_pos + 1 <= num_x -1:
                matrix_a[i, i+1] = matrix_a[i, i+1] + a

            if y_pos - 1 >= 0:
                ind = z_pos*num_x*num_y + (y_pos - 1)*num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + b
            if y_pos + 1 <= num_y - 1:
                ind = z_pos*num_x*num_y + (y_pos + 1)*num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + b

            if z_pos - 1 >= 0:
                ind = (z_pos - 1)*num_x*num_y + y_pos*num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + c
            if z_pos + 1 <= num_z - 1:
                ind = (z_pos + 1)*num_x*num_y + y_pos*num_x + x_pos
                matrix_a[i, ind] = matrix_a[i, ind] + c

            # boundary conditions

            if x_pos == 0: #  u(0, j, k) = u(1, j, k): the left face
                matrix_a[i, i] = matrix_a[i, i] + a
            if y_pos == num_y - 1: # u(i, num_y, k) = u(i, num_y - 1, k): the back face
                matrix_a[i, i] = matrix_a[i, i] + b
            if y_pos == 0: # u(i, 0, k) = u(i, 1, k): the front face
                matrix_a[i, i] = matrix_a[i, i] + b
            if z_pos == num_z - 1: # u(i, j, num_z) = u(i, j, num_z - 1): the top face
                matrix_a[i, i] = matrix_a[i, i] + c

            # heat source
            if z_pos == 0:
                if (x_pos >= heat_start_pos_x) and (x_pos <= heat_stop_pos_x) and \
                  (y_pos >= heat_start_pos_y) and (y_pos <= heat_stop_pos_y):
                    matrix_b[i, 0] = c
                else:
                    matrix_a[i, i] = matrix_a[i, i] + c

            # diffusion
            if x_pos == num_x - 1:
                matrix_a[i, i] = matrix_a[i, i] + a/(1+self.heat_exchange_const*step_x)

        return self.diffusity_const*(matrix_a.tocsc()), self.diffusity_const*(matrix_b.tocsc())


def make_init_star(ha, hylaa_settings, samples):
    '''returns a Star'''

    n = ha.dims
    bounds_list = [] # bounds on each dimension

    samples_cubed = samples**3

    for dim in xrange(n):
        if dim < samples_cubed:
            lb = ub = 0.0
        elif dim == samples_cubed: # input term
            lb = 0.9
            ub = 1.1
        elif dim == samples_cubed + 1: # time term
            lb = ub = 0.0
        elif dim == samples_cubed + 2: # affine term
            lb = ub = 1
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    if not hylaa_settings.simulation.krylov_seperate_constant_vars or \
            hylaa_settings.simulation.sim_mode != SimulationSettings.KRYLOV:
        init_mat, init_rhs = make_constraint_matrix(bounds_list)
        rv = Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs)
    else:
        init_mat, init_rhs, variable_dim_list, fixed_dim_tuples = make_seperated_constraints(bounds_list)

        rv = Star(hylaa_settings, ha.modes['mode'], init_mat, init_rhs, \
                  var_lists=[variable_dim_list], fixed_tuples=fixed_dim_tuples)

    return rv

def define_settings(samples_per_side):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE

    settings = HylaaSettings(step=0.5, max_time=500.0, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.KRYLOV

    #settings.simulation.krylov_use_odeint = False
    #settings.simulation.check_answer = True
    #settings.simulation.krylov_use_gpu = True
    #settings.simulation.krylov_profiling = True
    settings.simulation.krylov_multithreaded_arnoldi_expm = False # use multiple threads to pipeline arnoldi and expm
    settings.simulation.krylov_multithreaded_rel_error = False # use multiple threads to pipeline rel_error calculation

    settings.simulation.krylov_rel_error = 1e-5

    center_x = int(math.floor(samples_per_side/2.0))
    center_y = int(math.floor(samples_per_side/2.0))
    center_z = int(math.floor(samples_per_side/2.0))

    center_dim = center_z * samples_per_side * samples_per_side + center_y * samples_per_side + center_x

    plot_settings.xdim_dir = samples_per_side**3 + 1 # +0 = input, +1 = time, +2 = affine
    plot_settings.ydim_dir = center_dim

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    samples_per_side = 40

    ha = define_ha(samples_per_side)
    settings = define_settings(samples_per_side)
    init = make_init_star(ha, settings, samples_per_side)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
