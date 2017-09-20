'''
2D Heat Equation Based on Tran's model
'''

import math

import numpy as np
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
    heat_exchange_coeff = 0.5
    thermal_cond = 1.0

    len_x = 1.0
    len_y = 1.0
    has_heat_source = True
    heat_source_pos = np.array([0.0, 0.4], dtype=float)
    he = HeatTwoDimension2(diffusity_const, heat_exchange_coeff, thermal_cond, \
                                   len_x, len_y, has_heat_source, heat_source_pos)

    print "Making {}x{} 2d Heat Plate ODEs...".format(samples_per_side, samples_per_side)
    a_matrix, b_matrix = he.get_odes(samples_per_side, samples_per_side)
    print "Finished Making ODEs"

    assert isinstance(a_matrix, csc_matrix)

    dims = a_matrix.shape[0]
    col_ptr = [n for n in a_matrix.indptr]
    rows = [n for n in a_matrix.indices]
    data = [n for n in a_matrix.data]

    b_matrix = b_matrix.transpose()[0, :]

    # add single input effects column
    for u in xrange(1):
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

    center_x = int(math.floor(samples_per_side / 2.0))
    center_y = int(math.floor(samples_per_side / 2.0))
    center_dim = center_y * samples_per_side + center_x

    # x_center >= 0.9
    mat = csr_matrix(([-1], [center_dim], [0, 1]), dtype=float, shape=(1, dims))
    #rhs = np.array([-0.9], dtype=float) # 0.9 = safe
    rhs = np.array([-0.8], dtype=float) # 0.8 = unsafe
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)

    return ha

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

    def __init__(self, diffusity_const, heat_exchange_coeff, thermal_cond, \
                     len_x, len_y, has_heat_source, heat_source_pos):
        self.diffusity_const = diffusity_const if diffusity_const > 0 else 0 # diffusity constant
        self.heat_exchange_coeff = heat_exchange_coeff if heat_exchange_coeff > 0 else 0 # heat exchange coefficient
        self.thermal_cond = thermal_cond if thermal_cond > 0 else 0 # thermal conductivity
        self.len_x = len_x if len_x > 0 else 0 # length x
        self.len_y = len_y if len_y > 0 else 0 # length y

        if self.diffusity_const == 0 or self.heat_exchange_coeff == 0 or \
          self.thermal_cond == 0 or  self.len_x == 0 or self.len_y == 0:
            raise ValueError("inappropriate parameters")
        self.heat_lost_const = self.heat_exchange_coeff/self.thermal_cond

        self.has_heat_source = has_heat_source
        assert isinstance(heat_source_pos, np.ndarray), "heat source pos is not an ndarray"
        if heat_source_pos.shape != (2,):
            raise ValueError("heat source position is not 2x1 array")
        if heat_source_pos[0] < 0 or heat_source_pos[1] > self.len_x:
            raise ValueError("Heat source position value error")
        self.heat_source_pos = heat_source_pos # an array to indicate the position of heat source
        # heat_source_pos = ([[x_start, x_end])

    def get_odes(self, num_x, num_y):
        'obtain the linear model of 2-d heat equation'

        assert isinstance(num_x, int), "number of mesh point should be an integer"
        assert isinstance(num_y, int), "number of messh point should be an integer"

        if num_x <= 0 or num_y <= 0:
            raise ValueError('number of mesh points should be larger than zero')

        disc_step_x = self.len_x/(num_x+1) # dicrezation step along x axis
        disc_step_y = self.len_y/(num_y+1) # discrezation step along y axis

        if self.has_heat_source:
            heat_start_pos_x = int(math.floor(self.heat_source_pos[0]/disc_step_x))
            heat_end_pos_x = int(math.floor(self.heat_source_pos[1]/disc_step_x))

        # we use explicit semi- finite-difference method to obtain the
        # linear model of heat equation

        num_var = num_x*num_y # number of discrezation variables
        # changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient
        matrix_a = lil_matrix((num_var, num_var))
        matrix_b = lil_matrix((num_var, 2))
        a = 1/disc_step_x**2
        b = 1/disc_step_y**2
        c = -2*(a+b)
        k = self.heat_lost_const
        step_x = disc_step_x


        for i in xrange(0, num_var):
            matrix_a[i, i] = c # filling diagonal
            x_pos = i%num_x # x-position corresponding to i-th state variable
            y_pos = int((i - x_pos)/num_x) # y-position corresponding to i-th variable

            # fill along x - axis
            if x_pos - 1 >= 0:
                matrix_a[i, i-1] = a
            else:
                matrix_a[i, i] = matrix_a[i, i] + a

            if x_pos + 1 <= num_x -1:
                matrix_a[i, i+1] = a
            else:
                # fill diffusion term
                matrix_a[i, i] = matrix_a[i, i] + a/(1+k*step_x)
                matrix_b[i, 1] = a*(k*step_x)/(1+k*step_x)

            # fill along y-axis
            if y_pos - 1 >= 0:
                matrix_a[i, (y_pos-1)*num_x + x_pos] = b
            else:
                if self.has_heat_source and x_pos >= heat_start_pos_x and x_pos <= heat_end_pos_x:
                    matrix_b[i, 0] = b
                else:
                    matrix_a[i, i] = matrix_a[i, i] + b

            if y_pos + 1 <= num_y - 1:
                matrix_a[i, (y_pos+1)*num_x + x_pos] = b
            else:
                matrix_a[i, i] = matrix_a[i, i] + b

        return self.diffusity_const*(matrix_a.tocsc()), self.diffusity_const*(matrix_b.tocsc())


def make_init_star(ha, hylaa_settings, samples):
    '''returns a Star'''

    n = ha.dims
    bounds_list = [] # bounds on each dimension

    samples_sq = samples*samples

    for dim in xrange(n):
        if dim < samples_sq:
            lb = ub = 0.0
        elif dim == samples_sq: # input term
            lb = 0.9
            ub = 1.1
        elif dim == samples_sq + 1: # time term
            lb = ub = 0.0
        elif dim == samples_sq + 2: # affine term
            lb = ub = 1
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    if not hylaa_settings.simulation.seperate_constant_vars or \
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
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    settings = HylaaSettings(step=0.2, max_time=200.0, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.KRYLOV

    #settings.simulation.check_answer = True
    settings.simulation.krylov_use_gpu = True
    settings.simulation.krylov_profiling = True

    center_x = int(math.floor(samples_per_side/2.0))
    center_y = int(math.floor(samples_per_side/2.0))
    center_dim = center_y * samples_per_side + center_x

    plot_settings.xdim_dir = samples_per_side * samples_per_side + 1 # +0 = input, +1 = time, +2 = affine
    plot_settings.ydim_dir = center_dim

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    samples_per_side = 200

    ha = define_ha(samples_per_side)
    settings = define_settings(samples_per_side)
    init = make_init_star(ha, settings, samples_per_side)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
