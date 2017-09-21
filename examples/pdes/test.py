"""This is for testing"""


import math
import matplotlib.pyplot as plt
import numpy as np
from pdes import HeatOneDimension, HeatTwoDimension1, HeatTwoDimension2, sim_odeint_sparse, HeatThreeDimension

from scipy.io import loadmat

def heat_1d():
    'test 1-d heat equation'
    len_x = 200
    diffusity_const = 1.16 # cm2/sec 
    thermal_cond = 0.93  #cal/cm-sec-degree
    heat_exchange_coeff = 1 
    he = HeatOneDimension(diffusity_const, diffusity_const, heat_exchange_coeff, len_x)
    num_x = 59 # number of meshpoint between 0 and len_x
    matrix_a, matrix_b = he.get_odes(num_x)
    print "\nmatrix_a:\n{}".format(matrix_a.toarray())
    print "\nmatrix_b:\n{}".format(matrix_b.toarray())

    init_vec = np.zeros((matrix_a.shape[0],))    
    input_g = 20 #

    input_vec = matrix_b*[input_g]
    final_time = 2000
    num_steps = 100000
    time_step = float(final_time)/float(num_steps)
    discretization_step = float(len_x)/float(num_x + 1)

    # stability condition for numerical method
    
    print "\nsimulation time step is: {}".format(time_step)
    print "\ndiscrezation step is: {}".format(discretization_step)

    print "\nthe stability condition for numberical method is satisfied: time_step <= 0.5*discrezation_step^2"
    if (time_step > discretization_step**2/2):
        raise ValueError("\nThe stability condition for numerical method is not satisfied")

    times = np.linspace(0, final_time, num_steps)
    runtime, result = sim_odeint_sparse(matrix_a, init_vec, input_vec, final_time, num_steps)

    # central point temperature
    #center_point_index = int(math.ceil((len_x/2)/discretization_step))
    center_point_index = int(math.ceil(num_x/2))
    center_point_temp = result[:, center_point_index]
    print "\n the central point index is: {}".format(center_point_index)
    print "\nthe center point position is: x = {}cm".format((center_point_index + 1)*discretization_step)

    plt.plot(times, center_point_temp, 'b', label='center_point')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

    

def heat_2d1():
    'test 2-dimensional heat-flow benchmark'
    # parameters
    diffusity_const = 1.16 # cm^2/sec
    heat_exchange_coeff = 1
    thermal_cond = 0.93  # cal/cm-sec-degree
    len_x = 100 # cm
    len_y = 100 # cm
    he = HeatTwoDimension1(diffusity_const, heat_exchange_coeff, thermal_cond, len_x, len_y)
     # get linear ode model of 2-d heat equation
    num_x = 3 # number of discretized step points between 0 and len_x
    num_y = 3 # number of discretized step points between 0 and len_y
    matrix_a, matrix_b = he.get_odes(num_x, num_y)
    print "\nmatrix_a :\n{}".format(matrix_a.todense())
    print "\nmatrix_b :\n{}".format(matrix_b.todense())

    # simulate linear ode model of 2-d heat equation
    n = matrix_a.shape[0]
    init_vec = np.zeros((n,))
    # Initial condition IC: u(x,y,0) = sin(pi*x/100), 0 <= x <= 100
    for i in xrange(0, n):
        pos_x = i%num_x
        init_vec[i] = math.sin(math.pi*float((pos_x+1)/(num_x+1)))
    print "\ninitial vector v = {}".format(init_vec)

    # input vector: f1 = 1, g1 = 1, g2 = 10
    f1 = 1
    g1 = 1
    g2 = 10
    v_vec = np.array([f1, g1, g2])
    input_vec = matrix_b*v_vec
    final_time = 10000
    num_steps = 1000000
    times = np.linspace(0, final_time, num_steps)
    runtime, result = sim_odeint_sparse(matrix_a, init_vec, input_vec, final_time, num_steps)

    print "\n the result is: \n{}".format(result)
    print "\n result shape is: \n{}".format(result.shape)

    # plot the center point temperature
    center_point_pos_x = int(math.ceil(num_x/2)) - 1
    center_point_pos_y = int(math.ceil(num_y/2)) - 1

    center_point_state_pos = center_point_pos_y*num_x + center_point_pos_x
    print "\ncenter_point corresponds to the {}-th state variable".format(center_point_state_pos)

    center_point_temp = result[:, center_point_state_pos]
    plt.plot(times, center_point_temp, 'b', label='center_point')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def heat_2d2():
    'test 2-d heat equation ZhiHan benchmark'

    # parameters
    diffusity_const = 0.01
    heat_exchange_coeff = 0.5
    thermal_cond = 1
    len_x = 1
    len_y = 1
    has_heat_source = True
    heat_source_pos = np.array([0, 0.4])
    he = HeatTwoDimension2(diffusity_const, heat_exchange_coeff, thermal_cond,\
                                   len_x, len_y, has_heat_source, heat_source_pos)

    # get linear ode model of 2-d heat equation
    num_x = 4 # number of discretized steps between 0 and len_x
    num_y = 4 # number of discretized steps between 0 and len_y
    matrix_a, matrix_b = he.get_odes(num_x, num_y)
    #print "\nmatrix_a :\n{}".format(matrix_a.todense())
    #print "\nmatrix_b :\n{}".format(matrix_b.todense())

    matlab_matrices = loadmat('20x20.mat')


    matlab_a = matlab_matrices['A']

    print matrix_a.todense()

    assert matlab_a.shape == matrix_a.shape

    for y in xrange(matrix_a.shape[0]):
        for x in xrange(matrix_a.shape[1]):
            val_matrix = matrix_a[y, x]
            val_matlab = matlab_a[y, x]

            if abs(val_matrix - val_matlab) > 1e-6:
                print "mismatch in element {},{}, matrix val = {}, matlab val = {}".format(y, x, val_matrix, val_matlab)
                exit(1)

    print "TODO: also add comparisons for B and c from matlab"

    #matlab_b = matlab_matrices['b']
    #matlab_c = matlab_matrices['c']

    # simulate the linear ode model of 2-d heat equation
    heat_source = 1 # the value of heat source is 1 degree celcius
    envi_temp = 0   # environment temperature is 0 degree celcius
    inputs = np.array([heat_source, envi_temp]) # input to linear ode model
    print "\ninputs to the odes including heat_source = {} and environment temperature = {}".\
      format(heat_source, envi_temp)

    input_vec = matrix_b*inputs
    print "\input vector v = matrix_b*inputs is: \n{}".format(input_vec)

    init_vec = np.zeros((matrix_a.shape[0]),)
    final_time = 200
    num_steps = 1000
    times = np.linspace(0, final_time, num_steps)
    runtime, result = sim_odeint_sparse(matrix_a, init_vec, input_vec, final_time, num_steps)

    print "\n the result is: \n{}".format(result)
    print "\n result shape is: \n{}".format(result.shape)

    # plot the result

    #plot the center point temperature
    center_point_pos_x = int(math.floor(num_x/2))
    center_point_pos_y = int(math.floor(num_y/2))

    center_point_state_pos = center_point_pos_y*num_x + center_point_pos_x
    print "\ncenter_point corresponds to the {}-th state variable".format(center_point_state_pos)

    center_point_temp = result[:, center_point_state_pos]
    plt.plot(times, center_point_temp, 'b', label='center_point')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def ZhiHan_benchmark():
    'produce again ZhiHan benchmark, find out that Zhi Han matlab code was wrong'

    
    # parameters
    diffusity_const = 0.01
    heat_exchange_coeff = 0.5
    thermal_cond = 1
    len_x = 1
    len_y = 1
    has_heat_source = True
    heat_source_pos = np.array([0, 0.4])
    he = HeatTwoDimension2(diffusity_const, heat_exchange_coeff, thermal_cond,\
                                   len_x, len_y, has_heat_source, heat_source_pos)

    # get linear ode model of 2-d heat equation
    num_x = 4 # number of discretized steps between 0 and len_x
    num_y = 4 # number of discretized steps between 0 and len_y
    matrix_a, matrix_b = he.get_odes(num_x, num_y)
    print "\nmatrix_a :\n{}".format(matrix_a)
    print "\nmatrix_b :\n{}".format(matrix_b)

    # simulate the linear ode model of 2-d heat equation
    heat_source = 1 # the value of heat source is 1 degree celcius
    envi_temp = 0   # environment temperature is 0 degree celcius
    inputs = np.array([heat_source, envi_temp]) # input to linear ode model
    print "\ninputs to the odes including heat_source = {} and environment temperature = {}".\
      format(heat_source, envi_temp)

    # matrix b
    A = matrix_a
    b = matrix_b*inputs
    c = np.zeros((1, A.shape[0]))
    
    center_point_pos_x = int(math.floor(num_x/2))
    center_point_pos_y = int(math.floor(num_y/2))

    center_point_state_pos = center_point_pos_y*num_x + center_point_pos_x
    print "\ncenter_point corresponds to the {}-th state variable".format(center_point_state_pos)

    c[0, center_point_state_pos] = 1

    print "\n A = {}".format(A)
    print "\n b = {}".format(b)
    print "\n c = {}".format(c)

def heat_3d():
    'heat 3d benchmark'

    diffusity_const = 0.01
    heat_exchange_const = 0.5
    len_x = 1
    len_y = 1
    len_z = 1
    
    heat_source_pos = np.array([[0.2, 0.4], [0.2, 0.4]])
    he = HeatThreeDimension(diffusity_const, heat_exchange_const, len_x, len_y, len_z, heat_source_pos)

    # get linear ode model of 2-d heat equation
    num_x = 4 # number of discretized steps between 0 and len_x
    num_y = 4 # number of discretized steps between 0 and len_y
    num_z = 4 # number of discretized steps between 0 and len_z
    matrix_a, matrix_b = he.get_odes(num_x, num_y, num_z)
    
    A = matrix_a
    b = matrix_b
    c = np.zeros((1, A.shape[0]))
    
    c_pos_x = int(math.floor(num_x/2))
    c_pos_y = int(math.floor(num_y/2))
    c_pos_z = int(math.floor(num_z/2))

    c_state_pos = c_pos_z*num_x*num_y + c_pos_y*num_x + c_pos_x
    print"\n--------------------"
    print "\ncenter_point corresponds to the {}-th state variable".format(c_state_pos)

    c[0, c_state_pos] = 1

    print"\n--------------------"

    print "\n A = {}".format(A)
    print "\n b = {}".format(b)
    print "\n c = {}".format(c)
    
    

if __name__ == '__main__':
    #heat_1d()  # benchmark from the book: "Partial differential equations for scientists and engineers", page 39 
    #heat_2d1() # benchmark from the same book, page 40.
    #heat_2d2() # Zhi Han benchmark in his thesis, page 68.
    #ZhiHan_benchmark()
    heat_3d() # 3-dimensional heat equation benchmark
