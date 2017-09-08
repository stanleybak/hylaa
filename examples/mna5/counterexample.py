"Counter-example trace generated using HyLAA"

import sys
from numpy import array, int32
from scipy.sparse import csc_matrix
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    data = array([  1.00000000e+00,   1.00000000e+00,   1.00000000e+00, ...,
        -3.05779570e-12,  -3.36357527e-11,   1.00000000e+00])
    indices = array([   18,    45,    19, ..., 10911, 10912, 10913], dtype=int32)
    indptr = array([    0,     2,     4, ..., 54159, 54159, 54160], dtype=int32)
    a_matrix = csc_matrix((data, indices, indptr), dtype=float, shape=(10915, 10915))
    data = array([-1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int16)
    indices = array([18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=int32)
    indptr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
    b_matrix = csc_matrix((data, indices, indptr), dtype=float, shape=(10915, 9))

    inputs = []
    inputs += [[0.10000000000000001, 0.10000000000000001, 0.10000000000000001, 0.10000000000000001, 0.10000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001]] * 384

    step = 0.005
    max_time = 1.92

    start_point = array([  2.00000000e-04,   2.00000000e-04,   2.00000000e-04, ...,
         0.00000000e+00,   0.00000000e+00,   1.00000000e+00])
    normal_vec = array([-1.,  0.,  0., ...,  0.,  0.,  0.])
    normal_val = -0.1

    end_val = -0.100041875033
    sim_states, sim_times = check(a_matrix, b_matrix, step, max_time, start_point, inputs, normal_vec, end_val)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
