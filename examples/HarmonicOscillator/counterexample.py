"Counter-example trace generated using HyLAA"

import sys
from numpy import array, int32
from scipy.sparse import csc_matrix
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    data = array([-1.,  1.,  1.])
    indices = array([1, 0, 2], dtype=int32)
    indptr = array([0, 1, 2, 2, 3], dtype=int32)
    a_matrix = csc_matrix((data, indices, indptr), dtype=float, shape=(4, 4))
    b_matrix = None
    inputs = None

    step = 0.1
    max_time = 2.4

    start_point = array([-6.,  1.,  0.,  1.])
    normal_vec = array([-1.,  0.,  0.,  0.])
    normal_val = -5.0

    end_val = -5.0998254738
    sim_states, sim_times = check(a_matrix, b_matrix, step, max_time, start_point, inputs, normal_vec, end_val)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
