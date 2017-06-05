'''
ODEINT based simulation

Stanley Bak
June 2016
'''

import time

from scipy.integrate import odeint

import numpy as np

def sim_odeint_sparse(sparse_a_matrix, init_vec, step):
    'use odeint and keep the A matrix sparse'

    num_dims = sparse_a_matrix.shape[0]
    times = np.linspace(0, step, num=2)

    def der_func(state, _):
        'linear derivative function'

        rv = np.array(sparse_a_matrix * state)
        rv.shape = (num_dims,)

        return rv

    start = time.time()
    result = odeint(der_func, init_vec, times)

    # don't include step 0
    result = result[1:]

    runtime = time.time() - start

    return runtime, result
