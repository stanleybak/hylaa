'''
Simulation Speed Comparison

Stanley Bak
June 2017
'''

import time
import sys

from scipy.io import loadmat
from scipy.sparse import csc_matrix, random as random_sparse_matrix

import numpy as np

from sim_expm import sim_sparse_expm, sim_expm_mult, sim_dense_expm
from sim_odeint import sim_odeint_sparse
from sim_krylov import sim_krylov_sparse

def load_model(model_name):
    'load a model from the corresponding .mat file in the current directory'

    print("Loading {}...".format(model_name)),
    sys.stdout.flush()
    start = time.time()

    if model_name == 'ha':
        # 2-d harmonic oscillator, this one is hardcoded
        dense_matrix = np.array([[0, 1], [-1, 0]], dtype=float)
        rv = csc_matrix(dense_matrix)
    elif model_name.startswith('rand'):
        # random sparse matrix with given size and density, such as 'rand_100_0.1'
        rv = make_random_sparse_matrix(model_name)
    else:
        filename = model_name + ".mat"

        data = loadmat(filename)
        rv = data['A']

    print("done ({:.2f} sec), Matrix Size = {:.3f}MB".format(time.time() - start, rv.data.nbytes / 1024.0 / 1024.0))

    return rv

def make_random_sparse_matrix(model_name):
    '''
    make a random sparse matrix from the model name. The name contains both the matrix size and density.'

    For example rand_1000_0.1 is a random 1000x1000 matrix with a density of 0.1
    '''

    _, dims, density = model_name.split('_')

    dims = int(dims)
    density = float(density)

    result = random_sparse_matrix(dims, dims, density=density, format='csc')

    return result

def make_alternating_init_vector(num_dims):
    'make an initial vector consisting of an alternating sequence: [0, 1, 0, 1, 0, ...]'

    rv = np.zeros((num_dims,), dtype=float)

    zero_flag = True

    for n in xrange(num_dims):
        if not zero_flag:
            rv[n] = 1.0

        zero_flag = not zero_flag

    return rv

def make_random_init_vector(num_dims):
    'make a random initial vector'

    rv = np.random.random((num_dims,))

    return rv

def main():
    'compare simulation methods runtime and accuracy, printing to stdout'

    models = []
    #models += ['ha']
    #models += ['building']
    #models += ['iss']
    #models += ['fom']
    #models += ['MNA5']

    for dims in [10000, 100000, 1000000]:
        model_name = "rand_{}_{}".format(dims, 10.0 / dims)
        models += [model_name]
    #

    method_tuples = []
    #method_tuples += [('Dense Expm', sim_dense_expm)]
    #method_tuples += [('Sparse Expm', sim_sparse_expm)]
    method_tuples += [('Expm_mult', sim_expm_mult)]
    method_tuples += [('Sparse odeint', sim_odeint_sparse)]
    method_tuples += [('Krylov', sim_krylov_sparse)]

    # used in the loop
    correct_result = None

    for model in models:
        print("\n--- {} ---".format(model))
        sparse_a_matrix = load_model(model)
        num_dims = sparse_a_matrix.shape[0]

        #init_vector = make_alternating_init_vector(num_dims)
        init_vector = make_random_init_vector(num_dims)

        step = 0.1

        for method_tuple in method_tuples:
            name, func = method_tuple

            runtime, result = func(sparse_a_matrix, init_vector, step)
            assert isinstance(runtime, float)
            assert isinstance(result, np.ndarray)

            if method_tuple is method_tuples[0]:
                correct_result = result
                abs_error = 0
                rel_error = 0
            else:
                diff = result - correct_result

                numerator = np.linalg.norm(diff, ord=2)
                denominator = np.linalg.norm(correct_result, ord=2)

                abs_error = numerator
                rel_error = numerator / denominator

            full_runtime_mins = num_dims * runtime / 60.0

            if full_runtime_mins < 1:
                full_runtime = "{:.2f} secs".format(runtime)
            elif full_runtime_mins < 60:
                full_runtime = "{:.2f} mins".format(full_runtime_mins)
            else:
                full_runtime = "{:.2f} hours".format(full_runtime_mins / 60.0)

            print("{}: {:.1f}ms (full {} dims estimate is {}), rel_err={:.3g}, abs_err={:.3g}".format(
                name, runtime * 1000.0, num_dims, full_runtime, rel_error, abs_error))

if __name__ == '__main__':
    main()
