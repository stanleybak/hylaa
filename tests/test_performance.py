'''
A set of performance measurements for various linear algebra operations. These tests print output.
You can run them one-by-one using: python -m unittest test_performance.TestPerformance.test_pdot
'''

import ctypes
import math
import time
import unittest
import multiprocessing
from multiprocessing.pool import ThreadPool

from numpy.ctypeslib import ndpointer
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, dia_matrix

from test_krylov import random_five_diag_sym_matrix

from np_dot_benchmark import get_num_threads

# used for the multi-threadded cases
global_thread_pool = ThreadPool(multiprocessing.cpu_count())

def pmult(a_mat, vec, force_parallel=False):
    'parallel dot product'

    global global_thread_pool
    assert isinstance(a_mat, csr_matrix)
    dims = a_mat.shape[0]

    # will change back after
    preshape = vec.shape
    vec.shape = (dims, 1)

    if a_mat.shape[0] < 150 and not force_parallel:
        # single-threaded is fast if dims < 150
        rv = a_mat * vec
    else:
        # multi-threaded version

        def mult_func((sub_mat, vec)):
            'multiplication of matrices, for parallel map'

            return sub_mat * vec

        split = multiprocessing.cpu_count()
        args = []

        for i in xrange(split):
            start_row = i * dims / split
            end_row = (i+1) * dims / split

            if start_row == end_row:
                continue

            # I think this way would make a copy of the data, which we want to avoid
            #sub_a_mat = a_mat[start_row:end_row, :]

            # construct a sub csr_matrix (no copy... hopefully)
            start = a_mat.indptr[start_row]
            end = a_mat.indptr[end_row]

            # I think sub_ind_ptr we have to copy
            sub_ind_ptr = a_mat.indptr[start_row:end_row+1].copy()

            for j in xrange(sub_ind_ptr.shape[0]):
                sub_ind_ptr[j] -= start
            
            sub_a_mat_tuple = (a_mat.data[start:end], a_mat.indices[start:end], sub_ind_ptr)
            sub_a_mat = csr_matrix(sub_a_mat_tuple, dtype=a_mat.dtype, shape=(end_row-start_row, a_mat.shape[1]))

            print "share mem? {}".format(np.may_share_memory(sub_a_mat.data, a_mat.data))

            args.append((sub_a_mat, vec))

        result = global_thread_pool.map(mult_func, args)

        rv = np.concatenate(result)

    rv.shape = (dims,)
    vec.shape = preshape

    return rv

def paxpy(a, alpha, b, force_parallel=False):
    'parallel version of: a += alpha * b'

    global global_thread_pool
    assert len(a.shape) == 1
    assert len(b.shape) == 1
    assert a.shape == b.shape

    if a.shape[0] < 150 and not force_parallel:
        # single-threaded is fast if dims < 150
        a += alpha * b
    else:
        # multi-threaded version

        def axpy_func((a, alpha, b)):
            'axpy function, for parallel map'

            a += alpha * b

        size = a.shape[0]
        split = multiprocessing.cpu_count()
        args = []

        for i in xrange(split):
            start_index = i * size / split
            end_index = (i+1) * size / split

            if start_index == end_index:
                continue

            args.append((a[start_index:end_index], alpha, b[start_index:end_index]))

        global_thread_pool.map(axpy_func, args)

def pdot(a, b, force_parallel=False):
    'parallel dot product'

    global global_thread_pool
    assert len(a.shape) == 1
    assert len(b.shape) == 1
    assert a.shape == b.shape

    if a.shape[0] < 150 and not force_parallel:
        # single-threaded is fast if dims < 150
        rv = np.dot(a, b)
    else:
        # multi-threaded version

        def mult_func((num1, num2)):
            'multiplication of matrices, for parallel map'

            return np.dot(num1, num2)

        size = a.shape[0]
        split = multiprocessing.cpu_count()
        args = []

        for i in xrange(split):
            start_index = i * size / split
            end_index = (i+1) * size / split

            if start_index == end_index:
                continue

            args.append((a[start_index:end_index], b[start_index:end_index]))

        result = global_thread_pool.map(mult_func, args)

        rv = sum(result)

    return rv

def heat3d_tran(samples_per_side, diffusity_const, heat_exchange_const):
    'return a and b matrix for the heat3d system'

    len_x = len_y = len_z = 1.0
    num_x = num_y = num_z = samples_per_side

    heat_source_pos = np.array([[0.0, 0.4], [0.0, 0.2]])

    assert isinstance(heat_source_pos, np.ndarray)
    if heat_source_pos.shape != (2, 2):
        raise ValueError("heat source position should be 2 x 2 array")

    assert num_x > 0 and num_y > 0 and num_z > 0, "number of mesh points should be large than zero"

    step_x = len_x/(num_x + 1)
    step_y = len_y/(num_y + 1)
    step_z = len_z/(num_z + 1)

    a = 1/step_x**2
    b = 1/step_y**2
    c = 1/step_z**2
    d = -2*(a + b + c)

    heat_start_pos_x = int(math.ceil(heat_source_pos[0, 0]/step_x)) - 1
    heat_stop_pos_x = int(math.floor(heat_source_pos[0, 1]/step_x)) - 1

    heat_start_pos_y = int(math.ceil(heat_source_pos[1, 0]/step_y)) - 1
    heat_stop_pos_y = int(math.floor(heat_source_pos[1, 1]/step_y)) - 1

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
            #if (x_pos >= heat_start_pos_x) and (x_pos <= heat_stop_pos_x) and \
            #  (y_pos >= heat_start_pos_y) and (y_pos <= heat_stop_pos_y):
            #    matrix_b[i, 0] = c
            #else:
            matrix_a[i, i] = matrix_a[i, i] + c

        # diffusion
        if x_pos == num_x - 1:
            matrix_a[i, i] = matrix_a[i, i] + a/(1+heat_exchange_const*step_x)

    return diffusity_const*(matrix_a.tocsc()), diffusity_const*(matrix_b.tocsc())

def heat3d_dia(samples, diffusity_const, heat_exchange_const):
    'fast dia_matrix construction for heat3d dynamics'

    samples_sq = samples**2
    dims = samples**3
    step = 1.0/(samples + 1)

    a = diffusity_const * 1.0 / step**2
    d = -2.0 * (a + a + a)

    data = np.zeros((7, dims), dtype=float)
    offsets = np.array([-samples_sq, -samples, -1, 0, 1, samples, samples_sq], dtype=float)

    # element with z = -1
    data[0, :-samples_sq] = a

    # element with y = -1
    for s in xrange(samples):
        start = s*samples_sq
        end = (s+1)*(samples_sq) - samples
        data[1, start:end] = a

    # element with x = -1
    for s in xrange(samples_sq):
        start = s*samples
        end = (s+1)*(samples) - 1
        data[2, start:end] = a

    #### diagonal element ####
    data[3, :] = d # (prefill)

    # adjust when z = 0 or z = samples-1
    data[3, :samples_sq] += a
    data[3, -samples_sq:] += a

    # adjust when y = 0 or y = samples-1
    for z in xrange(samples):
        z_offset = z * samples_sq

        data[3, z_offset:z_offset + samples] += a
        data[3, z_offset+samples_sq-samples:z_offset+samples_sq] += a

    # adjust when x = 0 (and add diffusion term when x = samples-1)
    for z in xrange(samples):
        for y in xrange(samples):
            offset = z * samples_sq + y * samples

            data[3, offset] += a

            data[3, offset + samples - 1] += a/(1+heat_exchange_const*step)

    #### end diagnal element ####
    # element with x = +1
    for s in xrange(samples_sq):
        start = 1 + s * samples
        end = (s+1) * samples
        data[4, start:end] = a

    # element with y = +1
    for s in xrange(samples):
        start = s*samples_sq+samples
        end = (s+1)*(samples_sq)
        data[5, start:end] = a

    # element with z = +1
    data[6, samples_sq:] = a

    rv = dia_matrix((data, offsets), shape=(dims, dims))
    assert np.may_share_memory(rv.data, data) # make sure we didn't copy memory

    return rv

def my_dia_mult(dia, vec):
    'multiply a diagonal matrix and vector manually'

    assert isinstance(dia, dia_matrix)

    result = np.zeros(vec.shape, dtype=float)

    for index in xrange(len(dia.offsets)):
        offset = dia.offsets[index]

        if offset == 0:
            result += vec * dia.data[index]
        elif offset < 0:
            result[:offset] += vec[-offset:] * dia.data[index, :offset]
        else:
            result[offset:] += vec[:-offset] * dia.data[index, offset:]

    #print "py result: ",

    #for num in result:
    #    print "{:.6f} ".format(num),

    #print ""

    return result

class TestPerformance(unittest.TestCase):
    'Performance unit tests'

    def setUp(self):
        np.random.seed(1)
        np.set_printoptions(suppress=True) # suppress floating point printing

    def test_heat3d_dia_mult(self):
        '''test multiplication using different versions of the heat3d matrix'''

        # load the c library
        lib = ctypes.CDLL('./test_performance_c.so')

        c_dia_mult = lib.diaMult
        c_dia_mult.restype = None

        #double* result, double* vec, int matW, int matH, double* data, int* offsets, int numOffsets
        c_dia_mult.argtypes = \
            [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
             ctypes.c_int, ctypes.c_int,\
             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

        diffusity_const = 0.01
        heat_exchange_const = 0.5
        samples = 50
        dims = samples**3

        print "making {} dim random vector".format(dims)
        vec = np.random.random_sample((dims,))

        print "Making dynamics with dia function"
        start = time.time()
        a_mat_dia = heat3d_dia(samples, diffusity_const, heat_exchange_const)
        print "Dia construct time {:.3f}s".format(time.time() - start)

        ##################

        print "C-based multiplication"
        start = time.time()
        splits = multiprocessing.cpu_count() #
        result_c = np.zeros((dims,), dtype=float)

        for _ in xrange(10):
            c_dia_mult(result_c, vec, dims, dims, a_mat_dia.data, a_mat_dia.offsets, len(a_mat_dia.offsets), splits)
            
        print "C-based time {:.3f}s".format(time.time() - start)

        ##################

        print "Multiplying dia"
        start = time.time()

        for _ in xrange(10):
            result_dia = a_mat_dia * vec
            
        print "Dia mult time {:.3f}s".format(time.time() - start)

        #print "my_dia_mult with diagonal matrix"
        #start = time.time()
        #result_python = my_dia_mult(a_mat_dia, vec)
        #print "my_dia_mult time {:.3f}s".format(time.time() - start)

        # check result matches
        #self.assertTrue(np.allclose(result_dia, result_python))
        self.assertTrue(np.allclose(result_dia, result_c))
        

    def test_heat3d_dia_vs_csr(self):
        '''test multiplication using different versions of the heat3d matrix'''

        diffusity_const = 0.01
        heat_exchange_const = 0.5
        samples = 320
        dims = samples**3

        print "making {} dim random vector".format(dims)
        vec = np.random.random_sample((dims,))

        print "Making dynamics with dia function"
        start = time.time()
        a_mat_dia = heat3d_dia(samples, diffusity_const, heat_exchange_const)
        print "Dia construct time {:.3f}s".format(time.time() - start)

        print "Creating csr version"
        start = time.time()
        a_mat_csr = csr_matrix(a_mat_dia)
        print "Csr construct time {:.3f}s".format(time.time() - start)

        ##################

        print "Multiplying dia"
        start = time.time()
        result_dia = a_mat_dia * vec
        print "Dia mult time {:.3f}s".format(time.time() - start)

        print "Multiplying csr"
        start = time.time()
        result_csr = a_mat_csr * vec
        print "Csr mult time {:.3f}s".format(time.time() - start)

        # check result matches
        self.assertTrue(np.allclose(result_dia, result_csr))

    def test_heat3d_make(self):
        '''test making heat3d orignal vs diag_matrix'''

        diffusity_const = 0.01
        heat_exchange_const = 0.5
        samples = 20 #563 = limit on laptop (takes 3 secs) ~170 million dims

        print "Making {} dimensional dynamics with dia function".format(samples**3)
        start = time.time()
        a_mat_dia = heat3d_dia(samples, diffusity_const, heat_exchange_const)
        print "Dia construct time {:.3f}s".format(time.time() - start)

        print "Making dynamics with tran function"
        start = time.time()
        a_mat_tran, _ = heat3d_tran(samples, diffusity_const, heat_exchange_const)
        print "Tran construct time {:.3f}s".format(time.time() - start)

        print "Checking matrix equality..."
        a_mat_tran = dia_matrix(a_mat_tran)

        self.assertEqual(a_mat_tran.shape, a_mat_dia.shape)
        assert np.allclose(a_mat_dia.data, a_mat_tran.data)
        assert np.allclose(a_mat_dia.offsets, a_mat_tran.offsets)

    def test_norm(self):
        '''compare numpy's norm and norm(2)'''

        dims = int(5e7)
        vec = np.random.random_sample((1, dims))

        start = time.time()
        norm = np.linalg.norm(vec)
        print "norm(vec) time: {:.2}s".format(time.time() - start)

        start = time.time()
        norm2 = np.linalg.norm(vec, 2)
        print "norm(vec, 2) time: {:.2}s".format(time.time() - start)

        self.assertTrue(np.allclose(norm, norm2))

    def test_pmult(self):
        'test parallel csr matrix mult'

        dims = int(1e6) # 1e7, par = 0.4, serial = 0.1

        a_matrix = random_five_diag_sym_matrix(dims, True)

        start = time.time()
        vec = np.random.random_sample((dims,))
        print "allocated vector in {}s".format(time.time() - start)

        start = time.time()
        res_parallel = pmult(a_matrix, vec, force_parallel=True)
        print "parallel time = {}s".format(time.time() - start)

        start = time.time()
        vec.shape = (dims, 1)
        res_serial = a_matrix * vec
        res_serial.shape = (dims,)
        print "serial time = {}s".format(time.time() - start)

        self.assertTrue(np.allclose(res_parallel, res_serial))

    def test_paxpy(self):
        'test parallel axpy function'

        dims = int(1e8)

        start = time.time()
        a = np.random.random_sample((dims,))
        print "allocated first vector in {}s".format(time.time() - start)

        start = time.time()
        par_a = a.copy()
        print "copied first vector in {}s".format(time.time() - start)

        start = time.time()
        b = np.random.random_sample((dims,))
        print "allocated second vector in {}s".format(time.time() - start)

        start = time.time()
        paxpy(par_a, -0.5, b, force_parallel=True)
        print "parallel time = {}s".format(time.time() - start)

        start = time.time()
        a += -0.5 * b
        print "serial time = {}s".format(time.time() - start)

        self.assertTrue(np.allclose(par_a, a))

    def test_pdot(self):
        'test parallel dot function'

        print "openblas threads = {}".format(get_num_threads())

        dims = int(5e8)

        start = time.time()
        
        #a = np.zeros((dims,), dtype=float)
        #for i in xrange(dims):
        #    a[i] = random.random()
        a = np.random.random_sample((dims,))
        print "allocated first vector in {}s".format(time.time() - start)

        start = time.time()
        b = np.random.random_sample((dims,))
        print "allocated second vector in {}s".format(time.time() - start)

        start = time.time()
        res_parallel = pdot(a, b, force_parallel=True)
        print "parallel time = {}s".format(time.time() - start)

        start = time.time()
        res_serial = np.dot(a, b)
        print "serial time = {}s".format(time.time() - start)

        self.assertTrue(np.allclose(res_parallel, res_serial))
