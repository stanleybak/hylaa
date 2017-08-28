'''
Unit tests for Hylass's gpu_interface.py
Stanley Bak
August 2017
'''

import unittest
import random
import math
import time

from hylaa.krylov_interface import KrylovInterface
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def make_iss_matrix(num_copies):
    'create a matrix from the international space station system model'

    orig_dims = 271
    inds = [
        135, 407, 679, 951, 1223, 1495, 1767, 2039, 2311, 2583, 2855, 3127,
        3399, 3671, 3943, 4215, 4487, 4759, 5031, 5303, 5575, 5847, 6119, 6391,
        6663, 6935, 7207, 7479, 7751, 8023, 8295, 8567, 8839, 9111, 9383, 9655,
        9927, 10199, 10471, 10743, 11015, 11287, 11559, 11831, 12103, 12375,
        12647, 12919, 13191, 13463, 13735, 14007, 14279, 14551, 14823, 15095,
        15367, 15639, 15911, 16183, 16455, 16727, 16999, 17271, 17543, 17815,
        18087, 18359, 18631, 18903, 19175, 19447, 19719, 19991, 20263, 20535,
        20807, 21079, 21351, 21623, 21895, 22167, 22439, 22711, 22983, 23255,
        23527, 23799, 24071, 24343, 24615, 24887, 25159, 25431, 25703, 25975,
        26247, 26519, 26791, 27063, 27335, 27607, 27879, 28151, 28423, 28695,
        28967, 29239, 29511, 29783, 30055, 30327, 30599, 30871, 31143, 31415,
        31687, 31959, 32231, 32503, 32775, 33047, 33319, 33591, 33863, 34135,
        34407, 34679, 34951, 35223, 35495, 35767, 36039, 36311, 36583, 36585,
        36720, 36857, 36992, 37129, 37264, 37401, 37536, 37673, 37808, 37945,
        38080, 38217, 38352, 38489, 38624, 38761, 38896, 39033, 39168, 39305,
        39440, 39577, 39712, 39849, 39984, 40121, 40256, 40393, 40528, 40665,
        40800, 40937, 41072, 41209, 41344, 41481, 41616, 41753, 41888, 42025,
        42160, 42297, 42432, 42569, 42704, 42841, 42976, 43113, 43248, 43385,
        43520, 43657, 43792, 43929, 44064, 44201, 44336, 44473, 44608, 44745,
        44880, 45017, 45152, 45289, 45424, 45561, 45696, 45833, 45968, 46105,
        46240, 46377, 46512, 46649, 46784, 46921, 47056, 47193, 47328, 47465,
        47600, 47737, 47872, 48009, 48144, 48281, 48416, 48553, 48688, 48825,
        48960, 49097, 49232, 49369, 49504, 49641, 49776, 49913, 50048, 50185,
        50320, 50457, 50592, 50729, 50864, 51001, 51136, 51273, 51408, 51545,
        51680, 51817, 51952, 52089, 52224, 52361, 52496, 52633, 52768, 52905,
        53040, 53177, 53312, 53449, 53584, 53721, 53856, 53993, 54128, 54265,
        54400, 54537, 54672, 54809, 54944, 55081, 55216, 55353, 55488, 55625,
        55760, 55897, 56032, 56169, 56304, 56441, 56576, 56713, 56848, 56985,
        57120, 57257, 57392, 57529, 57664, 57801, 57936, 58073, 58208, 58345,
        58480, 58617, 58752, 58889, 59024, 59161, 59296, 59433, 59568, 59705,
        59840, 59977, 60112, 60249, 60384, 60521, 60656, 60793, 60928, 61065,
        61200, 61337, 61472, 61609, 61744, 61881, 62016, 62153, 62288, 62425,
        62560, 62697, 62832, 62969, 63104, 63241, 63376, 63513, 63648, 63785,
        63920, 64057, 64192, 64329, 64464, 64601, 64736, 64873, 65008, 65145,
        65280, 65417, 65552, 65689, 65824, 65961, 66096, 66233, 66368, 66505,
        66640, 66777, 66912, 67049, 67184, 67321, 67456, 67593, 67728, 67865,
        68000, 68137, 68272, 68409, 68544, 68681, 68816, 68953, 69088, 69225,
        69360, 69497, 69632, 69769, 69904, 70041, 70176, 70313, 70448, 70585,
        70720, 70857, 70992, 71129, 71264, 71401, 71536, 71673, 71808, 71945,
        72080, 72217, 72352, 72489, 72624, 72761, 72896, 73033, 73168
    ]

    data = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -0.3887, -0.0062346,
        -0.60078, -0.007751, -1.9781, -0.014065, -1.9785, -0.014066, -3.1943,
        -0.017872, -3.9682, -0.01992, -5.2756, -0.022969, -5.3176, -0.02306,
        -6.1969, -0.024894, -6.2041, -0.024908, -6.6376, -0.025763, -6.643,
        -0.025774, -14.915, -0.03862, -15.321, -0.039142, -28.924, -0.053781,
        -28.924, -0.053781, -31.36, -0.056, -31.666, -0.056273, -34.377,
        -0.058632, -34.377, -0.058632, -37.182, -0.060977, -37.229, -0.061016,
        -62.94, -0.079335, -65.885, -0.08117, -65.946, -0.081207, -71.925,
        -0.084809, -84.946, -0.092166, -85.054, -0.092225, -85.262, -0.092337,
        -91.273, -0.095537, -91.283, -0.095542, -95.44, -0.097693, -95.44,
        -0.097693, -95.457, -0.097702, -95.457, -0.097702, -95.492, -0.09772,
        -95.492, -0.09772, -105.24, -0.10258, -105.32, -0.10262, -115.82,
        -0.10762, -116.34, -0.10786, -178.25, -0.13351, -178.73, -0.13369,
        -233.48, -0.1528, -305.2, -0.1747, -306.11, -0.17496, -336, -0.1833,
        -338.12, -0.18388, -437.91, -0.20926, -444.23, -0.21077, -467.28,
        -0.21617, -468.98, -0.21656, -505.12, -0.22475, -505.14, -0.22475,
        -730.75, -0.27032, -730.76, -0.27033, -749.99, -0.27386, -749.99,
        -0.27386, -831.1, -0.28829, -831.1, -0.28829, -908.85, -0.30147,
        -909.06, -0.30151, -934.4, -0.30568, -934.43, -0.30568, -1013.9,
        -0.31842, -1013.9, -0.31842, -1089.9, -0.33013, -1089.9, -0.33013,
        -1122.1, -0.33498, -1122.1, -0.33498, -1147.7, -0.33878, -1147.7,
        -0.33878, -1147.9, -0.33881, -1147.9, -0.33881, -1148.2, -0.33886,
        -1148.2, -0.33886, -1219.6, -0.34923, -1230.4, -0.35077, -1442.9,
        -0.37986, -1532.3, -0.39144, -1532.6, -0.39148, -1608.8, -0.4011,
        -1609.1, -0.40114, -1752, -0.41857, -1752.2, -0.41859, -1782.2,
        -0.42216, -1782.2, -0.42216, -1808.4, -0.42525, -1809, -0.42532,
        -1846.1, -0.42967, -1846.1, -0.42967, -1910.9, -0.43714, -1910.9,
        -0.43714, -1948.5, -0.44142, -1948.5, -0.44142, -1949, -0.44147, -1949,
        -0.44147, -2050.7, -0.45285, -2050.8, -0.45285, -2091.3, -0.45731,
        -2091.4, -0.45732, -2180.6, -0.46697, -2180.8, -0.46699, -2191,
        -0.46808, -2196.9, -0.46871, -2288.2, -0.47835, -2302.1, -0.4798,
        -2575.4, -0.50748, -2576.1, -0.50756, -2757.5, -0.52512, -2758.1,
        -0.52518, -2772.3, -0.52653, -2797.8, -0.52894, -2926.6, -0.54098,
        -2926.8, -0.541, -3146.5, -0.56094, -3150.8, -0.56132, -3212.8,
        -0.56682, -3213.4, -0.56687, -3281.1, -0.57281, -3291.2, -0.57369,
        -3291.4, -0.5737, -3427.6, -0.58546, -3427.6, -0.58546, -3428.8,
        -0.58556, -3428.8, -0.58556, -3429.3, -0.5856, -3429.3, -0.5856,
        -3448.2, -0.58721, -3448.2, -0.58721, -3449, -0.58729, -3449, -0.58729,
        -3452.3, -0.58757, -3452.3, -0.58757, -3762.6, -0.6134
    ]

    total_dims = num_copies * orig_dims

    rows = []
    cols = []
    vals = []

    for copy in xrange(num_copies):
        offset = orig_dims * copy

        for i in xrange(0, len(inds)):
            row = offset + inds[i] / orig_dims
            col = offset + inds[i] % orig_dims
            val = data[i]

            rows.append(row)
            cols.append(col)
            vals.append(val)

    mat = coo_matrix((vals, (rows, cols)), shape=(total_dims, total_dims))

    return csr_matrix(mat)


def random_sparse_matrix(dims, entries_per_row, random_cols=True, print_progress=False):
    'make a random sparse matrix with the given number of entries per row'

    row_inds = []
    cols = []
    vals = []

    start = last_print = time.time()

    for row in xrange(dims):
        row_inds.append(len(vals))

        if print_progress and row % 10000 == 0 and time.time() - last_print > 1.0:
            last_print = time.time()
            elapsed = last_print - start
            print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s".format(row, dims, 100.0 * row / dims, elapsed)

        for entry_index in xrange(entries_per_row):

            if random_cols:
                r = random.random() * dims
                col = int(math.floor(r))
                cols.append(col)
            else:
                cols.append(entry_index)

            vals.append(random.random())

    row_inds.append(len(vals))

    if print_progress:
        elapsed = last_print - start
        print "Row {} / {} ({:.2f}%). Elapsed: {:.1f}s".format(dims, dims, 100.0, elapsed)

    start = time.time()
    rv = csr_matrix((vals, cols, row_inds), shape=(dims, dims))

    if print_progress:
        print "making csr_matrix time {:.1f}s".format(time.time() - start)

    return rv

def arnoldi(mat, vec, iterations):
    'arnoldi for a single initial vector'

    dims = mat.shape[0]
    assert vec.shape == (dims, 1), "vec.shape was {}, expected (1, {})".format(vec.shape, dims)

    vec = vec.transpose()

    v_mat, h_mat = arnoldi_parallel(mat.T, vec, iterations)

    v_mat.shape = (iterations + 1, dims)
    h_mat.shape = (iterations, iterations + 1)

    return v_mat.transpose(), h_mat.transpose()


def arnoldi_parallel(mat_transpose, vecs, iterations):
    'arnoldi with split multiple initial vecs'

    num_init = vecs.shape[0]
    size = vecs.shape[1]

    prev_v = np.zeros((num_init, (iterations + 1) * size))
    h_mat = np.zeros((num_init, (iterations + 1) * (iterations)))

    for c in xrange(num_init):
        vec = vecs[c, :]
        vec = vec / np.linalg.norm(vec)
        prev_v[c, 0:size] = vec

    # use a-transpose
    #a_transpose = mat.T.copy()

    for cur_it in xrange(1, iterations + 1):
        # do all the multiplications up front
        for cur_vec in xrange(num_init):
            #vec = np.dot(prev_v[cur_vec, (cur_it-1)*size:cur_it*size], a_transpose)

            vec = prev_v[cur_vec, (cur_it-1)*size:cur_it*size] * mat_transpose

            prev_v[cur_vec, cur_it*size:(cur_it+1)*size] = vec

        for cur_vec in xrange(num_init):
            vec = prev_v[cur_vec, cur_it*size:(cur_it+1)*size]

            prev_mat = prev_v[cur_vec, 0:cur_it*size]
            prev_mat.shape = (cur_it, size)

            dots = h_mat[cur_vec][(iterations + 1) * (cur_it-1):(iterations + 1) * (cur_it-1) + cur_it]
            dots[:] = np.dot(prev_mat, vec.T)

            sub_vecs = np.dot(np.diag(dots), prev_mat)

            for c in xrange(cur_it):
                vec -= sub_vecs[c]

            norm = np.linalg.norm(vec)

            h_mat[cur_vec][cur_it + (iterations+1) * (cur_it-1)] = norm

            if norm >= 1e-6:
                vec = vec / norm
                prev_v[cur_vec, cur_it*size:(cur_it+1)*size] = vec

    return prev_v, h_mat

class TestKrylovInterface(unittest.TestCase):
    'Unit tests for krylov utilities'

    def setUp(self):
        'test setup'

        random.seed(1)
        KrylovInterface.reset()
        

    def test_arnoldi_single(self):
        'compare the python implementation with the cusp implementation with a single initial vector'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 1

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)

        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec = np.array([[1.0] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        v_mat_testing, h_mat_testing = arnoldi(a_matrix, init_vec, iterations)

        projected_v_mat_testing = key_dir_mat * v_mat_testing

        # using cusp

        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        
        result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing), "Correct projected v matrix")

    def test_arnoldi_offset(self):
        'compare the python implementation with the cusp implementation with a single initial vector (2nd column)'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 1

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)

        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec = np.array([[1.0] if d == 1 else [0.0] for d in xrange(dims)], dtype=float)
        v_mat_testing, h_mat_testing = arnoldi(a_matrix, init_vec, iterations)

        projected_v_mat_testing = key_dir_mat * v_mat_testing

        # using cusp

        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        
        result_h, result_pv = KrylovInterface.arnoldi_parallel(1)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing), "Correct projected v matrix")

    def test_arnoldi_off_end(self):
        'test arnodli when doing 2 parallel vectors at a time when straddling the end (only 1 should be done)'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec4 = np.array([[1.0] if d == 4 else [0.0] for d in xrange(dims)], dtype=float)

        v_mat_testing4, h_mat_testing4 = arnoldi(a_matrix, init_vec4, iterations)
        projected_v_mat_testing4 = key_dir_mat * v_mat_testing4

        # using cusp
        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        
        result_h, result_pv = KrylovInterface.arnoldi_parallel(4)

        self.assertEqual(len(result_h), len(result_pv))
        self.assertEqual(len(result_h), 1)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing4), "Correct h matrix")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing4), "Correct projV matrix")

    def test_arnoldi_double(self):
        'compare the cusp implementation with a two initial vectors versus the python implementation'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        # using python
        init_vec1 = np.array([[1.0] if d == 0 else [0.0] for d in xrange(dims)], dtype=float)
        init_vec2 = np.array([[1.0] if d == 1 else [0.0] for d in xrange(dims)], dtype=float)

        v_mat_testing1, h_mat_testing1 = arnoldi(a_matrix, init_vec1, iterations)
        projected_v_mat_testing1 = key_dir_mat * v_mat_testing1

        v_mat_testing2, h_mat_testing2 = arnoldi(a_matrix, init_vec2, iterations)
        projected_v_mat_testing2 = key_dir_mat * v_mat_testing2

        # using cusp
        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        
        result_h, result_pv = KrylovInterface.arnoldi_parallel(0)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing1), "Correct h matrix init vec 1")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing1), "Correct projV matrix for init vec 1")

        self.assertTrue(np.allclose(result_h[1], h_mat_testing2), "Correct h matrix init vec 2")
        self.assertTrue(np.allclose(result_pv[1], projected_v_mat_testing2), "Correct projV matrix for init vec 2")

    def test_iss(self):
        'test the cusp implementation using the iss model'

        #KrylovInterface.set_use_profiling(True)
        #KrylovInterface.set_use_gpu(True)

        iterations = 10
        num_parallel = 3

        a_matrix = make_iss_matrix(1)
        dims = a_matrix.shape[0]

        dir1 = np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float)
        dir2 = np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float)
        key_dir_mat = csr_matrix([dir1, dir2])

        # use initial dimensions 100 and 101 and 102

        # using python
        init_vec1 = np.array([[1.0] if d == 100 else [0.0] for d in xrange(dims)], dtype=float)
        init_vec2 = np.array([[1.0] if d == 101 else [0.0] for d in xrange(dims)], dtype=float)
        init_vec3 = np.array([[1.0] if d == 102 else [0.0] for d in xrange(dims)], dtype=float)

        v_mat_testing1, h_mat_testing1 = arnoldi(a_matrix, init_vec1, iterations)
        projected_v_mat_testing1 = key_dir_mat * v_mat_testing1

        v_mat_testing2, h_mat_testing2 = arnoldi(a_matrix, init_vec2, iterations)
        projected_v_mat_testing2 = key_dir_mat * v_mat_testing2
        
        v_mat_testing3, h_mat_testing3 = arnoldi(a_matrix, init_vec3, iterations)
        projected_v_mat_testing3 = key_dir_mat * v_mat_testing3

        # using cusp
        KrylovInterface.preallocate_memory(iterations, num_parallel, dims, 2)
        KrylovInterface.load_a_matrix(a_matrix)
        KrylovInterface.load_key_dir_matrix(key_dir_mat)
        
        result_h, result_pv = KrylovInterface.arnoldi_parallel(100)

        self.assertTrue(np.allclose(result_h[0], h_mat_testing1), "Correct h matrix init vec 100")
        self.assertTrue(np.allclose(result_pv[0], projected_v_mat_testing1), "Correct projV matrix for init vec 100")

        self.assertTrue(np.allclose(result_h[1], h_mat_testing2), "Correct h matrix init vec 101")
        self.assertTrue(np.allclose(result_pv[1], projected_v_mat_testing2), "Correct projV matrix for init vec 101")

        self.assertTrue(np.allclose(result_h[2], h_mat_testing3), "Correct h matrix init vec 102")
        self.assertTrue(np.allclose(result_pv[2], projected_v_mat_testing3), "Correct projV matrix for init vec 102")

    def test_compare_gpu_cpu(self):
        'compare the cusp implementation gpu vs cpu (if a gpu is detected)'
        
        dims = 5
        iterations = 2
        key_dirs = 2
        num_parallel = 2

        a_matrix = random_sparse_matrix(dims, entries_per_row=2)
        key_dir_mat = random_sparse_matrix(dims, entries_per_row=2)[:key_dirs, :]

        if KrylovInterface.has_gpu():
            result_h_list = []
            result_pv_list = []

            for use_gpu in [False, True]:
                KrylovInterface.set_use_gpu(use_gpu)
                #print "\n---------------\n"
                #KrylovInterface.set_use_profiling(True)

                KrylovInterface.preallocate_memory(iterations, num_parallel, dims, key_dirs)
                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_parallel(2) # offset by 2 just because

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            for i in [0, 1]:
                self.assertTrue(np.allclose(result_h_list[0][i], result_h_list[1][i]), "bad h-matrix i={}".format(i))
                self.assertTrue(np.allclose(result_pv_list[0][i], result_pv_list[1][i]), "bad projV i={}".format(i))

    def test_compare_gpu_cpu_large(self):
        'compare the cusp implementation gpu vs cpu (if a gpu is detected) on a large example'

        # this test is manually enabled, since it can take a long time
        test_enabled = True
        
        if test_enabled:
            print "running cpu / gpu timing comparison on large random matrix"
            
            dims = 10 * 1000 * 1000
            iterations = 10

            print "making random matrix..."
            a_matrix = random_sparse_matrix(dims, entries_per_row=6, print_progress=True)
            print "done"

            dir1 = np.array([float(n) if n % 2 == 0 else 0.0 for n in xrange(dims)], dtype=float)
            dir2 = np.array([float(n) if n % 2 == 1 else 0.0 for n in xrange(dims)], dtype=float)
            key_dir_mat = csr_matrix([dir1, dir2])

            result_h_list = []
            result_pv_list = []

            for use_gpu in [False, True]:
                if use_gpu and not KrylovInterface.has_gpu():
                    break

                print "\n---------------\n"
                print "running with use_gpu = {}".format(use_gpu)
                
                KrylovInterface.set_use_gpu(use_gpu)
                KrylovInterface.set_use_profiling(True)

                KrylovInterface.preallocate_memory(iterations, 1, dims, 2)
                KrylovInterface.load_a_matrix(a_matrix)
                KrylovInterface.load_key_dir_matrix(key_dir_mat)
                result_h, result_pv = KrylovInterface.arnoldi_parallel(2) # offset by 2 just because

                result_h_list.append(result_h)
                result_pv_list.append(result_pv)

            if len(result_h_list) == 2:
                self.assertTrue(np.allclose(result_h_list[0][0], result_h_list[1][0]), "h-mat mismatch")
                self.assertTrue(np.allclose(result_pv_list[0][0], result_pv_list[1][0]), "mismatch projV")

if __name__ == '__main__':
    unittest.main()
