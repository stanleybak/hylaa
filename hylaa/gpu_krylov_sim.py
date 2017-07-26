'''
Dung Tran
June 2017

Simulation of linear system x' = Ax using krylov supspace method in CPU and GPU 

'''

import ctypes
import os
import time
import random
from krypy.utils import arnoldi

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.sparse import csr_matrix, coo_matrix, rand
from scipy.linalg import expm
from hylaa.util import Freezable, get_script_path


class GpuKrylovSim(Freezable):
    'GPU-enhanced matrix-vector multiplication using the python/c interface'

    # static member (library)
    _lib = None

    def __init__(self):
        raise RuntimeError('GpuKrylovSim is a static class and should not be instantiated')

    @staticmethod
    def _init_static():
        'open the library (if not opened already) and initialize the static members'

        if GpuKrylovSim._lib is None:
            lib_path = os.path.join(get_script_path(__file__), 'gpu_interface', 'gpu_krylov_sim.so')
            GpuKrylovSim._lib = lib = ctypes.CDLL(lib_path)

            GpuKrylovSim._has_gpu = lib.hasGpu
            GpuKrylovSim._has_gpu.restype = ctypes.c_int
            GpuKrylovSim._has_gpu.argtypes = None

            GpuKrylovSim._choose_GPU_or_CPU = lib.choose_GPU_or_CPU
            GpuKrylovSim._choose_GPU_or_CPU.restype = None
            GpuKrylovSim._choose_GPU_or_CPU.argtypes = [ctypes.c_char_p]            

            GpuKrylovSim._load_matrix = lib.loadMatrix
            GpuKrylovSim._load_matrix.restype = None
            GpuKrylovSim._load_matrix.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                             ctypes.c_int]

            GpuKrylovSim._load_keyMatrix = lib.loadKeyDirMatrix
            GpuKrylovSim._load_keyMatrix.restype = None
            GpuKrylovSim._load_keyMatrix.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                             ctypes.c_int]
                                             
            GpuKrylovSim._arnoldi_initVector = lib.arnoldi_initVector
            GpuKrylovSim._arnoldi_initVector.restype = ctypes.c_int
            GpuKrylovSim._arnoldi_initVector.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                            ctypes.c_int, ctypes.c_int]

            GpuKrylovSim._arnoldi_initVectorPos = lib.arnoldi_initVectorPos
            GpuKrylovSim._arnoldi_initVectorPos.restype = ctypes.c_int 
            GpuKrylovSim._arnoldi_initVectorPos.argtypes = [ctypes.c_int,
                                            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                            ctypes.c_int, ctypes.c_int]
            
            GpuKrylovSim._arnoldi_parallel = lib.arnoldi_parallel
            GpuKrylovSim._arnoldi_parallel.restype = ctypes.c_int 
            GpuKrylovSim._arnoldi_parallel.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

            GpuKrylovSim._sim = lib.sim
            GpuKrylovSim._sim.restype = None
            GpuKrylovSim._sim.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]

            GpuKrylovSim._sim2 = lib.sim2
            GpuKrylovSim._sim2.restype = None
            GpuKrylovSim._sim2.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]

            
            GpuKrylovSim._getKeySimResult = lib.getKeySimResult
            GpuKrylovSim._getKeySimResult.restype = None
            GpuKrylovSim._getKeySimResult.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

            
            GpuKrylovSim._getKeySimResult_parallel = lib.getKeySimResult_parallel
            GpuKrylovSim._getKeySimResult_parallel.restype = None
            GpuKrylovSim._getKeySimResult_parallel.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
            
        if GpuKrylovSim._has_gpu() == 0:
            raise RuntimeError("GPU not detected.")

    @staticmethod
    def choose_GPU_or_CPU(msg):
        GpuKrylovSim._init_static()
        GpuKrylovSim._choose_GPU_or_CPU(msg)
        
    @staticmethod
    def load_matrix(sparse_matrix):
        'load a sparse matrix'

        assert isinstance(sparse_matrix, csr_matrix)
        GpuKrylovSim._init_static()

        w, h = sparse_matrix.shape
        rows, cols = sparse_matrix.nonzero()
        entries = sparse_matrix[rows, cols].A1.copy()

        GpuKrylovSim._load_matrix(w, h, rows, cols, entries, len(rows))
        GpuKrylovSim._is_loaded = True
        GpuKrylovSim._loaded_w = w
        GpuKrylovSim._loaded_h = h

    @staticmethod
    def load_keyDirSparseMatrix(keyDirSparseMatrix):
        'load key sparse matrix'

        assert isinstance(keyDirSparseMatrix, csr_matrix)
        GpuKrylovSim._init_static()

        w, h = keyDirSparseMatrix.shape
        rows, cols = keyDirSparseMatrix.nonzero()
        entries = keyDirSparseMatrix[rows, cols].A1.copy()

        GpuKrylovSim._load_keyMatrix(w, h, rows, cols, entries, len(rows))
        GpuKrylovSim._is_keyLoaded = True
        GpuKrylovSim._keyLoaded_w = w
        GpuKrylovSim._keyLoaded_h = h

    @staticmethod
    def arnoldi_initVector(init_vector, size, numIter):
        'implement the aroldi algorithm using Gpu and return the matrix Hm, keep the matrix Vm in the device memory'

        assert isinstance(init_vector, np.ndarray)
        GpuKrylovSim._init_static()

        assert GpuKrylovSim._is_loaded
        assert init_vector.shape[0] == GpuKrylovSim._loaded_w

        result_H = np.zeros((numIter*numIter))

        actual_numIter = GpuKrylovSim._arnoldi_initVector(init_vector, result_H, size, numIter)

        result_H.shape = (numIter, numIter)

        actual_result_H = np.zeros((actual_numIter,actual_numIter))
        
        for i in range(0,actual_numIter):
            for j in range(0, actual_numIter):
                actual_result_H[i,j] = result_H[i,j]

        return actual_result_H, actual_numIter

    @staticmethod
    def arnoldi_initVectorPos(basic_initVector_pos, size, numIter):
        'implement the aroldi algorithm using Gpu and return the matrix Hm, keep the matrix Vm in the device memory'

        GpuKrylovSim._init_static()

        assert GpuKrylovSim._is_loaded

        result_H = np.zeros((numIter*numIter))

        actual_numIter = GpuKrylovSim._arnoldi_initVectorPos(basic_initVector_pos, result_H, size, numIter)
        
        result_H.shape = (numIter, numIter)

        actual_result_H = np.zeros((actual_numIter,actual_numIter))

        for i in range(0,actual_numIter):
            for j in range(0, actual_numIter):
                actual_result_H[i,j] = result_H[i,j]

        return actual_result_H, actual_numIter

    @staticmethod
    def arnoldi_parallel(size, numIter):
        'implement the aroldi algorithm in parallel'

        GpuKrylovSim._init_static()

        assert GpuKrylovSim._is_loaded

        result_H = np.zeros((size*numIter*numIter))

        actual_numIter = GpuKrylovSim._arnoldi_parallel(size, numIter, result_H)
        
        result_H.shape = (size, numIter, numIter)

        actual_result_H = np.zeros((size, actual_numIter, actual_numIter))

        for k in range(0, size):
            for i in range(0,actual_numIter):
                for j in range(0, actual_numIter):
                    actual_result_H[k,i,j] = result_H[k,i,j]

        return actual_result_H, actual_numIter



    
    @staticmethod
    def sim(matrix_Hf,size,numIter,numStep):
        'compute the simulation result from the matrix Hf = exp(i*timeStep*Hm)e1 and copy the result back to cpu'

        assert isinstance(matrix_Hf, np.ndarray)
        GpuKrylovSim._init_static()
        matrix_Hf.reshape(numIter*numStep)
        sim_result = np.zeros((size,numStep))
        GpuKrylovSim._sim(matrix_Hf, sim_result, size, numIter, numStep)
        
        return sim_result

    
    @staticmethod
    def sim2(matrix_Hf,size,numIter,numStep):
        'compute the simulation result from the matrix Hf = exp(i*timeStep*Hm)e1 and do not copy the result back to cpu'

        assert isinstance(matrix_Hf, np.ndarray)
        GpuKrylovSim._init_static()
        matrix_Hf.reshape(numIter*numStep)
        GpuKrylovSim._sim2(matrix_Hf, size, numIter, numStep)

    @staticmethod
    def getKeySimResult(DirMatrix_numRows,numSimStep):
        'get Simulation Result in a specific direction defined by a sparse matrix'
        keySimResult = np.zeros((DirMatrix_numRows,numSimStep))
        GpuKrylovSim._getKeySimResult(keySimResult)
        
        return  keySimResult
    
    @staticmethod
    def getKeySimResult_parallel(dirMatrix_numRows,size,numIter,expHt_tuples):
        'get Simulation Result in parallel in a specific direction defined by a sparse matrix'

        expHt_parallel = np.zeros((size,numIter))
        
        for i in range(0,size):
            expHt = expHt_tuples[i]
            expHt_parallel[i,:] = expHt[:,0] # get exp(H*t)*e1, e1 = [1 0 ...0]^T, get first column of exp(H*t)
        
            
        keySimResult_tuples = np.zeros((size,dirMatrix_numRows))
        GpuKrylovSim._getKeySimResult_parallel(size,numIter,expHt_parallel,keySimResult_tuples)

        keySimResult_tuples = np.transpose(keySimResult_tuples)
        
        return  keySimResult_tuples
        
def make_iss_matrix(num_copies):
    'create a matrix from the international space station system model'

    orig_dims = 271
    inds = [135, 407, 679, 951, 1223, 1495, 1767, 2039, 2311, 2583, 2855, 3127, 3399, 3671, 3943, 4215,
            4487, 4759, 5031, 5303, 5575, 5847, 6119, 6391, 6663, 6935, 7207, 7479, 7751, 8023, 8295,
            8567, 8839, 9111, 9383, 9655, 9927, 10199, 10471, 10743, 11015, 11287, 11559, 11831, 12103,
            12375, 12647, 12919, 13191, 13463, 13735, 14007, 14279, 14551, 14823, 15095, 15367, 15639,
            15911, 16183, 16455, 16727, 16999, 17271, 17543, 17815, 18087, 18359, 18631, 18903, 19175,
            19447, 19719, 19991, 20263, 20535, 20807, 21079, 21351, 21623, 21895, 22167, 22439, 22711,
            22983, 23255, 23527, 23799, 24071, 24343, 24615, 24887, 25159, 25431, 25703, 25975, 26247,
            26519, 26791, 27063, 27335, 27607, 27879, 28151, 28423, 28695, 28967, 29239, 29511, 29783,
            30055, 30327, 30599, 30871, 31143, 31415, 31687, 31959, 32231, 32503, 32775, 33047, 33319,
            33591, 33863, 34135, 34407, 34679, 34951, 35223, 35495, 35767, 36039, 36311, 36583, 36585,
            36720, 36857, 36992, 37129, 37264, 37401, 37536, 37673, 37808, 37945, 38080, 38217, 38352,
            38489, 38624, 38761, 38896, 39033, 39168, 39305, 39440, 39577, 39712, 39849, 39984, 40121,
            40256, 40393, 40528, 40665, 40800, 40937, 41072, 41209, 41344, 41481, 41616, 41753, 41888,
            42025, 42160, 42297, 42432, 42569, 42704, 42841, 42976, 43113, 43248, 43385, 43520, 43657,
            43792, 43929, 44064, 44201, 44336, 44473, 44608, 44745, 44880, 45017, 45152, 45289, 45424,
            45561, 45696, 45833, 45968, 46105, 46240, 46377, 46512, 46649, 46784, 46921, 47056, 47193,
            47328, 47465, 47600, 47737, 47872, 48009, 48144, 48281, 48416, 48553, 48688, 48825, 48960,
            49097, 49232, 49369, 49504, 49641, 49776, 49913, 50048, 50185, 50320, 50457, 50592, 50729,
            50864, 51001, 51136, 51273, 51408, 51545, 51680, 51817, 51952, 52089, 52224, 52361, 52496,
            52633, 52768, 52905, 53040, 53177, 53312, 53449, 53584, 53721, 53856, 53993, 54128, 54265,
            54400, 54537, 54672, 54809, 54944, 55081, 55216, 55353, 55488, 55625, 55760, 55897, 56032,
            56169, 56304, 56441, 56576, 56713, 56848, 56985, 57120, 57257, 57392, 57529, 57664, 57801,
            57936, 58073, 58208, 58345, 58480, 58617, 58752, 58889, 59024, 59161, 59296, 59433, 59568,
            59705, 59840, 59977, 60112, 60249, 60384, 60521, 60656, 60793, 60928, 61065, 61200, 61337,
            61472, 61609, 61744, 61881, 62016, 62153, 62288, 62425, 62560, 62697, 62832, 62969, 63104,
            63241, 63376, 63513, 63648, 63785, 63920, 64057, 64192, 64329, 64464, 64601, 64736, 64873,
            65008, 65145, 65280, 65417, 65552, 65689, 65824, 65961, 66096, 66233, 66368, 66505, 66640,
            66777, 66912, 67049, 67184, 67321, 67456, 67593, 67728, 67865, 68000, 68137, 68272, 68409,
            68544, 68681, 68816, 68953, 69088, 69225, 69360, 69497, 69632, 69769, 69904, 70041, 70176,
            70313, 70448, 70585, 70720, 70857, 70992, 71129, 71264, 71401, 71536, 71673, 71808, 71945,
            72080, 72217, 72352, 72489, 72624, 72761, 72896, 73033, 73168]

    data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, -0.3887, -0.0062346, -0.60078, -0.007751, -1.9781, -0.014065, -1.9785,
            -0.014066, -3.1943, -0.017872, -3.9682, -0.01992, -5.2756, -0.022969, -5.3176, -0.02306,
            -6.1969, -0.024894, -6.2041, -0.024908, -6.6376, -0.025763, -6.643, -0.025774, -14.915,
            -0.03862, -15.321, -0.039142, -28.924, -0.053781, -28.924, -0.053781, -31.36, -0.056,
            -31.666, -0.056273, -34.377, -0.058632, -34.377, -0.058632, -37.182, -0.060977, -37.229,
            -0.061016, -62.94, -0.079335, -65.885, -0.08117, -65.946, -0.081207, -71.925, -0.084809,
            -84.946, -0.092166, -85.054, -0.092225, -85.262, -0.092337, -91.273, -0.095537, -91.283,
            -0.095542, -95.44, -0.097693, -95.44, -0.097693, -95.457, -0.097702, -95.457, -0.097702,
            -95.492, -0.09772, -95.492, -0.09772, -105.24, -0.10258, -105.32, -0.10262, -115.82,
            -0.10762, -116.34, -0.10786, -178.25, -0.13351, -178.73, -0.13369, -233.48, -0.1528,
            -305.2, -0.1747, -306.11, -0.17496, -336, -0.1833, -338.12, -0.18388, -437.91, -0.20926,
            -444.23, -0.21077, -467.28, -0.21617, -468.98, -0.21656, -505.12, -0.22475, -505.14,
            -0.22475, -730.75, -0.27032, -730.76, -0.27033, -749.99, -0.27386, -749.99, -0.27386,
            -831.1, -0.28829, -831.1, -0.28829, -908.85, -0.30147, -909.06, -0.30151, -934.4, -0.30568,
            -934.43, -0.30568, -1013.9, -0.31842, -1013.9, -0.31842, -1089.9, -0.33013, -1089.9, -0.33013,
            -1122.1, -0.33498, -1122.1, -0.33498, -1147.7, -0.33878, -1147.7, -0.33878, -1147.9, -0.33881,
            -1147.9, -0.33881, -1148.2, -0.33886, -1148.2, -0.33886, -1219.6, -0.34923, -1230.4, -0.35077,
            -1442.9, -0.37986, -1532.3, -0.39144, -1532.6, -0.39148, -1608.8, -0.4011, -1609.1, -0.40114,
            -1752, -0.41857, -1752.2, -0.41859, -1782.2, -0.42216, -1782.2, -0.42216, -1808.4, -0.42525,
            -1809, -0.42532, -1846.1, -0.42967, -1846.1, -0.42967, -1910.9, -0.43714, -1910.9, -0.43714,
            -1948.5, -0.44142, -1948.5, -0.44142, -1949, -0.44147, -1949, -0.44147, -2050.7, -0.45285,
            -2050.8, -0.45285, -2091.3, -0.45731, -2091.4, -0.45732, -2180.6, -0.46697, -2180.8, -0.46699,
            -2191, -0.46808, -2196.9, -0.46871, -2288.2, -0.47835, -2302.1, -0.4798, -2575.4, -0.50748,
            -2576.1, -0.50756, -2757.5, -0.52512, -2758.1, -0.52518, -2772.3, -0.52653, -2797.8, -0.52894,
            -2926.6, -0.54098, -2926.8, -0.541, -3146.5, -0.56094, -3150.8, -0.56132, -3212.8, -0.56682,
            -3213.4, -0.56687, -3281.1, -0.57281, -3291.2, -0.57369, -3291.4, -0.5737, -3427.6, -0.58546,
            -3427.6, -0.58546, -3428.8, -0.58556, -3428.8, -0.58556, -3429.3, -0.5856, -3429.3, -0.5856,
            -3448.2, -0.58721, -3448.2, -0.58721, -3449, -0.58729, -3449, -0.58729, -3452.3, -0.58757,
            -3452.3, -0.58757, -3762.6, -0.6134]

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

def random_sparse_matrix(dims, entries_per_row, random_cols=True):
    'make a random sparse matrix with the given number of entries per row'

    num_rows = num_cols = dims

    rows = []
    cols = []
    vals = []

    for row in xrange(num_rows):

        for entry_index in xrange(entries_per_row):
            rows.append(row)

            if random_cols:
                cols.append(int(random.random() * num_cols))
            else:
                cols.append(entry_index)

            vals.append(random.random())

    mat = coo_matrix((vals, (rows, cols)), shape=(num_rows, num_cols))


    return csr_matrix(mat)

def test_gpu_krylov_sim():
    'compare timing of Arnoldi algorithm using gpu and cpu-krypy'

    print "making matrix..."
    start = time.time()
    a = random_sparse_matrix(10000, entries_per_row=5, random_cols=True)
    #a = make_iss_matrix(1)
    print "made in {:.2f} seconds".format(time.time() - start)

    m = 30 # number of iteration of Arnoldi Algorithm
    timeStep = 0.01  # simulation time step
    numStep = 1    # number of simulation step

    
    # get matrix H from Arnoldi Algorithm
    GpuKrylovSim.choose_GPU_or_CPU("CPU")
    GpuKrylovSim.load_matrix(a)

    print"\n -------------------------------------------------"

    print"\n Compare simulation using Krylov Subspace methods that are implemented using CPU-KRYPY and GPU-CUSPARSE..." 

    # Compare CPU- and GPU- based methods
    print"\n Compare CPU- and GPU- based methods..."

    print"\n Create random initial vector..." 
    #vec = np.random.random((a.shape[0],1)) # initial vector used for arnoldi_initVector() method
    #initial vector used for arnoldi_initVectorPos() method
    basic_initVector_pos = 1
    vec = np.zeros((a.shape[0],1))
    vec[basic_initVector_pos] = 1 
    
    print"\n Selected number of iteration of Arnoldi argorithm m = {}".format(m)
    print"\n Selected simulation time step, timStep = {}".format(timeStep)
    print"\n Seclected number step of simulation, numStep = {}".format(numStep)

    print"\n -------------------------------------------------"    

    #'Arnoldi algorithm using gpu, two methods can be used arnoldi_initVector() and arnoldi_initVectorPos()'
    print"\n Execute Arnoldi algorithm using gpu, the packet used is cusparse ..."
    start = time.time()
    #res_gpu_arnoldi_H, actual_numIter = GpuKrylovSim.arnoldi_initVector(vec,a.shape[0],m) # using arnoldi_initVector() method
    res_gpu_arnoldi_H, actual_numIter = GpuKrylovSim.arnoldi_initVectorPos(basic_initVector_pos,a.shape[0],m) # using arnoldi_initVectorPos() method
    print "\n Computation time of Arnoldi algorithm using gpu {:.1f}ms".format(1000 * (time.time() - start))

    print"\n Selected number of iteration of Arnoldi Algorithm m ={}".format(m)
    print"\n Actual number of interation of Arnoldi Algorithm actual_numIter = {}".format(actual_numIter)

    
    #'Arnoldi algorithm using cpu - krypy packet'
    print"\n Execute Arnoldi algorithm using cpu, the packet used is krypy..."
    arn_start = time.time()
    V, H = arnoldi(a,vec,m)
    Vm = V[:,0:m]
    Hm = H[0:m,:]
    print "\n Computation time of Arnoldi Algorithm using cpu - krypy packet = {}ms".format(1000 * (time.time() - arn_start))  

    print "\n Error between two H matrices of two approaches: norm (Hm - res_gpu_arnoldi_H) = {}".format(np.linalg.norm(Hm-res_gpu_arnoldi_H))

    
    # compute Hf = exp(i*timeStep*Hm)e1, i = 0 ... numStep
    print"\n Compute matrix Hf = exp(i*timStep*Hm)*e1, (i = 0.. numStep), e1 is basic vector e1 = (1, 0,..,0)^T..."
    start = time.time()
    Hf = np.zeros((actual_numIter, numStep))
    for i in range(0, numStep):
        Hs = expm((i+1)*timeStep*res_gpu_arnoldi_H)
        Hf[:,i] = Hs[:,1]
    print "\n Time for computing Hf = {}ms".format(1000 * (time.time() - start))

    # compute simulation result using cpu
    print"\n Compute simulation result using cpu..."
    start = time.time()
    beta  = np.linalg.norm(vec)
    cpu_sim_result = np.dot(Vm,Hf)
    cpu_sim_result = beta*cpu_sim_result
    
    print"\n Time for computing simulation result using cpu = {}ms".format(1000 * (time.time() - start))
        
    # compute simulation result using gpu_krylov_sim
    print"\n Compute simulation result using gpu..."
   
    start = time.time()
    gpu_sim_result = GpuKrylovSim.sim(Hf,a.shape[0],actual_numIter,numStep)
    print"\n Time for computing simulation result using gpu = {}ms".format(1000 * (time.time() - start))
    
    # compute the error of two approaches
    print"\n Compute the errors of simulation results of two approaches..."
    start = time.time()
    err_gpu_cpu = np.zeros((a.shape[0],numStep))
    err_gpu_cpu = gpu_sim_result - cpu_sim_result
    for i in range(0,numStep):
        print"\n 2-norm of the simulation error of two approaches at step {} = {}".format(i+1,np.linalg.norm(err_gpu_cpu[:,i]))
    print"\n Time for computing the norm of the error = {}ms".format(1000*(time.time()-start))  


    # Test get simulation result in some specific direction declared by a dense matrix

def test_getKeySimResult():
    
    print "making matrix..."
    start = time.time()
    #a = random_sparse_matrix(1000000, entries_per_row=5, random_cols=True)
    a = make_iss_matrix(1)
    print "made in {:.2f} seconds".format(time.time() - start)

    m = 5 # number of iteration of Arnoldi Algorithm
    timeStep = 0.01  # simulation time step
    numStep = 1    # number of simulation step

    systemSize = a.shape[0]
    
    basic_initVector_pos = 1
    vec = np.zeros((systemSize,1))
    vec[basic_initVector_pos] = 1

    GpuKrylovSim.load_matrix(a)
    
    print"\n Selected number of iteration of Arnoldi argorithm m = {}".format(m)
    print"\n Selected simulation time step, timStep = {}".format(timeStep)
    print"\n Seclected number step of simulation, numStep = {}".format(numStep)

    print"\n -------------------------------------------------"    
    
    #'Arnoldi algorithm using gpu, two methods can be used arnoldi_initVector() and arnoldi_initVectorPos()'
    print"\n Execute Arnoldi algorithm using gpu, the packet used is cusparse ..."
    start = time.time()
    #res_gpu_arnoldi_H, actual_numIter = GpuKrylovSim.arnoldi_initVector(vec,a.shape[0],m) # using arnoldi_initVector() method
    res_gpu_arnoldi_H, actual_numIter = GpuKrylovSim.arnoldi_initVectorPos(basic_initVector_pos,a.shape[0],m) # using arnoldi_initVectorPos() method
    print "\n Computation time of Arnoldi algorithm using gpu {:.1f}ms".format(1000 * (time.time() - start))

    print"\n Selected number of iteration of Arnoldi Algorithm m ={}".format(m)
    print"\n Actual number of interation of Arnoldi Algorithm actual_numIter = {}".format(actual_numIter)

    
    # compute Hf = exp(i*timeStep*Hm)e1, i = 0 ... numStep
    print"\n Compute matrix Hf = exp(i*timStep*Hm)*e1, (i = 0.. numStep), e1 is basic vector e1 = (1, 0,..,0)^T..."
    start = time.time()
    Hf = np.zeros((actual_numIter, numStep))
    for i in range(0, numStep):
        Hs = expm((i+1)*timeStep*res_gpu_arnoldi_H)
        Hf[:,i] = Hs[:,1]
    print "\n Time for computing Hf = {}ms".format(1000 * (time.time() - start))

         
    # compute simulation result using gpu_krylov_sim, the result is saved in device memory, not coppied back to CPU
    print"\n Compute simulation result using gpu..."
    start = time.time()
    GpuKrylovSim.sim2(Hf,a.shape[0],actual_numIter,numStep)
    print"\n Time for computing simulation result using gpu = {}ms".format(1000 * (time.time() - start))

    # compute the key simulation result corresponding to specific direction defined by direction matrix
   
    # generate random sparse direction matrix
    print"\n Compute the key simulation result corresponding to a specific direction defined by a sparse matrix..."
    print"\n Generate random sparse direction matrix"
    keyDirMatrix_Sparse = rand(2,systemSize,density = 0.01, format = 'csr',dtype = None, random_state = None)
    GpuKrylovSim.load_keyDirSparseMatrix(keyDirMatrix_Sparse)
    keySimResult_Sparse = GpuKrylovSim.getKeySimResult(2,numStep)
    print(keySimResult_Sparse)

def test_arnoldi_parallel():

    print "making matrix..."
    start = time.time()
    a = random_sparse_matrix(100, entries_per_row=5, random_cols=True)
    #a = make_iss_matrix(1)
    print "made in {:.2f} seconds".format(time.time() - start)

    m = 5 # number of iteration of Arnoldi Algorithm
    
    # get matrix H from Arnoldi Algorithm
    GpuKrylovSim.choose_GPU_or_CPU("CPU")
    GpuKrylovSim.load_matrix(a)

    print "running arnoldi algorithm in parallel for all initial vectors..."
    res_gpu_arnoldi_H, actual_numIter = GpuKrylovSim.arnoldi_parallel(a.shape[0],m)

    for i in range(0,a.shape[0]):
        print "running arnoldi algorithm for single initial vector..."
        res_gpu_arnoldi_H_i, actual_numIter_i = GpuKrylovSim.arnoldi_initVectorPos(i,a.shape[0],m) # using arnoldi_initVectorPos() method
        print "the actual number of iteration of Arnoldi_parallel is: {}".format(actual_numIter)
        print "the {}-th Hm computed by Arnoldi_parallel is:\n {}".format(i,res_gpu_arnoldi_H[i,:,:])
        print "the actual number of iteration of Arnoldi_initVectorPos is: {}".format(actual_numIter_i)
        print "the {}-th Hm computed by Arnoldi_initVectorPos is: \n {}".format(i,res_gpu_arnoldi_H_i)
        

def test_getKeySimResult_parallel():
    print "making matrix..."
    start = time.time()
    a = random_sparse_matrix(10, entries_per_row=5, random_cols=True)
    #a = make_iss_matrix(1)
    print "made in {:.2f} seconds".format(time.time() - start)
    numIter = 5  # number of iteration of Arnoldi algorithm
    step = 0.01  # simulation time step
    size = a.shape[0]
    
    # Load a matrix into device memory
    GpuKrylovSim.load_matrix(a)

    keyDirMatrix = np.zeros((2,size))
    keyDirMatrix_numRows = keyDirMatrix.shape[0]
    
    keyDirMatrix[0,0] = 1
    keyDirMatrix[0,1] = 1    # direction x0+x1

    keyDirMatrix[1,2] = 1
    keyDirMatrix[1,3] = -1   # direction x2-x3

    keyDirMatrix_Sparse = csr_matrix(keyDirMatrix)

    # Load key direction matrix 
    GpuKrylovSim.load_keyDirSparseMatrix(keyDirMatrix_Sparse)
    print "number of Rows of keyDirMatrix is: {}\n".format(keyDirMatrix_numRows)
    
    # Run Arnoldi algorithm in paralel
    H_tuples, actual_numIter = GpuKrylovSim.arnoldi_parallel(size,numIter)

    curStep = 2; 
    
    # Compute exp(H*t) at current time step, i.e., t = curStep*step

    expHt_tuples = []
    for i in range(0,size):
        curTimeStep = curStep*step
        expHt_tuples.insert(i,expm(curTimeStep*H_tuples[i]))
        print "The {}-th H matrix is: \n {}".format(i,H_tuples[i])
        print "The {}-th exp(H*t) at time t = {} is: \n {}".format(i,curTimeStep,expHt_tuples[i])

    keySimResult_tuples = GpuKrylovSim.getKeySimResult_parallel(keyDirMatrix_numRows,size,numIter,expHt_tuples)

        
if __name__ == '__main__':
   # test_gpu_krylov_sim()
   # test_getKeySimResult()
   # test_arnoldi_parallel()
    test_getKeySimResult_parallel()
