'''
Dung Tran & Stanley Bak
August 2017


Simulation of linear system x' = Ax using krylov supspace method in CPU and GPU
'''

import ctypes
import os
import math

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.sparse import csr_matrix

from hylaa.util import Freezable, get_script_path
from hylaa.timerutil import Timers

class CuspData(Freezable):
    'Function pointers / data for gpu or cpu'

    def __init__(self):
        self.set_use_profiling = None
        self.set_print_output = None
        self.load_a_matrix = None
        self.load_key_dir_matrix = None
        self.get_free_memory_mb = None
        self.preallocate_memory = None
        self.arnoldi_unit = None
        self.arnoldi_vec = None
        self.get_profiling_data = None
        self.print_profiling_data = None

        self.n = 0 # dimensions of a matrix
        self.k = 0 # number of key directions
        self.i = 0 # number of iterations

        self.freeze_attrs()

class KrylovInterface(object):
    'GPU (and CPU) interface for Krylov subspace simulations of a linear system'

    # static member (library)
    _lib = None
    _lib_path = os.path.join(get_script_path(__file__), 'krylov_interface', 'cusp_krylov_stan.so')

    float_type = ctypes.c_double

    def __init__(self):
        raise RuntimeError(
            'KrylovInterface is a static class and should not be instantiated')

    @staticmethod
    def _init_static():
        'open the library (if not opened already) and initialize the static members'

        if KrylovInterface._lib is None:
            KrylovInterface._lib = lib = ctypes.CDLL(KrylovInterface._lib_path)
            float_type = KrylovInterface.float_type

            # int hasGpu()
            KrylovInterface._has_gpu = lib.hasGpu
            KrylovInterface._has_gpu.restype = ctypes.c_ulong
            KrylovInterface._has_gpu.argtypes = None

            # void reset()
            KrylovInterface._reset = lib.reset
            KrylovInterface._reset.restype = None
            KrylovInterface._reset.argtypes = None

            # CPU and GPU container objects... _funcs stores the selected version
            KrylovInterface._cpu = CuspData()
            KrylovInterface._gpu = CuspData()
            KrylovInterface._cusp = KrylovInterface._cpu # can be changed programatically

            #void setUseProfilingCpu(unsigned long enabled)
            cpu_func = KrylovInterface._cpu.set_use_profiling = lib.setUseProfilingCpu
            gpu_func = KrylovInterface._gpu.set_use_profiling = lib.setUseProfilingGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [ctypes.c_ulong]

            #void setUsePrintOutputCpu(unsigned long enabled)
            cpu_func = KrylovInterface._cpu.set_print_output = lib.setPrintOutputCpu
            gpu_func = KrylovInterface._gpu.set_print_output = lib.setPrintOutputGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [ctypes.c_ulong]

            #void loadAMatrixGpu(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
            #        double* values, int valuesLen)
            cpu_func = KrylovInterface._cpu.load_a_matrix = lib.loadAMatrixCpu
            gpu_func = KrylovInterface._gpu.load_a_matrix = lib.loadAMatrixGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_ulong, ctypes.c_ulong,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong
            ]

            # void loadKeyDirMatrixGpu(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds,
            #                          int colIndsLen, double* values, int valuesLen)
            cpu_func = KrylovInterface._cpu.load_key_dir_matrix = lib.loadKeyDirMatrixCpu
            gpu_func = KrylovInterface._gpu.load_key_dir_matrix = lib.loadKeyDirMatrixGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_ulong, ctypes.c_ulong,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong
            ]

            # double getFreeMemoryMbCpu()
            cpu_func = KrylovInterface._cpu.get_free_memory_mb = lib.getFreeMemoryMbCpu
            gpu_func = KrylovInterface._gpu.get_free_memory_mb = lib.getFreeMemoryMbGpu
            gpu_func.restype = cpu_func.restype = float_type
            gpu_func.argtypes = cpu_func.argtypes = None

            # unsigned long preallocateMemoryGpu(unsigned long arnoldiIterations, unsigned long dims,
            #                       unsigned long keyDirMatSize)
            cpu_func = KrylovInterface._cpu.preallocate_memory = lib.preallocateMemoryCpu
            gpu_func = KrylovInterface._gpu.preallocate_memory = lib.preallocateMemoryGpu
            gpu_func.restype = cpu_func.restype = ctypes.c_ulong
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong
            ]

            #void arnoldiUnitGpu(unsigned long dim, FLOAT_TYPE *resultH, unsigned long sizeResultH,
            #                    FLOAT_TYPE *resultPV, unsigned long sizeResultPV)
            cpu_func = KrylovInterface._cpu.arnoldi_unit = lib.arnoldiUnitCpu
            gpu_func = KrylovInterface._gpu.arnoldi_unit = lib.arnoldiUnitGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_ulong,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong
            ]

            #void arnoldiVecGpu(FLOAT_TYPE *vec, unsigned long vecLen, FLOAT_TYPE *resultH,
            #       unsigned long sizeResultH, FLOAT_TYPE *resultPV, unsigned long sizeResultPV)
            cpu_func = KrylovInterface._cpu.arnoldi_vec = lib.arnoldiVecCpu
            gpu_func = KrylovInterface._gpu.arnoldi_vec = lib.arnoldiVecGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong
            ]

            # void getProfilingDataGpu(const char *name, FLOAT_TYPE *resultVec, unsigned long resultVecLen)
            cpu_func = KrylovInterface._cpu.get_profiling_data = lib.getProfilingDataCpu
            gpu_func = KrylovInterface._gpu.get_profiling_data = lib.getProfilingDataGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_char_p,
                ndpointer(float_type, flags="C_CONTIGUOUS"), ctypes.c_ulong,
            ]

            # void printProfilingDataGpu()
            cpu_func = KrylovInterface._cpu.print_profiling_data = lib.printProfilingDataCpu
            gpu_func = KrylovInterface._gpu.print_profiling_data = lib.printProfilingDataGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = None

    @staticmethod
    def set_use_gpu(use_gpu):
        'set if the GPU should be used (false = CPU)'

        KrylovInterface._init_static()

        if use_gpu:
            assert KrylovInterface.has_gpu(), "set_use_gpu(True) called, but no GPU hardware was detected"

            KrylovInterface._cusp = KrylovInterface._gpu
        else:
            KrylovInterface._cusp = KrylovInterface._cpu

    @staticmethod
    def has_gpu():
        'is a GPU detected?'

        KrylovInterface._init_static()
        return KrylovInterface._has_gpu() != 0

    @staticmethod
    def reset():
        'reset BOTH gpu and cpu variables used in c code'

        KrylovInterface._init_static()
        KrylovInterface._reset()

        KrylovInterface.set_use_gpu(False) # revert to CPU

    @staticmethod
    def cpu_get_free_memory_mb():
        'get the free memory on the cpu (main memory) in megabytes'

        KrylovInterface._init_static()

        return KrylovInterface._cpu.get_free_memory_mb()

    @staticmethod
    def get_free_memory_mb():
        'get the amount of free memory in megabytes (on gpu or cpu)'

        KrylovInterface._init_static()

        return KrylovInterface._cusp.get_free_memory_mb()

    @staticmethod
    def set_use_profiling(use_profiling):
        'set if timing profiling should be used'

        KrylovInterface._init_static()
        KrylovInterface._cusp.set_use_profiling(1 if use_profiling else 0)

    @staticmethod
    def set_print_output(print_output):
        'set if output should be printing to stdout (profiling also needs to be on)'

        KrylovInterface._init_static()
        KrylovInterface._cusp.set_print_output(1 if print_output else 0)

    @staticmethod
    def get_profiling_data(name):
        '''
        Get the profiling data for a single measurement (identified by the string). Use with set_use_profiling(True)

        This returns a pair (milliseconds, gflops)
        '''

        KrylovInterface._init_static()

        result = np.array([0.0, 0.0], dtype=KrylovInterface.float_type)
        KrylovInterface._cusp.get_profiling_data(name, result, len(result))

        return (result[0], result[1])

    @staticmethod
    def print_profiling_data():
        'print cached profiling data to stdout and reset. Use with set_use_profiling(True)'

        KrylovInterface._init_static()
        KrylovInterface._cusp.print_profiling_data()

    @staticmethod
    def load_a_matrix(matrix):
        'load dynamics A matrix (should be a csr sparse matrix)'

        KrylovInterface._init_static()
        h, w = matrix.shape

        assert isinstance(matrix, csr_matrix), "expected a_matrix to be csr_matrix, got {}".format(type(matrix))
        assert matrix.dtype == KrylovInterface.float_type, \
            "expected matrix dtype {}".format(type(KrylovInterface.float_type))
        assert w == h, "a matrix should be square"
        assert KrylovInterface._cusp.n != 0, "call preallocate() before load_a_matrix()"
        assert w == KrylovInterface._cusp.n, "a_matrix dims ({}) differs from preallocate dims ({})".format(
            w, KrylovInterface._cusp.n)

        values = matrix.data
        row_offsets = matrix.indptr
        col_inds = matrix.indices

        Timers.tic("load a matrix")
        KrylovInterface._cusp.load_a_matrix(w, h, row_offsets, len(row_offsets), col_inds, len(col_inds), values,
                                            len(values))
        Timers.toc("load a matrix")

    @staticmethod
    def load_key_dir_matrix(matrix):
        'load key direction sparse matrix'

        KrylovInterface._init_static()

        assert isinstance(matrix, csr_matrix)
        assert matrix.dtype == KrylovInterface.float_type, "expected matrix dtype {}".format(
            type(KrylovInterface.float_type))

        values = matrix.data
        row_offsets = matrix.indptr
        col_inds = matrix.indices

        h, w = matrix.shape
        assert w == KrylovInterface._cusp.n, "key dir matrix width ({}) should equal number of dimensions ({})".format(
            w, KrylovInterface._cusp.n)
        assert h == KrylovInterface._cusp.k, "key dir matrix height ({}) should equal keyDirMatSize ({})".format(
            h, KrylovInterface._cusp.k)

        Timers.tic("load key dir matrix")
        KrylovInterface._cusp.load_key_dir_matrix(w, h, row_offsets, len(row_offsets), col_inds, len(col_inds),
                                                  values, len(values))
        Timers.toc("load key dir matrix")

    @staticmethod
    def preallocate_memory(arnoldi_iterations, dims, key_dir_mat_size, error_on_fail=False):
        '''
        preallocate memory used in the arnoldi iteration
        returns True on sucess and False on (memory allocation) error
        '''

        assert isinstance(error_on_fail, bool)

        KrylovInterface._init_static()

        KrylovInterface._cusp.i = arnoldi_iterations
        KrylovInterface._cusp.n = dims
        KrylovInterface._cusp.k = key_dir_mat_size

        Timers.tic("preallocate memory")
        result = KrylovInterface._cusp.preallocate_memory(arnoldi_iterations, dims, key_dir_mat_size) != 0
        Timers.toc("preallocate memory")

        KrylovInterface._preallocated_memory = result
        KrylovInterface._arnoldi_iterations = arnoldi_iterations

        if error_on_fail and not result:
            raise RuntimeError("Memory allocation (preallocate) for krylov computation failed. " + \
                               "Run with krylov_profiling = True for details")

        return result


!!! HERE !!! Remove arnoldi_unit and arnoldi_vec... replace by a single function which takes in a csr_vector
!!! then, check if test_arnoldi still passes !!!
!!! then do the same thing for lanczos !!!
    @staticmethod
    def arnoldi_unit(dim):
        '''
        Run the arnoldi algorithm in for a single orthonormal vector
        Returns a tuple: h-matrix and projected-v matrix

        h_matrix is of size (arnoldi_iter * (arnoldi_iter + 1))
        projected_v_matrix is of size (init_vecs) x key_dirs
        '''

        KrylovInterface._init_static()

        # allocate results
        k = KrylovInterface._cusp.k
        i = KrylovInterface._cusp.i

        result_h = np.zeros((i * (i + 1)), dtype=KrylovInterface.float_type)
        result_pv = np.zeros(((i+1) * k), dtype=KrylovInterface.float_type)

        Timers.tic('arnoldi')
        KrylovInterface._cusp.arnoldi_unit(dim, result_h, len(result_h), result_pv, len(result_pv))
        Timers.toc('arnoldi')

        result_h.shape = (i, i+1)
        result_pv.shape = (i+1, k)

        result_h = result_h.transpose()
        result_pv = result_pv.transpose()

        return result_h, result_pv

    @staticmethod
    def arnoldi_vec(vec):
        '''
        Run the arnoldi algorithm starting at the passed in vector
        Returns a tuple: h-matrix and projected-v matrix

        h_matrix is of size (arnoldi_iter * (arnoldi_iter + 1))
        projected_v_matrix is of size init_vecs x key_dirs
        '''

        assert isinstance(vec, np.ndarray) and vec.dtype == KrylovInterface.float_type

        KrylovInterface._init_static()

        # allocate results
        k = KrylovInterface._cusp.k
        i = KrylovInterface._cusp.i

        result_h = np.zeros((i * (i + 1)), dtype=KrylovInterface.float_type)
        result_pv = np.zeros(((i+1) * k), dtype=KrylovInterface.float_type)

        # normalize vec
        norm = np.linalg.norm(vec)
        assert not math.isinf(norm) and not math.isnan(norm) and norm > 1e-9, "bad initial vec norm: {}".format(norm)

        normalized_vec = vec / norm
        normalized_vec.shape = (len(normalized_vec), )

        Timers.tic('arnoldi')
        KrylovInterface._cusp.arnoldi_vec(normalized_vec, len(normalized_vec), result_h, len(result_h), \
                                          result_pv, len(result_pv))
        Timers.toc('arnoldi')

        result_h.shape = (i, i+1)
        result_pv.shape = (i+1, k)

        result_h = result_h.transpose()
        result_pv = result_pv.transpose()

        # multiply the projection by the initial vec norm to retrieve the correct answer
        result_pv *= norm

        return result_h, result_pv
