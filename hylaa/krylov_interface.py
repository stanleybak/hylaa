'''
Dung Tran & Stanley Bak
August 2017


Simulation of linear system x' = Ax using krylov supspace method in CPU and GPU
'''

import ctypes
import os
import time
import random

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import expm
from hylaa.util import Freezable, get_script_path

class KrylovInterface(Freezable):
    'GPU (and CPU) interface for Krylov subspace simulations of a linear system'

    # static member (library)
    _lib = None
    _lib_path = os.path.join(get_script_path(__file__), 'gpu_interface', 'cusp_krylov_stan.so')

    def __init__(self):
        raise RuntimeError(
            'KrylovInterface is a static class and should not be instantiated')

    @staticmethod
    def _init_static():
        'open the library (if not opened already) and initialize the static members'

        if KrylovInterface._lib is None:

            KrylovInterface._lib = lib = ctypes.CDLL(KrylovInterface._lib_path)

            # int hasGpu()
            KrylovInterface._has_gpu = lib.hasGpu
            KrylovInterface._has_gpu.restype = ctypes.c_int
            KrylovInterface._has_gpu.argtypes = None

            # void reset()
            KrylovInterface._reset = lib.reset
            KrylovInterface._reset.restype = None
            KrylovInterface._reset.argtypes = None

            # CPU and GPU container objects... _funcs stores the selected version
            KrylovInterface._cpu = object()
            KrylovInterface._gpu = object()
            KrylovInterface._funcs = KrylovInterface._cpu # can be changed programatically

            #void loadATransposeGpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
            #           int nonZeroCount)
            cpu_func = KrylovInterface._cpu.load_a_transpose = lib.loadATransposeCpu
            gpu_func = KrylovInterface._gpu.load_a_transpose = lib.loadATransposeGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int
            ]

            # void loadKeyDirMatrixCpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
            #             int nonZeroCount)
            cpu_func = KrylovInterface._cpu.load_key_dir_matrix = lib.loadKeyDirMatrixCpu
            gpu_func = KrylovInterface._gpu.load_key_dir_matrix = lib.loadKeyDirMatrixGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int
            ]

            # double getFreeMemoryMbCpu()
            cpu_func = KrylovInterface._cpu.get_free_memory_mb = lib.getFreeMemoryMbCpu
            gpu_func = KrylovInterface._gpu.get_free_memory_mb = lib.getFreeMemoryMbGpu
            gpu_func.restype = cpu_func.restype = ctypes.c_float
            gpu_func.argtypes = cpu_func.argtypes = None

            # int preallocateMemoryGpu(int arnoldiIterations, int numTimeSteps, int numParallelInitVecs)
            cpu_func = KrylovInterface._cpu.preallocate_memory = lib.preallocateMemoryCpu
            gpu_func = KrylovInterface._gpu.preallocate_memory = lib.preallocateMemoryGpu
            gpu_func.restype = cpu_func.restype = ctypes.c_int
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]

            # void arnoldiParallelCpu(int startDim, double* resultH)
            cpu_func = KrylovInterface._cpu.arnoldi_parallel = lib.arnoldiParallelCpu
            gpu_func = KrylovInterface._gpu.arnoldi_parallel = lib.arnoldiParallelGpu
            gpu_func.restype = cpu_func.restype = None
            gpu_func.argtypes = cpu_func.argtypes = [
                ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
            ]

            # initialize GPU ?

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

    @staticmethod
    def get_free_memory_mb():
        'get the amount of free memory in megabytes'

        KrylovInterface._init_static()

        return KrylovInterface._funcs.get_free_memory_mb()

    @staticmethod
    def load_a_transpose(matrix):
        'load a transpose (should be a sparse matrix)'

        KrylovInterface._init_static()
        w, h = matrix.shape

        assert isinstance(matrix, csr_matrix)
        assert w == h, "a matrix should be square"

        rows, cols = matrix.nonzero()
        entries = matrix[rows, cols].A1.copy()

        KrylovInterface._funcs.load_a_transpose(w, h, rows, cols, entries, len(rows))

        KrylovInterface._loaded_a = True
        KrylovInterface._dims = h

    @staticmethod
    def load_key_dir_matrix(matrix):
        'load key direction sparse matrix'

        KrylovInterface._init_static()
        w, h = matrix.shape

        assert isinstance(matrix, csr_matrix)
        assert KrylovInterface._loaded_a
        assert KrylovInterface._dims == w

        rows, cols = matrix.nonzero()
        entries = matrix[rows, cols].A1.copy()

        KrylovInterface._funcs.load_key_dir_matrix(w, h, entries, len(entries))

        KrylovInterface._loaded_key = True
        KrylovInterface._key_h = h

    @staticmethod
    def preallocate_memory(arnoldi_iterations, time_steps, parallel_init_vecs):
        '''
        preallocate memory used in the parallel arnoldi iteration
        returns True on sucess and False on (memory allocation) error
        '''

        KrylovInterface._init_static()

        result = KrylovInterface._funcs.preallocate_memory(arnoldi_iterations, time_steps, parallel_init_vecs) != 0

        KrylovInterface._preallocated_memory = result
        KrylovInterface._arnoldi_iterations = arnoldi_iterations
        KrylovInterface._parallel_init_vecs = parallel_init_vecs

        return result

    @staticmethod
    def arnoldi_parallel(start_dim):
        '''
        Run the arnoldi algorithm in parallel for a certain number of orthonormal vectors
        Returns h_matrix, which is of size (arnoldi_iter * arnoldi_iter) * parallel_init_vecs

        The matrix may be partially assigned if start_dim + parallel_init_vecs > toal_dims
        '''

        KrylovInterface._init_static()

        assert KrylovInterface._loaded_a
        assert KrylovInterface._preallocated_memory

        # make sure result_h
        it = KrylovInterface._arnoldi_iterations
        result_h = np.zeros((KrylovInterface._parallel_init_vecs * it * it))

        KrylovInterface._funcs.arnoldi_parallel(start_dim, result_h)

        result_h.shape = (KrylovInterface._parallel_init_vecs, it, it)

        return result_h
