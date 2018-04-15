'''
Stanley Bak
Nov 2016
GLPK python <-> C++ interface
'''

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

from scipy.sparse import csr_matrix, csc_matrix

from hylaa.timerutil import Timers
from hylaa.util import Freezable, get_script_path

class LpInstance(Freezable):
    'Linear programm instance using the hylaa python/c++ glpk interface'

    # static member (library)
    _lib = None

    @staticmethod
    def _init_static():
        'open the library (if not opened already) and initialize the static members'

        if LpInstance._lib is None:
            lib_path = os.path.join(get_script_path(__file__), 'glpk_interface', 'hylaa_glpk.so')
            LpInstance._lib = lib = ctypes.CDLL(lib_path)

            # void* initLp(int numStandardVars, int numBasisVars)
            LpInstance._init_lp = lib.initLp
            LpInstance._init_lp.restype = ctypes.c_void_p
            LpInstance._init_lp.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

            # void delLp(void* lpdata)
            LpInstance._del_lp = lib.delLp
            LpInstance._del_lp.restype = None
            LpInstance._del_lp.argtypes = [ctypes.c_void_p]

            # void updateBasisMatrix(void* lpdata, double* matrix, int w, int h)
            LpInstance._update_basis_matrix = lib.updateBasisMatrix
            LpInstance._update_basis_matrix.restype = None
            LpInstance._update_basis_matrix.argtypes = \
                [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

            # void addInputEffectsMatrix(void* lpdata, double* matrix, int w, int h)
            LpInstance._add_input_effects_matrix = lib.addInputEffectsMatrix
            LpInstance._add_input_effects_matrix.restype = None
            LpInstance._add_input_effects_matrix.argtypes = \
                [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

            #void setInitConstraintsCsr(LpData* lpd, int w, int h, double* data, int dataLen, int* inds,
            #               int indsLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
            LpInstance._set_init_constraints_csr = lib.setInitConstraintsCsr
            LpInstance._set_init_constraints_csr.restype = None
            LpInstance._set_init_constraints_csr.argtypes = \
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,\
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setInputConstraintsCsc(LpData* lpd, int w, int h, double* data, int dataLen, int* inds,
            #               int indsLen, int* indptr, int indptrLen, double* rhs, int rhsLen)
            LpInstance._set_input_constraints_csc = lib.setInputConstraintsCsc
            LpInstance._set_input_constraints_csc.restype = None
            LpInstance._set_input_constraints_csc.argtypes = \
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,\
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setOutputConstraints(void* lpdata, double* matrix, int w, int h, double* rhs, int rhsLen)
            LpInstance._set_output_constraints_csr = lib.setOutputConstraintsCsr
            LpInstance._set_output_constraints_csr.restype = None
            LpInstance._set_output_constraints_csr.argtypes = \
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,\
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setNoOutputConstraints(LpData* lpd)
            LpInstance._set_no_output_constraints = lib.setNoOutputConstraints
            LpInstance._set_no_output_constraints.restype = None
            LpInstance._set_no_output_constraints.argtypes = \
                [ctypes.c_void_p]

            # int minimize(void* lpdata, double* direction, int dirLen, double* result, int resLen)
            LpInstance._minimize = lib.minimize
            LpInstance._minimize.restype = ctypes.c_int
            LpInstance._minimize.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
                ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void printLp(void* lpdata)
            LpInstance._print_lp = lib.printLp
            LpInstance._print_lp.restype = None
            LpInstance._print_lp.argtypes = [ctypes.c_void_p]

            # void resetLp(void* lpdata)
            LpInstance._reset_lp = lib.resetLp
            LpInstance._reset_lp.restype = None
            LpInstance._reset_lp.argtypes = [ctypes.c_void_p]

            # int totalIterations()
            LpInstance._total_iterations = lib.totalIterations
            LpInstance._total_iterations.restype = ctypes.c_int
            LpInstance._total_iterations.argtypes = []

            # int totalOptimizations()
            LpInstance._total_optimizations = lib.totalOptimizations
            LpInstance._total_optimizations.restype = ctypes.c_int
            LpInstance._total_optimizations.argtypes = []

            # void test()
            LpInstance._test = lib.test
            LpInstance._test.restype = None
            LpInstance._test.argtypes = []

    def __init__(self, num_output_vars, num_init_vars, num_inputs=0):
        LpInstance._init_static()
        self.lp_data = None

        assert num_output_vars > 0
        assert num_init_vars > 0

        print "initialized lp with {} output vars".format(num_output_vars)
        self.lp_data = LpInstance._init_lp(num_output_vars, num_init_vars, num_inputs)

        # put a copy of del_lp into the object for use in the destructor
        self.del_lp = LpInstance._del_lp

        # for very minor error checking
        self.num_init_vars = num_init_vars
        self.num_output_vars = num_output_vars
        self.num_inputs = num_inputs
        self.added_init_constraints = False
        self.added_input_constraints = False

        self.freeze_attrs()

    def __del__(self):
        if self.lp_data is not None:
            self.del_lp(self.lp_data)
            self.lp_data = None

    def set_init_constraints(self, constraint_mat, rhs):
        '''set the initial state constraints'''

        assert not self.added_init_constraints, "Init constraints attempted to be set twice"
        assert isinstance(constraint_mat, csr_matrix)
        assert isinstance(rhs, np.ndarray)
        assert rhs.shape == (constraint_mat.shape[0],)
        assert self.num_init_vars == constraint_mat.shape[1], "incorrect init constraints width"

        Timers.tic("lp overhead")
        LpInstance._set_init_constraints_csr(self.lp_data, constraint_mat.shape[1], constraint_mat.shape[0], \
            constraint_mat.data, len(constraint_mat.data), constraint_mat.indices, len(constraint_mat.indices), \
            constraint_mat.indptr, len(constraint_mat.indptr), rhs, rhs.shape[0])

        Timers.toc("lp overhead")
        self.added_init_constraints = True

    def set_output_constraints(self, constraint_mat, rhs):
        '''set the output state constraints'''

        assert isinstance(constraint_mat, csr_matrix)
        assert isinstance(rhs, np.ndarray)
        assert rhs.shape == (constraint_mat.shape[0],)
        assert self.num_output_vars == constraint_mat.shape[1], "incorrect output constraints width"

        Timers.tic("lp overhead")
        LpInstance._set_output_constraints_csr(self.lp_data, constraint_mat.shape[1], constraint_mat.shape[0], \
            constraint_mat.data, len(constraint_mat.data), constraint_mat.indices, len(constraint_mat.indices), \
            constraint_mat.indptr, len(constraint_mat.indptr), rhs, rhs.shape[0])

        Timers.toc("lp overhead")

    def set_no_output_constraints(self):
        '''set the output state constraints (no constriants, for plotting, for example)'''

        Timers.tic("lp overhead")
        LpInstance._set_no_output_constraints(self.lp_data)
        Timers.toc("lp overhead")

    def update_basis_matrix(self, matrix):
        'update the basis matrix in an lp'

        assert self.num_output_vars is not None
        assert self.num_init_vars is not None
        assert isinstance(matrix, np.ndarray)

        assert matrix.shape == (self.num_output_vars, self.num_init_vars), "Expected {}x{} basis mat, got {}x{}".format(
            self.num_output_vars, self.num_init_vars, matrix.shape[0], matrix.shape[1])

        Timers.tic("lp overhead")
        LpInstance._update_basis_matrix(self.lp_data, matrix, matrix.shape[1], matrix.shape[0])
        Timers.toc("lp overhead")

    def print_lp(self):
        '''print the lp constraint matrix to stdout (a debugging function)'''

        assert self.num_output_vars is not None
        assert self.num_init_vars is not None

        LpInstance._print_lp(self.lp_data)

    def reset_lp(self):
        '''reset the lp statuses'''

        assert self.num_output_vars is not None
        assert self.num_init_vars is not None

        LpInstance._reset_lp(self.lp_data)

    def minimize(self, direction, result, error_if_infeasible=False):
        '''
        minimize a constraint using the cur-time variables. this returns True of False, depending on
        whether the LP was feasible. If it was feasible, the passed-in 'result' vector is assigned.

        If the result vector is of size num_output_vars, then only the output_vars
        which minimize the objective will be set, otherwise, the complete LP result will be copied, for
        as many entries as exist in the passed-in result np.ndarray
        '''

        assert self.num_output_vars is not None
        assert self.num_init_vars is not None

        dir_len, = direction.shape

        assert dir_len == self.num_output_vars, \
            "minimize objective length({}) should match number of output variables({})".format(
                dir_len, self.num_output_vars)

        # result must be a 1-d np.array
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 1

        Timers.tic("lp minimize")
        is_feasible = LpInstance._minimize(self.lp_data, direction, dir_len, result, result.shape[0]) == 0
        Timers.toc("lp minimize")

        if not is_feasible and error_if_infeasible:
            raise RuntimeError('minimize LP was infeasible when error_if_infeasible=True')

        return is_feasible

    def set_input_constraints_csc(self, constraint_mat, rhs):
        '''set the input constraints'''

        assert isinstance(constraint_mat, csc_matrix)
        assert not self.added_input_constraints, "input constraints set twice"
        assert self.num_inputs == constraint_mat.shape[1]
        self.added_input_constraints = True

        LpInstance._set_input_constraints_csc(self.lp_data, constraint_mat.shape[1], constraint_mat.shape[0], \
        constraint_mat.data, len(constraint_mat.data), constraint_mat.indices, len(constraint_mat.indices), \
        constraint_mat.indptr, len(constraint_mat.indptr), rhs, rhs.shape[0])

    def add_input_effects_matrix(self, matrix):
        'update the basis matrix in an lp'

        assert self.added_input_constraints, "input constraints should be set before adding input effects matrix"
        assert self.num_output_vars is not None
        assert self.num_init_vars is not None
        assert isinstance(matrix, np.ndarray)

        assert matrix.shape == (self.num_output_vars, self.num_inputs), "Expected {}x{} basis mat, got {}x{}".format(
            self.num_output_vars, self.num_inputs, matrix.shape[0], matrix.shape[1])

        Timers.tic("lp overhead")
        LpInstance._add_input_effects_matrix(self.lp_data, matrix, matrix.shape[1], matrix.shape[0])
        Timers.toc("lp overhead")

    @staticmethod
    def total_iterations():
        '''returns the total number of lp iterations performed over all the problems'''
        LpInstance._init_static()

        return LpInstance._total_iterations()

    @staticmethod
    def total_optimizations():
        '''returns the total number of lp minimize operations performed over all the problems'''
        LpInstance._init_static()

        return LpInstance._total_optimizations()

    @staticmethod
    def test():
        'call the test() interface method'

        LpInstance._init_static()

        LpInstance._test()

    @staticmethod
    def print_stats():
        'print stats about lp solving to stdout'

        print "LP minimize calls: {}".format(LpInstance.total_optimizations())
        print "LP iterations: {}".format(LpInstance.total_iterations())
