'''
Stanley Bak
Nov 2016
GLPK python <-> C++ interface
'''

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

from scipy.sparse import csr_matrix

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
            LpInstance._init_lp.argtypes = [ctypes.c_int, ctypes.c_int]

            # void delLp(void* lpdata)
            LpInstance._del_lp = lib.delLp
            LpInstance._del_lp.restype = None
            LpInstance._del_lp.argtypes = [ctypes.c_void_p]

            # void updateTimeElapseMatrix(void* lpdata, double* matrix, int w, int h)
            LpInstance._update_time_elapse_matrix = lib.updateTimeElapseMatrix
            LpInstance._update_time_elapse_matrix.restype = ctypes.c_int
            LpInstance._update_time_elapse_matrix.argtypes = \
                [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

            # void setInitConstraints(void* lpdata, double* data, int dataLen, int* indices, int indicesLen,
            #                         int* indptr, int indptrLen, double* rhs, int rhsLen)

            LpInstance._set_init_constraints = lib.setInitConstraints
            LpInstance._set_init_constraints.restype = None
            LpInstance._set_init_constraints.argtypes = \
                [ctypes.c_void_p, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setCurTimeConstraints(void* lpdata, double* data, int dataLen, int* indices, int indicesLen,
            #                            int* indptr, int indptrLen, double* rhs, int rhsLen)
            LpInstance._set_cur_time_constraints = lib.setCurTimeConstraints
            LpInstance._set_cur_time_constraints.restype = None
            LpInstance._set_cur_time_constraints.argtypes = \
                [ctypes.c_void_p, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int minimize(void* lpdata, double* direction, int dirLen, double* result, int resLen)
            LpInstance._minimize = lib.minimize
            LpInstance._minimize.restype = ctypes.c_int
            LpInstance._minimize.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
                ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void printLp(void* lpdata)
            LpInstance._print_lp = lib.printLp
            LpInstance._print_lp.restype = None
            LpInstance._print_lp.argtypes = [ctypes.c_void_p]

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

    def __init__(self, num_cur_time_vars, num_init_vars):
        LpInstance._init_static()

        self.lp_data = LpInstance._init_lp(num_cur_time_vars, num_init_vars)

        # put a copy of del_lp into the object for use in the destructor
        self.del_lp = LpInstance._del_lp

        # for error-checking
        self.num_cur_time_vars = num_cur_time_vars
        self.num_init_vars = num_init_vars

        self.added_init_constraints = False
        self.added_cur_time_constraints = False
        self.added_time_elapse_matrix = False

        self.freeze_attrs()

    def __del__(self):
        self.del_lp(self.lp_data)
        self.lp_data = None

    def update_time_elapse_matrix(self, matrix):
        'update the time elapse matrix in an lp'

        assert self.added_init_constraints
        assert isinstance(matrix, np.ndarray)
        assert len(matrix.shape) == 2, "expected 2d matrix"

        assert matrix.shape[0] == self.num_cur_time_vars, "time-elapse matrix wrong height"
        assert matrix.shape[1] == self.num_init_vars, "time-elapse matrix wrong width"

        Timers.tic("lp update_time_elapse_matrix")
        rv = LpInstance._update_time_elapse_matrix(self.lp_data, matrix, matrix.shape[1], matrix.shape[0])
        Timers.toc("lp update_time_elapse_matrix")

        self.added_time_elapse_matrix = True

        if rv != 0:
            raise RuntimeError("update_basis_matrix failed")

    def set_init_constraints(self, constraint_mat, rhs):
        '''set the initial state constraints'''

        assert not self.added_init_constraints
        assert not self.added_cur_time_constraints
        assert not self.added_time_elapse_matrix
        assert isinstance(constraint_mat, csr_matrix)
        assert isinstance(rhs, np.ndarray)
        assert rhs.shape == (constraint_mat.shape[0],)

        Timers.tic("lp set_init_constraints")
        data = constraint_mat.data
        indices = constraint_mat.indices
        indptr = constraint_mat.indptr

        LpInstance._set_init_constraints(self.lp_data, data, data.shape[0], indices, indices.shape[0],
                                         indptr, indptr.shape[0], rhs, rhs.shape[0])

        Timers.toc("lp set_init_constraints")

        self.added_init_constraints = True

    def set_cur_time_constraints(self, constraint_mat, rhs):
        '''set the constraints to be checked at each time step'''

        assert self.added_init_constraints
        assert not self.added_cur_time_constraints
        assert not self.added_time_elapse_matrix

        assert isinstance(constraint_mat, csr_matrix)
        assert isinstance(rhs, np.ndarray)
        assert rhs.shape == (constraint_mat.shape[0],)

        Timers.tic("lp set_cur_time_constraints")
        data = constraint_mat.data
        indices = constraint_mat.indices
        indptr = constraint_mat.indptr

        LpInstance._set_cur_time_constraints(self.lp_data, data, data.shape[0], indices, indices.shape[0],
                                             indptr, indptr.shape[0], rhs, rhs.shape[0])
        Timers.toc("lp set_cur_time_constraints")

        self.added_cur_time_constraints = True

    def print_lp(self):
        '''print the lp constraint matrix to stdout (a debugging function)'''

        LpInstance._print_lp(self.lp_data)

    def minimize(self, direction, result, error_if_infeasible=False):
        '''
        minimize a constraint using the cur-time variables. this returns True of False, depending on
        whether the LP was feasible. If it was feasible, the passed-in 'result' vector is assigned.

        The result vector can be either of size num_cur_time_vars or (self.num_cur_time_vars + self.num_init_vars).
        If it's the smaller one, only the result for the current-time variables is used.
        '''

        assert self.added_time_elapse_matrix

        dir_len, = direction.shape
        res_len, = result.shape
        total_vars = (self.num_cur_time_vars + self.num_init_vars)

        assert dir_len == self.num_cur_time_vars, \
            "minimize objective length({}) should match number of cur-time variables({})".format(
                dir_len, self.num_cur_time_vars)

        assert res_len == self.num_cur_time_vars or res_len == total_vars, \
            ("result length({}) should match either number of cur-time variables({}) " + \
                "or total num variables({})").format(res_len, self.num_cur_time_vars, total_vars)

        Timers.tic("lp minimize")
        minimize_result = np.zeros(total_vars)
        res = LpInstance._minimize(self.lp_data, direction, dir_len, minimize_result, total_vars)
        Timers.toc("lp minimize")

        if res_len == self.num_cur_time_vars:
            result[:] = minimize_result[self.num_init_vars:]
        else:
            result[:] = minimize_result[:]

        is_feasible = (res == 0)

        if not is_feasible and error_if_infeasible:
            raise RuntimeError('minimize LP was infeasible when error_if_infeasible=True')

        return is_feasible

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
