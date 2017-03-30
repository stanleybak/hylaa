'''
Stanley Bak
Nov 2016
GLPK python <-> C++ interface
'''

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

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

            # void updateBasisMatrix(void* lpdata, double* matrix, int w, int h)
            LpInstance._update_basis_matrix = lib.updateBasisMatrix
            LpInstance._update_basis_matrix.restype = ctypes.c_int
            LpInstance._update_basis_matrix.argtypes = \
                [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

            # void addBasisConstraint(void* lpdata, double* aVec, int aVecLen, double bVal)
            LpInstance._add_basis_constraint = lib.addBasisConstraint
            LpInstance._add_basis_constraint.restype = None
            LpInstance._add_basis_constraint.argtypes = \
                [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_double]

            # void addBasisConstraint(void* lpdata, double* aVec, int aVecLen, double bVal)
            LpInstance._add_standard_constraint = lib.addStandardConstraint
            LpInstance._add_standard_constraint.restype = None
            LpInstance._add_standard_constraint.argtypes = \
                [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_double]

            # void addInputStar(void* lpdata, double* aMatrix, int aWidth, int aHeight, double* bVec, int bLen,
            #                   double* basisMatrix, int bmWidth, int bmHeight)
            LpInstance._add_input_star = lib.addInputStar
            LpInstance._add_input_star.restype = None
            LpInstance._add_input_star.argtypes = \
                [ctypes.c_void_p,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]

            # int minimize(void* lpdata, double* direction, int dirLen, double* result, int resLen)
            LpInstance._minimize = lib.minimize
            LpInstance._minimize.restype = ctypes.c_int
            LpInstance._minimize.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
                ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void getColStatuses(void* lpdata, char* store, int storeLen)
            LpInstance._get_col_statuses = lib.getColStatuses
            LpInstance._get_col_statuses.restype = ctypes.c_int
            LpInstance._get_col_statuses.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"),
                                                     ctypes.c_int]

            # void getRowStatuses(void* lpdata, char* store, int storeLen)
            LpInstance._get_row_statuses = lib.getRowStatuses
            LpInstance._get_row_statuses.restype = ctypes.c_int
            LpInstance._get_row_statuses.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"),
                                                     ctypes.c_int]

            # void setLastInputStatuses(void* lpdata, char* rowStats, int rLen, char* colStats, int cLen)
            LpInstance._set_last_input_statuses = lib.setLastInputStatuses
            LpInstance._set_last_input_statuses.restype = None
            LpInstance._set_last_input_statuses.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"), ctypes.c_int, \
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setStandardBasisStatuses(void* lpdata, char* rowStats, int rLen, char* colStats, int cLen)
            LpInstance._set_standard_basis_statuses = lib.setStandardBasisStatuses
            LpInstance._set_standard_basis_statuses.restype = None
            LpInstance._set_standard_basis_statuses.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"), ctypes.c_int, \
                ndpointer(ctypes.c_int8, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setStandardConstraintValues(void* lpdata, double* vals, int len)
            LpInstance._set_standard_constraint_values = lib.setStandardConstraintValues
            LpInstance._set_standard_constraint_values.restype = None
            LpInstance._set_standard_constraint_values.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # void setBasisConstraintValues(void* lpdata, double* vals, int len)
            LpInstance._set_basis_constraint_values = lib.setBasisConstraintValues
            LpInstance._set_basis_constraint_values.restype = None
            LpInstance._set_basis_constraint_values.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

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

    def __init__(self, num_standard_vars, num_basis_vars):
        LpInstance._init_static()

        self.lp_data = LpInstance._init_lp(num_standard_vars, num_basis_vars)

        # put a copy of del_lp into the object for use in the destructor
        self.del_lp = LpInstance._del_lp

        # for error-checking
        self.num_standard_vars = None
        self.num_basis_vars = None
        self.num_inputs = None
        self.added_standard_constraint = False

        self.freeze_attrs()

    def __del__(self):
        self.del_lp(self.lp_data)
        self.lp_data = None

    def get_row_statuses(self, store):
        'get the row statuses of the current lp solution, and store then in the passed-in variable'

        size = store.shape[0]

        Timers.tic("lp get_statuses")
        rv = LpInstance._get_row_statuses(self.lp_data, store, size)
        Timers.toc("lp get_statuses")

        if rv != 0:
            raise RuntimeError("get_row_statuses failed")

    def get_col_statuses(self, store):
        'get the col statuses of the current lp solution, and store then in the passed-in variable'

        size = store.shape[0]

        Timers.tic("lp get_statuses")
        rv = LpInstance._get_col_statuses(self.lp_data, store, size)
        Timers.toc("lp get_statuses")

        if rv != 0:
            raise RuntimeError("get_col_statuses failed")

    def set_last_input_statuses(self, row_statuses, col_statuses):
        'set the statuses for the last-added input star'

        row_size = row_statuses.shape[0]
        col_size = col_statuses.shape[0]

        Timers.tic("lp set_statuses")
        LpInstance._set_last_input_statuses(self.lp_data, row_statuses, row_size, col_statuses, col_size)
        Timers.toc("lp set_statuses")

    def set_standard_basis_statuses(self, row_statuses, col_statuses):
        'set the statuses for the standard variables / constraints and basis variables / constraints'

        row_size = row_statuses.shape[0]
        col_size = col_statuses.shape[0]

        Timers.tic("lp set_statuses")
        LpInstance._set_standard_basis_statuses(self.lp_data, row_statuses, row_size, col_statuses, col_size)
        Timers.toc("lp set_statuses")

    def update_basis_matrix(self, matrix):
        'update the basis matrix in an lp'

        assert isinstance(matrix, np.ndarray)
        assert len(matrix.shape) == 2, "expected 2d matrix"

        if self.num_standard_vars is None:
            self.num_basis_vars = matrix.shape[0]
            self.num_standard_vars = matrix.shape[1]

        Timers.tic("lp update_basis_matrix")
        rv = LpInstance._update_basis_matrix(self.lp_data, matrix, matrix.shape[1], matrix.shape[0])
        Timers.toc("lp update_basis_matrix")

        if rv != 0:
            raise RuntimeError("update_basis_matrix failed")

    def add_basis_constraint(self, a_vec, b_val):
        '''add a constraint in the star's basis'''

        if len(a_vec.shape) == 1:
            w = a_vec.shape[0]
        else:
            h, w = a_vec.shape
            assert h == 1, "expected 1-d vector in add_basis_constaint()"

        assert self.num_inputs is None, "add_basis_constraint() called after adding inputs to LP"
        assert w == self.num_basis_vars, "add_basis_constraint() had incorrect length: {}; expected: {}".format(
            w, self.num_basis_vars)

        Timers.tic("lp add_basis_constraint")
        LpInstance._add_basis_constraint(self.lp_data, a_vec, w, b_val)
        Timers.toc("lp add_basis_constraint")

    def add_standard_constraint(self, a_vec, b_val):
        '''add a constraint in the standard basis'''

        if len(a_vec.shape) == 1:
            w = a_vec.shape[0]
        else:
            h, w = a_vec.shape
            assert h == 1, "expected 1-d vector in add_standard_constaint()"

        Timers.tic("lp add_standard_constraint")
        LpInstance._add_standard_constraint(self.lp_data, a_vec, w, b_val)
        Timers.toc("lp add_standard_constraint")

        self.added_standard_constraint = True

    def add_input_star(self, a_matrix_t, b_vec, input_basis_matrix):
        '''minkowski add an input star into the lp (creates 1 new variable for each input)'''

        assert len(a_matrix_t.shape) == 2
        assert len(b_vec.shape) == 1

        assert a_matrix_t.shape[1] == b_vec.shape[0], "number of rows in constraints must match"
        assert a_matrix_t.shape[0] == input_basis_matrix.shape[0], \
            "number of columns in constraint matix / rows in input basis matrix must match"

        assert input_basis_matrix.shape[1] == self.num_standard_vars, "input basis matrix cols must match standard vars"

        if self.num_inputs is None:
            self.num_inputs = input_basis_matrix.shape[0]
        else:
            assert input_basis_matrix.shape[0] == self.num_inputs, "num_inputs changed between calls to add_input_star"

        Timers.tic("lp add_input_star")
        LpInstance._add_input_star(self.lp_data, a_matrix_t, a_matrix_t.shape[1], a_matrix_t.shape[0], b_vec, \
            b_vec.shape[0], input_basis_matrix, input_basis_matrix.shape[1], input_basis_matrix.shape[0])

        Timers.toc("lp add_input_star")

    def print_lp(self):
        '''print the lp constraint matrix to stdout (a debugging function)'''

        LpInstance._print_lp(self.lp_data)

    def set_standard_constraint_values(self, constraint_vals):
        '''set the values (right-hand-sides) of each of the standard var constraints'''

        LpInstance._set_standard_constraint_values(self.lp_data, constraint_vals, constraint_vals.shape[0])

    def set_basis_constraint_values(self, constraint_vals):
        '''set the values (right-hand-sides) of each of the basis var constraints'''

        LpInstance._set_basis_constraint_values(self.lp_data, constraint_vals, constraint_vals.shape[0])

    def minimize(self, direction, result, error_if_infeasible=False):
        '''
        minimize a constraint in the standard basis. this returns True of False, depending on
        whether the LP was feasible. If it was feasible, the passed-in 'result' vector is assigned
        '''

        assert len(direction) == self.num_standard_vars, \
            "minimize objective length({}) should match number of standard variables({})".format(
                len(direction), self.num_standard_vars)

        dir_len, = direction.shape
        res_len, = result.shape

        Timers.tic("lp minimize")
        res = LpInstance._minimize(self.lp_data, direction, dir_len, result, res_len)
        Timers.toc("lp minimize")

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
