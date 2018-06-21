'''
Stanley Bak
May 2018
GLPK python <-> C++ interface
'''

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

from scipy.sparse import csr_matrix, csc_matrix

def get_script_path(filename):
    '''get the path this script, pass in __file__ for the filename'''
    return os.path.dirname(os.path.realpath(filename))

class Freezable(object):
    'a class where you can freeze the fields (prevent new fields from being created)'

    _frozen = False

    def freeze_attrs(self):
        'prevents any new attributes from being created in the object'
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError("{} does not contain attribute '{}' (object was frozen)".format(self, key))

        object.__setattr__(self, key, value)

class LpInstance(Freezable):
    'Linear programm instance using the hylaa python/c++ glpk interface'

    # static member (library)
    _lib = None

    @staticmethod
    def _init_static():
        'open the library (if not opened already) and initialize the static members'

        if LpInstance._lib is None:
            lib_path = os.path.join(get_script_path(__file__), 'python_glpk.so')

            LpInstance._lib = lib = ctypes.CDLL(lib_path)

            # glp_prob* initLp()
            LpInstance._init_lp = lib.initLp
            LpInstance._init_lp.restype = ctypes.c_void_p
            LpInstance._init_lp.argtypes = []

            # void delLp(glp_prob* lp)
            LpInstance._del_lp = lib.delLp
            LpInstance._del_lp.restype = None
            LpInstance._del_lp.argtypes = [ctypes.c_void_p]

            # void resetLp(glp_prob* lp)
            LpInstance._reset_lp = lib.resetLp
            LpInstance._reset_lp.restype = None
            LpInstance._reset_lp.argtypes = [ctypes.c_void_p]

            #void printLp(glp_prob* lp)
            LpInstance._print_lp = lib.printLp
            LpInstance._print_lp.restype = None
            LpInstance._print_lp.argtypes = [ctypes.c_void_p]

            #void addCols(glp_prob* lp, int num)
            LpInstance._add_cols = lib.addCols
            LpInstance._add_cols.restype = None
            LpInstance._add_cols.argtypes = [ctypes.c_void_p, ctypes.c_int]

            #void addRowsLessEqual(glp_prob* lp, double* rhs, int rhsLen)
            LpInstance._add_rows_less_equal = lib.addRowsLessEqual
            LpInstance._add_rows_less_equal.restype = None
            LpInstance._add_rows_less_equal.argtypes = [ctypes.c_void_p,
                                                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                        ctypes.c_int]

            #void addRowsEqualZero(glp_prob* lp, int num)
            LpInstance._add_rows_equal_zero = lib.addRowsEqualZero
            LpInstance._add_rows_equal_zero.restype = None
            LpInstance._add_rows_equal_zero.argtypes = [ctypes.c_void_p, ctypes.c_int]

            #int setConstraintsCsr(glp_prob* lp, int rowOffset, int colOffset, double* data, int dataLen,
            #          int* indices, int indicesLen, int* indptr, int indptrLen, int numRows,
            #          int numCols)
            LpInstance._set_constraints_csr = lib.setConstraintsCsr
            LpInstance._set_constraints_csr.restype = ctypes.c_int
            LpInstance._set_constraints_csr.argtypes = \
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ctypes.c_int, ctypes.c_int]

            #int setConstraintsCsc(glp_prob* lp, int rowOffset, int colOffset, double* data, int dataLen,
            #          int* indices, int indicesLen, int* indptr, int indptrLen, int numRows,
            #          int numCols)
            LpInstance._set_constraints_csc = lib.setConstraintsCsc
            LpInstance._set_constraints_csc.restype = ctypes.c_int
            LpInstance._set_constraints_csc.argtypes = \
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                 ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int,
                 ctypes.c_int, ctypes.c_int]

            # int minimize(glp_prob* lp, double* result, int resLen)
            LpInstance._minimize = lib.minimize
            LpInstance._minimize.restype = ctypes.c_int
            LpInstance._minimize.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int setMinimizeDirection(glp_prob* lp, double* direction, int dirLen)
            LpInstance._set_minimize_direction = lib.setMinimizeDirection
            LpInstance._set_minimize_direction.restype = ctypes.c_int
            LpInstance._set_minimize_direction.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int getIterations(glp_prob* lp)
            LpInstance._get_iterations = lib.getIterations
            LpInstance._get_iterations.restype = ctypes.c_int
            LpInstance._get_iterations.argtypes = [ctypes.c_void_p]

            # int getNumRows(glp_prob* lp)
            LpInstance._get_num_rows = lib.getNumRows
            LpInstance._get_num_rows.restype = ctypes.c_int
            LpInstance._get_num_rows.argtypes = [ctypes.c_void_p]

            # int getNumCols(glp_prob* lp)
            LpInstance._get_num_cols = lib.getNumCols
            LpInstance._get_num_cols.restype = ctypes.c_int
            LpInstance._get_num_cols.argtypes = [ctypes.c_void_p]

            # int getMatrix(glp_prob* lp, double* mat, double* vec, int rows, int cols)
            LpInstance._get_matrix = lib.getMatrix
            LpInstance._get_matrix.restype = ctypes.c_int
            LpInstance._get_matrix.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
                ctypes.c_int, ctypes.c_int]

            # int test()
            LpInstance._test = lib.test
            LpInstance._test.restype = ctypes.c_int
            LpInstance._test.argtypes = []


    def __init__(self):
        LpInstance._init_static()
        self.lp_data = None

        self.lp_data = LpInstance._init_lp()

        # put a copy of del_lp into the object for use in the destructor
        self.del_lp = LpInstance._del_lp

        self.freeze_attrs()

    def __del__(self):
        if hasattr(self, 'lp_data') and self.lp_data is not None:
            self.del_lp(self.lp_data)
            self.lp_data = None

    def print_lp(self):
        'print the LP to stdout (useful for debugging small instances)'

        LpInstance._print_lp(self.lp_data)

    def reset_lp(self):
        'reset the lp statuses'

        LpInstance._reset_lp(self.lp_data)

    def add_cols(self, num):
        'add a certain number of columns to the LP'

        LpInstance._add_cols(self.lp_data, num)

    def add_rows_less_equal(self, rhs_vec):
        '''add rows to the LP with <= constraints

        rhs_vector is the right-hand-side values of the constriants
        '''

        if isinstance(rhs_vec, list):
            rhs_vec = np.array(rhs_vec, dtype=float)

        assert isinstance(rhs_vec, np.ndarray) and len(rhs_vec.shape) == 1, "expected 1-d right-hand-side vector"

        LpInstance._add_rows_less_equal(self.lp_data, rhs_vec, len(rhs_vec))

    def add_rows_equal_zero(self, num):
        '''add rows to the LP with == 0 constraints'''

        LpInstance._add_rows_equal_zero(self.lp_data, num)

    def set_constraints_csr(self, csr_mat, offset=None):
        '''set the constrains row by row to be equal to the passed-in csr matrix

        offset is an optional tuple (num_rows, num_cols) which tells you the top-left offset for the assignment
        '''

        assert isinstance(csr_mat, csr_matrix)

        if offset is None:
            offset = (0, 0)

        assert len(offset) == 2, "offset should be a 2-tuple (num_rows, num_cols)"

        res = LpInstance._set_constraints_csr(self.lp_data, offset[0], offset[1], csr_mat.data, len(csr_mat.data), \
            csr_mat.indices, len(csr_mat.indices), csr_mat.indptr, len(csr_mat.indptr), \
            csr_mat.shape[0], csr_mat.shape[1])

        if res != 0:
            raise RuntimeError("LP set_constraints_csr failed")

    def set_constraints_csc(self, csc_mat, offset=None):
        '''set the constrains column by column to be equal to the passed-in csc matrix

        offset is an optional tuple (num_rows, num_cols) which tells you the top-left offset for the assignment
        '''

        assert isinstance(csc_mat, csc_matrix)

        if offset is None:
            offset = (0, 0)

        assert len(offset) == 2, "offset should be a 2-tuple (num_rows, num_cols)"

        res = LpInstance._set_constraints_csc(self.lp_data, offset[0], offset[1], csc_mat.data, len(csc_mat.data), \
            csc_mat.indices, len(csc_mat.indices), csc_mat.indptr, len(csc_mat.indptr), \
            csc_mat.shape[0], csc_mat.shape[1])

        if res != 0:
            raise RuntimeError("LP set_constraints_csc failed")

    def set_minimize_direction(self, direction_vec):
        'set the direction for the optimization'

        if not isinstance(direction_vec, np.ndarray):
            direction_vec = np.array(direction_vec, dtype=float)

        assert len(direction_vec.shape) == 1

        res = LpInstance._set_minimize_direction(self.lp_data, direction_vec, len(direction_vec))

        if res != 0:
            raise RuntimeError("set_minimize_direction failed")

    def minimize_partial_result(self, result_vec, fail_on_unsat=False):
        '''minimize the LP, getting some of the columns as a result

        result_vec is an in/out parameter. It's initialized to the requested column numbers. It gets assigned
        with the result (if LP is SAT). It's also returned.
        '''

        if not isinstance(result_vec, np.ndarray):
            result_vec = np.array(result_vec, dtype=float)

        assert len(result_vec.shape) == 1

        assert isinstance(result_vec, np.ndarray) and len(result_vec.shape) == 1

        # minimize() returns 0 on success, 1 on unsat, -1 on error
        res = LpInstance._minimize(self.lp_data, result_vec, len(result_vec))

        if res == -1:
            raise RuntimeError("LP minimize() failed internally")

        if res == 1 and fail_on_unsat:
            raise RuntimeError("LP minimize() returned UNSAT, but fail_on_unsat was True")

        return result_vec

    def minimize_full_result(self, fail_on_unsat=False):
        'minimize the lp. returns the LP assigment for every column if SAT (as an np.ndarray), else None'

        num_cols = self.get_num_cols()

        result_vec = np.array([float(n) for n in range(num_cols)]) # get each column

        # minimize() returns 0 on success, 1 on unsat, -1 on error
        res = LpInstance._minimize(self.lp_data, result_vec, len(result_vec))

        if res == -1:
            raise RuntimeError("LP minimize() failed internally")

        if res == 1 and fail_on_unsat:
            raise RuntimeError("LP minimize() returned UNSAT, but fail_on_unsat was True")

        if res != 0:
            result_vec = None

        return result_vec

    def minimize(self, direction_vec, fail_on_unsat=False):
        'minimize the lp. This assigns the direction vec and returns the full result'

        self.set_minimize_direction(direction_vec)

        return self.minimize_full_result(fail_on_unsat=fail_on_unsat)

    def get_matrix(self):
        '''get the LP matrix as a dense np.ndarray (slow function, for testing)

        returns a tupleL (mat, vec), where mat is constraints, and vec is the right-hand side
        '''

        rows = self.get_num_rows()
        cols = self.get_num_cols()

        mat = np.zeros((rows, cols))
        vec = np.zeros((rows,))

        res = LpInstance._get_matrix(self.lp_data, mat, vec, rows, cols)

        if res != 0:
            raise RuntimeError("get_matrix failed")

        return mat, vec

    def get_num_rows(self):
        'get the number of rows in the lp'

        return LpInstance._get_num_rows(self.lp_data)

    def get_num_cols(self):
        'get the number of columns in the lp'

        return LpInstance._get_num_cols(self.lp_data)

    def get_iterations(self):
        'get the number of LP iterations performed so far'

        return LpInstance._get_iterations(self.lp_data)

    @staticmethod
    def test():
        '''call the c++ unit test function. returns 0 on success'''

        LpInstance._init_static()
        return LpInstance._test()
