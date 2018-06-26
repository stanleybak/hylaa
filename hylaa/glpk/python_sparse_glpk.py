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

    # static types for types in get_constraints
    GLP_FX = 5 # == constraint
    GLP_UP = 3 # <= constraint
    GLP_LO = 2 # >= constraint

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

            # glp_prob* copyLp(glp_prob* other)
            LpInstance._copy_lp = lib.copyLp
            LpInstance._copy_lp.restype = ctypes.c_void_p
            LpInstance._copy_lp.argtypes = [ctypes.c_void_p]

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

            # int minimize(glp_prob* lp, int* columns, double* result, int resLen)
            LpInstance._minimize = lib.minimize
            LpInstance._minimize.restype = ctypes.c_int
            LpInstance._minimize.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), \
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

            # int flipConstraint(glp_prob* lp, int rowIndex)
            LpInstance._flip_constraint = lib.flipConstraint
            LpInstance._flip_constraint.restype = ctypes.c_int
            LpInstance._flip_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int]

            # int delConstraint(glp_prob* lp, int rowIndex)
            LpInstance._del_constraint = lib.delConstraint
            LpInstance._del_constraint.restype = ctypes.c_int
            LpInstance._del_constraint.argtypes = [ctypes.c_void_p, ctypes.c_int]

            # int setConstraintRhs(glp_prob* lp, int rowIndex, double rhs)
            LpInstance._set_constraint_rhs = lib.setConstraintRhs
            LpInstance._set_constraint_rhs.restype = ctypes.c_int
            LpInstance._set_constraint_rhs.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

            # int getRhs(glp_prob* lp, double* vec, int vecLen)
            LpInstance._get_rhs = lib.getRhs
            LpInstance._get_rhs.restype = ctypes.c_int
            LpInstance._get_rhs.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int getTypes(glp_prob* lp, int* vec, int vecLen)
            LpInstance._get_types = lib.getTypes
            LpInstance._get_types.restype = ctypes.c_int
            LpInstance._get_types.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int getNumNz(glp_prob* lp)
            LpInstance._get_num_nz = lib.getNumNz
            LpInstance._get_num_nz.restype = ctypes.c_int
            LpInstance._get_num_nz.argtypes = [ctypes.c_void_p]

            #int getRow(glp_prob* lp, int row, double* data, int dataLen, int* inds, int indsLen)
            LpInstance._get_row = lib.getRow
            LpInstance._get_row.restype = ctypes.c_int
            LpInstance._get_row.argtypes = [ctypes.c_void_p, ctypes.c_int, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int getConstraints(glp_prob* lp, double* data, int dataLen, int* ind, int indLen,
            #                    int* indPtr, int indPtrLen)
            LpInstance._get_constraints = lib.getConstraints
            LpInstance._get_constraints.restype = ctypes.c_int
            LpInstance._get_constraints.argtypes = [ctypes.c_void_p, \
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int, \
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, \
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]

            # int getSubConstraints(glp_prob* lp, int x, int y, int w, int h, double* data, int dataLen)
            LpInstance._get_subconstraints = lib.getSubConstraints
            LpInstance._get_subconstraints.restype = ctypes.c_int
            LpInstance._get_subconstraints.argtypes = [ctypes.c_void_p, \
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]
                
            # int test()
            LpInstance._test = lib.test
            LpInstance._test.restype = ctypes.c_int
            LpInstance._test.argtypes = []

    def __init__(self, other_lpi=None):
        'create a copy of the passed-in lpi'
        LpInstance._init_static()
        self.lp_data = None

        if other_lpi is not None:
            self.lp_data = LpInstance._copy_lp(other_lpi.lp_data)
        else:
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

    def minimize_partial_result(self, columns, fail_on_unsat=True):
        '''minimize the LP, getting some of the columns as a result

        columns is the requested column numbers. These are the columns that
        get returned when SAT.

        If unsat, and fail_on_unsat is False, None is returned
        '''

        if not isinstance(columns, np.ndarray):
            columns = np.array(columns, dtype=np.int32)

        if columns.dtype != np.int32: # convert it
            assert columns.dtype != float, "expected array of integers"
            columns = np.array(columns, dtype=np.int32)

        assert len(columns.shape) == 1

        result = np.zeros((len(columns),), dtype=float)

        # minimize() returns 0 on success, 1 on unsat, -1 on error
        min_res = LpInstance._minimize(self.lp_data, columns, result, len(result))

        if min_res == -1:
            raise RuntimeError("LP minimize() failed internally")

        if min_res == 1:
            if fail_on_unsat:
                raise RuntimeError("LP minimize() returned UNSAT, but fail_on_unsat was True")
            else:
                result = None

        return result

    def minimize_full_result(self, fail_on_unsat=True):
        'minimize the lp. returns the LP assigment for every column if SAT (as an np.ndarray), else None'

        num_cols = self.get_num_cols()

        columns = np.array([i for i in range(num_cols)], dtype=np.int32)
        result_vec = np.zeros((num_cols,), dtype=float)

        # minimize() returns 0 on success, 1 on unsat, -1 on error
        res = LpInstance._minimize(self.lp_data, columns, result_vec, len(result_vec))

        if res == -1:
            raise RuntimeError("LP minimize() failed internally")

        if res == 1 and fail_on_unsat:
            raise RuntimeError("LP minimize() returned UNSAT, but fail_on_unsat was True")

        if res != 0:
            result_vec = None

        return result_vec

    def minimize(self, direction_vec, fail_on_unsat=True):
        'minimize the lp. This assigns the direction vec and returns the full result'

        self.set_minimize_direction(direction_vec)

        return self.minimize_full_result(fail_on_unsat=fail_on_unsat)

    def flip_constraint(self, row_index):
        '''flip a constraint from >= to <= or vis-versa

        returns True if the constraint is now a '<=' constraint
        '''

        res = LpInstance._flip_constraint(self.lp_data, row_index)

        if res < 0:
            raise RuntimeError("flip_constraint() failed internally")

        return res == 0

    def del_constraint(self, row_index):
        '''delete a constraint from the lp'''

        res = LpInstance._del_constraint(self.lp_data, row_index)

        if res != 0:
            raise RuntimeError("del_constraint() failed internally")

    def set_constraint_rhs(self, row_index, rhs):
        '''change an existing constraint's right hand side'''

        res = LpInstance._set_constraint_rhs(self.lp_data, row_index, rhs)

        if res != 0:
            raise RuntimeError("set_constraint_rhs() failed internally")

    def get_subconstraints(self, x, y, w, h):
        'get a subconstraint matrix from the lpi as a dense matrix'

        rv = np.zeros(w*h)

        res = LpInstance._get_subconstraints(self.lp_data, x, y, w, h, rv, len(rv))

        if res != 0:
            raise RuntimeError("get_subconstaints failed")

        rv.shape = (h, w)
        
        return rv

    def get_types(self):
        '''get the constraint types. See get_constraints() for the meaning of values in the returned array
        '''

        rows = self.get_num_rows()
                
        types = np.zeros((rows,), dtype=np.int32)
        res = LpInstance._get_types(self.lp_data, types, rows)

        if res != 0:
            raise RuntimeError("get_types failed")

        return types

    def get_constraints(self):
        '''get the LP matrix as a csr_matrix

        returns a 3-tuple, (mat, types, vec), where mat is constraints, vec is the right-hand side, and
        types is a np.array of integers corresponding to the type of constraint: 

        GLP_FX(== constraint): 5, GLP_UP (<= constraint): 3, GLP_LO (>= constraint): 2
        These constants are also defined as static members of LpInstance (for example, LpInstance.GLP_FX)
        '''

        rows = self.get_num_rows()
        cols = self.get_num_cols()

        vec = np.zeros((rows,), dtype=float)
        res = LpInstance._get_rhs(self.lp_data, vec, rows)

        if res != 0:
            raise RuntimeError("get_rhs failed")

        types = np.zeros((rows,), dtype=np.int32)
        res = LpInstance._get_types(self.lp_data, types, rows)

        if res != 0:
            raise RuntimeError("get_types failed")

        nnz = LpInstance._get_num_nz(self.lp_data)

        data = np.zeros((nnz,), dtype=float)
        inds = np.zeros((nnz,), dtype=np.int32)
        indptrs = np.zeros((rows+1,), dtype=np.int32)

        res = LpInstance._get_constraints(self.lp_data, data, len(data), inds, len(inds), indptrs, len(indptrs))

        if res != 0:
            raise RuntimeError("get_constraints failed")

        mat = csr_matrix((data, inds, indptrs), shape=(rows, cols), dtype=float)
        mat.check_format()

        return mat, types, vec

    def get_row(self, row):
        '''get a row of the LP matrix as a csr_matrix
        '''

        cols = self.get_num_cols()

        data = np.zeros((cols,), dtype=float)
        inds = np.zeros((cols,), dtype=np.int32)
        
        res = LpInstance._get_row(self.lp_data, row, data, len(data), inds, len(inds))

        if res == -1:
            raise RuntimeError("get_row failed")

        indptr = [0, res]

        row_mat = csr_matrix((data, inds, indptr), shape=(1, cols), dtype=float)
        row_mat.check_format()

        return row_mat

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
