'''
Unit tests for Hylass's glpk_interface.py
Stanley Bak
November 2016
'''

import unittest

from scipy.sparse import csr_matrix

import cvxopt

from hylaa.glpk_interface import LpInstance
import numpy as np

class TestGlpkInterface(unittest.TestCase):
    'Unit tests for optimization utilities'

    def test_cpp(self):
        'runs the c++ test() function for the glpk interface'

        start_op = LpInstance.total_optimizations()
        start_it = LpInstance.total_iterations()

        LpInstance.test()
        self.assertGreater(LpInstance.total_iterations() - start_it, 5) # the test ran a few iterations of lp
        self.assertGreater(LpInstance.total_optimizations() - start_op, 1) # the test ran a few lp

    def test_simple(self):
        '''test consistency with cvxopt on a simple problem'''

        # max 0.6x + 0.5y st.
        # x + 2y <= 1
        # 3x + y <= 2

        a_ub = [[1, 2], [3, 1]]
        b_ub = [1, 2]
        c = [-0.6, -0.5]

        self.compare_opt(a_ub, b_ub, c)

    def test_simple2(self):
        '''test another simple case (was failing)'''

        a_ub = [[1.0],
                [-1.0]]

        b_ub = [1.0, 1.0]

        num_vars = len(a_ub[0])
        c = [1.0 for _ in xrange(num_vars)]

        self.compare_opt(a_ub, b_ub, c)

    def test_underconstrained(self):
        'test an underconstrained case (fails for cvxopt)'

        a_ub = [[1.0, 0.0], [-1.0, 0.0]]
        b_ub = [1.0, 1.0]
        c = [1.0, 0.0]

        num_vars = 2

        lp = LpInstance(num_vars, num_vars)

        lp.set_init_constraints(csr_matrix(np.array(a_ub, dtype=float)), np.array(b_ub, dtype=float))
        lp.set_no_output_constraints()
        lp.update_basis_matrix(np.identity(num_vars))

        res_glpk = np.zeros(num_vars)
        lp.minimize(np.array(c, dtype=float), res_glpk)

        self.assertAlmostEqual(res_glpk[0], -1)

    def test_tricky(self):
        '''test consistency with cvxopt on a tricky problem (scipy.linprog fails)'''

        a_ub = [[-1.0, 0.0, 0.0, -2.1954134149515525e-08, 1.0000000097476742, 0.0],
                [1.0, -0.0, -0.0, 2.1954134149515525e-08, -1.0000000097476742, -0.0],
                [0.0, -1.0, 0.0, -1.000000006962809, 2.5063524589086228e-08, 0.0],
                [-0.0, 1.0, -0.0, 1.000000006962809, -2.5063524589086228e-08, -0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 1.0000000000000009],
                [-0.0, -0.0, 1.0, -0.0, -0.0, -1.0000000000000009],
                [0., 0., 0., 1.0, 0.0, 0.0],
                [0., 0., 0., -1.0, 0.0, 0.0],
                [0., 0., 0., 0.0, 1.0, 0.0],
                [0., 0., 0., 0.0, -1.0, 0.0],
                [0., 0., 0., 0.0, 0.0, 1.0],
                [0., 0., 0., 0.0, 0.0, -1.0]]

        b_ub = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, -0.0]

        num_vars = len(a_ub[0])
        c = [1.0 if i % 2 == 0 else 0.0 for i in xrange(num_vars)]

        self.compare_opt(a_ub, b_ub, c)

    def compare_opt(self, a_ub, b_ub, c):
        'compare cvx opt versus our glpk interface'

        # make sure we're using floats not ints
        a_ub = [[float(x) for x in row] for row in a_ub]
        b_ub = [float(x) for x in b_ub]
        c = [float(x) for x in c]

        num_vars = len(a_ub[0])

        # solve it with cvxopt
        options = {'show_progress': False}
        sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(a_ub).T, cvxopt.matrix(b_ub), options=options)

        if sol['status'] != 'optimal':
            raise RuntimeError("cvxopt LP failed: {}".format(sol['status']))

        res_cvxopt = [float(n) for n in sol['x']]

        # solve it with the glpk <-> hylaa interface
        lp = LpInstance(num_vars, num_vars)

        lp.set_init_constraints(csr_matrix(np.array(a_ub, dtype=float)), np.array(b_ub, dtype=float))
        lp.set_no_output_constraints()
        lp.update_basis_matrix(np.identity(num_vars))

        res_glpk = np.zeros(num_vars)

        lp.minimize(np.array(c, dtype=float), res_glpk)

        self.assertEqual(num_vars, len(res_cvxopt))
        self.assertAlmostEqual(np.dot(res_glpk, c), np.dot(res_cvxopt, c), places=5)

    def test_ha(self):
        'test based on harmonic oscillator dynamics'

        lp = LpInstance(2, 2)

        # x == 1, y == 0
        a_ub = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        b_ub = [1.0, -1.0, 0, 0]

        num_vars = 2

        lp = LpInstance(num_vars, num_vars)

        lp.set_init_constraints(csr_matrix(np.array(a_ub, dtype=float)), np.array(b_ub, dtype=float))
        lp.set_no_output_constraints()

        basis_mat = np.array([[-1, 0], [0, 1]], dtype=float)
        lp.update_basis_matrix(basis_mat)

        res = -np.ones(4)
        lp.minimize(np.array([0, 0], dtype=float), res)

        # result should be [0, -1] (standard basis) and [1, 0] (star basis)
        self.assertAlmostEqual(res[0], 1.0)
        self.assertAlmostEqual(res[1], 0.0)

        # star constraints
        self.assertAlmostEqual(res[2], -1.0)
        self.assertAlmostEqual(res[3], 0.0)

    def test_damping(self):
        'test based on damping dynamics'

        lp = LpInstance(1, 1)

        # x == 1
        lp.set_init_constraints(csr_matrix(np.array([[1], [-1]], dtype=float)), np.array([1, -1], dtype=float))
        lp.set_no_output_constraints()

        basis = np.array([[0.5]], dtype=float)
        lp.update_basis_matrix(basis)

        res = np.zeros(2)
        lp.minimize(np.array([0], dtype=float), res)

        self.assertAlmostEqual(res[0], 1.0)
        self.assertAlmostEqual(res[1], 0.5)

if __name__ == '__main__':
    unittest.main()
