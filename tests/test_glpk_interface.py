'''
Unit tests for Hylass's glpk_interface.py
Stanley Bak
November 2016
'''

import unittest

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

        orthonormal_basis = [[1.0 if d == index else 0.0 for d in xrange(num_vars)] for index in xrange(num_vars)]
        lp.update_basis_matrix(np.array(orthonormal_basis, dtype=float))

        # add each constraint
        for row in xrange(len(b_ub)):
            vec = np.matrix(a_ub[row], dtype=float)
            val = b_ub[row]

            lp.add_basis_constraint(vec, val)

        res_glpk = np.zeros(num_vars)
        lp.minimize(np.array(c, dtype=float), res_glpk)

        self.assertAlmostEqual(res_glpk[0], -1)
        #assert_allclose(res_glpk, [-1, 0], rtol=1e-10, atol=1e-10)

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

        #if sol['status'] == 'primal infeasible':
        #    res_cvxopt = None

        if sol['status'] != 'optimal':
            raise RuntimeError("cvxopt LP failed: {}".format(sol['status']))

        res_cvxopt = [float(n) for n in sol['x']]

        #print "cvxopt value = {}, result = {}".format(np.dot(res_cvxopt, c), repr(res_cvxopt))

        # solve it with the glpk <-> hylaa interface
        lp = LpInstance(num_vars, num_vars)

        orthonormal_basis = [[1.0 if d == index else 0.0 for d in xrange(num_vars)] for index in xrange(num_vars)]
        lp.update_basis_matrix(np.array(orthonormal_basis, dtype=float))

        # add each constraint
        for row in xrange(len(b_ub)):
            vec = np.matrix(a_ub[row], dtype=float)
            val = b_ub[row]

            lp.add_basis_constraint(vec, val)

        res_glpk = np.zeros(num_vars)
        lp.minimize(np.array(c, dtype=float), res_glpk)

        #print "glpk interface value = {}, result = {}".format(np.dot(res_glpk, c), repr(res_glpk))

        self.assertEqual(num_vars, len(res_cvxopt))
        self.assertAlmostEqual(np.dot(res_glpk, c), np.dot(res_cvxopt, c), places=5)

    def test_ha(self):
        'test based on harmonic oscillator dynamics'

        lp = LpInstance(2, 2)
        basis = np.array([[0, -1], [1, 0]], dtype=float)
        lp.update_basis_matrix(basis)

        # x == 1
        lp.add_basis_constraint(np.array([1, 0], dtype=float), 1.0)
        lp.add_basis_constraint(np.array([-1, 0], dtype=float), -1.0)

        # y == 0
        lp.add_basis_constraint(np.array([0, 1], dtype=float), 0)
        lp.add_basis_constraint(np.array([0, -1], dtype=float), -0)

        res = -np.ones(4)
        lp.minimize(np.array([0, 0], dtype=float), res)

        # result should be [0, -1] (standard basis) and [1, 0] (star basis)
        self.assertAlmostEqual(res[0], 0.0)
        self.assertAlmostEqual(res[1], -1.0)

        # star constraints
        self.assertAlmostEqual(res[2], 1.0)
        self.assertAlmostEqual(res[3], 0.0)

    def test_damping(self):
        'test based on damping dynamics'

        lp = LpInstance(1, 1)
        basis = np.array([[0.5]], dtype=float)
        lp.update_basis_matrix(basis)

        lp.add_basis_constraint(np.array([1.0], dtype=float), 1.0)
        lp.add_basis_constraint(np.array([-1.0], dtype=float), -1.0)

        res = np.zeros(2)
        lp.minimize(np.array([0], dtype=float), res)

        self.assertLess(res[0], 1.0)

if __name__ == '__main__':
    unittest.main()
