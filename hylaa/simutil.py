'''
Simulation utility functions for Hylaa. 
This is a version which performs simulations at fixed time steps.

Stanley Bak
September 2016
'''

from scipy.integrate import odeint

import numpy as np

from hylaa.timerutil import Timers

class SimulationBundle(object):
    'a simulation bundle of basis vectors in a fixed set of dynamics (single mode)'
    
    def __init__(self, a_matrix, b_vector, step_size, sim_tol=None):
        if isinstance(a_matrix, list):
            a_matrix = np.array(a_matrix, dtype=float)        

        if isinstance(b_vector, list):
            b_vector = np.array(b_vector, dtype=float)

        if sim_tol is not None:
            assert isinstance(sim_tol, float)

        assert isinstance(a_matrix, np.ndarray), "a matrix should be array of floats (no integers)"
        assert isinstance(b_vector, np.ndarray), "b vector should be array of floats (no integers)"
        
        assert step_size > 0
        self.step_size = step_size
            
        self.num_dims = b_vector.shape[0]
        assert self.num_dims > 0
        assert a_matrix.shape[0] == a_matrix.shape[1], "expected A matrix to be a square"
        assert len(b_vector.shape) == 1, "expected b_vector to be a single row: {}".format(b_vector)
        assert a_matrix.shape[0] == b_vector.shape[0], "A matrix and b vector sizes should match"
        
        self.sim_tol = sim_tol # tolerance for use in the simulation

        self.der_func = lambda state, dummy_t: np.add(np.dot(a_matrix, state), b_vector)

        banded_jac, max_upper, max_lower = make_banded_jacobian(a_matrix)

        if len(banded_jac) * len(banded_jac[0]) < a_matrix.shape[0] * a_matrix.shape[1]:
            # use banded jacobian
            
            banded_jac_transpose = np.array(banded_jac, dtype=float).T # pylint false positive
            self.jac_func = lambda dummy_state, dummy_t: banded_jac_transpose
            self.max_upper = max_upper
            self.max_lower = max_lower
        else:
            # use non-banded jacobian
            a_transpose = a_matrix.T
            self.jac_func = lambda dummy_state, dummy_t: a_transpose
            self.max_upper = self.max_lower = None

        # initialize simulation variables
        self.vec_values = [] # index is step, value is ndarray for the basis_matrix at that step
        self.last_states = []

        self.origin_sim = [np.array([0.0] * self.num_dims, dtype=float)]

        self.vec_values.append(np.empty([self.num_dims, self.num_dims]))

        for index in xrange(self.num_dims):
            orthonormal_vec = np.array([1.0 if d == index else 0.0 for d in xrange(self.num_dims)])
            self.last_states.append(orthonormal_vec)
            self.vec_values[0][index][:] = orthonormal_vec

    def get_vecs_origin_at_step(self, step, max_steps):
        '''
        get the exact state of the basis vectors and origin simulation 
        at a specific step number (multiply by self.size to get time)

        max_step is the maximum number of steps we'd ever want (optimization 
            so we don't waste time simulating too far)

        returns a tuple (list_of_basis_vecs, origin_sim)
        '''

        assert isinstance(max_steps, int), "max_steps should be an integer. Got: {} ({})".format(
            max_steps, type(max_steps))
        assert step <= max_steps

        Timers.tic("sim + overhead")

        # makes sure our simulations are long enough
        self._sim_all_past_step(step, max_steps)

        Timers.toc("sim + overhead")

        return (self.vec_values[step], self.origin_sim[step])

    def _sim_all_past_step(self, step, max_steps):
        '''
        Continue the simulations up to at least some given step
        '''

        num_completed_steps = len(self.vec_values) - 1

        if step > num_completed_steps:            
            num_new_steps = step - num_completed_steps

            # make sure we're at least doubling the number of steps (reduces overhead for repeated small extensions)
            if num_new_steps < num_completed_steps:
                num_new_steps = num_completed_steps # doubles the length

            # make sure we don't go past max_step (reduces overhead by not simulating too far)
            if num_completed_steps + num_new_steps > max_steps:
                num_new_steps = max_steps - num_completed_steps

            assert num_new_steps >= 1
            #num_new_steps = 300

            # pre-allocate result
            self.vec_values += [np.empty([self.num_dims, self.num_dims]) for _ in xrange(num_new_steps)]

            max_time = self.step_size * num_new_steps
            epsilon = self.step_size / 8.0 # to prevent round-off error on the end range
            times = np.arange(0.0, max_time + epsilon, self.step_size)

            # simulate origin
            origin_sim_start = self.origin_sim[-1]
            origin_sim_states = self._sim_one(origin_sim_start, times)[1:]

            for state in origin_sim_states:
                self.origin_sim.append(state)

            for vec_index in xrange(self.num_dims):
                last_state = self.last_states[vec_index]

                sim = self._sim_one(last_state, times)[1:]

                # compute the difference between origin_sim and sim, at each step
                diff = sim - origin_sim_states

                for sim_index in xrange(len(sim)):
                    absolute_step = sim_index + num_completed_steps + 1
                    self.vec_values[absolute_step][vec_index] = diff[sim_index]
                    #vec = diff[sim_index]
                    #self.vec_values[vec_index].append(vec)

                assert len(sim) > 0
                self.last_states[vec_index] = sim[-1]

    def _sim_one(self, start, times):
        '''
        simulate from a single point at the given times 

        return an nparray of states at those times
        '''

        Timers.tic("simulation")

        res = None

        if self.sim_tol is None:
            res = odeint(self.der_func, start, times, Dfun=self.jac_func, col_deriv=True, mxstep=5000, 
                         mu=self.max_upper, ml=self.max_lower)
        else:
            tol = self.sim_tol
            res = odeint(self.der_func, start, times, Dfun=self.jac_func, col_deriv=True, mxstep=5000, 
                         mu=self.max_upper, ml=self.max_lower, atol=tol, rtol=tol)

        Timers.toc("simulation")

        return res

    def sim_until_inv_violated(self, pt, inv_list, max_steps):
        '''
        simulate a point until the invariant is violated or max_steps is reached

        returns the list of points (each one is an ndarray)
        '''

        assert isinstance(pt, np.ndarray)

        num_steps = 1
        rv = [pt]

        inv_violated = False

        while len(rv) - 1 < max_steps:
            if len(rv) - 1 + num_steps > max_steps:
                num_steps = max_steps - len(rv) + 1

            max_time = self.step_size * num_steps

            epsilon = self.step_size / 8.0 # to prevent round-off error on the end range
            times = np.arange(0.0, max_time + epsilon, self.step_size)

            # simulate 
            start = rv[-1]
            new_states = self._sim_one(start, times)[1:]

            # for every new state, check if it violates the invariant
            for state in new_states:
                for inv in inv_list:
                    if np.dot(inv.vector, state) > inv.value:
                        inv_violated = True
                        break

                if inv_violated:
                    break
                else:
                    rv.append(state)

            if inv_violated:
                break
            else:
                num_steps *= 2

        return rv

def make_banded_jacobian(matrix):
    '''returns a banded jacobian list (in odeint's format), along with mu and ml parameters'''

    # first find the values of mu and ml
    dims = matrix.shape[0]
    assert dims == matrix.shape[1]
    mu = 0
    ml = 0

    for row in xrange(dims):
        for col in xrange(dims):
            if matrix[row][col] != 0:
                if col > row:
                    dif = col - row
                    mu = max(mu, dif)
                else:
                    dif = row - col
                    ml = max(ml, dif)

    banded = []

    for yoffset in xrange(-mu, ml+1):
        row = []

        for diag in xrange(dims):
            x_index = diag
            y_index = diag + yoffset

            if y_index < 0 or y_index >= dims:
                row.append(0.0)
            else:
                row.append(matrix[y_index][x_index])

        banded.append(row)

    return (banded, mu, ml)

