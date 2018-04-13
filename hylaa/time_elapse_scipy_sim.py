'''
scipy-based simulation for time elapse

Stanley Bak
April 2018
'''

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.util import Freezable
from hylaa.timerutil import Timers

class TimeElapseScipySim(Freezable):
    'container object for the numerical simulation method'

    def __init__(self, time_elapser):
        self.settings = time_elapser.settings
        self.sim_settings = self.settings.time_elapse.scipy
        self.a_matrix = time_elapser.a_matrix
        self.b_matrix = time_elapser.b_matrix
        self.time_elapser = time_elapser
        self.one_step_input_effects_matrix = None

        assert isinstance(self.a_matrix, csr_matrix)
        assert self.b_matrix is None or isinstance(self.b_matrix, csc_matrix)

        if time_elapser.use_init_space:
            sim_mat = self.a_matrix
            init_vecs = self.time_elapser.init_space_csc
            self.proj_mat = self.time_elapser.output_space_csr
        else:
            sim_mat = csr_matrix(self.a_matrix.transpose())
            init_vecs = self.time_elapser.output_space_csr.transpose()
            self.proj_mat = self.time_elapser.init_space_csc.transpose()

        assert isinstance(self.proj_mat, csr_matrix)

        der_func = lambda _, state: np.array(sim_mat * state, dtype=float)

        # a list of lists, the first index is the initial vector, second is dense_output objs
        self.dense_outputs = []

        # a list of simluation objects for each initial vector
        self.sim_objs = []

        for col_index in xrange(init_vecs.shape[1]):
            init = init_vecs[:, col_index].toarray()
            init.shape = (init.shape[0],)

            max_step = self.sim_settings.max_step
            atol, rtol = self.sim_settings.atol, self.sim_settings.rtol
            sim_obj = self.sim_settings.ode_class(der_func, 0, init, np.inf, max_step, atol, rtol)
            self.sim_objs.append(sim_obj)

        if self.b_matrix is not None:
            # list of lists, for each input vector
            self.input_dense_outputs = []

            # a list of simluation objects for each input
            self.input_sim_objs = []


    def step(self):
        'compute the current step basis matrix and store in self.time_elapser'

        init_space = self.time_elapser.init_space_csc
        output_space = self.time_elapser.output_space_csr

        if self.time_elapser.next_step == 0:
            # step zero

            self.time_elapser.cur_basis_mat = (output_space * init_space).toarray()
            self.time_elapser.cur_input_effects_matrix = None
        else:
            # ensure each simulation has at least one step
            if len(self.dense_outputs) == 0:
                for sim_obj in self.sim_objs:
                    Timers.tic('simulate.step()')
                    sim_obj.step()
                    Timers.toc('simulate.step()')
                    assert sim_obj.status == 'running', 'Simulation failed. Status was: {}'.format(sim_obj.status)
                    dense_output_list = [sim_obj.dense_output()]
                    self.dense_outputs.append(dense_output_list)

            # find the correct dense_output for each initial vec, compute the point, and project it and store result
            cur_time = self.time_elapser.settings.step * self.time_elapser.next_step

            for init_index in xrange(len(self.sim_objs)):
                dense_output = find_dense_output(init_index, cur_time, self.sim_objs, self.dense_outputs)

                Timers.tic('simulate.dense_output(time)')
                point = dense_output(cur_time)
                Timers.toc('simulate.dense_output(time)')

                Timers.tic('simulate.project')
                proj_point = (self.proj_mat * point)
                Timers.toc('simulate.project')

                Timers.tic('simulate.store_basis_matrix')
                if self.time_elapser.use_init_space:
                    # fixed column
                    self.time_elapser.cur_basis_mat[:, init_index] = proj_point
                else:
                    # fixed row
                    self.time_elapser.cur_basis_mat[init_index, :] = proj_point
                Timers.toc('simulate.store_basis_matrix')

            if self.b_matrix is not None:
                if len(self.input_dense_outputs) == 0:
                    self.initialize_input_sims()

                num_inputs = self.b_matrix.shape[1]

                if self.time_elapser.cur_input_effects_matrix is None:
                    self.time_elapser.cur_input_effects_matrix = np.zeros((output_space.shape[0], num_inputs))

                input_time = cur_time - self.time_elapser.settings.step # input time is offset by one step

                for input_index in xrange(num_inputs):
                    dense_output = find_dense_output(input_index, input_time, \
                                                 self.input_sim_objs, self.input_dense_outputs)

                    Timers.tic('simulate.dense_output(time)')
                    point = dense_output(input_time)
                    Timers.toc('simulate.dense_output(time)')

                    Timers.tic('simulate.project')
                    proj_point = output_space * point
                    Timers.toc('simulate.project')

                    Timers.tic('simulate.store_basis_matrix')
                    self.time_elapser.cur_input_effects_matrix[:, input_index] = proj_point
                    Timers.toc('simulate.store_basis_matrix')

    def initialize_input_sims(self):
        '''initializes input simulations with the first step of input effects'''

        # compute input_effects_matrix
        a_mat_csc = csc_matrix(self.a_matrix)
        original_der_func = lambda _, state: np.array(self.a_matrix * state, dtype=float)

        for input_index in xrange(self.b_matrix.shape[1]):
            # augment the a matrix with a fixed affine term to account for the effects of inputs
            # add one new column equal to the the b_matrix column, with row == 0
            start_index = self.b_matrix.indptr[input_index]
            end_index = self.b_matrix.indptr[input_index + 1]
            dims = self.a_matrix.shape[0]

            data = [x for x in a_mat_csc.data] + [x for x in self.b_matrix.data[start_index:end_index]]
            indices = [n for n in a_mat_csc.indices] + \
                      [n for n in self.b_matrix.indices[start_index:end_index]]
            indptr = [n for n in a_mat_csc.indptr] + [len(data)]

            augmented_a_mat = csc_matrix((data, indices, indptr), dtype=float, shape=(dims + 1, dims + 1))

            der_func = lambda _, state, mat=augmented_a_mat: np.array(mat * state, dtype=float)

            # the initial state is just setting the affine variable to one
            init = np.array([1. if dim == dims else 0 for dim in xrange(dims + 1)], dtype=float)

            max_step = self.sim_settings.max_step
            atol, rtol = self.sim_settings.atol, self.sim_settings.rtol
            t_bound = self.time_elapser.settings.step
            sim_obj = self.sim_settings.ode_class(der_func, 0, init, t_bound, max_step, atol, rtol)

            # simulate up to one time step
            while sim_obj.status == 'running' and sim_obj.t < self.time_elapser.settings.step:
                Timers.tic('simulate.step()')
                sim_obj.step()
                Timers.toc('simulate.step()')

            assert sim_obj.status == 'finished', 'Simulation failed. Status was: {}'.format(sim_obj.status)

            # start will be the initial state for the input-effects simulation
            start = sim_obj.y[:-1]

            sim_obj = self.sim_settings.ode_class(original_der_func, 0, start, np.inf, max_step, atol, rtol)

            Timers.tic('simulate.step()')
            sim_obj.step()
            Timers.toc('simulate.step()')
            assert sim_obj.status == 'running', 'Simulation failed. Status was: {}'.format(sim_obj.status)
            dense_output_list = [sim_obj.dense_output()]
            self.input_dense_outputs.append(dense_output_list)
            self.input_sim_objs.append(sim_obj)

            do = sim_obj.dense_output()

def find_dense_output(index, cur_time, sim_objs, dense_outputs):
    'find (or compute) the dense output instance at the passed-in time'

    do_list = dense_outputs[index]
    sim_obj = sim_objs[index]

    if cur_time >= do_list[-1].t_min and cur_time <= do_list[-1].t_max:
        # most common case: use last dense_output
        rv = do_list[-1]
    elif cur_time >= do_list[-1].t_max:
        # also common: need to simulate more
        while cur_time >= do_list[-1].t_max:
            Timers.tic('simulate.step()')
            sim_obj.step()
            Timers.toc('simulate.step()')
            assert sim_obj.status == 'running', 'Simulation failed. Status was: {}'.format(sim_obj.status)
            do_list.append(sim_obj.dense_output())

        rv = do_list[-1]
    else:
        # middle case, do a binary search, this can be optimized if it becomes too slow with multiple modes
        Timers.tic('binary_search_do_list')
        rv = binary_search_do_list(cur_time, do_list, 0, len(do_list) - 1)
        Timers.toc('binary_search_do_list')

    assert rv.t_min <= cur_time <= rv.t_max

    return rv

def binary_search_do_list(cur_time, do_list, min_index, max_index):
    '''do a binary search on a dense_output list
    min_index and max_index are both inclusive
    '''

    if min_index == max_index:
        rv = do_list[min_index]
    else:
        middle_index = (min_index + max_index) / 2
        middle_do = do_list[middle_index]

        if cur_time < middle_do.t_min:
            rv = binary_search_do_list(cur_time, do_list, min_index, middle_index - 1)
        elif cur_time > middle_do.t_max:
            rv = binary_search_do_list(cur_time, do_list, middle_index + 1, max_index)
        else:
            rv = middle_do

    return rv
