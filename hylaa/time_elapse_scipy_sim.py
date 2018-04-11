'''
scipy-based simulation for time elapse

Stanley Bak
April 2018
'''

import numpy as np
from scipy.sparse import csr_matrix

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

        assert isinstance(self.a_matrix, csr_matrix)

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

    def step(self):
        'compute the current step basis matrix and store in self.time_elapser'

        if self.time_elapser.next_step == 0:
            # step zero
            init_space = self.time_elapser.init_space_csc
            output_space = self.time_elapser.output_space_csr

            self.time_elapser.cur_basis_mat = (output_space * init_space).toarray()
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
                dense_output = self.find_dense_output(init_index, cur_time)

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

    def find_dense_output(self, init_index, cur_time):
        'find (or compute) the dense output instance at the passed-in time'

        do_list = self.dense_outputs[init_index]
        sim_obj = self.sim_objs[init_index]

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
