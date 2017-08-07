'''
Time Elapse for the Krylov method using CPU or GPU
'''

import numpy as np
from scipy.sparse.linalg import expm

from hylaa.gpu_krylov_sim import GpuKrylovSim
from hylaa.containers import SimulationSettings

def make_cur_time_elapse_mat_list(self, time_elapser):
    'updates self.cur_time_elapse_mat and, if there are inputs, self.cur_input_effects_matrix'

    settings = time_elapser.settings
    dims = time_elapser.dims
    step_time = settings.step
    a_matrix = time_elapser.a_matrix
    krylov_iter = min(time_elapser.settings.simulation.krylov_dimension, a_matrix.shape[0])

    # Choose using CPU (host_memory) or GPU (device_memory)
    
    use_gpu = settings.simulation.sim_mode is SimulationSettings.KRYLOV_GPU
    GpuKrylovSim.set_use_gpu(use_gpu)

    rv = []
    rv.append(np.array(self.key_dir_mat.todense(), dtype=float)) # step zero

    GpuKrylovSim.load_matrix(self.a_matrix.tocsr()) # load a_matrix into device memory
    GpuKrylovSim.load_keyDirSparseMatrix(self.key_dir_mat) # load key direction matrix into device memory
    cusp_h_tuples, _ = GpuKrylovSim.arnoldi_parallel(self.dims, krylov_iter)

    def get_exp_h_tuples(step_num):
        'gets a list of e^{H * step_num * step_time}, for each initial vector'

        rv = []

        for i in xrange(0, dims):
            rv.append(expm(step_num * step_time * self.cusp_h_tuples[i, :, :]))

        return rv

    for step in xrange(1, time_elapser.settings.num_steps):

        if self.settings.simulation.krylov_compute_exp_Ht == SimulationSettings.KRYLOV_H_MULT:
        # first step matrix exp, other step matrix multiplication
            if self.next_step == 0:
                self.cur_time_elapse_mat = np.array(self.key_dir_mat.todense(), dtype=float)
            elif self.one_step_expH_tuples is None:  # compute exp(H*step)
                assert self.next_step == 1
                assert isinstance(self.key_dir_mat, csr_matrix)
                Timers.tic('time_elapse.step first step')
                get_one_step_expH_tuples()

                self.cur_step_expH_tuples = self.one_step_expH_tuples # save for next step
                # to do: get keySimResult in parallel
                self.cur_time_elapse_mat = GpuKrylovSim.getKeySimResult_parallel(self.key_dir_mat.shape[0],self.dims,self.settings.simulation.krylov_numIter,self.cur_step_expH_tuples)    

                Timers.toc('time_elapse.step first step')

            else:               
                Timers.tic('time_elapse.step other steps')
                for i in range(0, self.dims):
                    cur_step_expH = self.cur_step_expH_tuples[i]*self.one_step_expH_tuples[i]
                    self.cur_step_expH_tuples[i] = cur_step_expH # save for next step

                # todo: get keySimResult in parallel
                self.cur_time_elapse_mat = GpuKrylovSim.getKeySimResult_parallel(self.key_dir_mat.shape[0],self.dims,self.settings.simulation.krylov_numIter,self.cur_step_expH_tuples)     
                Timers.toc('time_elapse.step other steps')

        elif self.settings.simulation.krylov_compute_exp_Ht == SimulationSettings.KRYLOV_H_EXP:
        # matrix exponential for all steps
            get_any_step_expH_tuples(self.next_step)
            self.cur_time_elapse_mat = GpuKrylovSim.getKeySimResult_parallel(self.key_dir_mat.shape[0],self.dims,self.settings.simulation.krylov_numIter,self.cur_step_expH_tuples)
