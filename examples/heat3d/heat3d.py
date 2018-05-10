'''
3D Heat Equation Based on Tran's model
'''

import math

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, dia_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, TimeElapseSettings
from hylaa.star import Star

def define_ha(samples_per_side):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')

    # parameters
    diffusity_const = 0.01
    heat_exchange_const = 0.5

    print "Making {}x{}x{} ({} dims) 3d Heat Plate ODEs...".format(samples_per_side, samples_per_side, \
                                                               samples_per_side, samples_per_side**3)
    a_matrix = heat3d_dia(samples_per_side, diffusity_const, heat_exchange_const)

    print "Finished Making ODEs"

    mode.set_dynamics(a_matrix)

    #error = ha.new_mode('error')
    #dims = combined_mat.shape[0]

    #center_x = int(math.floor(samples_per_side/2.0))
    #center_y = int(math.floor(samples_per_side/2.0))
    #center_z = int(math.floor(samples_per_side/2.0))

    #center_dim = center_z * samples_per_side * samples_per_side + center_y * samples_per_side + center_x

    # x_center >= 0.9
    #mat = csr_matrix(([-1], [center_dim], [0, 1]), dtype=float, shape=(1, dims))
    #rhs = np.array([-0.5], dtype=float) # 0.5 = safe
    #rhs = np.array([-0.4], dtype=float) # 0.4 = unsafe
    #trans1 = ha.new_transition(mode, error)
    #trans1.set_guard(mat, rhs)

    return ha

def heat3d_dia(samples, diffusity_const, heat_exchange_const):
    'fast dia_matrix construction for heat3d dynamics'

    samples_sq = samples**2
    dims = samples**3
    step = 1.0 / (samples + 1)

    a = diffusity_const * 1.0 / step**2
    d = -2.0 * (a + a + a)

    data = np.zeros((7, dims))
    offsets = np.array([-samples_sq, -samples, -1, 0, 1, samples, samples_sq], dtype=float)

    # element with z = -1
    data[0, :-samples_sq] = a

    # element with y = -1
    for s in xrange(samples):
        start = s * samples_sq
        end = (s + 1) * (samples_sq) - samples
        data[1, start:end] = a

    # element with x = -1
    for s in xrange(samples_sq):
        start = s * samples
        end = (s + 1) * (samples) - 1
        data[2, start:end] = a

    #### diagonal element ####
    data[3, :] = d     # (prefill)

    # adjust when z = 0 or z = samples-1
    data[3, :samples_sq] += a
    data[3, -samples_sq:] += a

    # adjust when y = 0 or y = samples-1
    for z in xrange(samples):
        z_offset = z * samples_sq

        data[3, z_offset:z_offset + samples] += a
        data[3, z_offset + samples_sq - samples:z_offset + samples_sq] += a

    # adjust when x = 0 (and add diffusion term when x = samples-1)
    for z in xrange(samples):
        for y in xrange(samples):
            offset = z * samples_sq + y * samples

            data[3, offset] += a

            data[3, offset + samples - 1] += a / (1+heat_exchange_const * step)

    #### end diagnal element ####
    # element with x = +1
    for s in xrange(samples_sq):
        start = 1 + s * samples
        end = (s + 1) * samples
        data[4, start:end] = a

    # element with y = +1
    for s in xrange(samples):
        start = s * samples_sq + samples
        end = (s + 1) * (samples_sq)
        data[5, start:end] = a

    # element with z = +1
    data[6, samples_sq:] = a

    rv = dia_matrix((data, offsets), shape=(dims, dims))
    assert np.may_share_memory(rv.data, data)     # make sure we didn't copy memory

    return rv

def make_init_star(ha, hylaa_settings, samples):
    '''returns a Star'''

    dims = ha.modes.values()[0].a_matrix_csr.shape[0]

    data = []
    inds = []
    indptrs = [0]

    assert samples >= 10 and samples % 10 == 0, "init region isn't evenly divided by discretization"

    for z in xrange(samples / 10 + 1):
        zoffset = z * samples * samples

        for y in xrange(2 * samples / 10 + 1):
            yoffset = y * samples

            for x in xrange(4 * samples / 10 + 1):
                dim = x + yoffset + zoffset

                data.append(1)
                inds.append(dim)

    indptrs.append(len(data))
    init_space = csc_matrix((data, inds, indptrs), dtype=float, shape=(dims, 1))

    init_mat = csr_matrix(np.array([[1], [-1.]], dtype=float))
    init_mat_rhs = np.array([1.1, -0.9], dtype=float)

    return Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_mat_rhs)

def define_settings(samples_per_side, stdout, use_arnoldi):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_GNUPLOT

    settings = HylaaSettings(step=0.02, max_time=20.0, plot_settings=plot_settings)
    settings.time_elapse.method = TimeElapseSettings.KRYLOV
    settings.skip_step_times = True
    kryset = settings.time_elapse.krylov
    #settings.time_elapse.check_answer = True

    kryset.use_lanczos_eigenvalues = True
    kryset.stdout = stdout
    kryset.force_arnoldi = use_arnoldi

    center_x = int(math.floor(samples_per_side/2.0))
    center_y = int(math.floor(samples_per_side/2.0))
    center_z = int(math.floor(samples_per_side/2.0))

    center_dim = center_z * samples_per_side * samples_per_side + center_y * samples_per_side + center_x

    plot_settings.xdim_dir = None
    plot_settings.ydim_dir = center_dim

    return settings

def run_hylaa(samples_per_side=20, stdout=True, use_arnoldi=False):
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha = define_ha(samples_per_side)
    settings = define_settings(samples_per_side, stdout, use_arnoldi)
    init = make_init_star(ha, settings, samples_per_side)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(20, True, False)
