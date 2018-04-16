'''
Building Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, TimeElapseSettings
from hylaa.star import Star

def define_ha(limit):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('build.mat')
    a_matrix = dynamics['A']
    b_matrix = csc_matrix(dynamics['B'])

    mode.set_dynamics(csr_matrix(a_matrix))

    # 0 <= u1 <= 0.1
    bounds_list = [(0.8, 1.0)]
    _, u_mat, u_rhs, u_range_tuples = bounds_list_to_init(bounds_list)

    mode.set_inputs(b_matrix, u_mat, u_rhs, u_range_tuples)

    error = ha.new_mode('error')

    y1 = dynamics['C'][0]
    output_space = csr_matrix(y1)

    mode.set_output_space(output_space)

    trans1 = ha.new_transition(mode, error)
    mat = csr_matrix(([-1], [0], [0, 1]), dtype=float, shape=(1, 1))
    rhs = np.array([-limit], dtype=float) # safe
    trans1.set_guard(mat, rhs) # y3 >= limit

    return ha

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    bounds_list = []
    dims = ha.modes.values()[0].a_matrix_csr.shape[0]

    for dim in xrange(dims):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim == 25:
            lb = -0.0001
            ub = 0.0001
        else:
            lb = ub = 0

        bounds_list.append((lb, ub))

    init_space, init_mat, init_mat_rhs, init_range_tuples = bounds_list_to_init(bounds_list)

    return Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_mat_rhs, \
                init_range_tuples=init_range_tuples)

def define_settings(ha, limit):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    #plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.plot_mode = PlotSettings.PLOT_NONE

    max_time = 20.0
    step_size = 0.1
    settings = HylaaSettings(step=step_size, max_time=max_time, plot_settings=plot_settings)

    settings.time_elapse.method = TimeElapseSettings.SCIPY_SIM
    settings.time_elapse.check_answer = False

    #settings.interval_guard_optimization = False
    #settings.time_elapse.scipy_sim.max_step = 0.001

    #settings.time_elapse.scipy_sim.rtol = 1e-9
    #settings.time_elapse.scipy_sim.atol = 1e-12

    #settings.skip_step_times = False

    plot_settings.xdim_dir = None
    plot_settings.ydim_dir = ha.modes.values()[0].output_space_csr[0]

    plot_settings.max_shown_polys = None
    plot_settings.label.y_label = '$y_{1}$'
    plot_settings.label.x_label = 'Time'
    plot_settings.label.title = 'Building (Uncertain Inputs)'
    #plot_settings.label.axes_limits = (0.4, 0.6, -0.0002, -0.0001)
    plot_settings.plot_size = (12, 8)
    plot_settings.label.big(size=36)

    plot_settings.extra_lines = [[(0.0, limit), (max_time, limit)]]

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    #limit = 0.004 # reachable
    limit = 0.005 # unreachable

    ha = define_ha(limit)
    settings = define_settings(ha, limit)
    init = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
