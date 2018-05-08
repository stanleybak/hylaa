'''
International Space Station Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, bounds_list_to_init
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, TimeElapseSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    mode = ha.new_mode('mode')
    dynamics = loadmat('iss.mat')
    a_matrix = dynamics['A']
    b_matrix = dynamics['B']

    mode.set_dynamics(csr_matrix(a_matrix))

    # 0 <= u1 <= 0.1
    # 0.8 <= u2 <= 1.0
    # 0.9 <= u3 <= 1.0
    bounds_list = [(0, 0.1), (0.8, 1.0), (0.9, 1.0)]
    _, u_mat, u_rhs, u_range_tuples = bounds_list_to_init(bounds_list)

    mode.set_inputs(b_matrix, u_mat, u_rhs, u_range_tuples)

    error = ha.new_mode('error')

    y3 = dynamics['C'][2]
    output_space = csr_matrix(y3)

    mode.set_output_space(output_space)

    limit = 0.0005
    #limit = 0.0007

    trans1 = ha.new_transition(mode, error)
    mat = csr_matrix(([1], [0], [0, 1]), dtype=float, shape=(1, 1))
    rhs = np.array([-limit], dtype=float) # safe
    trans1.set_guard(mat, rhs) # y3 <= -limit

    trans2 = ha.new_transition(mode, error)
    mat = csr_matrix(([-1], [0], [0, 1]), dtype=float, shape=(1, 1))
    rhs = np.array([-limit], dtype=float) # safe
    trans2.set_guard(mat, rhs) # y3 >= limit

    return ha

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    bounds_list = []
    dims = ha.modes.values()[0].a_matrix_csr.shape[0]

    for dim in xrange(dims):
        if dim < 270:
            lb = -0.0001
            ub = 0.0001
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        bounds_list.append((lb, ub))

    init_space, init_mat, init_mat_rhs, init_range_tuples = bounds_list_to_init(bounds_list)

    return Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_mat_rhs, \
                init_range_tuples=init_range_tuples)

def define_settings(ha):
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    #plot_settings.plot_mode = PlotSettings.PLOT_NONE
    plot_settings.filename = "iss_uncertain_inputs.png"

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
    plot_settings.label.y_label = '$y_{3}$'
    plot_settings.label.x_label = 'Time'
    plot_settings.label.title = 'Space Station (Uncertain Inputs)'
    #plot_settings.label.axes_limits = (0.4, 0.6, -0.0002, -0.0001)
    plot_settings.plot_size = (12, 8)
    plot_settings.label.big(size=36)

    plot_settings.extra_lines = [[(0.0, -0.0005), (20.0, -0.0005)], [(0.0, 0.0005), (20.0, 0.0005)]]

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    #print "!!! in iss.py run_hylaa(). Check if early break in arnoldi loop actually helps performance on this example"

    ha = define_ha()
    settings = define_settings(ha)
    init = make_init_star(ha, settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
