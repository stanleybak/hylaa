'''
International Space Station Example in Hylaa-Continuous
'''

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix

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

    # a is a csc_matrix
    col_ptr = [n for n in a_matrix.indptr]
    rows = [n for n in a_matrix.indices]
    data = [n for n in a_matrix.data]

    b_matrix = dynamics['B']
    num_inputs = b_matrix.shape[1]

    for u in xrange(num_inputs):
        rows += [n for n in b_matrix[:, u].indices]
        data += [n for n in b_matrix[:, u].data]
        col_ptr.append(len(data))

    combined_mat = csc_matrix((data, rows, col_ptr), \
        shape=(a_matrix.shape[0] + num_inputs, a_matrix.shape[1] + num_inputs))

    mode.set_dynamics(csr_matrix(combined_mat))

    error = ha.new_mode('error')

    y3 = dynamics['C'][2]
    col_ptr = [n for n in y3.indptr] + num_inputs * [y3.data.shape[0]]
    y3 = csc_matrix((y3.data, y3.indices, col_ptr), shape=(1, y3.shape[1] + num_inputs))
    output_space = csr_matrix(y3)

    #print "y3.data = {}, y3.indices = {}, y3.col_ptr = {}".format(y3.data, y3.indices, y3.col_ptr)

    mode.set_output_space(output_space)

    limit = 0.0005
    #limit = 0.00017
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
        if dim == 270: # input 1
            lb = 0
            ub = 0.1
        elif dim == 271: # input 2
            lb = 0.8
            ub = 1.0
        elif dim == 272: # input 3
            lb = 0.9
            ub = 1.0
        elif dim < 270:
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
    plot_settings.filename = "iss_fixed_inputs.png"

    max_time = 20.0
    step_size = 0.01
    settings = HylaaSettings(step=step_size, max_time=max_time, plot_settings=plot_settings)

    settings.time_elapse.method = TimeElapseSettings.SCIPY_SIM
    #settings.time_elapse.method = TimeElapseSettings.KRYLOV
    #settings.time_elapse.krylov.use_lanczos_eigenvalues = False
    #settings.time_elapse.check_answer = True

    #settings.interval_guard_optimization = False

    #settings.skip_step_times = False

    plot_settings.xdim_dir = None
    plot_settings.ydim_dir = ha.modes.values()[0].output_space_csr[0].toarray()

    plot_settings.max_shown_polys = None
    plot_settings.label.y_label = '$y_{3}$'
    plot_settings.label.x_label = 'Time'
    plot_settings.label.title = 'Space Station (Fixed Inputs)'
    #plot_settings.label.axes_limits = (0.4, 0.6, -0.0002, -0.0001)
    plot_settings.plot_size = (12, 8)
    plot_settings.label.big(size=36)

    plot_settings.extra_lines = [[(0.0, -0.00017), (20.0, -0.00017)], [(0.0, 0.00017), (20.0, 0.00017)]]

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
