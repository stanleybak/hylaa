'''
Harmonic Oscillator (with time) Example in Hylaa-Continuous
'''

import math

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.settings import PlotSettings, TimeElapseSettings
from hylaa.star import Star

def define_ha(sparse_definition):
    '''make the hybrid automaton'''

    ha = LinearHybridAutomaton()

    # with time and affine variable
    a_matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
    a_matrix = csr_matrix(a_matrix, dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    if sparse_definition:
        # x1 >= 4.0 & x1 <= 4.0
        output_space = csr_matrix(([1.], [0], [0, 1]), shape=(1, 4), dtype=float)

        mat = csr_matrix(np.array([[1.], [-1.]], dtype=float))
        rhs = np.array([4.0, -4.0], dtype=float)
    else:
        # use full dimensional output space
        output_space = csr_matrix(np.identity(4))

        mat = csr_matrix(np.array([[1., 0, 0, 0], [-1., 0, 0, 0]], dtype=float))
        rhs = np.array([4.0, -4.0], dtype=float)

    mode.set_output_space(output_space)
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)


    return ha

def make_init_star(ha, hylaa_settings, sparse_definition):
    '''returns a star'''

    rv = None

    if sparse_definition:
        # vec1 is <0, 1, 0, 0> with the constraint that 0 <= vec1 <= 1
        # vec2 is <-5, 0, 0, 1> with the constraint that vec2 == 1

        init_space = csc_matrix(np.array([[0., 1, 0, 0], [-5, 0, 0, 1]], dtype=float).transpose())
        init_mat = csr_matrix(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]], dtype=float))
        init_rhs = np.array([[1], [0], [1], [-1.]], dtype=float)
    else:
        init_space = csc_matrix(np.identity(4))
        init_mat = csr_matrix(np.array([[1., 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], \
            [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]], dtype=float))

        init_rhs = np.array([[-5.], [5], [1], [0], [0], [0], [1], [-1]], dtype=float)

    rv = Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_rhs)

    return rv

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_NONE #PlotSettings.PLOT_INTERACTIVE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    # save a video file instead
    #plot_settings.make_video("vid.mp4", frames=20, fps=5)

    plot_settings.num_angles = 128
    plot_settings.max_shown_polys = 2048
    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Reachable States'
    plot_settings.plot_size = (12, 7)
    plot_settings.label.big(size=48)

    plot_settings.reachable_poly_width = 10
    plot_settings.extra_lines = [[(4.0, 10.0), (4.0, -10.0)]]
    plot_settings.extra_lines_color = 'red'
    plot_settings.extra_lines_width = 4

    settings = HylaaSettings(step=math.pi/4, max_time=3 * math.pi / 4, plot_settings=plot_settings)
    #settings.time_elapse.method = TimeElapseSettings.SCIPY_SIM
    settings.time_elapse.method = TimeElapseSettings.KRYLOV

    settings.time_elapse.check_answer = True

    #settings.time_elapse.krylov.stdout = True

    #settings.time_elapse.force_init_space = True

    return settings

def run_hylaa():
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    hylaa_settings = define_settings()
    sparse_definition = True

    ha = define_ha(sparse_definition)
    init = make_init_star(ha, hylaa_settings, sparse_definition)

    engine = HylaaEngine(ha, hylaa_settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa()
