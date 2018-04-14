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

def define_ha():
    '''make the hybrid automaton'''

    ha = LinearHybridAutomaton()

    # with time and affine variable
    a_matrix = csr_matrix(np.array([[0, 1], [-1, 0]], dtype=float))

    b_matrix = csc_matrix(np.array([[1, 0], [0, 1]], dtype=float))

    # -0.5 <= u1 <= 0.5
    # -0.5 <= u2 <= 0.5
    u_mat = csr_matrix(np.array([[-1., 0], [1., 0], [0, -1], [0, 1]], dtype=float))
    u_rhs = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix)
    mode.set_inputs(b_matrix, u_mat, u_rhs)

    error = ha.new_mode('error')

    # error: x1 == 7
    output_space = csr_matrix(([1.], [0], [0, 1]), shape=(1, a_matrix.shape[0]), dtype=float)

    mat = csr_matrix(np.array([[-1.], [1.]], dtype=float))
    rhs = np.array([-7, 7], dtype=float)

    mode.set_output_space(output_space)
    trans1 = ha.new_transition(mode, error)
    trans1.set_guard(mat, rhs)

    return ha

def make_init_star(ha, hylaa_settings):
    '''returns a star'''

    rv = None

    # vec1 is <0, 1, 0, 0> with the constraint that 0 <= vec1 <= 1
    # vec2 is <-5, 0, 0, 1> with the constraint that vec2 == 1

    #init_space = csc_matrix(np.array([[0., 1, 0, 0], [-5, 0, 0, 1]], dtype=float).transpose())
    #init_mat = csr_matrix(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]], dtype=float))
    #init_rhs = np.array([[1], [0], [1], [-1.]], dtype=float)

    init_space = csc_matrix(np.array([[1., 0], [0, 1]], dtype=float))
    init_mat = csr_matrix(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]], dtype=float))
    init_rhs = np.array([[-5], [6], [1], [0.]], dtype=float)

    rv = Star(hylaa_settings, ha.modes['mode'], init_space, init_mat, init_rhs)

    return rv

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_INTERACTIVE
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

    settings = HylaaSettings(step=math.pi/4, max_time=2 * math.pi, plot_settings=plot_settings)
    settings.time_elapse.method = TimeElapseSettings.SCIPY_SIM
    #settings.time_elapse.method = TimeElapseSettings.KRYLOV
    #settings.time_elapse.method = TimeElapseSettings.EXP_MULT

    settings.time_elapse.check_answer = True

    #settings.time_elapse.krylov.stdout = True

    #settings.time_elapse.force_init_space = True

    return settings

def run_hylaa(hylaa_settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha = define_ha()
    init = make_init_star(ha, hylaa_settings)

    engine = HylaaEngine(ha, hylaa_settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(define_settings())
