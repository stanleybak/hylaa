'''
Harmonic Oscillator (with time) Example in Hylaa-Continuous
'''

import numpy as np
from scipy.sparse import csc_matrix

from hylaa.hybrid_automaton import LinearHybridAutomaton, SparseLinearConstraint
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.containers import PlotSettings, SimulationSettings
from hylaa.star import Star

def define_ha():
    '''make the hybrid automaton'''

    # 4 variables: x, y, time, affine_term
    ha = LinearHybridAutomaton(4)

    a_matrix = csc_matrix(np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float))

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    trans.condition_list.append(SparseLinearConstraint([-1, 0, 0, 0], -6.5)) # x0 >= 6.5

    return ha

def define_init(ha, hylaa_settings):
    '''returns a star'''
    constraints = []
    constraints.append(SparseLinearConstraint([-1, 0, 0, 0], 6)) # x0 >= -6
    constraints.append(SparseLinearConstraint([1, 0, 0, 0], -5)) # x0 <= -5
    constraints.append(SparseLinearConstraint([0, -1, 0, 0], 0)) # x1 >= 0
    constraints.append(SparseLinearConstraint([0, 1, 0, 0], 1)) # x1 <= 1

    # time == 0
    constraints.append(SparseLinearConstraint([0, 0, -1, 0], 0))
    constraints.append(SparseLinearConstraint([0, 0, 1, 0], 0))

    # affine_term = 1
    constraints.append(SparseLinearConstraint([0, 0, 0, -1], -1))
    constraints.append(SparseLinearConstraint([0, 0, 0, 1], 1))

    return Star(hylaa_settings, constraints, ha.modes['mode'])

def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_INTERACTIVE
    plot_settings.xdim_dir = [1, 0, 0, 0]
    plot_settings.ydim_dir = [0, 1, 0, 0]

    # save a video file instead
    # plot_settings.make_video("building.mp4", frames=220, fps=40)

    plot_settings.num_angles = 128
    plot_settings.max_shown_polys = 2048
    plot_settings.label.y_label = 'y'
    plot_settings.label.x_label = 'x'
    plot_settings.label.title = ''
    plot_settings.plot_size = (8, 8)
    #plot_settings.label.big(size=40)

    settings = HylaaSettings(step=0.1, max_time=6.0, plot_settings=plot_settings)
    settings.simulation.sim_mode = SimulationSettings.MATRIX_EXP

    return settings

def run_hylaa(hylaa_settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha = define_ha()
    init = define_init(ha, hylaa_settings)

    engine = HylaaEngine(ha, hylaa_settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(define_settings())
