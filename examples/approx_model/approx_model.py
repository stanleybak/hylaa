'''
Harmonic Oscillator Example in Hylaa, demonstrating using
various approximation models for continuous-time reachability


dynamics are:
x' = y + u1
y' = -x
starting from [-5, -4], [0, 1]
with u1 in [-0.2, 0.2]
'''

import math

import numpy as np
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha():
    '''make the hybrid automaton'''

    ha = HybridAutomaton()

    a_matrix = np.array([[0, 1], [-1, 0]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)

    b_mat = [[1], [0]]
    b_constraints = [[1], [-1]]
    b_rhs = [0.2, 0.2]

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)
    mode.set_inputs(b_mat, b_constraints, b_rhs)

    return ha

def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    # init states: x in [-5, -4], y in [0, 1]
    init_lpi = lputil.from_box([[-5, -4], [0, 1]], mode)
    #init_lpi = lputil.from_box([[-5, -5], [0, 0]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list

def define_settings():
    'get the hylaa settings object'

    step = math.pi/4
    max_time = math.pi / 2
    settings = HylaaSettings(step, max_time)

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    #plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    #plot_settings.filename = 'ha.mp4'
    #plot_settings.video_fps = 2
    #plot_settings.video_extra_frames = 10 # extra frames at the end of a video so it doesn't end so abruptly
    #plot_settings.video_pause_frames = 5 # frames to render in video whenever a 'pause' occurs
    
    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Harmonic Oscillator'

    return settings

def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()

    tuples = []
    #tuples.append((HylaaSettings.APPROX_NONE, "approx_none.png"))
    tuples.append((HylaaSettings.APPROX_CHULL, "approx_chull.png"))
    #tuples.append((HylaaSettings.APPROX_LGG, "approx_lgg.png"))

    for model, filename in tuples: 
        settings.approx_model, settings.plot.filename = model, filename

        init_states = make_init(ha)
        print(f"\nMaking {filename}...")
        Core(ha, settings).run(init_states)

if __name__ == '__main__':
    run_hylaa()
