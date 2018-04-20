'''
Hylaa Settings File
Stanley Bak
September 2016
'''

import math

from hylaa.util import Freezable

import numpy as np
from scipy.integrate import RK45

class HylaaSettings(Freezable):
    'Settings for the computation'

    def __init__(self, step, max_time, plot_settings=None):
        if plot_settings is None:
            plot_settings = PlotSettings()

        assert isinstance(plot_settings, PlotSettings), "plot_settings was of type {}".format(type(plot_settings))

        self.step = step # simulation step size
        self.num_steps = int(math.ceil(max_time / step))

        self.plot = plot_settings
        self.time_elapse = TimeElapseSettings(step)

        self.print_output = True # print status and waiting list information to stdout
        self.skip_step_times = False # print the times at each step

        self.print_lp_on_error = False # upon reaching an error mode, print LP and exit (no counter-example)
        self.counter_example_filename = 'counterexample.py' # the counter-example filename to create on errors

        ### COMPUTATION OPTIMIZATIONS ###

        # if initial states are intervals and single guard condition, we don't need an LP solver
        self.interval_guard_optimization = True

        self.freeze_attrs()

class TimeElapseSettings(Freezable):
    'time elapse settings container'

    # simulation mode (matrix-exp)
    MATRIX_EXP = 0 # matrix exp every step (slow)
    EXP_MULT = 1 # first step matrix exp, remaining steps matrix-vector multiplication
    KRYLOV = 2 # krylov simulation method
    SCIPY_SIM = 3 # numerical simulation using an instance of scipy.integrate.OdeSolver

    def __init__(self, step):
        self.step = step
        self.method = TimeElapseSettings.SCIPY_SIM

        self.force_init_space = None # True -> use init space when possible, False -> output space, None -> auto detect

        self.check_answer = False # double-check answer using MATRIX_EXP at each step (slow!)
        self.check_answer_abs_tol = 1e-5 # absolute tolerance when checking answer

        self.krylov = KrylovSettings() # used only with krylov method
        self.scipy_sim = ScipySimSettings() # used only with the scipy_sim method

        self.freeze_attrs()

class ScipySimSettings(Freezable):
    'scipy-based simulation settings'

    def __init__(self):
        self.ode_class = RK45

        # settings for the simulation
        self.max_step = np.inf
        self.rtol = 1e-7
        self.atol = 1e-10

class KrylovSettings(Freezable):
    'krylov simulation settings'

    def __init__(self):
        self.stdout = False # additional printing for krylov method
        self.target_error = 1e-6 # desired krylov simulation error

        # simulation settings for computation on smaller H Matrix
        self.ode_class = RK45 # simulation class object. if None, will use expm_mult

        # settings for the simulation (if ode_class is not None)
        self.max_step = np.inf
        self.rtol = 1e-7
        self.atol = 1e-10

        self.freeze_attrs()

class PlotSettings(Freezable):
    'plot settings container'

    PLOT_NONE = 0 # don't plot (for performance measurement)
    PLOT_FULL = 1 # plot the computation video live
    PLOT_INTERACTIVE = 2 # step-by-step live plotting with buttons
    PLOT_IMAGE = 3 # save the image plot to a file
    PLOT_VIDEO = 4 # save animation to a video file
    PLOT_MATLAB = 5 # create a matlab script which visualizes the reachable region
    PLOT_GNUPLOT = 6 # plot gnuplot polygon data file

    def __init__(self):
        self.plot_mode = PlotSettings.PLOT_NONE

        self.xdim_dir = 0 # plotting x dimension direction (if None, use time)
        self.ydim_dir = 1 # plotting y dimension direction (if None, use time)

        self.plot_size = (12, 8) # inches
        self.label = LabelSettings() # plot title, axis labels, font sizes, ect.

        self.num_angles = 512 # how many evenly-spaced angles to put into plot_vecs

        self.extra_lines = None # extra lines to draw on the plot. list of lists of x,y pairs
        self.extra_lines_color = 'gray' # color of extra lines
        self.extra_lines_width = 2 # width of extra lines
        self.reachable_poly_width = 2 # width of reachable polygon outlines

        self.min_frame_time = 0.025 # max 40 fps. This allows multiple frames to be drawn at once if they're fast.

        self.extend_plot_range_ratio = 0.1 # extend plot axis range 10% at a time
        self.anim_delay_interval = 0 # milliseconds, extra delay between frames

        self.filename = None # filename to print data to for certain plot modes

        self.video = None # instance of VideoSettings

        self.grid = True
        self.grid_xtics = None # override default xtics value, for example np.linspace(0.0, 5.0, 1.0)
        self.grid_ytics = None # override default ytics value, for example np.linspace(0.0, 5.0, 1.0)

        self.plot_traces = True
        self.max_shown_polys = 512 # thin out the reachable set if we go over this number of polys (optimization)
        self.draw_stride = 1 # draw every 2nd poly, 4th, ect.

        self.use_markers_for_small = True # draw markers when the reachable set is tiny instead of invisible polygons

        # these are useful for testing / debugging
        self.skip_frames = 0 # number of frames to process before we start drawing
        self.skip_show_gui = False # should we skip showing the graphical interface

        self.freeze_attrs()

    def make_video(self, filename, frames=100, fps=20):
        'update the plot settings to produce a video'

        self.plot_mode = PlotSettings.PLOT_VIDEO

        # turn off artificial delays
        self.anim_delay_interval = 0 # no extra delay between frames
        self.min_frame_time = 0 # draw every frame

        self.filename = filename

        # this should make it look a little sharper, but it will take longer to plot
        self.num_angles = 1024

        # looks better (but takes longer to plot)
        self.extend_plot_range_ratio = 0.0

        video = VideoSettings()
        video.frames = frames
        video.fps = fps

        self.video = video

class VideoSettings(Freezable):
    'settings for video'

    def __init__(self):
        self.codec = 'libx264'
        self.frames = None # number of frames in the video, matplotlib maxes out at 100 frame if not set
        self.fps = 20 # number of frames per second

        self.freeze_attrs()

class LabelSettings(Freezable):
    'settings for labels such as plot title, plot font size, ect.'

    def __init__(self):
        self.x_label = None
        self.y_label = None
        self.title = None

        self.title_size = 32
        self.label_size = 24
        self.tick_label_size = 18
        self.axes_limits = None # fixed axes limits; a 4-tuple (xmin, xmax, ymin, ymax) or None for auto

        self.freeze_attrs()

    def big(self, size=30):
        'increase sizes of labels'

        self.title_size = size
        self.label_size = size
        self.tick_label_size = int(0.9 * size)

    def turn_off(self):
        'turn off plot labels'

        self.x_label = ''
        self.y_label = ''
        self.title = ''
