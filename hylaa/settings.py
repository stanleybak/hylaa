'''
Hylaa Settings File
Stanley Bak, 2018
'''

import math

from matplotlib import animation

from hylaa.util import Freezable
from hylaa import aggstrat

# Force matplotlib to not use any Xwindows backend.
#import matplotlib
#matplotlib.use('Agg')

# Suppress floating point printing overflow/underflow printing in numpy
#import numpy as np
#np.set_printoptions(suppress=True)

class HylaaSettings(Freezable):  # pylint: disable=too-few-public-methods
    'Settings for the computation'

    STDOUT_NONE, STDOUT_NORMAL, STDOUT_VERBOSE, STDOUT_DEBUG = range(4)

    # Approximation Models: None: discrete-time, chull: convex hull only (different simulation semantics),
    #                       lgg: support function method from Le Guernic'10
    APPROX_NONE, APPROX_CHULL, APPROX_LGG = range(3)

    def __init__(self, step_size, max_time):
        plot_settings = PlotSettings()

        self.step_size = step_size # simulation step size
        self.num_steps = int(math.ceil(max_time / step_size))

        self.plot = plot_settings
        self.stdout = HylaaSettings.STDOUT_NORMAL
        self.stdout_colors = [None, "white", "blue", "yellow"] # colors for each level of printing

        ### SIMULATION-EQUIVALENT SEMANTICS / COMPUTATION PARAMETERS ###
        self.process_urgent_guards = False # allow zero continuous-post steps between transitions?
        self.do_guard_strengthening = True # add invariants of target modes to each guard?
        self.optimize_tt_transitions = True # auto-detect time-triggered transitions and use single-step semantics?
        self.approx_model = HylaaSettings.APPROX_NONE
        self.skip_zero_dynamics_modes = True

        # what to do when an error appears reachable
        self.stop_on_aggregated_error = False # stop whenever any state (aggregated or not) reaches an error mode
        self.stop_on_concrete_error = True # stop whenver a concrete state reaches an error
        self.make_counterexample = True # save counter-example to data structure / file?
        
        self.aggstrat = aggstrat.Aggregated() # aggregation strategy class

        # for deterministic random numbers (simulations / color selection)
        self.random_seed = 0

        self.freeze_attrs()

class PlotSettings(Freezable): # pylint: disable=too-few-public-methods,too-many-instance-attributes
    'plot settings container'

    PLOT_NONE = 0 # don't plot (safety checking only; for performance measurement)
    PLOT_LIVE = 1 # plot the computation live as we're computing
    PLOT_INTERACTIVE = 2 # live plotting with pauses and buttons upon certain events
    PLOT_IMAGE = 3 # save the image plot to a file
    PLOT_VIDEO = 4 # save a video to a file

    def __init__(self):
        self.plot_mode = PlotSettings.PLOT_NONE
        
        self.store_plot_result = False # store the reachable plot data inside the computation result object?

        self.filename = None # filename to print data to for certain plot modes

        self.plot_size = (8, 8) # inches

        # these settings can be lists, in which case we'll make multiple plots
        self.xdim_dir = 0 # plotting x dimension number, direction (np.array), None (time), or dict: mode_name -> dir
        self.ydim_dir = 1 # plotting y dimension number, direction (np.array), None (time), or dict: mode_name -> dir
        self.label = LabelSettings() # plot title, axis labels, font sizes, ect.
        self.extra_collections = None # list of extra animation collections

        self.num_angles = 512 # how many evenly-spaced angles to put into plot_vecs

        self.draw_stride = 1 # draw every n frames (good to inscrease if number of steps is large)
        
        self.reachable_poly_width = 2 # width of reachable polygon outlines
        self.extend_plot_range_ratio = 0.1 # extend plot axis range 10% at a time

        self.sim_line_color = 'black' # simulation line color
        self.sim_line_width = 0.2 # width of simulation lines

        self.grid = True
        self.grid_xtics = None # override default xtics value, for example np.linspace(0.0, 5.0, 1.0)
        self.grid_ytics = None # override default ytics value, for example np.linspace(0.0, 5.0, 1.0)

        self.use_markers_for_small = True # draw markers when the reachable set is tiny instead of (invisible) polygons

        self.show_counterexample = True # draw concrete counter-example in the last frame?

        self.interactive_skip_count = 0 # when using PLOT_INTERACTIVE, auto-click 'next' this many times

        self.video_fps = 40
        self.video_extra_frames = 40 # extra frames at the end of a video so it doesn't end so abruptly
        self.video_pause_frames = 20 # frames to render in video whenever a 'pause' occurs
        self.video_show_frame = True # show the frame counter?

        # function which returns the Writer with the desired settings used to create a video, used for video export
        def make_video_writer():
            'returns the Writer to create a video for export'

            writer_class = animation.writers['ffmpeg']
            return writer_class(fps=self.video_fps, metadata=dict(artist='Me'), bitrate=1800)

        self.make_video_writer_func = make_video_writer

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
        self.tick_label_size = int(0.8 * size)

    def turn_off(self):
        'turn off plot labels'

        self.x_label = ''
        self.y_label = ''
        self.title = ''
