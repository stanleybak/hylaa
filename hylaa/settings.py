'''
Hylaa Settings File
Stanley Bak, 2018
'''

import math

from matplotlib import animation

from hylaa.util import Freezable

# Force matplotlib to not use any Xwindows backend.
#import matplotlib
#matplotlib.use('Agg')

# Suppress floating point printing overflow/underflow printing in numpy
#import numpy as np
#np.set_printoptions(suppress=True)

class HylaaSettings(Freezable):  # pylint: disable=too-few-public-methods
    'Settings for the computation'

    STDOUT_NONE, STDOUT_NORMAL, STDOUT_VERBOSE, STDOUT_DEBUG = range(4)

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
        self.stop_on_error = True
        
        self.aggregation = AggregationSettings()

        self.freeze_attrs()

class AggregationSettings(Freezable): # pylint: disable=too-few-public-methods
    'aggregation settings container'

    AGG_NONE, AGG_BOX, AGG_ARNOLDI_BOX = range(3)
    POP_LOWEST_MINTIME, POP_LOWEST_AVGTIME, POP_LARGEST_MAXTIME = range(3)

    def __init__(self):
        
        self.agg_mode = AggregationSettings.AGG_ARNOLDI_BOX # transition aggregation method
        self.add_guard = True # when performing aggregation, also add the guard direction?

        self.custom_pop_func = None # function taking in waiting list and returns list of states to aggregate
        # if custom_pop_func was None, these two settings guide which state gets popped off of waiting list
        self.require_same_path = True # only aggregate states with same discrete-transition path? (False=all)
        self.pop_strategy = AggregationSettings.POP_LOWEST_AVGTIME

        self.deaggregation = False # perform concrete-trace guided deaggregation

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

        # these two can also be lists, in which case we'll make multiple plots
        self.xdim_dir = 0 # plotting x dimension number, direction (np.array), None (time), or dict: mode_name -> dir
        self.ydim_dir = 1 # plotting y dimension number, direction (np.array), None (time), or dict: mode_name -> dir

        self.plot_size = (8, 8) # inches
        self.label = LabelSettings() # plot title, axis labels, font sizes, ect.

        self.num_angles = 512 # how many evenly-spaced angles to put into plot_vecs

        self.draw_stride = 1 # draw every n frames (good to inscrease if number of steps is large)

        self.extra_draw_func = lambda ax: None # extra draw function that gets called each frame, param is axis object
        
        self.reachable_poly_width = 2 # width of reachable polygon outlines
        self.extend_plot_range_ratio = 0.1 # extend plot axis range 10% at a time

        self.grid = True
        self.grid_xtics = None # override default xtics value, for example np.linspace(0.0, 5.0, 1.0)
        self.grid_ytics = None # override default ytics value, for example np.linspace(0.0, 5.0, 1.0)

        self.use_markers_for_small = True # draw markers when the reachable set is tiny instead of invisible polygons

        # function which returns the Writer with the desired settings used to create a video, used for video export
        def make_video_writer():
            'returns the Writer to create a video for export'

            writer_class = animation.writers['ffmpeg']
            return writer_class(fps=50, metadata=dict(artist='Me'), bitrate=1800)

        self.make_video_writer_func = make_video_writer
        self.video_extra_frames = 20 # extra frames at the end of a video so it doesn't end so abruptly

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

class StaticSettings(): # pylint: disable=too-few-public-methods
    'Static settings'

    # swiglpk has a memory leak: https://github.com/biosustain/swiglpk/issues/31
    # how much memory should we allow to be used before we print a message and quit
    MAX_MEMORY_SWIGLPK_LEAK_GB = 1.0 # 2.0
