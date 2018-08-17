'''
Hylaa Settings File
Stanley Bak, 2018
'''

import math
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

    AGG_NONE, AGG_BOX, AGG_ARNOLDI_BOX = range(3)

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
        
        self.aggregation = HylaaSettings.AGG_ARNOLDI_BOX # transition aggregation method
        self.aggregation_add_guard = True # when performing aggregation, also add the guard direction?

        self.freeze_attrs()

class PlotSettings(Freezable): # pylint: disable=too-few-public-methods
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

        self.xdim_dir = 0 # plotting x dimension number, direction (np.array), None (time), or dict: mode_name -> dir
        self.ydim_dir = 1 # plotting y dimension number, direction (np.array), None (time), or dict: mode_name -> dir

        self.plot_size = (8, 8) # inches
        self.label = LabelSettings() # plot title, axis labels, font sizes, ect.

        self.num_angles = 512 # how many evenly-spaced angles to put into plot_vecs

        self.extra_draw_func = lambda ax: None # extra draw function that gets called each frame, param is axis object
        
        self.reachable_poly_width = 2 # width of reachable polygon outlines
        self.extend_plot_range_ratio = 0.1 # extend plot axis range 10% at a time

        self.grid = True
        self.grid_xtics = None # override default xtics value, for example np.linspace(0.0, 5.0, 1.0)
        self.grid_ytics = None # override default ytics value, for example np.linspace(0.0, 5.0, 1.0)

        self.use_markers_for_small = True # draw markers when the reachable set is tiny instead of invisible polygons

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
