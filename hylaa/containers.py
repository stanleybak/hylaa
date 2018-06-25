'''
Hylaa Containers File
Stanley Bak
September 2016
'''

import math

from hylaa.util import Freezable

class HylaaSettings(Freezable):
    'Settings for the computation'

    def __init__(self, step, max_time, plot_settings=None):
        if plot_settings is None:
            plot_settings = PlotSettings()

        assert isinstance(plot_settings, PlotSettings)

        self.step = step # simulation step size
        self.num_steps = int(math.ceil(max_time / step))

        self.plot = plot_settings

        self.aggregation = True # perform sucessor aggregation
        self.deaggregation = True # perform trace-guided deaggregation

        self.add_guard_during_aggregation = True # add guard constraints during aggregation
        self.add_box_during_aggregation = True # add box constraints during aggregation
        #self.trim_redundant_inv_constraints = True # perform redundant invariant trimming
        self.process_urgent_guards = False # should urgent transition (where 0 time elapses in a mode) be allowed?
        self.stop_when_error_reachable = True # should we stop computing immediately when an error mode is reached?

        self.print_output = True # print status and waiting list information to stdout
        self.skip_step_times = False # print the times at each step

        self.opt_decompose_lp = True # use the Minkowski sum decomposition optimization (for systems with inputs)
        self.opt_warm_start_lp = True # reuse the LP instances between guard checks (warm-start LP)

        self.do_guard_strengthening = True

        self.counter_example_filename = None # the counter-example filename to create on errors: "counterexample.py"
        self.simulation = SimulationSettings(step)

        self.freeze_attrs()

class SimulationSettings(Freezable):
    'simulation settings container'

    SIMULATION = 0 # full time range simulation (default)
    MATRIX_EXP = 1 # use matrix exponential(expm) for first step then do matrix multiplication

    def __init__(self, step):
        self.use_presimulation = False # this is faster, but less interactive (automatically set if plot is off)
        self.step = step
        self.sim_tol = None # odeint simulation tolerance setting (default is around 1.5e-8)
        self.threads = 1 # Threads used for simulation, None = auto-detect number of system cores
        self.sparse = False # use sparse matrices for simulation
        self.sim_mode = SimulationSettings.SIMULATION # use simulations or use matrix exp

        self.sim_in_memory_mb = 4 * 1024 # simulations size in memory per mode (roughly, not a strict limit)

        self.stdout = True # print output during simulations
        self.print_interval_secs = 2 # how often to print to stdout during parallel simulations

        self.freeze_attrs()

class PlotSettings(Freezable):
    'plot settings container'

    PLOT_NONE = 0 # don't plot (for performance measurement)
    PLOT_FULL = 1 # plot the computation video live
    PLOT_INTERACTIVE = 2 # step-by-step live plotting with buttons
    PLOT_IMAGE = 3 # save the image plot to a file
    PLOT_VIDEO = 4 # save animation to a video file
    PLOT_MATLAB = 5 # create a matlab script which visualizes the reachable region

    def __init__(self):
        self.plot_mode = PlotSettings.PLOT_FULL

        self.xdim = 0 # plotting x dimendion index
        self.ydim = 1 # plotting y dimension index

        self.plot_size = (12, 8) # inches
        self.label = LabelSettings() # plot title, axis labels, font sizes, ect.

        self.num_angles = 512 # how many evenly-spaced angles to put into plot_vecs

        self.extra_lines = None # extra lines to draw on the plot. list of lists of x,y pairs
        self.min_frame_time = 0.025 # max 40 fps. This allows multiple frames to be drawn at once if they're fast.

        self.extend_plot_range_ratio = 0.1 # extend plot axis range 10% at a time
        self.anim_delay_interval = 0 # milliseconds, extra delay between frames

        self.filename = None # used with PLOT_VIDEO AND PLOT_IMAGE

        self.video = None # instance of VideoSettings

        self.grid = True
        self.plot_traces = True
        self.max_shown_polys = 512 # thin out the reachable set if we go over this number of polys (optimization)
        self.draw_stride = 1 # draw every 2nd poly, 4th, ect.

        # these are useful for testing / debugging
        self.skip_frames = 0 # number of frames to process before we start drawing
        self.skip_show_gui = False # should we skip showing the graphical interface
        self.print_lp_at_each_step = False # should we print the LP being plotted at each step?

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

class HylaaResult(object):
    'Result, assigned to engine.result after computation'

    def __init__(self):
        self.time = None
