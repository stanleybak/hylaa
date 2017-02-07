'''
Hylaa Containers File
Stanley Bak
September 2016
'''

import math
from collections import OrderedDict

import numpy as np

from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.star import AggregationParent, add_guard_to_star, add_box_to_star
from hylaa.optutil import MultiOpt

class SimulationSettings(object):
    'empty object, so newer versions of Hyst files dont produce errors when running'
    
    def __init__(self):
        pass

class HylaaSettings(object):
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
        self.trim_redundant_inv_constraints = True # perform redundant invariant trimming
        self.process_urgent_guards = False # should urgent transition (where 0 time elapses in a mode) be allowed?

        self.sim_tol = None # odeint simulation tolerance setting (default is around 1.5e-8)
        self.solver = 'glpk-multi' # LP solver; one of 'glpk-multi', 'cvxopt-glpk'
        
        self.print_output = True # print status and waiting list information to stdout
        self.print_step_times = False # print the times at each step

        self.use_presimulation = False # this is faster, but less interactive (automatically set if plot is off)
        self.use_guard_strengthening = True # add the invariants of scucessor modes into the guards?

class PlotSettings(object):
    'plot settings container'

    PLOT_NONE = 0 # don't plot (for performance measurement)
    PLOT_FULL = 1 # plot the computation video live
    PLOT_INTERACTIVE = 2 # step-by-step live plotting with buttons
    PLOT_VIDEO = 3 # save animation to a video file
    PLOT_IMAGE = 4 # save the image plot to a file

    def __init__(self):
        self.plot_mode = PlotSettings.PLOT_FULL

        self.xdim = 0 # plotting x dimendion index
        self.ydim = 1 # plotting y dimension index
        
        self.plot_size = (12, 8) # inches
        self.label = LabelSettings() # plot title, axis labels, font sizes, ect.

        self.plot_vecs = None # vectors of directions we should maximize for plotting, set in init_plot_vecs()
        self.num_angles = 256 # how many evenly-spaced angles to put into plot_vecs

        self.extra_lines = None # extra lines to draw on the plot. list of lists of x,y pairs
        self.min_frame_time = 0.025 # max 40 fps. This allows multiple frames to be drawn at once if they're fast.

        self.extend_plot_range_ratio = 0.1 # extend plot axis range 10% at a time
        self.anim_delay_interval = 0 # milliseconds, extra delay between frames (useful if plotting to a video)
        
        self.video_filename = None # for use with plot_video, can force a format (try .mp4)
        self.video_codec = 'libx264'
        self.video_frames = None # number of frames in the video, matplotlib maxes out at 100 frame if not set

        self.filename = None # used with PLOT_IMAGE
        self.grid = True
        self.plot_traces = True
        self.max_shown_polys = 256 # thin out the reachable set if we go over this number of polys (optimization)
        self.draw_stride = 1 # draw every 2nd poly, 4th, ect.

        # these are useful for testing / debugging
        self.skip_frames = 0 # number of frames to process before we start drawing
        self.skip_show_gui = False # don't show the gui

    def init_plot_vecs(self, num_dims):
        'delayed initialization for the plot vectors'

        assert num_dims >= 1

        if self.plot_mode != self.PLOT_NONE:
            self.plot_vecs = []

            if self.xdim is not None and (self.xdim < 0 or self.xdim >= num_dims):
                self.xdim = None

            if self.ydim < 0 or self.ydim >= num_dims:
                self.ydim = 0

            if self.xdim is None:
                # use time for xdim and just minimize / maximize ydim
                self.plot_vecs.append(np.array([0.0 if d != self.ydim else 1.0 for d in xrange(num_dims)]))
                self.plot_vecs.append(np.array([0.0 if d != self.ydim else -1.0 for d in xrange(num_dims)]))
            else:
                step = 2.0 * math.pi / self.num_angles

                for theta in np.arange(0.0, 2.0*math.pi, step):
                    x = math.cos(theta)
                    y = math.sin(theta)

                    vec = np.array([0.0] * num_dims)
                    vec[self.xdim] = x
                    vec[self.ydim] = y

                    self.plot_vecs.append(vec)

class LabelSettings(object):
    'settings for labels such as plot title, plot font size, ect.'

    def __init__(self):
        self.x_label = None
        self.y_label = None
        self.title = None

        self.title_size = 32
        self.label_size = 24
        self.tick_label_size = 18

    def turn_off(self):
        'turn off plot labels'

        self.xlabel = ''
        self.ylabel = ''
        self.title = ''

class HylaaResult(object):
    'Result, assigned to engine.result after computation'

    def __init__(self):
        self.time = None

class SymbolicState(object):
    'a container object for states in the waiting list and other places'

    def __init__(self, mode, star):
        assert isinstance(mode, LinearAutomatonMode)

        self.mode = mode
        self.star = star

class WaitingList(object):
    '''
    The set of to-be computed values (discrete sucessors).

    There are aggregated states, and deaggregated states. The idea is states first
    go into the aggregrated ones, but may be later split and placed into the 
    deaggregated list. Thus, deaggregated states, if they exist, are popped first.
    The states here are SymbolicStates
    '''

    def __init__(self):
        self.aggregated_mode_to_state = OrderedDict()
        self.deaggregated_list = []

    def pop(self):
        'pop a state from the waiting list'

        assert len(self.aggregated_mode_to_state) + len(self.deaggregated_list) > 0, \
            "pop() called on empty waiting list"

        if len(self.deaggregated_list) > 0:
            rv = self.deaggregated_list[0]
            self.deaggregated_list = self.deaggregated_list[1:]
        else:
            # pop from aggregated list
            rv = self.aggregated_mode_to_state.popitem(last=False)[1]

            assert isinstance(rv, SymbolicState)
    
            # pylint false positive
            if isinstance(rv.star.parent, AggregationParent):
                rv.star.trim_redundant_perm_constraints()

        # assert temp_constraints is 0

        assert len(rv.star.temp_constraints) == 0

        return rv

    def print_stats(self):
        'print statistics about the waiting list'

        total = len(self.aggregated_mode_to_state) + len(self.deaggregated_list)

        print "Waiting list contains {} states ({} aggregated and {} deaggregated):".format(
            total, len(self.aggregated_mode_to_state), len(self.deaggregated_list))

        counter = 1

        for ss in self.deaggregated_list:
            print " {}. Deaggregated Successor in Mode '{}'".format(counter, ss.mode.name)
            counter += 1

        for mode, ss in self.aggregated_mode_to_state.iteritems():
            if isinstance(ss.star.parent, AggregationParent):
                print " {}. Aggregated Sucessor in Mode '{}': {} stars".format(counter, mode, len(ss.star.parent.stars))
            else:
                # should be a DiscretePostParent
                print " {}. Aggregated Sucessor in Mode '{}': single star".format(counter, mode)

            counter += 1

    def is_empty(self):
        'is the waiting list empty'

        return len(self.deaggregated_list) == 0 and len(self.aggregated_mode_to_state) == 0

    def add_deaggregated(self, state):
        'add a state to the deaggregated list'

        assert isinstance(state, SymbolicState)

        self.deaggregated_list.append(state)

    def add_aggregated(self, new_state, hylaa_settings):
        'add a state to the aggregated map'

        assert isinstance(new_state, SymbolicState)

        mode_name = new_state.mode.name

        existing_state = self.aggregated_mode_to_state.get(mode_name)

        if existing_state is None:
            self.aggregated_mode_to_state[mode_name] = new_state
        else:
            # combine the two stars
            cur_star = existing_state.star

            cur_star.current_step = min(
                cur_star.total_steps, new_state.star.total_steps)

            # if the parent of this star is not an aggregation, we need to create one
            # otherwise, we need to add it to the list of parents

            if isinstance(cur_star.parent, AggregationParent):
                # add it to the list of parents
                cur_star.parent.stars.append(new_state.star)

                cur_star.eat_star(new_state.star)
            else:
                # create the aggregation parent
                hull_star = cur_star.clone()
                hull_star.parent = AggregationParent(new_state.mode, [cur_star, new_state.star])

                if hylaa_settings.add_guard_during_aggregation:
                    add_guard_to_star(hull_star, cur_star.parent.transition.condition_list)

                if hylaa_settings.add_box_during_aggregation:
                    add_box_to_star(hull_star)

                # there may be temp constraints from invariant trimming
                hull_star.commit_temp_constraints()

                hull_star.eat_star(new_state.star)
                existing_state.star = hull_star


