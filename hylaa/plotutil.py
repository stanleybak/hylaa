'''
Hylaa Plot Utilities
Stanley Bak
Sept 2016
'''

import sys
import time
import random
from collections import OrderedDict

import numpy as np

from matplotlib import collections, animation, colors, rcParams
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

from hylaa import lpplot
from hylaa.timerutil import Timers
from hylaa.settings import PlotSettings
from hylaa.util import Freezable
from hylaa.result import replay_counterexample

class AxisLimits(Freezable):
    '''the axis limits'''

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        self.freeze_attrs()

    def __str__(self):
        return "({}, {}, {}, {})".format(self.xmin, self.xmax, self.ymin, self.ymax)

class InteractiveState(Freezable): # pylint: disable=too-few-public-methods
    '''the state during PLOT_INTERACTIVE'''

    def __init__(self):
        self.paused = False
        self.step = False

        self.freeze_attrs()

class ModeColors(Freezable):
    'maps mode names -> colors'

    def __init__(self, settings):
        self.settings = settings
        self.mode_to_color = {} # map mode name -> color string

        self.init_colors()
        self.freeze_attrs()

    @staticmethod
    def lighter(rgb_col):
        'return a lighter variant of an rgb color'

        return [(2.0 + val) / 3.0 for val in rgb_col]

    @staticmethod
    def darker(rgb_col):
        'return a darker variant of an rgb color'

        return [val / 2.0 for val in rgb_col]

    def init_colors(self):
        'initialize all_colors'

        self.all_colors = []

        # remove any colors with 'white' or 'yellow in the name
        skip_colors_substrings = ['white', 'yellow']
        skip_colors_exact = ['black', 'red', 'blue']
        skip_colors_threshold = 0.75 # skip colors lighter than this threshold (avg rgb value)

        for col in colors.cnames:
            skip = False

            r, g, b, _ = colors.to_rgba(col)
            avg = (r + g + b) / 3.0
            if avg > skip_colors_threshold:
                skip = True
            else:
                for col_substring in skip_colors_substrings:
                    if col_substring in col:
                        skip = True
                        break

            if not skip and not col in skip_colors_exact:
                self.all_colors.append(col)

        # we'll re-add these later; remove them before shuffling
        first_colors = ['lime', 'orange', 'cyan', 'magenta', 'green']

        for col in first_colors:
            self.all_colors.remove(col)

        # deterministic shuffle of all remaining colors
        random.seed(self.settings.random_seed)
        random.shuffle(self.all_colors)

        # prepend first_colors so they get used first
        self.all_colors = first_colors + self.all_colors

    def get_edge_face_colors(self, mode_name):
        '''
        get the edge and face colors from a mode name

        returns a tuple: (face_col, edge_col)
        '''

        col_name = self.mode_to_color.get(mode_name)

        if col_name is None:
            # pick the next color and save it
            col_name = self.all_colors[len(self.mode_to_color) % len(self.all_colors)]
            self.mode_to_color[mode_name] = col_name

        edge_col = colors.colorConverter.to_rgb(col_name)

        # make the faces a little lighter
        face_col = ModeColors.lighter(edge_col)

        return (face_col, edge_col)

class DrawnShapes(Freezable):
    'maintains shapes to be drawn'

    def __init__(self, plotman, subplot):
        self.plotman = plotman
        self.axes = plotman.axes_list[subplot]
        self.mode_colors = plotman.mode_colors
        self.subplot = subplot

        # parent is a string
        # for modes, prefix is 'mode_{modename}'
        # for sims, prefix is 'sim_states'
        # for cur state, parent is 'cur_state'
        self.parent_to_polys = OrderedDict()
        self.parent_to_markers = OrderedDict()

        self.cur_sim_lines = [] # list of verts (list of 2d points)
        self.cur_sim_line2ds = [] # list of Line2D objects

        self.extra_collection_list = []

        if plotman.settings.extra_collections:
            if isinstance(plotman.settings.extra_collections[0], list):
                self.extra_collection_list = plotman.settings.extra_collections[subplot]
            elif subplot == 0:
                self.extra_collection_list = plotman.settings.extra_collections

            for col in self.extra_collection_list:
                self.axes.add_collection(col)

        self.freeze_attrs()

    def get_artists(self):
        'get the list of artists, to be returned by animate function'

        rv = []

        # objects later in the list will be drawn after (above)
        draw_above = ['gray_state', 'cur_state']
        
        for name, polys in self.parent_to_polys.items():
            if not name in draw_above:
                rv.append(polys)

        for name in draw_above:
            if name in self.parent_to_polys:
                rv.append(self.parent_to_polys.get(name))

        for line2d in self.cur_sim_line2ds:
            rv.append(line2d)

        for markers in self.parent_to_markers.values():
            rv.append(markers)

        for collection in self.extra_collection_list:
            rv.append(collection)

        return rv

    def set_gray_state(self, verts_list):
        'set the dasehd set of states for one frame'

        def init_polycollection_func():
            "initialization function if polycollection doesn't exist"
            
            return collections.PolyCollection([], lw=self.plotman.settings.reachable_poly_width + 3, animated=True,
                                              edgecolor='gray', facecolor=(0., 0., 0., 0.), zorder=2)

        self._set_named_state(verts_list, 'gray_state', init_polycollection_func)

    def set_cur_state(self, verts_list):
        'set the currently tracked set of states for one frame'

        def init_polycollection_func():
            "initialization function if polycollection doesn't exist"
            
            return collections.PolyCollection([], lw=self.plotman.settings.reachable_poly_width, animated=True,
                                              edgecolor='k', facecolor=(0., 0., 0., 0.0), zorder=3)

        self._set_named_state(verts_list, 'cur_state', init_polycollection_func)

    def _set_named_state(self, verts_list, name, init_polycollection_func):
        'set a tracked set of states by name for one frame (for example "cur_state")'

        assert verts_list is None or isinstance(verts_list, list)

        polys = self.parent_to_polys.get(name)
        
        if polys is None:
            # setup for first time drawing cur_state

            polys = init_polycollection_func()
            self.axes.add_collection(polys)

            self.parent_to_polys[name] = polys

        paths = polys.get_paths()

        while polys is not None and paths:
            paths.pop() # remove the old polygon(s)

        if verts_list is not None:
            for verts in verts_list:
                # create a new polygon
                codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
                paths.append(Path(verts, codes))

    def commit_cur_sims(self):
        '''save the current simulation lines (stop appending to them)'''

        rv_verts = self.cur_sim_lines
        
        lines = self.parent_to_polys.get('sim_lines')

        if lines is None:
            lw = self.plotman.settings.sim_line_width
            color = self.plotman.settings.sim_line_color
            lines = collections.LineCollection([], lw=lw, animated=True, color=color, zorder=4)
            self.axes.add_collection(lines)
            self.parent_to_polys['sim_lines'] = lines

        # append all cur_sim_lines to lines
        paths = lines.get_paths()

        for verts in self.cur_sim_lines:
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
            paths.append(Path(verts, codes))

        self.cur_sim_lines = []
        for line2d in self.cur_sim_line2ds:
            # clear line2d
            line2d.set_xdata([])
            line2d.set_ydata([])

        markers = self.parent_to_markers.get('sim_states')

        if markers:
            markers.set_xdata([])
            markers.set_ydata([])

        return rv_verts

    def set_cur_sim(self, verts):
        '''set the current simulation pts'''

        num_sims = len(verts)

        if not self.cur_sim_lines:
            # first point
            while len(self.cur_sim_line2ds) < num_sims:
                lw = self.plotman.settings.sim_line_width
                color = self.plotman.settings.sim_line_color
                line2d = Line2D([], [], lw=lw, animated=True, color=color, zorder=4)
                self.axes.add_line(line2d)
                self.cur_sim_line2ds.append(line2d)

            self.cur_sim_lines = [[] for _ in range(len(verts))]

        # append point to line2ds
        for i, pt in enumerate(verts):
            if pt is None:
                continue
            
            self.cur_sim_lines[i].append(pt)

            line2d = self.cur_sim_line2ds[i]
            xdata = line2d.get_xdata()
            ydata = line2d.get_ydata()
            xdata.append(pt[0])
            ydata.append(pt[1])
            line2d.set_xdata(xdata)
            line2d.set_ydata(ydata)

        # append point to sim_states markers list
        markers = self.parent_to_markers.get('sim_states')

        if markers is None:
            markers = Line2D([], [], animated=True, ls='None', marker='o', mew=2, ms=2,
                             mec='red', mfc='red', zorder=5)
            self.axes.add_line(markers)
            self.parent_to_markers['sim_states'] = markers

        if verts is None:
            markers.set_xdata([])
            markers.set_ydata([])
        else:
            xs = [pt[0] for pt in verts if pt is not None]
            ys = [pt[1] for pt in verts if pt is not None]
        
            markers.set_xdata(xs)
            markers.set_ydata(ys)

    def add_reachable_poly(self, stateset):
        '''add a polygon which was reachable'''

        poly_verts = stateset.verts(self.plotman, self.subplot)
        mode_name = stateset.mode.name

        if len(poly_verts) <= 2 and self.plotman.settings.use_markers_for_small:
            markers = self.parent_to_markers.get('mode_' + mode_name)

            if markers is None:
                face_col, edge_col = self.mode_colors.get_edge_face_colors(mode_name)

                markers = Line2D([], [], animated=True, ls='None', alpha=0.5, marker='o', mew=2, ms=5,
                                 mec=edge_col, mfc=face_col)
                self.axes.add_line(markers)
                self.parent_to_markers['mode_' + mode_name] = markers

            xdata = markers.get_xdata()
            ydata = markers.get_ydata()
            xdata.append(poly_verts[0][0])
            ydata.append(poly_verts[0][1])
            markers.set_xdata(xdata)
            markers.set_ydata(ydata)
        else:
            polys = self.parent_to_polys.get(mode_name)

            if polys is None:
                lw = self.plotman.settings.reachable_poly_width
                face_col, edge_col = self.mode_colors.get_edge_face_colors(mode_name)
                polys = collections.PolyCollection([], lw=lw, animated=True, alpha=0.5,
                                                   edgecolor=edge_col, facecolor=face_col)
                self.axes.add_collection(polys)
                self.parent_to_polys[mode_name] = polys

            paths = polys.get_paths()

            codes = [Path.MOVETO] + [Path.LINETO] * (len(poly_verts) - 2) + [Path.CLOSEPOLY]
            paths.append(Path(poly_verts, codes))

            # save the Path list in the StateSet
            stateset.set_plot_path(self.subplot, paths, len(paths) - 1)

class PlotManager(Freezable):
    'manager object for plotting during or after computation'

    def __init__(self, hylaa_core):
        # matplotlib default rcParams caused incorrect trace output due to interpolation
        rcParams['path.simplify'] = False

        self.core = hylaa_core
        self.settings = hylaa_core.settings.plot

        self.fig = None
        self.axes_list = None
        
        self.actual_limits = None # AxisLimits object
        self.drawn_limits = None # AxisLimits object

        self.mode_colors = ModeColors(hylaa_core.settings)
        self.shapes = None # instance of DrawnShapes
        self.interactive = InteractiveState()

        self._anim = None # animation object
        self.num_frames_drawn = 0

        self.pause_frames = None # for video plots
        self.frame_text = None # for video plots when settings.video_show_frame = True

        self.plot_vec_list = [] # a list of plot_vecs for each subplot
        self.num_subplots = None
        self.init_plot_vecs()

        self.freeze_attrs()

    def pause(self):
        '''pause the plot animation'''

        self.interactive.step = False
        self.interactive.paused = True

    def init_plot_vecs(self):
        'initialize plot_vecs'

        if isinstance(self.settings.xdim_dir, list):
            assert isinstance(self.settings.ydim_dir, list)
            assert len(self.settings.xdim_dir) == len(self.settings.ydim_dir)
        else:
            self.settings.xdim_dir = [self.settings.xdim_dir]
            self.settings.ydim_dir = [self.settings.ydim_dir]

        for xdim_dir, ydim_dir in zip(self.settings.xdim_dir, self.settings.ydim_dir):
            assert not (xdim_dir is None and ydim_dir is None)

            plot_vecs = []

            if xdim_dir is None:
                plot_vecs.append(np.array([0, 1.], dtype=float))
                plot_vecs.append(np.array([0, -1.], dtype=float))
            elif self.settings.ydim_dir is None:
                plot_vecs.append(np.array([1., 0], dtype=float))
                plot_vecs.append(np.array([-1., 0], dtype=float))
            else:
                assert self.settings.num_angles >= 3, "needed at least 3 directions in plot_settings.num_angles"

                plot_vecs = lpplot.make_plot_vecs(self.settings.num_angles)

            self.plot_vec_list.append(plot_vecs)
            
        self.num_subplots = len(self.plot_vec_list)

    def draw_counterexample(self, ce_segments):
        '''we got a concrete counter example, draw it (if we're plotting)
        
        currently this only works for simple plots
        '''

        if self.settings.plot_mode != PlotSettings.PLOT_NONE and self.settings.show_counterexample:
            pts, times = replay_counterexample(ce_segments, self.core.hybrid_automaton, self.core.settings)

            for i, shapes in enumerate(self.shapes):
                w = self.settings.reachable_poly_width

                # project pts onto axis
                xdim = self.settings.xdim_dir[i]
                ydim = self.settings.ydim_dir[i]

                if xdim is None:
                    xs = times
                elif isinstance(xdim, int):
                    xs = [pt[xdim] for pt in pts]
                else:
                    continue

                if ydim is None:
                    ys = times
                elif isinstance(ydim, int):
                    ys = [pt[ydim] for pt in pts]
                else:
                    continue

                verts = [v for v in zip(xs, ys)]

                if self.settings.label[i].axes_limits is None:
                    self.update_axis_limits(verts, i)

                lc = collections.LineCollection([verts], lw=1.0, color='black', zorder=4)

                shapes.axes.add_collection(lc)
                shapes.extra_collection_list.append(lc)

    def state_popped(self):
        'a state was popped off the waiting list'

        if self.settings.plot_mode != PlotSettings.PLOT_NONE:
            for subplot in range(self.num_subplots):
                self.shapes[subplot].set_cur_state(None)

    def update_axis_limits(self, points_list, subplot):
        'update the axes limits to include the passed-in point list'

        first_draw = False

        if self.drawn_limits is None:
            self.drawn_limits = [None] * self.num_subplots
            self.actual_limits = [None] * self.num_subplots

        if self.drawn_limits[subplot] is None:
            first_draw = True
            first_x, first_y = points_list[0]
            self.actual_limits[subplot] = AxisLimits()
            self.drawn_limits[subplot] = AxisLimits()

            lim = self.actual_limits[subplot]
            drawn = self.drawn_limits[subplot]
            
            lim.xmin = lim.xmax = first_x
            lim.ymin = lim.ymax = first_y
            drawn.xmin = drawn.xmax = first_x
            drawn.ymin = drawn.ymax = first_y

        lim = self.actual_limits[subplot]
        drawn = self.drawn_limits[subplot]

        for p in points_list:
            x, y = p

            if x < lim.xmin:
                lim.xmin = x
            elif x > lim.xmax:
                lim.xmax = x

            if y < lim.ymin:
                lim.ymin = y
            elif y > lim.ymax:
                lim.ymax = y

        is_outside = lim.xmin < drawn.xmin or lim.xmax > drawn.xmax or lim.ymin < drawn.ymin or lim.ymax > drawn.ymax
        if first_draw or (is_outside and lim.xmin != lim.xmax and lim.ymin != lim.ymax):

            # expand drawn limits to surpass actual
            dx = lim.xmax - lim.xmin
            dy = lim.ymax - lim.ymin
            ratio = self.settings.extend_plot_range_ratio

            drawn.xmin = lim.xmin - dx * ratio
            drawn.xmax = lim.xmax + dx * ratio
            drawn.ymin = lim.ymin - dy * ratio
            drawn.ymax = lim.ymax + dy * ratio

            if drawn.xmin == drawn.xmax:
                self.axes_list[subplot].set_xlim(drawn.xmin - 1e-1, drawn.xmax + 1e-1)
            else:
                self.axes_list[subplot].set_xlim(drawn.xmin, drawn.xmax)

            if drawn.ymin == drawn.ymax:
                self.axes_list[subplot].set_ylim(drawn.ymin - 1e-1, drawn.ymax + 1e-1)
            else:
                self.axes_list[subplot].set_ylim(drawn.ymin, drawn.ymax)

    def create_plot(self):
        'create the plot'

        if not self.settings.plot_mode in [PlotSettings.PLOT_NONE]:
            
            self.fig, axes_list = plt.subplots(nrows=self.num_subplots, ncols=1, figsize=self.settings.plot_size, \
                                                    squeeze=False)

            self.axes_list = []
            
            for row in axes_list:
                for axes in row:
                    self.axes_list.append(axes)

            ha = self.core.hybrid_automaton

            if not isinstance(self.settings.label, list):
                self.settings.label = [self.settings.label]

            # only use title for the first subplot the first plot
            title = self.settings.label[0].title
            title = title if title is not None else ha.name
            self.axes_list[0].set_title(title, fontsize=self.settings.label[0].title_size)

            if self.settings.video_show_frame and self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
                ax = self.axes_list[0]
                self.frame_text = ax.text(0.01, 0.99, '', transform=ax.transAxes, verticalalignment='top')

            for i in range(self.num_subplots):
                labels = []
                label_settings = [self.settings.xdim_dir[i], self.settings.ydim_dir[i]]
                label_strings = [self.settings.label[i].x_label, self.settings.label[i].y_label]

                for label_setting, text in zip(label_settings, label_strings):
                    if text is not None:
                        labels.append(text)
                    elif label_setting is None:
                        labels.append('Time')
                    elif isinstance(label_setting, int):
                        labels.append('$x_{{ {} }}$'.format(label_setting))
                    else:
                        labels.append('')

                self.axes_list[i].set_xlabel(labels[0], fontsize=self.settings.label[i].label_size)
                self.axes_list[i].set_ylabel(labels[1], fontsize=self.settings.label[i].label_size)

                if self.settings.label[i].axes_limits is not None:
                    # hardcoded axes limits
                    xmin, xmax, ymin, ymax = self.settings.label[i].axes_limits

                    self.axes_list[i].set_xlim(xmin, xmax)
                    self.axes_list[i].set_ylim(ymin, ymax)

            if self.settings.grid:
                for axes in self.axes_list:
                    axes.grid(True, linestyle='dashed')

                    if self.settings.grid_xtics is not None:
                        axes.set_xticks(self.settings.grid_xtics)

                    if self.settings.grid_ytics is not None:
                        axes.set_yticks(self.settings.grid_ytics)

            # make the x and y axis animated in case of rescaling
            for i, axes in enumerate(self.axes_list):
                axes.xaxis.set_animated(True)
                axes.yaxis.set_animated(True)

                axes.tick_params(axis='both', which='major', labelsize=self.settings.label[i].tick_label_size)

            plt.tight_layout()

            self.shapes = [DrawnShapes(self, i) for i in range(self.num_subplots)]

    def plot_current_sim(self):
        '''
        plot the current simulation according to the plot settings.
        '''

        if self.core.sim_states:
            for subplot in range(self.num_subplots):
                xdim, ydim = self.settings.xdim_dir[subplot], self.settings.ydim_dir[subplot]
                plot_pts = []

                for obj in self.core.sim_states:
                    if obj is None:
                        plot_pts.append(None)
                    else:
                        _, pt, steps = obj
                        cur_time = steps * self.core.settings.step_size 
                        x, y = lpplot.pt_to_plot_xy(pt, xdim, ydim, cur_time)

                        plot_pts.append((x, y))

                self.shapes[subplot].set_cur_sim(plot_pts)

                if self.settings.label[subplot].axes_limits is None:
                    non_none_verts = [pt for pt in plot_pts if pt is not None]
                    self.update_axis_limits(non_none_verts, subplot)

    def commit_cur_sims(self):
        'commit the simulation points lines (finished reachability for a mode)'

        # setup result if it's not setup already
        if self.core.result.sim_lines is None:

            self.core.result.sim_lines = []

            for _ in self.shapes:
                self.core.result.sim_lines.append([])

        for i, shapes in enumerate(self.shapes):
            line = shapes.commit_cur_sims()

            self.core.result.sim_lines[i].append(line)

    def plot_current_state(self):
        '''
        plot the current StateSet according to the plot settings.
        '''

        state = self.core.aggdag.get_cur_state()

        for subplot in range(self.num_subplots):
            if self.settings.store_plot_result:
                verts = state.verts(self, subplot=subplot)

                self.core.result.plot_data.add_state(state, verts, subplot)

            if self.settings.plot_mode != PlotSettings.PLOT_NONE:
                verts = state.verts(self, subplot=subplot)
                verts_list = [verts]

                # if it's an aggregation, also add the predecessors to the plot
                if state.cur_step_in_mode == 0 and len(state.aggdag_op_list) > 1:
                    for op in state.aggdag_op_list:
                        if op is not None:
                            verts_list.append(op.poststate.verts(self, subplot=subplot))

                self.shapes[subplot].set_cur_state(verts_list)

                if self.settings.label[subplot].axes_limits is None:
                    self.update_axis_limits(verts, subplot)

        # finally, add a reachable poly for the current state
        if self.settings.plot_mode != PlotSettings.PLOT_NONE:
            self.add_reachable_poly(state)

    def add_reachable_poly(self, state):
        'add a reacahble poly to all subplots'

        for shape in self.shapes:
            shape.add_reachable_poly(state)

    def delete_plotted_state(self, stateset, step):
        'delete an already-plotted state'

        if self.settings.plot_mode != PlotSettings.PLOT_NONE:
            verts = stateset.del_plot_path(step)

            if verts is None:
                pass # this happens during normal operation (nested deaggregation)
                #print(f".#######plotutil plot was already deleted: {stateset}")
            else:
                # delete it from the result object
                if self.core.result.plot_data is not None:
                    self.core.result.plot_data.remove_state(stateset, step)

                # plot the verts of the deleted old plotted state
                self.highlight_states_gray([verts])

    def add_plotted_states(self, stateset_list):
        'add the passed-in statesets to be plotted'


        if self.settings.plot_mode != PlotSettings.PLOT_NONE:
            for state in stateset_list:
                self.add_reachable_poly(state)

                # add it to result as well
                if self.settings.store_plot_result:
                    for subplot in range(self.num_subplots):
                        verts = state.verts(self, subplot=subplot)
                        self.core.result.plot_data.add_state(state, verts, subplot)

            self.highlight_states(stateset_list)

    def highlight_states_gray(self, states):
        '''highlight the passed-in states (using gray line settings)

        states is a list of either StateSet objects or a list of verts for each subplot
        '''

        if self.settings.plot_mode != PlotSettings.PLOT_IMAGE:
            for subplot in range(self.num_subplots):
                verts_list = []

                for state in states:
                    if isinstance(state, list):
                        verts = state[subplot] # list of vertices
                        verts_list.append(verts)
                    else:
                        verts = state.verts(self, subplot=subplot)
                        verts_list.append(verts)

                    if self.settings.label[subplot].axes_limits is None:
                        self.update_axis_limits(verts, subplot)

                self.shapes[subplot].set_gray_state(verts_list)

    def highlight_states(self, states):
        '''highlight the passed-in states (using current_state settings)

        states is a list of either StateSet objects or a list of verts for each subplot
        '''

        for subplot in range(self.num_subplots):
            verts_list = []

            for state in states:
                if isinstance(state, list):
                    verts_list.append(state[subplot])
                else:
                    verts = state.verts(self, subplot=subplot)
                    verts_list.append(verts)

                if self.settings.label[subplot].axes_limits is None:
                    self.update_axis_limits(verts, subplot)

            self.shapes[subplot].set_cur_state(verts_list)

    def anim_func(self, frame):
        'animation draw function'

        if self.settings.video_show_frame and self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
            self.frame_text.set_text(f'Frame: {frame}')

        if self.interactive.paused and self.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            Timers.tic("paused")
            time.sleep(0.1)
            Timers.toc("paused")

            if self.settings.interactive_skip_count > 0:
                self.settings.interactive_skip_count -= 1 # modify the setting in-place... probably okay
                self.interactive.paused = False
                self.core.print_verbose(f"Unpausing plot. Skip count is now {self.settings.interactive_skip_count}.")
                
        elif self.interactive.paused and self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
            if self.pause_frames is None:
                self.pause_frames = self.settings.video_pause_frames
            else:
                self.pause_frames -= 1

                if self.pause_frames <= 0:
                    self.pause_frames = None
                    self.interactive.paused = False
        else:
            Timers.tic("frame")

            for ds in self.shapes:
                ds.set_cur_state(None)
                ds.set_gray_state(None)

            for _ in range(self.settings.draw_stride):
                self.core.do_step()

                # if we just wanted a single step (or do_step() caused paused to be set to True)
                if (self.interactive.step or self.interactive.paused) and \
                       (self.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE or
                        self.settings.plot_mode == PlotSettings.PLOT_VIDEO):
                    self.core.print_verbose(f"Paused due to interactive.step = {self.interactive.step}, " + \
                                            f"i.paused = {self.interactive.paused}")
                    self.pause()
                    break

            if self.core.aggdag.get_cur_state() is not None:
                self.plot_current_state()

            if self.core.sim_states is not None:
                self.plot_current_sim()

            Timers.toc("frame")

            if self.interactive.paused:
                self.core.print_normal("Paused After Frame #{}".format(self.num_frames_drawn))

            self.num_frames_drawn += 1

        # return a list of animated artists
        rv = []

        for axes in self.axes_list:
            rv.append(axes.xaxis)
            rv.append(axes.yaxis)

        for subplot in range(self.num_subplots):
            rv += self.shapes[subplot].get_artists()

        if self.settings.video_show_frame and self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
            rv.append(self.frame_text)

        return rv

    def anim_init_func(self):
        'animation init function'

        rv = []

        for axes in self.axes_list:
            rv.append(axes.xaxis)
            rv.append(axes.yaxis)

        for subplot in range(self.num_subplots):
            rv += self.shapes[subplot].get_artists()

        return rv

    def anim_iterator(self):
        'generator for the computation iterator'
        Timers.tic("anim_iterator")

        frame_counter = 0

        # do the computation until its done
        while not self.core.is_finished():
            if self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
                self.core.print_verbose("Saving Video Frame #{}".format(frame_counter))
                
            yield frame_counter
            frame_counter += 1

        if self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
            for _ in range(self.settings.video_extra_frames):
                frame_counter += 1
                yield frame_counter

        Timers.toc("anim_iterator")


    def compute_and_animate(self):
        'do the computation, plotting during the process'

        def next_pressed(_):
            'event function for next button press'
            self.interactive.paused = False

            if self.core.is_finished():
                self.core.print_normal("Computation is finished")

        def step_pressed(_):
            'event function for step button press'
            
            self.interactive.paused = False
            self.interactive.step = True

            if self.core.is_finished():
                self.core.print_normal("Computation is finished")

        def aggdag_pressed(_):
            'event function for save aggdag button press'
            
            filename = self.core.aggdag.save_viz()

            self.core.print_normal(f"Saved to filename: {filename}")

        def on_click(event):
            'on-click listener during interactive plots'

            x, y, ax = event.xdata, event.ydata, event.inaxes

            try:
                subplot = self.axes_list.index(ax)
            except ValueError:
                subplot = None

            if subplot is not None:
                # check which polygon you're in
                d = self.core.result.plot_data.get_plot_data(x, y, subplot)

                print(d[1:])

        if self.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            # do one frame
            self.interactive.paused = False
            self.interactive.step = True

            #plt.figure(2)

            # shrink plot, add buttons
            plt.subplots_adjust(bottom=0.12)

            axnext = plt.axes([0.81, 0.02, 0.1, 0.05])
            bnext = Button(axnext, 'Next', color='0.85', hovercolor='0.85')
            bnext.on_clicked(next_pressed)

            axstep = plt.axes([0.61, 0.02, 0.1, 0.05])
            bstep = Button(axstep, 'Step', color='0.85', hovercolor='0.85')
            bstep.on_clicked(step_pressed)

            axaggdag = plt.axes([0.31, 0.02, 0.15, 0.05])
            baggdag = Button(axaggdag, 'Save Aggdag', color='0.85', hovercolor='0.85')
            baggdag.on_clicked(aggdag_pressed)

            # add mouse-click listener
            cid = self.fig.canvas.mpl_connect('button_press_event', on_click)

        if self.settings.plot_mode == PlotSettings.PLOT_IMAGE:
            self.run_to_completion()
            self.save_image()
        else:
            interval = 1 if self.settings.plot_mode == PlotSettings.PLOT_VIDEO else 0

            self.num_frames_drawn = 0
            self._anim = animation.FuncAnimation(self.fig, self.anim_func, self.anim_iterator, \
                init_func=self.anim_init_func, interval=interval, blit=True, repeat=False, save_count=sys.maxsize)

            if self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
                self.save_video()
            else:
                plt.show()

    def run_to_completion(self, compute_plot=True):
        'run to completion, creating the plot at each step'

        Timers.tic("run_to_completion")

        while not self.core.is_finished():
            if compute_plot and self.shapes is not None:
                for subplot in range(self.num_subplots):
                    self.shapes[subplot].set_cur_state(None)

            self.core.do_step()

            if compute_plot and self.core.aggdag.get_cur_state():
                self.plot_current_state()

            if self.core.sim_states is not None:
                self.plot_current_sim()

        Timers.toc("run_to_completion")

    def save_video(self):
        'save a video file'

        writer = self.settings.make_video_writer_func()

        filename = self.settings.filename if self.settings.filename is not None else "anim.mp4"

        self._anim.save(filename, writer=writer)

        if not self.core.is_finished():
            raise RuntimeError("saving video exited before computation completed (is_finished() returned false)")

    def save_image(self):
        'save an image file'

        filename = self.settings.filename

        if filename is None:
            filename = "plot.png"

        plt.savefig(filename, bbox_inches='tight')

# monkey patch function for blitting tick-labels
# see http://stackoverflow.com/questions/17558096/animated-title-in-matplotlib
def _blit_draw(_self, artists, bg_cache):
    'money-patch version of animation._blit_draw'
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

animation.Animation._blit_draw = _blit_draw # pylint: disable=protected-access
