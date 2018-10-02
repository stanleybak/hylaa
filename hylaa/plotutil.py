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

    def __init__(self):
        self.init_colors()

        self.mode_to_color = {} # map mode name -> color string
        self.freeze_attrs()

    @staticmethod
    def lighter(rgb_col):
        'return a lighter variant of an rgb color'

        return [(2.0 + val) / 3.0 for val in rgb_col]

    @staticmethod
    def darker(rgb_col):
        'return a darker variant of an rgb color'

        return [val / 1.2 for val in rgb_col]

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
        random.seed(0)
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

        # parent is a string
        # for modes, prefix is 'mode_'
        # for cur state, parent is 'cur_state'
        self.parent_to_polys = OrderedDict()
        self.parent_to_markers = OrderedDict()

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
        
        for name, polys in self.parent_to_polys.items():
            if name != "cur_state":
                rv.append(polys)

        if "cur_state" in self.parent_to_polys:
            rv.append(self.parent_to_polys.get('cur_state'))

        for markers in self.parent_to_markers.values():
            rv.append(markers)

        for collection in self.extra_collection_list:
            rv.append(collection)

        return rv

    def set_cur_state(self, verts_list):
        'set the currently tracked set of states for one frame'

        assert verts_list is None or isinstance(verts_list, list)

        polys = self.parent_to_polys.get('cur_state')
        
        if polys is None:
            # setup for first time drawing cur_state
            lw = self.plotman.settings.reachable_poly_width
            polys = collections.PolyCollection([], lw=lw, animated=True, edgecolor='k', facecolor=(0., 0., 0., 0.))
            self.axes.add_collection(polys)

            self.parent_to_polys['cur_state'] = polys

        paths = polys.get_paths()

        while polys is not None and paths:
            paths.pop() # remove the old polygon(s)

        if verts_list is not None:
            for verts in verts_list:
                # create a new polygon
                codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
                paths.append(Path(verts, codes))

    def add_reachable_poly(self, poly_verts, stateset, subplot, num_subplots):
        '''add a polygon which was reachable'''

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
            if stateset.plot_paths is None:
                stateset.plot_paths = [None] * num_subplots
                stateset.plot_paths_indices = [None] * num_subplots
                
            if stateset.plot_paths[subplot] is None:
                stateset.plot_paths[subplot] = paths
                stateset.plot_paths_indices[subplot] = []

            stateset.plot_paths_indices[subplot].append(len(paths) - 1)

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

        self.mode_colors = ModeColors()
        self.shapes = None # instance of DrawnShapes
        self.interactive = InteractiveState()

        self._anim = None # animation object
        self.num_frames_drawn = 0

        self.plot_vec_list = [] # a list of plot_vecs for each subplot
        self.num_subplots = None
        self.init_plot_vecs()

        self.freeze_attrs()

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
        self.core.print_verbose("Num subplots = {}".format(self.num_subplots))

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

    def plot_current_state(self):
        '''
        plot the current StateSet according to the plot settings.
        '''

        state = self.core.aggdag.get_cur_state()

        for subplot in range(self.num_subplots):
            if self.settings.store_plot_result and subplot == 0: # only subplot 0 is saved
                verts = state.verts(self, subplot=subplot)

                self.core.result.mode_to_polys[state.mode.name].append(verts)

            if self.settings.plot_mode != PlotSettings.PLOT_NONE:
                Timers.tic("add to plot")

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

                self.shapes[subplot].add_reachable_poly(verts, state, subplot, self.num_subplots)

                Timers.toc("add to plot")

    def anim_func(self, _):
        'animation draw function'

        if self.interactive.paused:
            Timers.tic("paused")
            time.sleep(0.1)
            Timers.toc("paused")
        else:
            Timers.tic("frame")

            for subplot in range(self.num_subplots):
                self.shapes[subplot].set_cur_state(None)

            for _ in range(self.settings.draw_stride):
                self.core.do_step()

                # if we just wanted a single step (or do_step() caused paused to be set to True)
                if self.interactive.step or self.interactive.paused:
                    self.interactive.step = False
                    self.interactive.paused = True
                    break

            if self.core.aggdag.get_cur_state() is not None:
                self.plot_current_state()

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
