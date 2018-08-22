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

class InteractiveState(Freezable):
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

    def __init__(self, plotman):
        self.plotman = plotman
        self.axes = plotman.axes
        self.mode_colors = plotman.mode_colors

        # parent is a string
        # for modes, prefix is 'mode_'
        # for cur state, parent is 'cur_state'
        self.parent_to_polys = OrderedDict()
        self.parent_to_markers = OrderedDict()

        self.freeze_attrs()

    def get_artists(self):
        'get the list of artists, to be returned by animate function'

        rv = []

        # make sure cur_state is last
        
        for name, polys in self.parent_to_polys.items():
            if name != "cur_state":
                rv.append(polys)

        if "cur_state" in self.parent_to_polys:
            rv.append(self.parent_to_polys.get('cur_state'))

        for markers in self.parent_to_markers.values():
            rv.append(markers)

        return rv

    def set_cur_state(self, verts):
        'set the currently tracked set of states for one frame'

        polys = self.parent_to_polys.get('cur_state')
        
        if polys is None:
            lw = self.plotman.settings.reachable_poly_width
            polys = collections.PolyCollection([], lw=lw, animated=True, edgecolor='k', facecolor=(0., 0., 0., 0.))
            self.axes.add_collection(polys)

            self.parent_to_polys['cur_state'] = polys

        paths = polys.get_paths()

        if polys is not None and paths:
            paths.pop() # remove the old polygon

        if verts is not None:
            # create a new polygon
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
            paths.append(Path(verts, codes))

    def add_reachable_poly(self, poly_verts, mode_name):
        '''add a polygon which was reachable'''

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

class PlotManager(Freezable):
    'manager object for plotting during or after computation'

    def __init__(self, hylaa_core):
        # matplotlib default rcParams caused incorrect trace output due to interpolation
        rcParams['path.simplify'] = False

        self.core = hylaa_core
        self.settings = hylaa_core.settings.plot

        self.fig = None
        self.axes = None
        self.actual_limits = None # AxisLimits object
        self.drawn_limits = None # AxisLimits object

        self.mode_colors = ModeColors()
        self.shapes = None # instance of DrawnShapes
        self.interactive = InteractiveState()

        self.drew_first_frame = False # one-time flag
        self._anim = None # animation object

        self.plot_vecs = []
        self.init_plot_vecs()

        self.freeze_attrs()

    def init_plot_vecs(self):
        'initialize plot_vecs'

        assert not (self.settings.xdim_dir is None and self.settings.ydim_dir is None)

        if self.settings.xdim_dir is None:
            self.plot_vecs.append(np.array([0, 1.], dtype=float))
            self.plot_vecs.append(np.array([0, -1.], dtype=float))
        elif self.settings.ydim_dir is None:
            self.plot_vecs.append(np.array([1., 0], dtype=float))
            self.plot_vecs.append(np.array([-1., 0], dtype=float))
        else:
            assert self.settings.num_angles >= 3, "needed at least 3 directions in plot_settings.num_angles"

            self.plot_vecs = lpplot.make_plot_vecs(self.settings.num_angles)

    def state_popped(self):
        'a state was popped off the waiting list'

        if self.settings.plot_mode != PlotSettings.PLOT_NONE:
            self.shapes.set_cur_state(None)

    def update_axis_limits(self, points_list):
        'update the axes limits to include the passed-in point list'

        first_draw = False

        if self.drawn_limits is None:
            first_draw = True
            first_x, first_y = points_list[0]
            self.actual_limits = AxisLimits()
            self.drawn_limits = AxisLimits()
            self.actual_limits.xmin = self.actual_limits.xmax = first_x
            self.actual_limits.ymin = self.actual_limits.ymax = first_y
            self.drawn_limits.xmin = self.drawn_limits.xmax = first_x
            self.drawn_limits.ymin = self.drawn_limits.ymax = first_y

        lim = self.actual_limits
        drawn = self.drawn_limits

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
                self.axes.set_xlim(drawn.xmin - 1e-1, drawn.xmax + 1e-1)
            else:
                self.axes.set_xlim(drawn.xmin, drawn.xmax)

            if drawn.ymin == drawn.ymax:
                self.axes.set_ylim(drawn.ymin - 1e-1, drawn.ymax + 1e-1)
            else:
                self.axes.set_ylim(drawn.ymin, drawn.ymax)

    def create_plot(self):
        'create the plot'

        if not self.settings.plot_mode in [PlotSettings.PLOT_NONE]:
            self.fig, self.axes = plt.subplots(nrows=1, figsize=self.settings.plot_size)
            ha = self.core.hybrid_automaton

            title = self.settings.label.title
            title = title if title is not None else ha.name

            labels = []
            label_settings = [self.settings.xdim_dir, self.settings.ydim_dir]
            label_strings = [self.settings.label.x_label, self.settings.label.y_label]

            for label_setting, text in zip(label_settings, label_strings):
                if text is not None:
                    labels.append(text)
                elif label_setting is None:
                    labels.append('Time')
                elif isinstance(label_setting, int):
                    labels.append('$x_{{ {} }}$'.format(label_setting))
                else:
                    labels.append('')

            self.axes.set_xlabel(labels[0], fontsize=self.settings.label.label_size)
            self.axes.set_ylabel(labels[1], fontsize=self.settings.label.label_size)
            self.axes.set_title(title, fontsize=self.settings.label.title_size)

            if self.settings.label.axes_limits is not None:
                # hardcoded axes limits
                xmin, xmax, ymin, ymax = self.settings.label.axes_limits

                self.axes.set_xlim(xmin, xmax)
                self.axes.set_ylim(ymin, ymax)

            if self.settings.grid:
                self.axes.grid(True, linestyle='dashed')

                if self.settings.grid_xtics is not None:
                    self.axes.set_xticks(self.settings.grid_xtics)

                if self.settings.grid_ytics is not None:
                    self.axes.set_xticks(self.settings.grid_ytics)

            # make the x and y axis animated in case of rescaling
            self.axes.xaxis.set_animated(True)
            self.axes.yaxis.set_animated(True)

            plt.tick_params(axis='both', which='major', labelsize=self.settings.label.tick_label_size)
            plt.tight_layout()

            self.shapes = DrawnShapes(self)

    def plot_current_state(self, state):
        '''
        plot the current SymbolicState according to the plot settings. returns still_feasible
        '''

        rv = True

        if self.settings.plot_mode != PlotSettings.PLOT_NONE or self.settings.store_plot_result:

            Timers.tic('verts()')
            verts = state.verts(self)
            Timers.toc('verts()')

            if self.settings.store_plot_result:
                if state.mode.name in self.core.result.mode_to_polys:
                    self.core.result.mode_to_polys[state.mode.name].append(verts)
                else:
                    self.core.result.mode_to_polys[state.mode.name] = [verts]

            if self.settings.plot_mode != PlotSettings.PLOT_NONE:
                Timers.tic("add to plot")
                self.shapes.set_cur_state(verts)

                if self.settings.label.axes_limits is None:
                    self.update_axis_limits(verts)

                self.shapes.add_reachable_poly(verts, state.mode.name)

                Timers.toc("add to plot")

        return rv

    def anim_func(self, force_single_frame):
        'animation draw function'

        if not force_single_frame and self.interactive.paused:
            Timers.tic("paused")
            time.sleep(0.1)
            Timers.toc("paused")
        else:
            Timers.tic("frame")

            self.shapes.set_cur_state(None)
            self.core.do_step()

            if self.core.cur_state is not None:
                if not self.plot_current_state(self.core.cur_state):
                    self.core.print_verbose("Continuous state discovered to be UNSAT during plot, removing state")
                    self.core.cur_state = None

            # if we just wanted a single step
            if self.interactive.step:
                self.interactive.step = False
                self.interactive.paused = True

            Timers.toc("frame")

            if self.interactive.paused and not force_single_frame:
                frame_timer = Timers.top_level_timer.get_children_recursive('frame')[0]
                self.core.print_normal("Paused After Frame #{}".format(frame_timer.num_calls))

        return [self.axes.xaxis, self.axes.yaxis] + self.shapes.get_artists()

    def anim_init_func(self):
        'animation init function'

        return [self.axes.xaxis, self.axes.yaxis] + self.shapes.get_artists()

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
                self.shapes.set_cur_state(None)

            self.core.do_step()

            if compute_plot and self.core.cur_state is not None:
                if not self.plot_current_state(self.core.cur_state):
                    self.core.print_verbose("Continuous state discovered to be UNSAT during plot, removing state")
                    self.core.cur_state = None

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
