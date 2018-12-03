'''
Hylaa Simuation Utilities
'''

import matplotlib.pyplot as plt

from hylaa.util import Freezable
from hylaa.settings import PlotSettings


class Simulation(Freezable):
    'main simulation container class. Initialize and call run()'

    def __init__(self, ha, settings, init, num_sims):
        self.ha = ha # pylint: disable=invalid-name
        self.settings = settings
        self.init = init
        
        self.num_sims = num_sims

        self.fig = None
        self.axes_list = None
        self.num_subplots = None
        self.frame_text = None

        self.freeze_attrs()

    def create_plot(self):
        'create the plot'

        if isinstance(self.settings.xdim_dir, list):
            assert isinstance(self.settings.ydim_dir, list)
            assert len(self.settings.xdim_dir) == len(self.settings.ydim_dir)
        else:
            self.settings.xdim_dir = [self.settings.xdim_dir]
            self.settings.ydim_dir = [self.settings.ydim_dir]

        self.num_subplots = len(self.settings.xdim_dir)

        if not self.settings.plot_mode in [PlotSettings.PLOT_NONE]:
            self.fig, axes_list = plt.subplots(nrows=self.num_subplots, ncols=1, figsize=self.settings.plot_size, \
                                                    squeeze=False)

            self.axes_list = []
            
            for row in axes_list:
                for axes in row:
                    self.axes_list.append(axes)

            if not isinstance(self.settings.label, list):
                self.settings.label = [self.settings.label]

            # only use title for the first subplot the first plot
            title = self.settings.label[0].title
            title = title if title is not None else self.ha.name
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

    def run(self, box):
        'run simulations from the given initial box'

        self.create_plot()
