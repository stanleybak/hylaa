'''
Stanley Bak
June 2017
Mass-Spring Simulation
'''

import random
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import colors
from matplotlib.widgets import Button, Slider

from scipy.linalg import expm
import numpy as np

class PlotStatus(object):
    'Container for the plot status'

    def __init__(self):
        self.max_time_axes = None
        self.num_mass_axis = None
        self.default_max_time = 20.0

        self.num_masses = None
        self.a_matrix = None
        self.history = None
        self.history_times = None
        self.max_time = None
        self.set_num_masses(1) # sets the above block of variables

        self.paused = True
        self.time = 0.0

        self.speed = 10 # steps per second (lower = faster simulation)

    def restart(self):
        'restart the animation'

        num = self.num_masses
        self.num_masses = None

        self.set_num_masses(num)

    def set_num_masses(self, num):
        'set the number of masses'

        if num != self.num_masses:
            self.num_masses = num
            self.paused = True
            self.time = 0.0
            self.history_times = [0.0]

            state = np.array([0] * 2 * num, dtype=float)
            state[1] = 0.8 # initial velocity

            self.history = [state]
            self.a_matrix = make_a_matrix(2 * num)

            if self.num_mass_axis is not None:
                self.num_mass_axis.axis([-1, num, -0.75, 0.75])

            self.update_max_time(self.default_max_time) # will call plt.draw()

    def set_num_mass_axis(self, axis):
        'set the matplotlib axis that is updated whenever num_masses changes'

        self.num_mass_axis = axis

    def set_time_dependent_axes(self, axes_list):
        'set the axes to be updated whenever max_time is changed'

        self.max_time_axes = axes_list

    def update_max_time(self, t):
        'update the max time on the axis'

        self.max_time = t

        if self.max_time_axes is not None:
            for a in self.max_time_axes:
                a.axis([0, t, -1, 1])

        plt.draw() # redraw everything (slow)

    def step(self):
        'advance the states by one time step if not paused'

        if not self.paused:
            delta_t = self.speed / 1000.0
            self.time += delta_t

            start = self.history[0]

            new_state = np.dot(expm(self.a_matrix * self.time), start)

            self.history.append(new_state)
            self.history_times.append(self.time)

            if self.time > self.max_time:
                self.update_max_time(self.max_time * 2.0)

def update_spring(pts, startx, endx):
    'update the list of points for a single spring in the animation'

    xs = []
    ys = []

    num_tips = 6
    difx = endx - startx

    stepx = difx / (num_tips + 1)
    is_top = True

    xs.append(startx)
    ys.append(0)

    for tip in xrange(num_tips):
        xpos = startx + (tip + 1) * stepx
        ypos = 0.3 if is_top else -0.3
        is_top = not is_top

        xs.append(xpos)
        ys.append(ypos)

    xs.append(endx)
    ys.append(0)

    pts.set_data(xs, ys)

def make_a_matrix(num_dims):
    '''get the A matrix corresponding to the dynamics'''

    a = np.zeros((num_dims, num_dims))

    for d in xrange(num_dims / 2):
        a[2*d][2*d+1] = 1 # pos' = vel

        a[2*d+1][2*d] = -2 # cur-state

        if d > 0:
            a[2*d+1][2*d-2] = 1 # prev-state

        if d < num_dims / 2 - 1:
            a[2*d+1][2*d+2] = 1 # next state

    return a

def get_colors():
    'get a list of colors'

    all_colors = []

    # remove any colors with 'white' or 'yellow in the name
    skip_colors_substrings = ['white', 'yellow']
    skip_colors_exact = ['black']

    for col in colors.cnames:
        skip = False

        for col_substring in skip_colors_substrings:
            if col_substring in col:
                skip = True
                break

        if not skip and not col in skip_colors_exact:
            all_colors.append(col)

    # we'll re-add these later; remove them before shuffling
    first_colors = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta', 'lime']

    for col in first_colors:
        all_colors.remove(col)

    # deterministic shuffle of all remaining colors
    random.seed(0)
    random.shuffle(all_colors)

    # prepend first_colors so they get used first
    all_colors = first_colors + all_colors

    return all_colors

def main():
    'main entry point'

    # make the figure before creating PlotState (otherwise draw() creates a new figure)
    fig = plt.figure(figsize=(16, 10))

    status = PlotStatus()
    max_num_masses = 10

    labelsize = 22
    all_colors = get_colors()

    ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    ax.grid()
    ax.axis([-1, 1, -1, 1])
    ax.set_xlabel('$x_0$', fontsize=labelsize)
    ax.set_ylabel('$v_0$', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    points, = ax.plot([], [], '-', color=all_colors[0])
    cur_point, = ax.plot([], [], 'o', color=all_colors[0])

    annotation = ax.annotate("Time: 0", xy=(0.05, 0.93), xycoords="axes fraction", fontsize=labelsize)
    annotation.set_animated(True)

    #####
    ax_pos = plt.subplot2grid((3, 2), (0, 1))
    ax_pos.grid()
    ax_pos.axis([0, status.max_time, -1, 1])
    plt.setp(ax_pos.get_xticklabels(), visible=False)
    ax_pos.set_ylabel('Position (offset)', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    ax_vel = plt.subplot2grid((3, 2), (1, 1))
    ax_vel.grid()
    ax_vel.axis([0, status.max_time, -1, 1])
    ax_vel.set_xlabel('Time', fontsize=labelsize)
    ax_vel.set_ylabel('Velocity', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    ax_pos_points = []
    ax_vel_points = []

    status.set_time_dependent_axes([ax_pos, ax_vel])

    for m in xrange(max_num_masses):
        pts, = ax_pos.plot([], [], '-', color=all_colors[m % len(all_colors)])
        ax_pos_points.append(pts)

        pts, = ax_vel.plot([], [], '-', color=all_colors[m % len(all_colors)])
        ax_vel_points.append(pts)

    #####
    ax_im = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax_im.grid()
    ax_im.axis([-1, status.num_masses, -0.75, 0.75])

    status.set_num_mass_axis(ax_im)
    plt.setp(ax_im.get_yticklabels(), visible=False)
    ax_im.set_xlabel('Position', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    patch_list = []
    mass_width = 0.3

    # masses
    for m in xrange(max_num_masses):
        patch = patches.Rectangle((0, 0), mass_width, 1.0, fc=all_colors[m % len(all_colors)], ec='black')
        patch_list.append(patch)
        ax_im.add_patch(patch)

    # springs
    spring_list = []

    for _ in xrange(max_num_masses + 1):
        pts, = ax_im.plot([], [], '-', color='black')

        spring_list.append(pts)

    def init():
        """initialize animation"""

        points.set_data([], [])
        cur_point.set_data([], [])
        annotation.set_text("")

        for pts in ax_pos_points:
            pts.set_data([], [])

        for pts in ax_vel_points:
            pts.set_data([], [])

        for patch in patch_list:
            patch.set_visible(False)

        for spring in spring_list:
            spring.set_data([], [])

        return ax_pos_points + ax_vel_points + [points, cur_point, annotation] + patch_list + spring_list

    def animate(_):
        """perform animation step"""

        status.step()

        annotation.set_text("Time: {:.2f}".format(status.time))

        xs = []
        ys = []

        for state in status.history:
            xs.append(state[0])
            ys.append(state[1])

        points.set_data(xs, ys)

        xs = []
        ys = []

        xs.append(status.history[-1][0])
        ys.append(status.history[-1][1])

        cur_point.set_data(xs, ys)

        ###########

        num_plot = status.num_masses if status.num_masses <= 3 else 1

        for m in xrange(max_num_masses):
            xs_pos = []
            ys_pos = []

            xs_vel = []
            ys_vel = []

            if m == 0 or (m < status.num_masses and status.num_masses <= 3):
                for i in xrange(len(status.history)):
                    state = status.history[i]
                    t = status.history_times[i]

                    xs_pos.append(t)
                    ys_pos.append(state[2*m])

                    xs_vel.append(t)
                    ys_vel.append(state[2*m + 1])

            ax_pos_points[m].set_data(xs_pos, ys_pos)
            ax_vel_points[m].set_data(xs_vel, ys_vel)

        ##### set patches
        prev_xpos = -1

        for m in xrange(max_num_masses):
            patch = patch_list[m]

            if m >= status.num_masses:
                patch.set_visible(False)
            else:
                patch.set_visible(True)

                xpos = m + status.history[-1][2*m]
                patch.set_xy([xpos - mass_width/2.0, -0.5])

                update_spring(spring_list[m], prev_xpos, xpos - mass_width/2.0)

                prev_xpos = xpos + mass_width / 2.0

        # update final spring
        update_spring(spring_list[status.num_masses], prev_xpos, status.num_masses)

        return ax_pos_points + ax_vel_points + [points, cur_point, annotation] + patch_list + spring_list

    def start_stop_pressed(_):
        'button event function'
        status.paused = not status.paused

    def restart_pressed(_):
        'button event function'

        status.restart()

    def update_speed(val):
        'slider moved event function'
        status.speed = val

    def update_masses(val):
        'slider moved event function'
        status.set_num_masses(int(round(val)))

    # shrink plot, add buttons
    plt.tight_layout()

    plt.subplots_adjust(bottom=0.15)

    axstart = plt.axes([0.8, 0.02, 0.15, 0.05])
    start_button = Button(axstart, 'Start / Stop', color='0.85', hovercolor='0.85')
    start_button.on_clicked(start_stop_pressed)
    start_button.label.set_fontsize(labelsize)

    axrestart = plt.axes([0.05, 0.02, 0.15, 0.05])
    restart_button = Button(axrestart, 'Restart', color='0.85', hovercolor='0.85')
    restart_button.on_clicked(restart_pressed)
    restart_button.label.set_fontsize(labelsize)

    axspeed = plt.axes([0.25, 0.02, 0.15, 0.05])
    sspeed = Slider(axspeed, 'Speed', 10.0, 100.0, valinit=10.0)
    sspeed.on_changed(update_speed)

    axmasses = plt.axes([0.6, 0.02, 0.15, 0.05])
    smasses = Slider(axmasses, 'Num Masses', 1, 10, valinit=1, valfmt='%0.0f')
    smasses.on_changed(update_masses)

    anim = animation.FuncAnimation(fig, animate, interval=1, blit=True, init_func=init)

    plt.show()
    #anim.save('heat.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # print "Done"

if __name__ == "__main__":
    main()
