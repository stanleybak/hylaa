'''
Main Hylaa Reachability Implementation
Stanley Bak
Aug 2016
'''

import numpy as np

from hylaa.plotutil import PlotManager
from hylaa.star import Star
from hylaa.hybrid_automaton import LinearHybridAutomaton
from hylaa.timerutil import Timers
from hylaa.containers import HylaaSettings, PlotSettings, HylaaResult
from hylaa.file_io import write_counter_example

class HylaaEngine(object):
    'main computation object. initialize and call run()'

    def __init__(self, ha, hylaa_settings):
        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(ha, LinearHybridAutomaton)

        self.hybrid_automaton = ha
        self.settings = hylaa_settings

        if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            Star.init_plot_vecs(self.settings.plot)

        self.plotman = PlotManager(self, self.settings.plot)

        # computation
        self.cur_star = None # a Star object

        self.cur_step_in_mode = None # how much dwell time in current continuous post
        self.max_steps_remaining = None # bound on num steps left in current mode ; assigned on pop

        self.reached_error = False
        self.result = None # a HylaaResult... assigned on run()

    def is_finished(self):
        'is the computation finished'

        return self.cur_star is None or self.reached_error

    def check_guards(self):
        '''check for discrete successors with the guards'''

        for i in xrange(len(self.cur_star.mode.transitions)):
            lp_solution = self.cur_star.get_guard_intersection(i)

            if lp_solution is not None:
                if self.settings.counter_example_filename is not None:
                    # print out the counter-example trace

                    filename = self.settings.counter_example_filename
                    mode = self.cur_star.mode
                    step_size = self.settings.step
                    total_steps = self.cur_star.time_elapse.next_step - 1
                    start_pt = lp_solution[:self.cur_star.dims]
                    norm_vec_sparse = self.cur_star.mode.transitions[i].guard_matrix[0]
                    normal_vec = np.array(norm_vec_sparse.toarray(), dtype=float)
                    normal_vec.shape = shape=(self.cur_star.dims,)
                    normal_val = self.cur_star.mode.transitions[i].guard_rhs[0]
                    end_val = lp_solution[self.cur_star.dims]

                    if self.settings.print_output:
                        print 'Writing counter-example trace file: "{}"'.format(filename)

                    write_counter_example(filename, mode, step_size, total_steps, start_pt,
                                          normal_vec, normal_val, end_val)

                self.reached_error = True
                break # no need to keep checking

    def do_step_continuous_post(self):
        '''do a step where it's part of a continuous post'''

        # advance time by one step
        if self.cur_star.time_elapse.next_step > self.settings.num_steps:
            self.cur_star = None
        else:
            if self.settings.print_output and not self.settings.skip_step_times:
                step_num = self.cur_star.time_elapse.next_step
                print "Step: {} / {} ({})".format(step_num, self.settings.num_steps, self.settings.step * step_num)

            self.cur_star.step()
            self.check_guards()

    def do_step(self):
        'do a single step of the computation'

        skipped_plot = False # if we skip the plot, do multiple steps

        while True:
            self.do_step_continuous_post()

            if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE or \
                                    not skipped_plot or self.is_finished():
                break

        if self.is_finished() and self.settings.print_output:
            if self.reached_error:
                print "Result: Error modes are reachable.\n"
            else:
                print "Result: Error modes are NOT reachable.\n"

    def run(self, init_star):
        '''
        run the computation
        '''

        assert isinstance(init_star, Star), "initial states should be a Star object"

        print "Run called..."

        self.result = HylaaResult()
        self.plotman.create_plot()

        self.cur_star = init_star

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.plotman.run_to_completion(self.do_step, self.is_finished, compute_plot=False)
        else:
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

        self.result.time = Timers.timers["total"].total_secs
