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

        self.result = None # a HylaaResult... assigned on run()

    def is_finished(self):
        'is the computation finished'

        return self.cur_star is None or not self.result.safe

    def reconstruct_full_start_pt(self, lp_solution):
        '''
        Reconstruct a full-dimensional start point from an lp solution in the current star.

        For use when settings.simulation.seperate_constant_vars == True in order to create the counter-example trace.
        '''

        star = self.cur_star
        fixed_tuples = star.fixed_tuples
        dims = star.dims

        start_pt = []
        fixed_index = 0
        var_index = 0

        for dim in xrange(dims):
            if fixed_index < len(fixed_tuples) and fixed_tuples[fixed_index][0] == dim:
                # fixed dimension
                start_pt.append(fixed_tuples[fixed_index][1])
                fixed_index += 1
            else:
                # variable dim, extract from lp solution
                start_pt.append(lp_solution[var_index])
                var_index += 1

        return np.array(start_pt, dtype=float)

    def check_guards(self):
        '''check for discrete successors with the guards'''

        if self.cur_star.time_elapse.next_step == 1 and self.settings.print_output:
            print "Solving first step guard LP..."

        for i in xrange(len(self.cur_star.mode.transitions)):
            lp_solution = self.cur_star.get_guard_intersection(i)

            if lp_solution is not None:
                if self.settings.print_lp_on_error:
                    # print the LP solution and exit
                    lpi = self.cur_star.get_guard_lpi(i)
                    lpi.print_lp()
                elif self.settings.counter_example_filename is not None:
                    # print out the counter-example trace to a counter-example file

                    filename = self.settings.counter_example_filename
                    star = self.cur_star
                    mode = star.mode
                    step_size = self.settings.step
                    total_steps = star.time_elapse.next_step - 1
                    start_pt = lp_solution[:star.lp_dims]

                    #print ".engine start point from lp = {}".format(start_pt)

                    if self.cur_star.var_lists is not None:
                        # reconstruct start_pt based on the fixed and non-fixed dims
                        start_pt = self.reconstruct_full_start_pt(start_pt)

                    #print ".engine reconstructed start point = {}".format(start_pt)

                    norm_vec_sparse = self.cur_star.mode.transitions[i].guard_matrix[0]
                    normal_vec = np.array(norm_vec_sparse.toarray(), dtype=float)
                    normal_vec.shape = (self.cur_star.dims,)
                    normal_val = self.cur_star.mode.transitions[i].guard_rhs[0]

                    end_val = lp_solution[self.cur_star.lp_dims]

                    num_constraints = len(self.cur_star.mode.transitions[i].guard_rhs)
                    input_vals = lp_solution[self.cur_star.lp_dims + num_constraints:]

                    if self.settings.print_output:
                        print 'Writing counter-example trace file: "{}"'.format(filename)

                    # construct inputs, which are in backwards order
                    inputs = []

                    for step in xrange(total_steps):
                        offset = len(input_vals) - (self.cur_star.inputs * (1 + step))
                        inputs.append(input_vals[offset:offset+self.cur_star.inputs])

                    write_counter_example(filename, mode, step_size, total_steps, start_pt, inputs,
                                          normal_vec, normal_val, end_val)

                self.result.safe = False
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
            if not self.result.safe:
                print "Result: Error modes are reachable.\n"
            else:
                print "Result: System is safe. Error modes are NOT reachable.\n"

    def run(self, init_star):
        '''
        Run the computation (main entry point)

        init_star is the initial state

        fixed_dim_list, if used, is a list of dimensions with fixed initial values
        '''

        assert isinstance(init_star, Star), "initial states should be a Star object"
        Timers.reset()

        self.result = HylaaResult()
        self.plotman.create_plot()

        self.cur_star = init_star

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.plotman.run_to_completion(self.do_step, self.is_finished, compute_plot=False)
        else:
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

        # assign results
        self.result.timers = Timers.timers
        self.result.krylov_stats = init_star.time_elapse.stats
