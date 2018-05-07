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
from hylaa.settings import HylaaSettings, PlotSettings, TimeElapseSettings
from hylaa.file_io import write_counter_example
from hylaa.util import Freezable

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
                step_num = self.cur_star.time_elapse.next_step - 1
                print "Unsafe at Step: {} / {} ({})".format(step_num, self.settings.num_steps, \
                                                            self.settings.step * step_num)

                self.result.init_vars = lp_solution[:self.cur_star.num_init_vars]

                end_output_lp_col = self.cur_star.num_init_vars + self.cur_star.mode.output_space_csr.shape[0]
                self.result.output_vars = lp_solution[self.cur_star.num_init_vars:end_output_lp_col]

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

                    output_space = self.cur_star.mode.output_space_csr
                    guard_mat = self.cur_star.mode.transitions[i].guard_matrix_csr

                    first_constraint = (guard_mat[0] * output_space).toarray()
                    first_constraint.shape = (first_constraint.shape[1],)

                    init_space_csc = self.cur_star.init_space_csc
                    guard_threshold = self.cur_star.mode.transitions[i].guard_rhs[0]


                    # multiply this by the output constraint matrix...
                    first_output_val = guard_mat[0] * self.result.output_vars

                    # construct inputs, which are in backwards order
                    inputs = []

                    if self.cur_star.inputs > 0:
                        # skip total input effects
                        input_start_col = end_output_lp_col + 1

                        input_vals = lp_solution[input_start_col:]

                        for step in xrange(total_steps):
                            offset = len(input_vals) - (self.cur_star.inputs * (1 + step))
                            inputs.append(input_vals[offset:offset+self.cur_star.inputs])

                    if self.settings.print_output:
                        print 'Writing counter-example trace file: "{}"'.format(filename)

                    write_counter_example(filename, mode, step_size, total_steps, self.result.init_vars, \
                        init_space_csc, inputs, first_constraint, guard_threshold, first_output_val)

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
        np.set_printoptions(suppress=True) # suppress floating point printing

        self.result = HylaaResult()
        self.plotman.create_plot()

        self.cur_star = init_star

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.plotman.run_to_completion(self.do_step, self.is_finished, compute_plot=False)
        else:
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

        # assign results
        self.result.top_level_timer = Timers.top_level_timer
        Timers.reset()

        if self.settings.time_elapse.method == TimeElapseSettings.KRYLOV:
            self.result.krylov_stats = init_star.time_elapse.time_elapse_obj.stats

        if self.plotman.reach_poly_data is not None:
            self.result.reachable_poly_data = self.plotman.reach_poly_data

class HylaaResult(Freezable):
    'Result, assigned to engine.result after computation'

    def __init__(self):
        self.top_level_timer = None # TimerData for total time
        self.safe = True # was the verification result safe?

        self.krylov_stats = None # krylov statistics, map of string -> value, copy of TimerElapse.stats
        self.reachable_poly_data = None # set to the vertices of the reachble plot (if plot mode is GNUPLOT or MATLAB)

        self.init_vars = None # counterexample: assigned when unsafe state is reachable
        self.output_vars = None # counterexample: assigned when unsafe state is reachable

        self.freeze_attrs()
