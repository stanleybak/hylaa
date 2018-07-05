'''
Main Hylaa Reachability Implementation
Stanley Bak, 2018
'''

from hylaa.settings import HylaaSettings, PlotSettings

from hylaa.plotutil import PlotManager
from hylaa.stateset import StateSet
from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.timerutil import Timers

from hylaa.util import Freezable

class Core(Freezable):
    'main computation object. initialize and call run()'

    def __init__(self, ha, hylaa_settings):
        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(ha, HybridAutomaton)

        self.hybrid_automaton = ha
        self.settings = hylaa_settings

        self.plotman = PlotManager(self)

        # computation
        self.waiting_list = None # list of State Set objects
        
        self.cur_state = None # a StateSet object
        self.cur_step_in_mode = None # how much dwell time in current continuous post
        self.max_steps_remaining = None # bound on num steps left in current mode ; assigned on pop

        self.result = None # a HylaaResult... assigned on run() to store verification result

        self.freeze_attrs()

    def is_finished(self):
        'is the computation finished'

        return self.cur_state is None and len(self.waiting_list) == 0 or not self.result.safe

    def check_guards(self):
        '''check for discrete successors with the guards'''

        if self.cur_state.time_elapse.next_step == 1 and self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            print("Solving first step guard LP...")

        for i in xrange(len(self.cur_state.mode.transitions)):
            lp_solution = self.cur_star.get_guard_intersection(i)

            if lp_solution is not None:
                step_num = self.cur_star.time_elapse.next_step - 1

                if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
                    print("Unsafe at Step: {} / {} ({})".format(step_num, self.settings.num_steps, \
                                                            self.settings.step * step_num))

                self.result.init_vars = lp_solution[:self.cur_star.num_init_vars]

                end_output_lp_col = self.cur_star.num_init_vars + self.cur_star.mode.output_space_csr.shape[0]
                self.result.output_vars = lp_solution[self.cur_star.num_init_vars:end_output_lp_col]

                if self.settings.print_lp_on_error:
                    # print the LP solution and exit
                    lpi = self.cur_star.get_guard_lpi(i)
                    lpi.print_lp()

                self.result.safe = False
                break # no need to keep checking

    def do_step_continuous_post(self):
        '''do a step where it's part of a continuous post'''

        # advance time by one step
        if self.cur_state.time_elapse.next_step > self.settings.num_steps:
            self.cur_state = None
        else:
            if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
                step_num = self.cur_state.time_elapse.next_step
                print("Step: {} / {} ({})".format(step_num, self.settings.num_steps, self.settings.step * step_num))

            self.cur_star.step()
            self.check_guards()
    def do_step(self):
        'do a single step of the computation'

        skipped_plot = False # if we skip the plot, do multiple steps

        while True:
            output = self.settings.print_output
            self.plotman.reset_temp_polys()

            if self.cur_state is None:
                self.do_step_pop(output)
            else:
                try:
                    self.do_step_continuous_post(output)
                except FoundErrorTrajectory: # this gets raised if an error mode is reachable and we should quit early
                    pass

            if self.cur_state is not None:
                skipped_plot = self.plotman.plot_current_star(self.cur_state)

            if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE or \
                                    not skipped_plot or self.is_finished():
                break

        if self.is_finished() and self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            if not self.result.safe:
                print("Result: Error modes are reachable.\n")
            else:
                print("Result: System is safe. Error modes are NOT reachable.\n")

    def run(self, init_state_list):
        '''
        Run the computation (main entry point)

        init_star is the initial state

        fixed_dim_list, if used, is a list of dimensions with fixed initial values
        '''

        for state in init_state_list:
            assert isinstance(state, StateSet), "initial states should be a list of StateSet objects"

        assert len(init_state_list) > 0, "expected list of initial states"

        self.result = HylaaResult()

        # initialize time elapse in each mode of the hybrid automaton
        ha = init_state_list[0].mode.parent

        for mode in ha.modes.values():
            mode.init_time_elapse(self.settings.step_size)

        if self.settings.do_guard_strengthening:
            ha.do_guard_strengthening()
        
        self.plotman.create_plot()

        self.waiting_list = init_state_list

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.plotman.run_to_completion(self.do_step, self.is_finished, compute_plot=False)
        else:
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

        # assign results
        self.result.top_level_timer = Timers.top_level_timer
        Timers.reset()

        return self.result

class CounterExampleSegmant(Freezable):
    'a part of a counter-example trace'

    def __init__(self):
        self.mode_name = None
        self.start = None
        self.end = None
        
        self.freeze_attrs()

class HylaaResult(Freezable):
    'Result, assigned to engine.result after computation'

    def __init__(self):
        self.top_level_timer = None # TimerData for total time
        self.safe = True # was the verification result safe?

        self.counterexample = [] # if unsafe, a list of CounterExampleSegment objects

        self.freeze_attrs()
