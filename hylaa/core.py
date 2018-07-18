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

    def print_waiting_list(self):
        'print out the waiting list'

        print("Waiting list has {} states".format(len(self.waiting_list)))

        for state in self.waiting_list:
            print(" {}".format(state))

    def is_finished(self):
        'is the computation finished'

        return (self.cur_state is None and len(self.waiting_list) == 0) or not self.result.safe

    def check_guards(self):
        '''check for discrete successors with the guards'''

        transitions = self.cur_state.mode.transitions

        for t in transitions:
            lp_solution = t.lpi.minimize(fail_on_unsat=False)

            if lp_solution is not None:
                step_num = self.cur_state.cur_step_since_start
                                    
                if t.to_mode.a_csr is not None: # add discrete successor
                    succesor_state = StateSet(t.lpi, t.to_mode)
                    self.waiting_list.append(succesor_state)

                    if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
                        print("Added Discrete Successor to '{}' at step {}".format(t.to_mode.name, step_num))

                else: # unsafe state
                    if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
                        print("Unsafe at Step: {} / {} ({})".format(step_num, self.settings.num_steps, \
                                                                self.settings.step_size * step_num))

                                                                
                    # TODO: print out counter-example

                    self.result.safe = False
                    self.assign_counterexample(lp_solution)
                    
                    break # no need to keep checking transitions

    def print_current_step_time(self):
        'print the current step and time'

        step_num = self.cur_state.cur_step_since_start
        total_time = self.settings.step_size * step_num
        print("Step: {} / {} ({})".format(step_num, self.settings.num_steps, total_time))

    def do_step_continuous_post(self):
        '''do a step where it's part of a continuous post'''

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            self.print_current_step_time()

        # first check guards
        self.check_guards()

        if not self.is_finished():
            # next advance time by one step
            if self.cur_state.cur_step_since_start >= self.settings.num_steps:
                self.cur_state = None
            else:
                self.cur_state.step()

    def do_step_pop(self):
        'do a step where we pop from the waiting list'

        self.plotman.state_popped() # reset certain per-mode plot variables

        self.cur_step_in_mode = 0

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            self.print_waiting_list()

        self.cur_state = self.waiting_list.pop()

        if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            print("Removed state in mode '{}' at time {:.2f}".format(
                self.cur_state.mode.name, self.cur_state.cur_step_since_start * self.settings.step_size))

        # TODO: check if cur_state has a true invariant

        # if a_matrix is None, it's an error mode
        if self.cur_state is not None and self.cur_state.mode.a_csr is None:
            if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
                print("Mode '{}' was an error mode; skipping.".format(self.cur_state.mode.name))

            self.cur_state = None

        if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            if self.cur_state is None:
                print("Popped state was refined away.")
            else:
                print("Doing continuous post in mode '{}'".format(self.cur_state.mode.name))

        # setup the lpi for each outgoing transition
        if self.cur_state is not None:
            self.max_steps_remaining = self.settings.num_steps - self.cur_state.cur_step_since_start
                        
            for transition in self.cur_state.mode.transitions:
                transition.make_lpi(self.cur_state)

            if not self.settings.process_urgent_guards:
                # force one continuous post step in each mode
                if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
                    self.print_current_step_time()
                    
                self.plotman.plot_current_state(self.cur_state)
                self.cur_state.step()

        # pause after discrete post when using PLOT_INTERACTIVE
        if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            self.plotman.interactive.paused = True
            
    def do_step(self):
        'do a single step of the computation'

        while not self.is_finished():
            if self.cur_state is None:
                self.do_step_pop()
            else:
                self.do_step_continuous_post()

            if self.cur_state is not None:
                self.plotman.plot_current_state(self.cur_state)

        if self.is_finished() and self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            if not self.result.safe:
                print("Result: Error modes are reachable.\n")
            else:
                print("Result: System is safe. Error modes are NOT reachable.\n")

    def assign_counterexample(self, lp_solution):
        '''create and assign the result counter-example from the lp
        this assignes to self.result.counterexample
        '''

        names = self.cur_state.lpi.get_names()
        
        self.result.counterexample = []
        seg = None

        for name, value in zip(names, lp_solution):

            # if first initial variable of mode then assign the segment.mode variable
            if name.startswith('m') and name.endswith('_i0'):
                if seg is not None: # starting a new segment, append the previous segment
                    self.result.counterexample.append(seg)

                seg = CounterExampleSegment()
                
                mode_id = int(name[1:-3])

                for mode in self.cur_state.mode.ha.modes.values():
                    if mode.mode_id == mode_id:
                        seg.mode = mode
                        break

                assert seg.mode is not None, "mode id {} not found in automaton".format(mode_id)
            
            if name.startswith('m'): # mode variable
                if '_i' in name:
                    seg.start.append(value)
                elif '_c' in name:
                    seg.end.append(value)

        # add the last segment
        self.result.counterexample.append(seg)

    def run(self, init_state_list):
        '''
        Run the computation (main entry point)

        init_star is the initial state

        fixed_dim_list, if used, is a list of dimensions with fixed initial values
        '''

        for state in init_state_list:
            assert isinstance(state, StateSet), "initial states should be a list of StateSet objects"

        assert init_state_list, "expected list of initial states"

        self.result = HylaaResult()

        # initialize time elapse in each mode of the hybrid automaton
        ha = init_state_list[0].mode.ha

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

class CounterExampleSegment(Freezable):
    'a part of a counter-example trace'

    def __init__(self):
        self.mode = None
        self.start = []
        self.end = []

        # TODO: inputs[]
        
        self.freeze_attrs()

    def __str__(self):
        return "[CE_Segment: {} -> {} in '{}']".format( \
            self.start, self.end, "<None>" if self.mode is None else self.mode.name)

    def __repr__(self):
        return str(self)

class HylaaResult(Freezable):
    'result object returned by core.run()'

    def __init__(self):
        self.top_level_timer = None # TimerData for total time
        self.safe = True # was the verification result safe?

        self.counterexample = [] # if unsafe, a list of CounterExampleSegment objects

        # assigned if setting.plot.store_plot_result is True, a map name -> list of lists (the verts at each step)
        self.mode_to_polys = {}

        self.freeze_attrs()
