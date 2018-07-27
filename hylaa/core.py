'''
Main Hylaa Reachability Implementation
Stanley Bak, 2018
'''

import numpy as np

from hylaa.settings import HylaaSettings, PlotSettings

from hylaa.plotutil import PlotManager
from hylaa.stateset import StateSet
from hylaa.hybrid_automaton import HybridAutomaton, was_tt_taken
from hylaa.timerutil import Timers
from hylaa.util import Freezable
from hylaa import lputil

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
        self.max_steps_remaining = None # bound on num steps left in current mode ; assigned on pop

        self.took_tt_transition = False # flag for if a tt was taken and the cur_state should be removed

        self.result = None # a HylaaResult... assigned on run() to store verification result

        self.freeze_attrs()

    def print_normal(self, msg):
        'print function for STDOUT_NORMAL and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            print(msg)

    def print_verbose(self, msg):
        'print function for STDOUT_VERBOSE and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            print(msg)

    def print_waiting_list(self):
        'print out the waiting list'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            print("Waiting list has {} states".format(len(self.waiting_list)))

            for state in self.waiting_list:
                print(" {}".format(state))

    def is_finished(self):
        'is the computation finished'

        finished = not self.result.safe

        if not finished:
            finished = self.cur_state is None and not self.waiting_list

        return finished

    def take_transition(self, t, transition_index):
        '''take the passed-in transition from the current state (may add to the waiting list)'''

        new_lpi = t.lpi.clone()

        lputil.add_reset_variables(new_lpi, t.to_mode.mode_id, transition_index, \
            reset_csr=t.reset_csr, minkowski_csr=t.reset_minkowski_csr, \
            minkowski_constraints_csr=t.reset_minkowski_constraints_csr, \
            minkowski_constraints_rhs=t.reset_minkowski_constraints_rhs)

        # make sure the successor is feasible
        if new_lpi.is_feasible():
            successor_state = StateSet(new_lpi, t.to_mode, self.cur_state.cur_step_since_start)
            self.waiting_list.append(successor_state)

            self.print_verbose("Added Discrete Successor to '{}' at step {}".format( \
                t.to_mode.name, self.cur_state.cur_step_since_start))

            # if it's a time-triggered transition, we may remove the state now
            if self.settings.optimize_tt_transitions and t.time_triggered:
                if was_tt_taken(t):
                    self.print_verbose("Transtion was time-triggered, finished with current state analysis")
                    self.took_tt_transition = True
                else:
                    self.print_verbose("Transtion was NOT taken as time-triggered, due to runtime checks not passing")
        else:
            # succesor is infeasible, check if it's the reset's fault, or due to numerical precision
            t.lpi.reset_lp()

            if t.lpi.is_feasible():
                # it was due to the reset. The user probably provided bad reset parameters.
                raise RuntimeError(("Continuous state was empty after applying reset in transition {}, " + \
                                   "was the reset correctly specified?").format(t))
            else:
                # it was due to numerical issues, it should be ok to remove the original (unsat) state
                self.print_verbose("Continuous state discovered to be UNSAT during transition, removing state")

                self.cur_state = None

    def error_reached(self, t):
        'an error mode was reached after taking transition t, report and create counterexample'

        step_num = self.cur_state.cur_step_since_start
        self.print_normal("Unsafe at Step: {} / {} ({})".format(step_num, self.settings.num_steps, \
                            self.settings.step_size * step_num))

        self.result.safe = False
        self.result.counterexample = make_counterexample(self.hybrid_automaton, t)

    def check_guards(self):
        '''check for discrete successors with the guards'''

        transitions = self.cur_state.mode.transitions

        for transition_index, t in enumerate(transitions):
            if t.lpi.is_feasible():                                    
                if t.to_mode.is_error():
                    self.error_reached(t)
                    break
                else:
                    self.take_transition(t, transition_index)

                    # current state may have become infeasible
                    if self.cur_state is None:
                        break

    def print_current_step_time(self):
        'print the current step and time'

        step_num = self.cur_state.cur_step_since_start
        total_time = self.settings.step_size * step_num
        self.print_verbose("Step: {} / {} ({})".format(step_num, self.settings.num_steps, total_time))

    def do_step_continuous_post(self):
        '''do a step where it's part of a continuous post'''

        self.print_current_step_time()

        if not self.is_finished():
            # next advance time by one step
            if self.cur_state.cur_step_since_start >= self.settings.num_steps:
                self.cur_state = None
            elif self.took_tt_transition:
                self.took_tt_transition = False
                self.cur_state = None
            else:
                still_feasible = self.cur_state.intersect_invariant()
                
                if not still_feasible:
                    self.print_normal("State left the invariant after {} steps".format(self.cur_state.cur_step_in_mode))
                        
                    self.cur_state = None
                else:
                    self.cur_state.step()
                    self.check_guards()

    def pop_waiting_list(self):
        'pop a state off the waiting list, possibly doing state-set aggreation'

        if not self.settings.aggregation:
            rv = self.waiting_list.pop(0)
        else:
            # aggregation is on, first find the state with the minimum time on the waiting list
            first = None

            for state in self.waiting_list:
                if first is None or state.cur_step_since_start < first.cur_step_since_start:
                    first = state

            self.print_verbose("Minimum time state on waiting list: {}".format(first))

            # remove all states with the same mode as 'first' for aggregation
            new_waiting_list = []
            agg_list = []

            for state in self.waiting_list:
                if state.mode == first.mode:
                    agg_list.append(state)
                else:
                    new_waiting_list.append(state)

            self.waiting_list = new_waiting_list # assign new waiting list

            if len(agg_list) == 1:
                rv = agg_list[0]
                self.print_verbose("Removed single state: {}".format(rv))
            else:
                self.print_verbose("Removed {} states for aggregation".format(len(agg_list)))
                # create a new state from the aggregation
                mid_index = len(agg_list) // 2
                mid_lpi = agg_list[mid_index].lpi
                pt = lputil.get_box_center(mid_lpi)
                direction_matrix = lputil.make_direction_matrix(pt, agg_list[mid_index].mode.a_csr)

                lpi_list = [state.lpi for state in agg_list]
                new_lpi = lputil.aggregate(lpi_list, direction_matrix)

                rv = StateSet(new_lpi, first.mode, cur_step_since_start=first.cur_step_since_start)
                
        return rv

    def do_step_pop(self):
        'do a step where we pop from the waiting list'

        self.plotman.state_popped() # reset certain per-mode plot variables
        self.print_waiting_list()

        self.result.last_cur_state = self.cur_state = self.pop_waiting_list()
        self.cur_state.cur_step_in_mode = 0

        self.print_normal("Removed state in mode '{}' at time {:.2f} (Waiting list has {} left)".format( \
                self.cur_state.mode.name, self.cur_state.cur_step_since_start * self.settings.step_size, \
                len(self.waiting_list)))

        # if a_matrix is None, it's an error mode
        if self.cur_state.mode.a_csr is None:
            self.print_normal("Mode '{}' was an error mode; skipping.".format(self.cur_state.mode.name))

            self.cur_state = None
        else:
            # setup the lpi for each outgoing transition
            self.max_steps_remaining = self.settings.num_steps - self.cur_state.cur_step_since_start

            still_feasible = self.cur_state.intersect_invariant()

            if not still_feasible:
                self.print_normal("Continuous state was outside of the mode's invariant; skipping.")
                    
                self.cur_state = None
            else:
                for transition in self.cur_state.mode.transitions:
                    transition.make_lpi(self.cur_state)

        # pause after discrete post when using PLOT_INTERACTIVE
        if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            self.plotman.interactive.paused = True
            
    def do_step(self):
        'do a single step of the computation'

        if not self.is_finished():
            if self.cur_state is None:
                self.do_step_pop()

                if self.settings.process_urgent_guards and self.cur_state is not None:
                    self.check_guards()
            else:
                self.do_step_continuous_post()

            if self.is_finished():
                if not self.result.safe:
                    self.print_normal("Result: Error modes are reachable.\n")
                else:
                    self.print_normal("Result: System is safe. Error modes are NOT reachable.\n")

    def run(self, init_state_list):
        '''
        Run the computation (main entry point)

        init_star is the initial state

        fixed_dim_list, if used, is a list of dimensions with fixed initial values
        '''

        Timers.reset()
        Timers.tic("total")

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

        if self.settings.optimize_tt_transitions:
            ha.detect_tt_transitions()
        
        self.plotman.create_plot()

        # populate waiting list
        self.waiting_list = []

        for state in init_state_list:
            is_feasible = state.lpi.minimize(columns=[], fail_on_unsat=False) is not None

            if not is_feasible:
                self.print_normal("Removed infeasible initial set in mode {}".format(state.mode.name))
                    
                continue
            
            still_feasible = state.intersect_invariant()

            if still_feasible:
                self.waiting_list.append(state)
            else:
                self.print_normal("Removed infeasible initial set after invariant intersection in mode {}".format( \
                        state.mode.name))

        if not self.waiting_list:
            raise RuntimeError("Error: No valid initial states were defined.")

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.plotman.run_to_completion(self.do_step, self.is_finished, \
                                           compute_plot=self.settings.plot.store_plot_result)
        else:
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

        Timers.toc("total")

        if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            Timers.print_stats()

        # assign results
        self.result.top_level_timer = Timers.top_level_timer
        Timers.reset()

        return self.result

def make_counterexample(ha, transition_to_error):
    '''make and return the result counter-example from the lp solution'''

    lpi = transition_to_error.lpi
    lp_solution = lpi.minimize() # resolves the LP to get the full unsafe solution
    names = lpi.get_names()

    counterexample = []
    seg = None

    for name, value in zip(names, lp_solution):

        # if first initial variable of mode then assign the segment.mode variable
        if name.startswith('m') and '_i0' in name:
            if seg is not None: # starting a new segment, append the previous segment
                counterexample.append(seg)

            seg = CounterExampleSegment()

            parts = name.split('_')

            if len(parts) == 2:
                assert not counterexample, "only the initial mode has not predecessor transition"
            else:
                assert len(parts) == 3

                # assign precessor transition
                transition_index = int(parts[2][1:])
                t = counterexample[-1].mode.transitions[transition_index]
                counterexample[-1].outgoing_transition = t

            mode_id = int(parts[0][1:])

            for mode in ha.modes.values():
                if mode.mode_id == mode_id:
                    seg.mode = mode
                    break

            assert seg.mode is not None, "mode id {} not found in automaton".format(mode_id)

        if name.startswith('m'): # mode variable
            if '_i' in name:
                seg.start.append(value)
            elif '_c' in name:
                seg.end.append(value)

    # add the final transition which is not encoded in the names of the variables
    seg.outgoing_transition = transition_to_error

    # add the last segment
    counterexample.append(seg)

    return counterexample

class CounterExampleSegment(Freezable):
    'a part of a counter-example trace'

    def __init__(self):
        self.mode = None # Mode object
        self.start = []
        self.end = []
        self.outgoing_transition = None # Transition object

        # TODO: inputs[]
        
        self.freeze_attrs()

    def __str__(self):
        return "[CE_Segment: {} -> {} in '{}'; out-trans='{}']".format( \
            self.start, self.end, "<None>" if self.mode is None else self.mode.name, \
            "<None>" if self.outgoing_transition is None else self.outgoing_transition.name)

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

        # the last core.cur_state object... used for unit testing
        self.last_cur_state = None

        self.freeze_attrs()
