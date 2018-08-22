'''
Main Hylaa Reachability Implementation
Stanley Bak, 2018
'''

import math
from collections import deque

import numpy as np
from termcolor import cprint

from hylaa.settings import HylaaSettings, PlotSettings, AggregationSettings

from hylaa.plotutil import PlotManager
from hylaa.stateset import StateSet, TransitionPredecessor, AggregationPredecessor
from hylaa.hybrid_automaton import HybridAutomaton, was_tt_taken
from hylaa.timerutil import Timers
from hylaa.util import Freezable
from hylaa.lpinstance import LpInstance, UnsatError
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

        self.num_steps = 0

        # make random number generation (for example, to find orthogonal directions) deterministic
        np.random.seed(seed=0)

        LpInstance.print_normal = self.print_normal
        LpInstance.print_verbose = self.print_verbose
        LpInstance.print_debug = self.print_debug

        self.freeze_attrs()

    def print_normal(self, msg):
        'print function for STDOUT_NORMAL and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_NORMAL:
            cprint(msg, self.settings.stdout_colors[HylaaSettings.STDOUT_NORMAL])

    def print_verbose(self, msg):
        'print function for STDOUT_VERBOSE and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            cprint(msg, self.settings.stdout_colors[HylaaSettings.STDOUT_VERBOSE])

    def print_debug(self, msg):
        'print function for STDOUT_DEBUG and above'

        if self.settings.stdout >= HylaaSettings.STDOUT_DEBUG:
            cprint(msg, self.settings.stdout_colors[HylaaSettings.STDOUT_DEBUG])

    def print_waiting_list(self):
        'print out the waiting list'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            col = self.settings.stdout_colors[HylaaSettings.STDOUT_VERBOSE]
            
            cprint("Waiting list has {} states".format(len(self.waiting_list)), col)

            if len(self.waiting_list) < 10:
                for state in self.waiting_list:
                    cprint(" {}".format(state), col)

    def is_finished(self):
        'is the computation finished'

        finished = not self.result.safe

        if not finished:
            finished = self.cur_state is None and not self.waiting_list

        return finished

    def take_transition(self, t, t_lpi):
        '''take the passed-in transition from the current state (may add to the waiting list)'''

        predecessor = TransitionPredecessor(self.cur_state.clone(keep_computation_path_id=True), t, t_lpi.clone())

        successor_has_inputs = t.to_mode.b_csr is not None

        lputil.add_reset_variables(t_lpi, t.to_mode.mode_id, t.transition_index, \
            reset_csr=t.reset_csr, minkowski_csr=t.reset_minkowski_csr, \
            minkowski_constraints_csr=t.reset_minkowski_constraints_csr, \
            minkowski_constraints_rhs=t.reset_minkowski_constraints_rhs, successor_has_inputs=successor_has_inputs)

        if t_lpi.is_feasible():
            successor_state = StateSet(t_lpi, t.to_mode, self.cur_state.cur_steps_since_start, predecessor)
            self.waiting_list.append(successor_state)

            self.print_verbose("Added Discrete Successor to '{}' at step {}".format( \
                t.to_mode.name, self.cur_state.cur_steps_since_start))

            # if it's a time-triggered transition, we may remove cur_state immediately
            if self.settings.optimize_tt_transitions and t.time_triggered:
                if was_tt_taken(self.cur_state.lpi, t):
                    self.print_verbose("Transition was time-triggered, finished with current state analysis")
                    self.took_tt_transition = True
                else:
                    self.print_verbose("Transition was NOT taken as time-triggered, due to runtime checks")
        else:
            # successor is infeasible, check if it's the reset's fault, or due to numerical precision
            # if it's due to numerical precision, the current state's lpi is barely feasible. If we reset it, it
            # becomes infeasible
            self.cur_state.lpi.reset_lp()

            if self.cur_state.lpi.is_feasible():
                # it was due to the reset. The user probably provided bad reset parameters.
                raise RuntimeError(("Continuous state was empty after applying reset in transition {}, " + \
                                   "was the reset correctly specified?").format(t))
            else:
                # it was due to numerical issues, it should be ok to remove the original (unsat) state
                self.print_normal("Continuous state discovered to be UNSAT during transition, removing state")

                self.cur_state = None

    def error_reached(self, t, lpi):
        'an error mode was reached after taking transition t, report and create counterexample'

        step_num = self.cur_state.cur_steps_since_start
        times = [round(self.settings.step_size * step_num[0], 12), round(self.settings.step_size * step_num[1], 12)]

        if step_num[0] == step_num[1]:
            step_num = step_num[0]
            times = times[0]
        
        self.print_normal("Unsafe at Step: {} / {}, time {}".format(step_num, self.settings.num_steps, times))

        self.result.safe = False

        if self.cur_state.has_aggregation_precessor():
            self.result.counterexample = None
        else:
            self.result.counterexample = make_counterexample(self.hybrid_automaton, t, lpi)

    def check_guards(self):
        '''check for discrete successors with the guards'''

        Timers.tic("check_guards")

        try:
            transitions = self.cur_state.mode.transitions

            for t in transitions:
                t_lpi = t.get_guard_intersection(self.cur_state.lpi)

                if t_lpi:
                    if t.to_mode.is_error():
                        self.error_reached(t, t_lpi)
                        break
                    else:
                        self.take_transition(t, t_lpi)

                    # current state may have become infeasible
                    if self.cur_state is None:
                        break

        except UnsatError:
            self.print_normal("State became infeasible after checking guards. " + \
                                      "Likely was barely feasible + numerical issues); removing state.")
            self.cur_state = None

        Timers.toc("check_guards")

    def print_current_step_time(self):
        'print the current step and time'

        step_num = self.cur_state.cur_steps_since_start
        times = [round(self.settings.step_size * step_num[0], 12), round(self.settings.step_size * step_num[1], 12)]

        if step_num[0] == step_num[1]:
            step_num = step_num[0]
            times = times[0]

        self.print_verbose("Step: {} / {} ({})".format(step_num, self.settings.num_steps, times))

    def do_step_continuous_post(self):
        '''do a step where it's part of a continuous post'''

        Timers.tic('do_step_continuous_post')

        self.print_current_step_time()

        if not self.is_finished():
            # next advance time by one step
            if self.cur_state.cur_steps_since_start[0] >= self.settings.num_steps:
                self.print_verbose("cur_state reached time bound, removing")
                self.cur_state = None

                if self.is_finished():
                    self.print_normal("Computation finished after {} steps.".format(self.num_steps))
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

        Timers.toc('do_step_continuous_post')

    def compute_pop_score(self, state):
        '''
        returns a score used to decide which state to pop off the waiting list. The state with the highest score
        will be removed first.
        '''

        if self.settings.aggregation.pop_strategy == AggregationSettings.POP_LOWEST_MINTIME:
            score = -(state.cur_steps_since_start[0])
        elif self.settings.aggregation.pop_strategy == AggregationSettings.POP_LOWEST_AVGTIME:
            score = -(state.cur_steps_since_start[0] + state.cur_steps_since_start[1])
        elif self.settings.aggregation.pop_strategy == AggregationSettings.POP_LARGEST_MAXTIME:
            score = state.cur_steps_since_start[1]
        else:
            raise RuntimeError("Unknown waiting list pop strategy: {}".format(self.settings.aggregation.pop_strategy))

        return score

    def get_agggreation_states(self):
        '''get the states from the waiting list we are going to aggregate

        This also updates the waiting list.

        returns agg_list, step_range (of states in agg_list)
        '''

        to_remove = self.waiting_list[0]
        to_remove_score = self.compute_pop_score(to_remove)

        for state in self.waiting_list:
            score = self.compute_pop_score(state)

            if score > to_remove_score or score == to_remove_score and state.mode.name < to_remove.mode.name:
                to_remove_score = score
                to_remove = state

        self.print_verbose("Aggregating with state at time {} in mode {}".format(to_remove.cur_steps_since_start,
                                                                                 to_remove.mode.name))

        # remove all states for aggregation
        new_waiting_list = []
        agg_list = []
        min_steps = float('inf')
        max_steps = -float('inf')

        for state in self.waiting_list:
            should_add = False

            if state is to_remove:
                should_add = True
            elif state.mode is to_remove.mode:
                if self.settings.aggregation.require_same_path:
                    # look at the parents
                    if (isinstance(to_remove.predecessor, TransitionPredecessor) and \
                        isinstance(state.predecessor, TransitionPredecessor) and \
                        to_remove.predecessor.transition is state.predecessor.transition and \
                        to_remove.predecessor.state.computation_path_id == state.predecessor.state.computation_path_id):
                        
                        should_add = True
                else: # require_same_path is False
                    should_add = True

            if should_add:
                agg_list.append(state)
                
                min_steps = min(min_steps, state.cur_steps_since_start[0])
                max_steps = max(max_steps, state.cur_steps_since_start[1])
            else:
                new_waiting_list.append(state)

        self.waiting_list = new_waiting_list

        assert agg_list, "returning empty agg_list?"

        return agg_list, [min_steps, max_steps]

    def perform_aggregation(self, agg_list, step_interval):
        '''
        perform aggregation on the passed-in list of states
        '''

        # create a new state from the aggregation
        postmode = agg_list[0].mode
        postmode_dims = postmode.a_csr.shape[0]
        mid_index = len(agg_list) // 2
        mid_state = agg_list[mid_index]
        pred = mid_state.predecessor

        if self.settings.aggregation.agg_mode == AggregationSettings.AGG_BOX or pred is None:
            agg_dir_mat = np.identity(postmode_dims)
        elif self.settings.aggregation.agg_mode == AggregationSettings.AGG_ARNOLDI_BOX:
            # aggregation with a predecessor, use arnoldi directions in predecessor mode in center of
            # middle aggregagted state, then project using the reset, and reorthogonalize

            assert isinstance(pred, TransitionPredecessor)
            premode = pred.state.mode
            pt = lputil.get_box_center(pred.premode_lpi)
            self.print_debug("aggregation point: {}".format(pt))

            premode_dir_mat = lputil.make_direction_matrix(pt, premode.a_csr)
            self.print_debug("premode dir mat:\n{}".format(premode_dir_mat))

            if pred.transition.reset_csr is None:
                agg_dir_mat = premode_dir_mat
            else:
                projected_dir_mat = premode_dir_mat * pred.transition.reset_csr.transpose()

                self.print_debug("projected dir mat:\n{}".format(projected_dir_mat))

                # re-orthgohonalize (and create new vectors if necessary)
                agg_dir_mat = lputil.reorthogonalize_matrix(projected_dir_mat, postmode_dims)

            # also add box directions in target mode (if they don't already exist)
            box_dirs = []
            for dim in range(postmode_dims):
                direction = [0 if d != dim else 1 for d in range(postmode_dims)]
                exists = False

                for row in agg_dir_mat:
                    if np.allclose(direction, row):
                        exists = True
                        break

                if not exists:
                    box_dirs.append(direction)

            if box_dirs:
                agg_dir_mat = np.concatenate((agg_dir_mat, box_dirs), axis=0)

        if pred and self.settings.aggregation.add_guard:
            # add all the guard conditions to the agg_dir_mat

            if pred.transition.reset_csr is None: # identity reset
                guard_dir_mat = pred.transition.guard_csr
            else:
                # multiply each direction in the guard by the guard
                guard_dir_mat = pred.transition.guard_csr * pred.transition.reset_csr.transpose()

            if guard_dir_mat.shape[0] > 0:
                agg_dir_mat = np.concatenate((agg_dir_mat, guard_dir_mat.toarray()), axis=0)
        else:
            raise RuntimeError("Unknown aggregatinon mode: {}".format(self.settings.aggregation.agg_mode))

        self.print_debug("agg dir mat:\n{}".format(agg_dir_mat))
        lpi_list = [state.lpi for state in agg_list]

        new_lpi = lputil.aggregate(lpi_list, agg_dir_mat, postmode)

        predecessor = AggregationPredecessor(agg_list) # Note: these objects weren't clone()'d
        return StateSet(new_lpi, agg_list[0].mode, step_interval, predecessor)

    def pop_waiting_list(self):
        'pop a state off the waiting list, possibly doing state-set aggreation'

        if self.settings.aggregation.agg_mode == AggregationSettings.AGG_NONE:
            rv = self.waiting_list.pop(0)
        else:
            agg_list, step_interval = self.get_agggreation_states()

            self.print_verbose("Removed {} state{} for aggregation at steps {}".format(
                len(agg_list), "s" if len(agg_list) > 1 else "", step_interval))

            if len(agg_list) == 1:
                rv = agg_list[0]
            else:
                rv = self.perform_aggregation(agg_list, step_interval)
                
        return rv

    def do_step_pop(self):
        'do a step where we pop from the waiting list'

        Timers.tic('do_step_pop')

        self.plotman.state_popped() # reset certain per-mode plot variables
        self.print_waiting_list()

        self.result.last_cur_state = self.cur_state = self.pop_waiting_list()

        self.print_normal("Removed state in mode '{}' at step {} ({} in mode) (Waiting list has {} left)".format( \
                self.cur_state.mode.name, self.cur_state.cur_steps_since_start, self.cur_state.cur_step_in_mode, \
                len(self.waiting_list)))

        # if a_matrix is None, it's an error mode
        if self.cur_state.mode.a_csr is None:
            self.print_normal("Mode '{}' was an error mode; skipping.".format(self.cur_state.mode.name))

            self.cur_state = None
        else:
            self.max_steps_remaining = self.settings.num_steps - self.cur_state.cur_steps_since_start[0]

            still_feasible = self.cur_state.intersect_invariant()

            if not still_feasible:
                self.print_normal("Continuous state was outside of the mode's invariant; skipping.")
                self.cur_state = None

        # pause after discrete post when using PLOT_INTERACTIVE
        if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            self.plotman.interactive.paused = True

        Timers.toc('do_step_pop')
                
    def do_step(self):
        'do a single step of the computation'

        Timers.tic('do_step')
        self.num_steps += 1

        if not self.is_finished():
            if self.cur_state is None:
                self.do_step_pop()

                if self.settings.process_urgent_guards and self.cur_state is not None:
                    self.check_guards()
            else:
                self.do_step_continuous_post()

        Timers.toc('do_step')

    def setup(self, init_state_list):
        'setup the computation (called by run())'

        Timers.tic('setup')

        for state in init_state_list:
            assert isinstance(state, StateSet), "initial states should be a list of StateSet objects"

        assert init_state_list, "expected list of initial states"

        self.result = HylaaResult()

        # initialize time elapse in each mode of the hybrid automaton
        ha = init_state_list[0].mode.ha

        for mode in ha.modes.values():
            mode.init_time_elapse(self.settings.step_size)

        if self.settings.optimize_tt_transitions:
            ha.detect_tt_transitions(self.print_debug)

        if self.settings.do_guard_strengthening:
            ha.do_guard_strengthening()

        ha.check_transition_dimensions()
        
        self.plotman.create_plot()

        # populate waiting list
        self.waiting_list = []

        for state in init_state_list:
            if not state.lpi.is_feasible():
                self.print_normal("Removed an infeasible initial set in mode {}".format(state.mode.name))
                continue
            
            still_feasible = state.intersect_invariant()

            if still_feasible:
                self.waiting_list.append(state)
            else:
                self.print_normal("Removed an infeasible initial set after invariant intersection in mode {}".format( \
                        state.mode.name))

        if not self.waiting_list:
            raise RuntimeError("Error: No valid initial states were defined.")

        Timers.toc('setup')

    def run_to_completion(self):
        'run the model to completion (called by run() if not plot is desired)'

        self.plotman.run_to_completion(compute_plot=self.settings.plot.store_plot_result)

    def run(self, init_state_list):
        '''
        Run the computation (main entry point)

        init_star is the initial state

        fixed_dim_list, if used, is a list of dimensions with fixed initial values
        '''

        Timers.reset()
        Timers.tic("total")

        self.setup(init_state_list)

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.run_to_completion()
        else:
            self.plotman.compute_and_animate()

        Timers.toc("total")

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            Timers.print_stats()

        if not self.result.safe:
            self.print_normal("Result: Error modes are reachable.\n")
        else:
            self.print_normal("Result: System is safe. Error modes are NOT reachable.\n")

        self.print_normal("Total Runtime: {:.2f} sec".format(Timers.top_level_timer.total_secs))

        # assign results
        self.result.top_level_timer = Timers.top_level_timer
        Timers.reset()

        return self.result

def make_counterexample(ha, transition_to_error, lpi):
    '''make and return the result counter-example from the lp solution'''

    lp_solution = lpi.minimize() # resolves the LP to get the full unsafe solution
    names = lpi.get_names()

    counterexample = []

    for name, value in zip(names, lp_solution):

        # if first initial variable of mode then assign the segment.mode variable
        if name.startswith('m') and '_i0' in name:
            seg = CounterExampleSegment()
            counterexample.append(seg)

            parts = name.split('_')

            if len(parts) == 2:
                assert len(counterexample) == 1, "only the initial mode should have no predecessor transition"
            else:
                assert len(parts) == 3

                # assign outgoing transition of previous counterexample segment
                transition_index = int(parts[2][1:])
                t = counterexample[-2].mode.transitions[transition_index]
                counterexample[-2].outgoing_transition = t

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
            elif '_I' in name:
                if '_I0' in name:
                    seg.inputs.appendleft([])
                    
                # inputs are in backwards order due to how the LP is constructed, prepend it
                seg.inputs[0].append(value)

        elif name.startswith('reset'):
            seg.reset_minkowski_vars.append(value)

    # add the final transition which is not encoded in the names of the variables
    seg.outgoing_transition = transition_to_error

    return counterexample

class CounterExampleSegment(Freezable):
    'a part of a counter-example trace'

    def __init__(self):
        self.mode = None # Mode object
        self.start = []
        self.end = []
        self.outgoing_transition = None # Transition object
        self.reset_minkowski_vars = [] # a list of minkowski variables in the outgoing reset

        self.inputs = deque() # inputs at each step (a deque of m-tuples, where m is the number of inputs)
        
        self.freeze_attrs()

    def __str__(self):
        return "[CE_Segment: {} -> {} in '{}']".format( \
            self.start, self.end, "<None>" if self.mode is None else self.mode.name)

    def __repr__(self):
        return str(self)

class HylaaResult(Freezable): # pylint: disable=too-few-public-methods
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
