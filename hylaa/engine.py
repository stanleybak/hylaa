'''
Main Hylaa Reachability Implementation
Stanley Bak
Aug 2016
'''

from collections import OrderedDict
import numpy as np

from hylaa.plotutil import PlotManager
from hylaa.star import Star
from hylaa.star import init_hr_to_star, init_constraints_to_star
from hylaa.starutil import InitParent, AggregationParent, ContinuousPostParent, DiscretePostParent
from hylaa.starutil import add_guard_to_star, add_box_to_star
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearAutomatonMode, LinearConstraint, HyperRectangle
from hylaa.timerutil import Timers
from hylaa.containers import HylaaSettings, SymbolicState, PlotSettings, HylaaResult
from hylaa.glpk_interface import LpInstance

class HylaaEngine(object):
    'main computation object. initialize and call run()'

    def __init__(self, ha, hylaa_settings):

        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(ha, LinearHybridAutomaton)

        self.ha = ha
        self.settings = hylaa_settings
        self.num_vars = len(ha.variables)

        if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            Star.init_plot_vecs(self.num_vars, self.settings.plot)

        self.plotman = PlotManager(self, self.settings.plot)

        # computation
        self.waiting_list = WaitingList()

        self.cur_state = None # a SymbolicState object
        self.cur_step_in_mode = None # how much dwell time in current continuous post
        self.max_steps_remaining = None # bound on num steps left in current mode ; assigned on pop
        self.cur_sim_bundle = None # set on pop

        self.reached_error = False
        self.result = None # a HylaaResult... assigned on run()

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.settings.simulation.use_presimulation = True

    def load_waiting_list(self, init_list):
        '''convert the init list into self.waiting_list'''

        assert len(init_list) > 0, "initial list length is 0"

        for mode, shape in init_list:
            assert isinstance(mode, LinearAutomatonMode)

            if isinstance(shape, HyperRectangle):
                star = init_hr_to_star(self.settings, shape, mode)
            elif isinstance(shape, list):
                assert len(shape) > 0, "initial constraints in mode '{}' was empty list".format(mode.name)
                assert isinstance(shape[0], LinearConstraint)

                star = init_constraints_to_star(self.settings, shape, mode)
            else:
                raise RuntimeError("Unsupported initial state type '{}': {}".format(type(shape), shape))

            self.waiting_list.add_deaggregated(SymbolicState(mode, star))

    def is_finished(self):
        'is the computation finished'

        rv = self.waiting_list.is_empty() and self.cur_state is None

        return rv or (self.settings.stop_when_error_reachable and self.reached_error)

    def check_guards(self, state):
        'check for discrete successors with the guards'

        assert state is not None

        for i in xrange(len(state.mode.transitions)):
            result = state.star.get_guard_intersection(i)

            if result is not None:
                transition = state.star.mode.transitions[i]
                successor_mode = transition.to_mode

                if successor_mode.is_error:
                    self.reached_error = True

                    if self.settings.stop_when_error_reachable:
                        raise FoundErrorTrajectory("Found error trajectory")

                # copy the current star to be the frozen pre-state of the discrete post operation
                discrete_prestate_star = state.star.clone()
                discrete_prestate_star.parent = ContinuousPostParent(state.mode, state.star.parent.star)

                discrete_poststate_star = state.star.clone()
                discrete_poststate_star.fast_forward_steps = 0 # reset fast forward on transitions
                basis_center = state.star.vector_to_star_basis(state.star.center)
                discrete_poststate_star.parent = DiscretePostParent(state.mode, discrete_prestate_star,
                                                                    basis_center, transition)

                print "~ converting each guard condition to the star's basis"

                # convert each of the guard conditions to the star's basis
                for g in transition.condition_list:

                    # basis vectors (non-transpose) * standard_condition
                    basis_influence = np.dot(state.star.basis_matrix, g.vector)

                    standard_center = state.star.vector_to_star_basis(state.star.center)
                    center_value = np.dot(standard_center, g.vector)
                    remaining_value = g.value - center_value

                    lc = LinearConstraint(basis_influence, remaining_value)

                    discrete_poststate_star.temp_constraints.append(lc)

                if transition.reset_matrix is not None:
                    raise RuntimeError("only empty resets are currently supported")

                # there may be minor errors when converting the guard to the star's basis, so
                # re-check for feasibility

                if discrete_poststate_star.is_feasible():
                    violation_basis_vec = result[star.num_dims:]

                    if not self.settings.aggregation or not self.settings.deaggregation or \
                       self.has_counterexample(state.star, violation_basis_vec, state.star.total_steps):

                        # convert origin offset to star basis and add to basis_center
                        successor = discrete_poststate_star
                        #successor.start_center = successor.center
                        successor.center_into_constraints(basis_center)

                        self.plotman.cache_star_verts(successor) # do this before committing temp constriants
                        successor.commit_temp_constraints()

                        successor.start_basis_matrix = state.star.basis_matrix
                        successor.sim_basis_matrix = None # gets assigned from sim_bundle on pop

                        new_state = SymbolicState(transition.to_mode, successor)

                        if self.settings.aggregation:
                            self.waiting_list.add_aggregated(new_state, self.settings)
                        else:
                            self.waiting_list.add_deaggregated(new_state)
                    else:
                        # a refinement occurred, stop processing guards
                        self.cur_state = state = None
                        break

    def deaggregate_star(self, star, steps_in_cur_star):
        'split an aggregated star in half, and place each half into the waiting list'

        Timers.tic('deaggregate star')

        assert isinstance(star.parent, AggregationParent)

        elapsed_aggregated_steps = steps_in_cur_star - star.total_steps - 1

        if elapsed_aggregated_steps < 0: # happens on urgent transitions
            elapsed_aggregated_steps = 0

        mode = star.parent.mode
        all_stars = star.parent.stars

        # fast forward stars
        for s in all_stars:
            s.total_steps += elapsed_aggregated_steps
            s.fast_forward_steps += elapsed_aggregated_steps

        mid = len(all_stars) / 2
        left_stars = all_stars[:mid]
        right_stars = all_stars[mid:]

        for stars in [left_stars, right_stars]:
            discrete_post_star = stars[0]
            assert isinstance(discrete_post_star.parent, DiscretePostParent)
            discrete_pre_star = discrete_post_star.parent.prestar
            assert isinstance(discrete_pre_star.parent, ContinuousPostParent)

            if len(stars) == 1:
                # this might be parent of parent
                cur_star = discrete_post_star
            else:
                cur_star = self.make_aggregated_star(mode, stars)

            # does aggregation work if start_basis matrix is different for the stars?
            ss = SymbolicState(mode, cur_star)
            self.waiting_list.add_deaggregated(ss)

        Timers.toc('deaggregate star')

    def make_aggregated_star(self, mode, star_list):
        '''
        make an aggregated star from a star list

        This returns a typle Star with parent of type AggregatedParent
        '''

        first_star_parent = star_list[0].parent
        hull_star = star_list[0].clone()

        assert len(hull_star.temp_constraints) == 0

        hull_star.parent = AggregationParent(mode, star_list)

        if self.settings.add_guard_during_aggregation:
            assert isinstance(first_star_parent, DiscretePostParent)
            add_guard_to_star(hull_star, first_star_parent.transition.guard_list)

        if self.settings.add_box_during_aggregation:
            add_box_to_star(hull_star)

        # there may be temp constraints from invariant trimming
        hull_star.commit_temp_constraints()

        for star_index in xrange(1, len(star_list)):
            star = star_list[star_index]
            hull_star.eat_star(star)

        return hull_star

    def has_counterexample(self, star, basis_point, steps_in_cur_star):
        'check if the given basis point in the given star corresponds to a real trace'

        # if the parent is an initial state, then we're done and plot
        if isinstance(star.parent, InitParent):
            rv = True
        elif isinstance(star.parent, ContinuousPostParent):
            rv = self.has_counterexample(star.parent.star, basis_point, steps_in_cur_star)

            if not rv:
                # rv was false, some refinement occurred and we need to delete this star

                print "; make this a setting, deleting aggregated from plot"
                #self.plotman.del_parent_successors(star.parent)
        elif isinstance(star.parent, DiscretePostParent):
            # we need to modify basis_point based on the parent's center
            basis_point = basis_point - star.parent.prestar_basis_center

            rv = self.has_counterexample(star.parent.prestar, basis_point, steps_in_cur_star)
        elif isinstance(star.parent, AggregationParent):
            # we need to SPLIT this aggregation parent
            rv = False

            self.deaggregate_star(star, steps_in_cur_star)
        else:
            raise RuntimeError("Concrete trace for parent type '{}' not implemented.".format(type(star.parent)))

        if rv and self.plotman.settings.plot_mode != PlotSettings.PLOT_NONE:
            if isinstance(star.parent, ContinuousPostParent):
                sim_bundle = star.parent.mode.get_sim_bundle(self.settings.simulation)
                num_steps = star.fast_forward_steps + star.total_steps - star.parent.star.total_steps
                start_basis_matrix = star.start_basis_matrix

                self.plotman.plot_trace(num_steps, sim_bundle, start_basis_matrix, basis_point)

        return rv

    def do_step_pop(self, output):
        'do a step where we pop from the waiting list'

        self.plotman.state_popped() # reset certain per-mode plot variables

        self.cur_step_in_mode = 0

        if output:
            self.waiting_list.print_stats()

        state = self.waiting_list.pop()

        if output:
            print "Removed state in mode '{}' at time {:.2f}; fast_forward_steps = {}".format(
                state.mode.name, state.star.total_steps * self.settings.step, state.star.fast_forward_steps)

        self.max_steps_remaining = self.settings.num_steps - state.star.total_steps + state.star.fast_forward_steps
        self.cur_sim_bundle = state.mode.get_sim_bundle(self.settings.simulation, state.star, self.max_steps_remaining)

        parent_star = state.star
        state.star = parent_star.clone()
        state.star.parent = ContinuousPostParent(state.mode, parent_star)
        self.cur_state = state

        if self.settings.process_urgent_guards:
            self.check_guards(self.cur_state)

            if self.cur_state is None:
                if output:
                    print "After urgent checking guards, state was refined away."
            elif output:
                print "Doing continuous post in mode '{}': ".format(self.cur_state.mode.name)

        if self.cur_state is not None and state.mode.is_error:
            self.cur_state = None

            if output:
                print "Mode '{}' was an error mode; skipping.".format(state.mode.name)

        # pause after discrete post when using PLOT_INTERACTIVE
        if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            self.plotman.interactive.paused = True

    def do_step_continuous_post(self, output):
        '''do a step where it's part of a continuous post'''

        # advance current state by one time step
        state = self.cur_state

        if state.star.total_steps >= self.settings.num_steps:
            self.cur_state = None
        else:
            sim_bundle = self.cur_sim_bundle

            if self.settings.print_step_times:
                print "Step: {} / {} ({})".format(self.cur_step_in_mode + 1, self.settings.num_steps,
                                                  self.settings.step * (self.cur_step_in_mode))

            sim_step = self.cur_step_in_mode + 1 + state.star.fast_forward_steps

            new_basis_matrix, new_center = sim_bundle.get_vecs_origin_at_step(
                sim_step, self.max_steps_remaining)

            state.star.update_from_sim(new_basis_matrix, new_center)

            # increment step
            self.cur_step_in_mode += 1
            state.star.total_steps += 1

            self.check_guards(self.cur_state)

            # refinement may occur while checking guards, which sets cur_state to None
            if self.cur_state is None:
                if output:
                    print "After checking guards, state was refined away."
            else:
                is_still_feasible, inv_vio_star_list = self.cur_state.star.trim_to_invariant()

                for star in inv_vio_star_list:
                    self.plotman.add_inv_violation_star(star)

                if not is_still_feasible:
                    self.cur_state = None

        # after continuous post completes
        if self.cur_state is None:
            if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
                self.plotman.interactive.paused = True

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
                skipped_plot = self.plotman.plot_current_state(self.cur_state)

            if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE or \
                                    not skipped_plot or self.is_finished():
                break

        if self.is_finished():
            if self.reached_error:
                print "Result: Error modes are reachable.\n"
            else:
                print "Result: Error modes are NOT reachable.\n"

    def run_to_completion(self):
        'run the computation until it finishes (without plotting)'
    
        Timers.tic("total")

        while not self.is_finished():
            self.do_step()

        Timers.toc("total")

        LpInstance.print_stats()
        Timers.print_stats()

        self.result.time = Timers.timers["total"].total_secs

    def run(self, init_list):
        '''
        run the computation

        init is a list of (LinearAutomatonMode, HyperRectangle)
        '''

        assert len(init_list) > 0

        # strengthen guards to inclunde invariants of targets
        ha = init_list[0][0].parent
        ha.do_guard_strengthening()

        self.result = HylaaResult()
        self.plotman.create_plot()

        # convert init states to stars
        self.load_waiting_list(init_list)

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            # run without plotting
            self.run_to_completion()
        else:
            # plot during computation
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

class WaitingList(object):
    '''
    The set of to-be computed values (discrete sucessors).

    There are aggregated states, and deaggregated states. The idea is states first
    go into the aggregrated ones, but may be later split and placed into the 
    deaggregated list. Thus, deaggregated states, if they exist, are popped first.
    The states here are SymbolicStates
    '''

    def __init__(self):
        self.aggregated_mode_to_state = OrderedDict()
        self.deaggregated_list = []

    def pop(self):
        'pop a state from the waiting list'

        assert len(self.aggregated_mode_to_state) + len(self.deaggregated_list) > 0, \
            "pop() called on empty waiting list"

        if len(self.deaggregated_list) > 0:
            rv = self.deaggregated_list[0]
            self.deaggregated_list = self.deaggregated_list[1:]
        else:
            # pop from aggregated list
            rv = self.aggregated_mode_to_state.popitem(last=False)[1]

            assert isinstance(rv, SymbolicState)
    
            # pylint false positive
            if isinstance(rv.star.parent, AggregationParent):
                rv.star.trim_redundant_perm_constraints()

        return rv

    def print_stats(self):
        'print statistics about the waiting list'

        total = len(self.aggregated_mode_to_state) + len(self.deaggregated_list)

        print "Waiting list contains {} states ({} aggregated and {} deaggregated):".format(
            total, len(self.aggregated_mode_to_state), len(self.deaggregated_list))

        counter = 1

        for ss in self.deaggregated_list:
            print " {}. Deaggregated Successor in Mode '{}'".format(counter, ss.mode.name)
            counter += 1

        for mode, ss in self.aggregated_mode_to_state.iteritems():
            if isinstance(ss.star.parent, AggregationParent):
                print " {}. Aggregated Sucessor in Mode '{}': {} stars".format(counter, mode, len(ss.star.parent.stars))
            else:
                # should be a DiscretePostParent
                print " {}. Aggregated Sucessor in Mode '{}': single star".format(counter, mode)

            counter += 1

    def is_empty(self):
        'is the waiting list empty'

        return len(self.deaggregated_list) == 0 and len(self.aggregated_mode_to_state) == 0

    def add_deaggregated(self, state):
        'add a state to the deaggregated list'

        assert isinstance(state, SymbolicState)

        self.deaggregated_list.append(state)

    def add_aggregated(self, new_state, hylaa_settings):
        'add a state to the aggregated map'

        assert isinstance(new_state, SymbolicState)

        mode_name = new_state.mode.name

        existing_state = self.aggregated_mode_to_state.get(mode_name)

        if existing_state is None:
            self.aggregated_mode_to_state[mode_name] = new_state
        else:
            # combine the two stars
            cur_star = existing_state.star

            cur_star.current_step = min(
                cur_star.total_steps, new_state.star.total_steps)

            # if the parent of this star is not an aggregation, we need to create one
            # otherwise, we need to add it to the list of parents

            if isinstance(cur_star.parent, AggregationParent):
                # add it to the list of parents
                cur_star.parent.stars.append(new_state.star)

                cur_star.eat_star(new_state.star)
            else:
                # create the aggregation parent
                hull_star = cur_star.clone()
                hull_star.parent = AggregationParent(new_state.mode, [cur_star, new_state.star])

                if hylaa_settings.add_guard_during_aggregation:
                    add_guard_to_star(hull_star, cur_star.parent.transition.condition_list)

                if hylaa_settings.add_box_during_aggregation:
                    add_box_to_star(hull_star)

                # there may be temp constraints from invariant trimming
                hull_star.commit_temp_constraints()

                hull_star.eat_star(new_state.star)
                existing_state.star = hull_star

class FoundErrorTrajectory(RuntimeError):
    'gets thrown if a trajectory to the error states is found when settings.stop_when_error_reachable is True'
