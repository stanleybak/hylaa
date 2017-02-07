'''
Main Hylaa Reachability Implementation
Stanley Bak
Aug 2016
'''

import numpy as np

import hylaa.optutil as optutil
from hylaa.plotutil import PlotManager
from hylaa.star import init_hr_to_star, init_constraints_to_star, Star
from hylaa.star import InitParent, AggregationParent, ContinuousPostParent, DiscretePostParent
from hylaa.star import add_guard_to_star, add_box_to_star
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearAutomatonMode, LinearConstraint, HyperRectangle
from hylaa.simutil import SimulationBundle
from hylaa.timerutil import Timers
from hylaa.containers import HylaaSettings, SymbolicState, PlotSettings, HylaaResult, WaitingList

class HylaaEngine(object):
    'main computation object. initialize and call run()'

    def __init__(self, ha, hylaa_settings):

        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(ha, LinearHybridAutomaton)

        self.ha = ha
        self.settings = hylaa_settings
        self.num_vars = len(ha.variables)
        Star.solver = hylaa_settings.solver # set LP solver

        self.settings.plot.init_plot_vecs(self.num_vars)
        self.plotman = PlotManager(self, self.settings.plot, Star.solver)

        # computation
        self.sim_bundles = {} # map of mode_name -> SimulationBundle

        self.waiting_list = WaitingList()

        self.cur_state = None # a SymbolicState object
        self.cur_step_in_mode = None # how much dwell time in current continuous post
        self.max_steps_remaining = None # bound on num steps left in current mode ; assigned on pop
        self.cur_sim_bundle = None # set on pop

        self.selected_stars = [] # set

        self.result = None # a HylaaResult... assigned on run()

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.settings.use_presimulation = True

    def load_waiting_list(self, init_list):
        '''convert the init list into self.waiting_list'''

        assert len(init_list) > 0, "initial list length is 0"

        for mode, shape in init_list:
            assert isinstance(mode, LinearAutomatonMode)

            if isinstance(shape, HyperRectangle):
                star = init_hr_to_star(shape, mode)
            elif isinstance(shape, list):
                assert len(shape) > 0, "initial constraints in mode '{}' was empty list".format(mode.name)
                assert isinstance(shape[0], LinearConstraint)

                star = init_constraints_to_star(shape, mode)
            else:
                raise RuntimeError("Unsupported initial state type '{}': {}".format(type(shape), shape))

            self.plotman.cache_star_verts(star)
            self.waiting_list.add_deaggregated(SymbolicState(mode, star))

            optutil.MultiOpt.reset_per_mode_vars()

    def is_finished(self):
        'is the computation finished'

        rv = self.waiting_list.is_empty() and self.cur_state is None

        return rv

    def plot_invariant_violation(self, star, invariant_lc):
        '''
        plot the invariant violation star
        
        star is the original (untrimmed) star
        invariant_lc is a LinearConstraint (the invariant condition that was violated)
        lp_result is the linear programming result point showing the violation
        '''

        if self.plotman.settings.plot_mode != PlotSettings.PLOT_NONE:
            inv_vio_star = star.clone()

            # add the inverse of the linear constraint as the invariant-violation
            inv_lc = LinearConstraint([-v for v in invariant_lc.vector], -invariant_lc.value)
            inv_vio_star.temp_constraints.append(inv_lc)

            # re-check for feasibility after adding the constraint

            if inv_vio_star.is_feasible() is not None:
                self.plotman.add_inv_violation_star(inv_vio_star)

    def trim_to_invariant(self, state):
        '''
        trim the passed-in state to the invariant.

        returns true if the star is still feasible
        '''

        star = state.star
        still_feasible = True
        added_invariant_constraint = False

        if len(state.mode.inv_list) > 0:
            # if any of the invariant conditions are violated, we need to trim the star
            standard_center = star.center()
            num_dims = len(standard_center)

            lp_constraints = star.to_lp_constraints()
            c_list = []

            for lin_con in state.mode.inv_list:
                c_list.append([-ele for ele in lin_con.vector] + [0.0] * star.num_dims)

            result_list = optutil.optimize_multi(Star.solver, c_list, lp_constraints)

            for inv_index in xrange(len(result_list)):
                lin_con = state.mode.inv_list[inv_index]
                result = result_list[inv_index]
                offset = result[0:num_dims]
                point = standard_center + offset

                val = np.dot(point, lin_con.vector)

                if val > lin_con.value:
                    # convert the condition to the star's basis

                    # basis vectors (non-transpose) * standard_condition
                    basis_influence = np.dot(star.basis_matrix, lin_con.vector)
                    center_value = np.dot(standard_center, lin_con.vector)
                    remaining_value = lin_con.value - center_value

                    lc = LinearConstraint(basis_influence, remaining_value)

                    self.plot_invariant_violation(star, lc)

                    star.temp_constraints.append(lc)
                    added_invariant_constraint = True
                
                    # we added a new constraint to the star, check if it's still feasible
                    if not star.is_feasible():
                        still_feasible = False
                        break # don't check the remaining invariant linear conditions

        if still_feasible:
            if len(star.temp_constraints) > 0 and self.settings.trim_redundant_inv_constraints:
                star.trim_redundant_temp_constraints()

                if added_invariant_constraint is False:
                    star.commit_temp_constraints()

        return still_feasible

    def check_guards(self, state):
        'check for discrete successors with the guards'

        assert state is not None

        transitions = state.mode.transitions

        if len(transitions) > 0:
            standard_center = state.star.center()
            num_dims = len(standard_center)

        for transition in transitions:
            # all the guards must be true at the same time for it to be a successor

            lp_constraints = state.star.to_lp_constraints()
            guard_constraints_a = []
            guard_constraints_b = []

            for g in transition.condition_list:
                empty = [0.0] * num_dims
                center_value = np.dot(standard_center, g.vector)

                guard_constraints_a.append([ele for ele in g.vector] + empty)
                guard_constraints_b.append(g.value - center_value)

            lp_constraints.a_temp_ub += guard_constraints_a
            lp_constraints.b_temp_ub += guard_constraints_b

            c = [0.0] * (2 * num_dims) # objective function is not important

            # use the first guard as the objective function
            #first_guard = transition.condition_list[0]
            #c = [-ele for ele in first_guard.vector] + [0.0] * num_dims

            result = optutil.optimize_multi(Star.solver, [c], lp_constraints)[0]

            if result is not None: # if a guard intersection exists
                print "; guard intersection exists at {}".format(result[0:num_dims])

                # copy the current star to be the frozen pre-state of the discrete post operation
                discrete_prestate_star = state.star.clone()
                discrete_prestate_star.parent = ContinuousPostParent(state.mode, state.star.parent.star)

                discrete_poststate_star = state.star.clone()
                discrete_poststate_star.fast_forward_steps = 0 # reset fast forward on transitions
                basis_center = state.star.vector_to_star_basis(state.star.sim_center)
                discrete_poststate_star.parent = DiscretePostParent(state.mode, discrete_prestate_star, 
                                                                    basis_center, transition)

                # convert each of the guard conditions to the star's basis
                for g in transition.condition_list:

                    # basis vectors (non-transpose) * standard_condition
                    basis_influence = np.dot(state.star.basis_matrix, g.vector)

                    center_value = np.dot(standard_center, g.vector)
                    remaining_value = g.value - center_value

                    lc = LinearConstraint(basis_influence, remaining_value)

                    discrete_poststate_star.temp_constraints.append(lc)

                if transition.reset_matrix is not None:
                    raise RuntimeError("only empty resets are currently supported")

                # there may be minor errors when converting the guard to the star's basis, so
                # re-check for feasibility

                if discrete_poststate_star.is_feasible():
                    violation_basis_vec = result[num_dims:]

                    if not self.settings.aggregation or not self.settings.deaggregation or \
                       self.has_counterexample(state.star, violation_basis_vec, state.star.total_steps):

                        # convert origin offset to star basis and add to basis_center
                        successor = discrete_poststate_star
                        #successor.start_center = successor.center()
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

        # star is
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
            add_guard_to_star(hull_star, first_star_parent.transition.condition_list)

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
                sim_bundle = self.sim_bundles[star.parent.mode.name]
                num_steps = star.fast_forward_steps + star.total_steps - star.parent.star.total_steps
                start_basis_matrix = star.start_basis_matrix

                self.plotman.plot_trace(num_steps, sim_bundle, start_basis_matrix, basis_point)

        return rv

    def presimulate(self, sim_bundle, point_in_star, inv_list):
        'this is an optimation where we try to guess the dwell time using a simulation, so we avoid repeated calls'

        origin_sim = sim_bundle.sim_until_inv_violated(point_in_star, inv_list, self.max_steps_remaining)
        num_presimulation_steps = int(len(origin_sim) * 1.2)

        if num_presimulation_steps > self.max_steps_remaining:
            num_presimulation_steps = self.max_steps_remaining

        self.cur_sim_bundle.get_vecs_origin_at_step(num_presimulation_steps, self.max_steps_remaining)

    def do_step_pop(self, output):
        'do a step where we pop from the waiting list'

        self.plotman.state_popped() # reset certain per-mode plot variables

        optutil.MultiOpt.reset_per_mode_vars()
        self.cur_step_in_mode = 0

        if output:
            self.waiting_list.print_stats()

        state = self.waiting_list.pop()

        if output:
            print "Removed state in mode '{}' at time {:.2f}; fast_forward_steps = {}".format(
                state.mode.name, state.star.total_steps * self.settings.step, state.star.fast_forward_steps)

        self.max_steps_remaining = self.settings.num_steps - state.star.total_steps + state.star.fast_forward_steps
        self.cur_sim_bundle = self.sim_bundles.get(state.mode.name)
        
        if self.cur_sim_bundle is None:
            self.cur_sim_bundle = SimulationBundle(state.mode.a_matrix, state.mode.b_vector, self.settings.step, 
                                                   sim_tol=self.settings.sim_tol)
            self.sim_bundles[state.mode.name] = self.cur_sim_bundle

            # optimization: presimulate sim bundle until after invariant becomes false
            if self.settings.use_presimulation:
                pt_in_star = np.array(state.star.get_feasible_point(), dtype=float)
                self.presimulate(self.cur_sim_bundle, pt_in_star, state.mode.inv_list)

        parent_star = state.star        
        state.star = parent_star.clone()
        state.star.parent = ContinuousPostParent(state.mode, parent_star)
        self.cur_state = state

        is_still_feasible = self.trim_to_invariant(state)

        if not is_still_feasible:
            self.cur_state = None

            if output:
                print "State after invariant trimming was not feasible; skipping."

        if self.cur_state is not None and self.settings.process_urgent_guards:
            self.check_guards(self.cur_state)

            if self.cur_state is None:
                if output:
                    print "After urgent checking guards, state was refined away."
            else:
                # cur_state is not Null
                print "Doing continuous post in mode '{}': ".format(self.cur_state.mode.name)

        if self.cur_state is not None and state.mode.is_null_dynamics():
            self.cur_state = None

            if output:
                print "Mode dynamics in '{}' were null; skipping.".format(state.mode.name)

        # pause after discrete post
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
                print "{}".format(self.settings.step * (self.cur_step_in_mode + 1))

            sim_step = self.cur_step_in_mode + 1 + state.star.fast_forward_steps

            sim_basis_matrix, sim_center = sim_bundle.get_vecs_origin_at_step(
                sim_step, self.max_steps_remaining)

            state.star.update_from_sim(sim_basis_matrix, sim_center)

            # increment step            
            self.cur_step_in_mode += 1
            state.star.total_steps += 1 

            self.check_guards(self.cur_state)

            # refinement may occur while checking guards, which sets cur_state to None
            if self.cur_state is None:
                if output:
                    print "After checking guards, state was refined away."
            else:
                is_still_feasible = self.trim_to_invariant(self.cur_state)

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
                self.do_step_continuous_post(output)

            if self.cur_state is not None:
                skipped_plot = self.plotman.plot_current_state(self.cur_state)

            if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE or \
                                    not skipped_plot or self.is_finished():
                break

    def run_to_completion(self):
        'run the computation until it finishes (without plotting)'
    
        Timers.tic("total")

        while not self.is_finished():
            self.do_step()

        Timers.toc("total")

        self.result.time = Timers.timers["total"].total_secs
        
    def run(self, init_list):
        '''
        run the computation
    
        init is a list of (LinearAutomatonMode, HyperRectangle)
        '''

        assert len(init_list) > 0

        # strengthen guards to inclunde invariants of targets
        if self.settings.use_guard_strengthening:
            ha = init_list[0][0].parent
            ha.do_guard_strengthening()

        self.result = HylaaResult()
        self.plotman.create_plot()

        # convert init states to stars
        self.load_waiting_list(init_list)

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            # proceed without plotting
            Timers.tic("total")
            
            while not self.is_finished():
                self.do_step()

            Timers.toc("total")
            Timers.print_time_stats()

            self.result.time = Timers.timers["total"].total_secs
        else:
            # plot during computation
            self.plotman.compute_and_animate(self.do_step, self.is_finished)
