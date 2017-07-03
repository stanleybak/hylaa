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
from hylaa.glpk_interface import LpInstance

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

    def check_guards(self, state):
        'check for discrete successors with the guards'

        assert state is not None

        for i in xrange(len(state.mode.transitions)):
            lp_solution = state.get_guard_intersection(i)

            if lp_solution is not None:
                transition = state.mode.transitions[i]
                successor_mode = transition.to_mode

                if successor_mode.is_error:
                    self.reached_error = True

                    if self.settings.stop_when_error_reachable:
                        raise FoundErrorTrajectory("Found error trajectory")

                # copy the current star to be the frozen pre-state of the discrete post operation
                discrete_prestate_star = state.clone()
                discrete_prestate_star.parent = ContinuousPostParent(state.mode, state.parent.star)

                discrete_poststate_star = state.clone()
                basis_center = state.vector_to_star_basis(state.center)

                discrete_poststate_star.parent = DiscretePostParent(state.mode, discrete_prestate_star,
                                                                    basis_center, transition)

                # add each of the guard conditions to discrete_poststate_star
                for lin_con in transition.condition_list:

                    # first, convert the condition to the star's basis

                    # basis vectors (non-transpose) * standard_condition
                    basis_influence = np.dot(state.basis_matrix, lin_con.vector)
                    center_value = np.dot(state.center, lin_con.vector)
                    remaining_value = lin_con.value - center_value

                    basis_lc = LinearConstraint(basis_influence, remaining_value)
                    discrete_poststate_star.add_basis_constraint(basis_lc)

                if transition.reset_matrix is not None:
                    raise RuntimeError("only empty resets are currently supported")

                # there may be minor errors when converting the guard to the star's basis, so
                # re-check for feasibility

                if discrete_poststate_star.is_feasible():
                    violation_basis_vec = lp_solution[state.num_dims:]

                    if not self.settings.aggregation or not self.settings.deaggregation or \
                       self.has_counterexample(state, violation_basis_vec, state.total_steps):

                        # convert origin offset to star basis and add to basis_center
                        successor = discrete_poststate_star
                        #successor.start_center = successor.center

                        successor.center_into_constraints(basis_center)

                        #self.plotman.cache_star_verts(successor) # do this before committing temp constriants

                        successor.start_basis_matrix = state.basis_matrix
                        #successor.basis_matrix = None # gets assigned from sim_bundle on pop

                        successor.mode = transition.to_mode

                        if self.settings.aggregation:
                            self.waiting_list.add_aggregated(successor, self.settings)
                        else:
                            self.waiting_list.add_deaggregated(successor)
                    else:
                        # a refinement occurred, stop processing guards
                        self.cur_star = state = None
                        break

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

            #self.check_guards()

    def do_step(self):
        'do a single step of the computation'

        skipped_plot = False # if we skip the plot, do multiple steps

        while True:

            try:
                self.do_step_continuous_post()
            except FoundErrorTrajectory: # this gets raised if an error mode is reachable and we should quit early
                pass

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

        self.result = HylaaResult()
        self.plotman.create_plot()

        self.cur_star = init_star

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            # run without plotting
            Timers.tic("total")

            while not self.is_finished():
                self.do_step()

            Timers.toc("total")

            if self.settings.print_output:
                LpInstance.print_stats()
                Timers.print_stats()
        else:
            # use plot (will print states after completion (before gui closes)
            self.plotman.compute_and_animate(self.do_step, self.is_finished)

        self.result.time = Timers.timers["total"].total_secs

class FoundErrorTrajectory(RuntimeError):
    'gets thrown if a trajectory to the error states is found when settings.stop_when_error_reachable is True'
