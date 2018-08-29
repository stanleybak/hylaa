'''
Stanley Bak
Aggregation Directed Acyclic Graph (DAG) implementation
'''

from collections import namedtuple

import numpy as np

from termcolor import cprint

from hylaa.settings import HylaaSettings, AggregationSettings
from hylaa.util import Freezable
from hylaa.stateset import AggregationPredecessor, TransitionPredecessor, StateSet
from hylaa.timerutil import Timers
from hylaa import lputil

class AggDag(Freezable):
    'Aggregation directed acyclic graph (DAG) used to manage the deaggregation process'

    def __init__(self, settings, core):
        self.settings = settings
        self.core = core
        
        self.roots = [] # list of root AggDagNode where the computation begins
        self.cur_node = None # the aggdag_node currently under a continuous post operation

        self.waiting_list = [] # a list of StateSet objects
        self.node_parents = [] # a list of AggDagNode objects, the origin nodes of each of the waiting_list states
        
        self.freeze_attrs()

    def add_init_state(self, state):
        'add an initial state'

        self.waiting_list.append(state)
        self.node_parents.append(None)

    def add_to_waiting_list(self, state):
        'add a StateSet to the waiting list (a transition was feasible)'
        
        self.waiting_list.append(state)
        self.node_parents.append(self.cur_node)

    def print_waiting_list(self):
        'print out the waiting list'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            col = self.settings.stdout_colors[HylaaSettings.STDOUT_VERBOSE]
            
            cprint("Waiting list has {} states".format(len(self.waiting_list)), col)

            if len(self.waiting_list) < 10:
                for state in self.waiting_list:
                    cprint(" {}".format(state), col)

    @staticmethod
    def pop_score(state, strategy):
        '''
        returns a score used to decide which state to pop off the waiting list. The state with the highest score
        will be removed first.
        '''

        if strategy == AggregationSettings.POP_LOWEST_MINTIME:
            score = -(state.cur_steps_since_start[0])
        elif strategy == AggregationSettings.POP_LOWEST_AVGTIME:
            score = -(state.cur_steps_since_start[0] + state.cur_steps_since_start[1])
        elif strategy == AggregationSettings.POP_LARGEST_MAXTIME:
            score = state.cur_steps_since_start[1]
        else:
            raise RuntimeError("Unknown waiting list pop strategy: {}".format(strategy))

        return score

    def default_pop_func(self):
        '''
        Get the states to remove from the waiting list based on a score-based method
        '''

        # use score method to decide which mode to pop
        to_remove = self.waiting_list[0]
        to_remove_score = AggDag.pop_score(to_remove, self.settings.aggregation.pop_strategy)

        for state in self.waiting_list:
            score = AggDag.pop_score(state, self.settings.aggregation.pop_strategy)

            if score > to_remove_score or score == to_remove_score and state.mode.name < to_remove.mode.name:
                to_remove_score = score
                to_remove = state

        self.core.print_verbose("Aggregating with state at time {} in mode {}".format(to_remove.cur_steps_since_start,
                                                                                      to_remove.mode.name))

        # remove all states for aggregation
        agg_list = []

        for state in self.waiting_list:
            should_add = False

            if state is to_remove:
                should_add = True
            elif state.mode is to_remove.mode:
                if self.settings.aggregation.require_same_path:
                    if (isinstance(to_remove.predecessor, TransitionPredecessor) and \
                        isinstance(state.predecessor, TransitionPredecessor) and \
                        to_remove.predecessor.transition is state.predecessor.transition and \
                        to_remove.predecessor.state.computation_path_id == state.predecessor.state.computation_path_id):

                        should_add = True
                else: # require_same_path is False
                    should_add = True

            if should_add:
                agg_list.append(state)

        return agg_list

    def get_aggregation_states(self):
        '''get the states from the waiting list we are going to aggregate

        This also updates the internal data structures (waiting_list and AggDag nodes)

        returns agg_list, parent_node
        '''

        Timers.tic('get_aggregation_states')

        if self.settings.aggregation.custom_pop_func:
            self.core.print_verbose("Aggregating based on custom pop function")

            agg_list = self.settings.aggregation.custom_pop_func(self.waiting_list)
        else:
            agg_list = self.default_pop_func()

        new_waiting_list = []
        new_node_parents = []
        parent_node = None

        for state, node in zip(self.waiting_list, self.node_parents):
            if not state in agg_list:
                new_waiting_list.append(state)
                new_node_parents.append(node)
            elif parent_node is None:
                parent_node = node
            else:
                assert node is parent_node, "aggregation attempted from states with different aggdag parent nodes"

        self.core.print_verbose("agg_list had {} states, new_waiting_list has {} states".format(
            len(agg_list), len(new_waiting_list)))

        assert agg_list, "agg_list was empty"
        assert len(agg_list) + len(new_waiting_list) == len(self.waiting_list), "agg_list had new states in it?"

        self.waiting_list = new_waiting_list
        self.node_parents = new_node_parents

        Timers.toc('get_aggregation_states')

        return agg_list, parent_node

    def pop_waiting_list(self):
        'pop a state off the waiting list, possibly doing state-set aggregation'

        aggregated = False
        parent_node = None

        if self.settings.aggregation.agg_mode == AggregationSettings.AGG_NONE:
            agg_list = [self.waiting_list.pop(0)]
            parent_node = self.node_parents.pop(0)
        else:
            agg_list, parent_node = self.get_aggregation_states()
            
            self.core.print_verbose("Removed {} state{} for aggregation".format(
                len(agg_list), "s" if len(agg_list) > 1 else ""))

        if len(agg_list) == 1:
            rv = agg_list[0]
        else:
            rv = perform_aggregation(agg_list, self.settings.aggregation, self.core.print_debug)
            aggregated = True

        assert len(self.waiting_list) == len(self.node_parents)

        # create a new AggDagNode for the current computation
        self.cur_node = AggDagNode()

        if parent_node is None:
            self.roots.append(self.cur_node)
        else:
            # update transitions involving the aggregated states
            for state in agg_list:
                for op in parent_node.op_list:
                    if isinstance(op, OpTransition) and op.state is state:
                        assert op.child_node is None
                        op.child_node = self.cur_node

        if not aggregated:
            self.cur_node.concrete_state = rv
        else:
            self.cur_node.aggregated_state = rv

        return rv

class AggDagNode(Freezable): # pylint: disable=too-few-public-methods
    'A node of the Aggregation DAG'

    def __init__(self):
        self.op_list = [] # list of Op* objects

        self.concrete_state = None # StateSet
        self.aggregated_state = None # StateSet, or None if this is a non-aggergated set
        
        self.freeze_attrs()

# Operation types
OpInvIntersect = namedtuple('OpInvIntersect', ['step', 'mode', 'i_index', 'is_stronger'])
OpTransition = namedtuple('OpTransition', ['step', 'premode_lpi', 'transition', 'state', 'child_node'])
        
def perform_aggregation(agg_list, agg_settings, print_debug):
    '''
    perform aggregation on the passed-in list of states
    '''

    min_step = min([state.cur_steps_since_start[0] for state in agg_list])
    max_step = max([state.cur_steps_since_start[1] for state in agg_list])
    step_interval = [min_step, max_step]

    print_debug("Aggregation step interval: {}".format(step_interval))

    # create a new state from the aggregation
    postmode = agg_list[0].mode
    postmode_dims = postmode.a_csr.shape[0]
    mid_index = len(agg_list) // 2
    mid_state = agg_list[mid_index]
    pred = mid_state.predecessor

    if agg_settings.agg_mode == AggregationSettings.AGG_BOX or pred is None:
        agg_dir_mat = np.identity(postmode_dims)
    elif agg_settings.agg_mode == AggregationSettings.AGG_ARNOLDI_BOX:
        # aggregation with a predecessor, use arnoldi directions in predecessor mode in center of
        # middle aggregagted state, then project using the reset, and reorthogonalize

        assert isinstance(pred, TransitionPredecessor)
        premode = pred.state.mode
        pt = lputil.get_box_center(pred.premode_lpi)
        print_debug("aggregation point: {}".format(pt))

        premode_dir_mat = lputil.make_direction_matrix(pt, premode.a_csr)
        print_debug("premode dir mat:\n{}".format(premode_dir_mat))

        if pred.transition.reset_csr is None:
            agg_dir_mat = premode_dir_mat
        else:
            projected_dir_mat = premode_dir_mat * pred.transition.reset_csr.transpose()

            print_debug("projected dir mat:\n{}".format(projected_dir_mat))

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

    if pred and agg_settings.add_guard:
        # add all the guard conditions to the agg_dir_mat

        if pred.transition.reset_csr is None: # identity reset
            guard_dir_mat = pred.transition.guard_csr
        else:
            # multiply each direction in the guard by the guard
            guard_dir_mat = pred.transition.guard_csr * pred.transition.reset_csr.transpose()

        if guard_dir_mat.shape[0] > 0:
            agg_dir_mat = np.concatenate((agg_dir_mat, guard_dir_mat.toarray()), axis=0)
    else:
        raise RuntimeError("Unknown aggregation mode: {}".format(agg_settings.agg__mode))

    print_debug("agg dir mat:\n{}".format(agg_dir_mat))
    lpi_list = [state.lpi for state in agg_list]

    new_lpi = lputil.aggregate(lpi_list, agg_dir_mat, postmode)

    print("TODO: GET RID OF PREDECESSORS (USE AGGDAG ONLY)")
    predecessor = AggregationPredecessor(agg_list) # Note: these objects weren't clone()'d

    return StateSet(new_lpi, agg_list[0].mode, step_interval, predecessor)
