'''
Deaggregation Management implementation

Stanley Bak
Nov 2018
'''

from collections import deque, namedtuple

from hylaa.util import Freezable
from hylaa.timerutil import Timers

# Operation types
OpInvIntersect = namedtuple('OpInvIntersect', ['step', 'node', 'i_index', 'is_stronger'])
OpLeftInvariant = namedtuple('OpLeftInvariant', ['step', 'node', 'reached_time_bound'])

class OpTransition(): # pylint: disable=too-few-public-methods
    'a transition operation'

    def __init__(self, step, parent_node, child_node, transition, poststate):
        self.step = step
        self.parent_node = parent_node
        self.child_node = child_node
        self.transition = transition
        self.poststate = poststate

    def __str__(self):
        return f"[OpTransition({self.step}, {self.transition})]"

class DeaggregationManager(Freezable):
    'manager for deaggregation data'

    def __init__(self, aggdag):
        self.aggdag = aggdag
        
        self.waiting_nodes = deque() # a deque of 2-tuples (parent, children_list) of AggDagNodes awaiting deaggregation

        # associated with the current computaiton
        self.deagg_parent = None # a AggDagNode
        self.deagg_children = None # a list of AggDagNode

        # during replay, transition sucessors may have common children nodes that got aggregated
        # this maps the overapproximation node to the new unaggregated components (list of stateset and list of ops)
        # str(AggDagNode.id()) -> list_of_TransitionOps 
        self.nodes_to_ops = None # maps (newop.parent, oldop.child) to list of transtion_ops

        self.replay_step = None # current step during a replay

    def doing_replay(self):
        'are we in the middle of a refinement replay?'

        return self.replay_step is not None or self.waiting_nodes

    def plot_replay_deaggregated(self, cur_step_in_mode):
        '''plot the updated states during a replay

        child_plot_mask is a list of booleans, indicating which deagg chidlren should be plotted.
        plots may be skipped if the child left the invariant on an earlier step

        note: invariant can be false for one step
        '''

        plotman = self.aggdag.core.plotman

        # add children plots
        plot_child_states = []
        
        for i, child in enumerate(self.deagg_children):
            if not child.node_left_invariant():
                plot_child_states.append(child.stateset)
                assert child.stateset.cur_step_in_mode == cur_step_in_mode, \
                  f"expected cur_step_in_mode {cur_step_in_mode}, but it was {child.stateset.cur_step_in_mode}"

        plotman.add_plotted_states(plot_child_states)

        # delete parent plot
        plotman.delete_plotted_state(self.deagg_parent.stateset, cur_step_in_mode)

    def init_replay(self):
        'initialize a replay action'

        assert self.replay_step is None

        self.deagg_parent, self.deagg_children = self.waiting_nodes.popleft()
        self.replay_step = 0
        self.nodes_to_ops = {}

        # pause plot
        self.aggdag.core.print_verbose("Pausing due to start of do_step_replay()")
        self.aggdag.core.plotman.pause()

    def do_step_replay(self):
        'do a refinement replay step'

        assert self.doing_replay()

        if self.replay_step is None:
            self.init_replay()
        
        assert self.replay_step is not None

        op_list = self.deagg_parent.op_list
        op = op_list[self.replay_step]
        cur_step_in_mode = op.step

        # This part is a bit complicated due to discrete-time sematnics.
        # Basically, in continuous post the operation order is:
        # 1. inv intersect
        # 2. step()
        # 3. check transitions
        # 4. plot
        #
        # in order to recreate this during a replay, we need conditional branching:
        # if first op is transition, process all transitions [which does step() first] at the current step,
        #                                                    but no invariant intersections
        # if first op is invariant intersection, process all ops at the current step and transitions at the next step
        #  
        only_process_transitions = isinstance(op_list[self.replay_step], OpTransition)

        parent_left_invariant = False
        
        for op in op_list[self.replay_step:]:
            if op.step > cur_step_in_mode and not only_process_transitions:
                cur_step_in_mode += 1
                only_process_transitions = True
            
            if op.step != cur_step_in_mode:
                #print(".deagg breaking because next op step is not cur_step_in_mode")
                break

            if only_process_transitions and not isinstance(op, OpTransition):
                #print(f".deagg breaking because next op is not OpTransition: {op}")
                break

            #print(f".deagg replaying step {self.replay_step}/{len(self.deagg_parent.op_list)}: {op}")

            if isinstance(op, OpLeftInvariant):
                parent_left_invariant = True

            for child in self.deagg_children:
                if not child.node_left_invariant():
                    child.replay_op(self.deagg_parent.op_list, self.replay_step)
                else:
                    self.aggdag.core.print_verbose("child has left invariant")

            self.replay_step += 1

        # advance the children to the current step if in case we only had inv_intersection operations
        for child in self.deagg_children:
            if not child.node_left_invariant():
                child.stateset.step(cur_step_in_mode)

        if not parent_left_invariant:
            # do the plot after the replay actions, as this can affect the plot (for example, invariant intersection)
            self.plot_replay_deaggregated(cur_step_in_mode)

        was_last_step = self.replay_step >= len(self.deagg_parent.op_list)

        if was_last_step or all([c.node_left_invariant() for c in self.deagg_children]):
            # update recursive children
            
            for pair, ops in self.nodes_to_ops.items():
                _, child = pair
                self.aggdag.core.print_verbose(f"making node for recursive deaggregation with t={ops[0].transition}")

                # aggregate all ops into a single node, using same aggregation as before
                node = self.aggdag.make_node(ops, child.agg_type_from_parents, child.stateset.aggstring)
                
                self.waiting_nodes.append((child, [node]))
                
            self.deagg_parent = self.deagg_children = self.replay_step = self.nodes_to_ops = None

            #print(".deaggregation calling aggdag.save_viz()")
            #self.aggdag.save_viz()

    def update_transition_successors(self, old_op, new_op):
        '''
        during a replay, update a node's successsors recursively
        '''

        # aggregation will only be done if both new_op.parent and old_op.child match
        parent_node = new_op.parent_node
        child_node = old_op.child_node
        pair = (parent_node, child_node)

        if pair not in self.nodes_to_ops:
            ops = []
            self.nodes_to_ops[pair] = ops
        else:
            ops = self.nodes_to_ops[pair]

        ops.append(new_op)

    def begin_replay(self, node):
        'begin a deaggregation replay with the passed-in node'

        Timers.tic('begin deagg replay')

        # remove all states in the waiting list that come from this node
        self.aggdag.remove_node_decendants_from_waiting_list(node)

        # start to populate waiting_nodes
        self.waiting_nodes.append((node, node.split()))

        Timers.toc('begin deagg replay')
