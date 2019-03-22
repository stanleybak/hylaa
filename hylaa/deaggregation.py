'''
Deaggregation Management implementation

Stanley Bak
Nov 2018
'''

from collections import deque

from hylaa.util import Freezable

class DeaggregationManager(Freezable):
    'manager for deaggregation data'

    def __init__(self, aggdag, plot_on):
        self.aggdag = aggdag
        self.plot_on = plot_on
        
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

    def do_step_replay(self):
        'do a refinement replay step'

        assert self.doing_replay()

        # init
        if self.replay_step is None:
            self.deagg_parent, self.deagg_children = self.waiting_nodes.popleft()
            self.replay_step = 0
            self.nodes_to_ops = {}

        # replay up to the next OpTransition
        finished_replay = True
        
        for i in range(self.replay_step, len(self.deagg_parent.op_list)):
            op = self.deagg_parent.op_list[i]
            is_transition = self.aggdag.is_op_transition(op)

            for child in self.deagg_children:
                if not child.node_left_invariant():
                    child.replay_op(self.deagg_parent.op_list, i)
                else:
                    self.aggdag.core.print_verbose("child has left invariant")

            if is_transition:
                should_break = self.post_replay_op_transition(op)

                if should_break:
                    # break (to draw a frame)
                    self.replay_step = i+1
                    finished_replay = False
                    break

        if finished_replay:
            # update recursive children
            plot_state_list = []
            
            for pair, ops in self.nodes_to_ops.items():
                _, child = pair
                self.aggdag.core.print_verbose(f"making node for recursive deaggregation with t={ops[0].transition}")

                # aggregate all ops into a single node
                node = self.aggdag.make_node(ops, child.agg_type_from_parents)
                
                self.waiting_nodes.append((child, [node]))
                plot_state_list.append(node.stateset)

            if self.plot_on:
                self.aggdag.core.plotman.highlight_states(plot_state_list)
                
            self.deagg_parent = self.deagg_children = self.replay_step = self.nodes_to_ops = None

            #print(".deaggregation calling aggdag.save_viz()")
            #self.aggdag.save_viz()

        if self.doing_replay():
            self.aggdag.core.plotman.interactive.paused = True
            self.aggdag.core.print_verbose("Pausing due to step_replay()")

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

    def post_replay_op_transition(self, op):
        '''
        perform operations after an op transition. returns true if we should break the replay loop (to draw a frame).
        '''

        should_break = False

        # update plot
        if self.deagg_parent.stateset.step_to_paths: # step_to_paths is None if plotting is off
            # first clear the old plotted state
            verts = self.deagg_parent.stateset.del_plot_path(op.step)

            # plot the verts of the deleted old plotted state
            self.aggdag.core.plotman.highlight_states_gray([verts])

            plot_state_list = []

            # also plot the new states (if they're feasible)
            for child in self.deagg_children:
                if not child.node_left_invariant():
                    plot_state_list.append(child.stateset)
                    self.aggdag.core.plotman.add_reachable_poly(child.stateset)

            self.aggdag.core.plotman.highlight_states(plot_state_list)

        # always break (to get an accurate frame count)
        should_break = True

        return should_break

    def begin_replay(self, node):
        'begin a deaggregation replay with the passed-in node'

        # remove all states in the waiting list that come from this node
        self.aggdag.remove_node_from_waiting_list(node)

        self.waiting_nodes.append((node, node.split()))
