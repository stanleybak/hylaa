'''
Stanley Bak
Aggregation Directed Acyclic Graph (DAG) implementation
'''

from collections import namedtuple

import numpy as np

from termcolor import cprint

from graphviz import Digraph

from hylaa.settings import HylaaSettings
from hylaa.util import Freezable
from hylaa.stateset import StateSet
from hylaa import lputil, aggregate

# Operation types
OpInvIntersect = namedtuple('OpInvIntersect', ['step', 'node', 'i_index', 'is_stronger'])
OpLeftInvariant = namedtuple('OpLeftInvariant', ['step', 'node'])

class OpTransition(): # pylint: disable=too-few-public-methods
    'a transition operation'

    def __init__(self, step, parent_node, child_node, transition, poststate):
        self.step = step
        self.parent_node = parent_node
        self.child_node = child_node
        self.transition = transition
        self.poststate = poststate

class AggDag(Freezable):
    'Aggregation directed acyclic graph (DAG) used to manage the deaggregation process'

    def __init__(self, settings, core):
        self.settings = settings
        self.core = core
        
        self.roots = [] # list of root AggDagNode where the computation begins
        self.cur_node = None # the aggdag_node currently under a continuous post operation

        self.waiting_list = [] # a list of tuples: (StateSet, OpTransition)
        
        self.freeze_attrs()

    def get_cur_state(self):
        '''get the current state being propagated

        This may be None, an aggregated state, or a concrete state
        '''

        rv = None

        if self.cur_node is not None:
            rv = self.cur_node.get_cur_state()

        return rv

    def cur_state_left_invariant(self):
        '''called when the current state left the invariant'''

        state = self.get_cur_state()

        op = OpLeftInvariant(state.cur_step_in_mode, self.cur_node)
        self.cur_node.op_list.append(op)

        self.cur_node = None        

    def add_init_state(self, state):
        'add an initial state'

        self.waiting_list.append((state, None))

    def print_waiting_list(self):
        'print out the waiting list'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            col = self.settings.stdout_colors[HylaaSettings.STDOUT_VERBOSE]
            
            cprint("Waiting list has {} states".format(len(self.waiting_list)), col)

            if len(self.waiting_list) < 20:
                for state, op in self.waiting_list:
                    trans = ''

                    if isinstance(op, OpTransition):
                        trans = op.transition
                    
                    cprint(" [Mode: {} (from op {}) at steps {}]".format(state.mode.name, trans,
                                                                         state.cur_steps_since_start), col)

    def add_invariant_op(self, step, i_index, is_stronger):
        '''
        an invariant intersection occured, add the invariant intersection op to the current node's op list
        '''

        # OpInvIntersect = namedtuple('OpInvIntersect', ['step', 'node', 'i_index', 'is_stronger'])

        op = OpInvIntersect(step, self.cur_node, i_index, is_stronger)
        self.cur_node.op_list.append(op)

    def add_transition_successor(self, t, t_lpi):
        '''take the passed-in transition from the current state (add to the waiting list)

        t is the AutomatonTransition onject
        t_lpi is an LpInstance consisting of the states in the premode where the current set and guard intersect
        '''

        cur_state = self.get_cur_state()
        successor_has_inputs = t.to_mode.b_csr is not None

        op = OpTransition(cur_state.cur_step_in_mode, self.cur_node, None, t, None)

        self.settings.aggstrat.pretransition(t, t_lpi, op)

        lputil.add_reset_variables(t_lpi, t.to_mode.mode_id, t.transition_index, \
            reset_csr=t.reset_csr, minkowski_csr=t.reset_minkowski_csr, \
            minkowski_constraints_csr=t.reset_minkowski_constraints_csr, \
            minkowski_constraints_rhs=t.reset_minkowski_constraints_rhs, successor_has_inputs=successor_has_inputs)

        if not t_lpi.is_feasible():
            raise RuntimeError("cur_state became infeasible after reset was applied")

        op_list = [op]
        state = StateSet(t_lpi, t.to_mode, cur_state.cur_steps_since_start, op_list, cur_state.is_concrete)
        op.poststate = state

        self.cur_node.op_list.append(op)
        self.waiting_list.append((state, op))

    def pop_waiting_list(self):
        'pop a state off the waiting list, possibly doing state-set aggregation'

        aggregated = False

        agg_pair_list = self.settings.aggstrat.pop_waiting_list(self.waiting_list)

        assert agg_pair_list, "pop_waiting_list should return non-empty list"
        assert len(agg_pair_list[0]) == 2, "expected pop_waiting_list to return list of pairs: (StateSet, OpTransition)"
        assert isinstance(agg_pair_list[0][0], StateSet)
        assert isinstance(agg_pair_list[0][1], (OpTransition, type(None)))

        # remove each element of agg_pair_list from the waiting_list
        waiting_list_presize = len(self.waiting_list)
        self.waiting_list = [pair for pair in self.waiting_list if not pair in agg_pair_list]
        assert waiting_list_presize == len(self.waiting_list) + len(agg_pair_list), "pop_waiting_list returned " + \
                                                                            "elements not orignally in waiting_list"

        agg_list, op_list = zip(*agg_pair_list)

        if len(agg_list) == 1:
            rv = agg_list[0]
        else:
            self.core.print_verbose(f"Aggregating {len(agg_list)} states")
            aggregated = True

            at = self.settings.aggstrat.get_agg_type(agg_list, op_list)

            if at.is_chull:
                assert not at.is_box and not at.is_arnoldi_box

                rv = aggregate.aggregate_chull(agg_list, op_list, self.core.print_debug)

            elif at.is_box or at.is_arnoldi_box:
                rv = aggregate.aggregate_box_arnoldi(agg_list, op_list, at.is_box, at.is_arnoldi_box, at.add_guard,
                                                     self.core.print_debug)
            else:
                raise RuntimeError(f"Unsupported aggregation type: {at}")

        # create a new AggDagNode for the current computation
        self.cur_node = AggDagNode(op_list)

        # update the OpTransition objects with the child information
        all_none = True

        for op in op_list:
            if op:
                all_none = False
                op.child_node = self.cur_node

        if all_none:
            self.roots.append(self.cur_node)

        if not aggregated:
            self.cur_node.concrete_state = rv
        else:
            self.cur_node.aggregated_state = rv

        return rv

    def show(self, lr=True):
        'visualize the aggdag using graphviz'

        g = Digraph(name='aggdag')

        if lr:
            g.graph_attr['rankdir'] = 'LR'

        #g.edge_attr.update(arrowhead='dot', arrowsize='2')
        
        already_drawn_nodes = []

        for i, root in enumerate(self.roots):
            preroot = 'preroot{}'.format(i)
            g.node(preroot, style="invis")

            name = "node_{}".format(id(root))
            g.edge(preroot, name)

            root.viz(g, already_drawn_nodes)

        g.view(cleanup=True)

class AggDagNode(Freezable):
    'A node of the Aggregation DAG'

    def __init__(self, parent_ops):
        self.op_list = [] # list of Op* objects

        self.concrete_state = None # StateSet
        self.aggregated_state = None # StateSet, or None if this is a non-aggergated set

        # parent information
        self.parent_ops = parent_ops
        
        self.freeze_attrs()

    def get_mode(self):
        'get the mode for this aggdag node'

        return self.get_cur_state().mode

    def get_cur_state(self):
        'get the current state for this node (aggregated if it exists, else concrete)'

        if self.aggregated_state is not None:
            rv = self.aggregated_state
        else:
            rv = self.concrete_state

        return rv

    def viz(self, g, already_drawn_nodes):
        '''draw the aggdag node using the graphiz library

        g is the DiGraph object to draw to
        '''

        already_drawn_nodes.append(self)
        name = "node_{}".format(id(self))
        label = self.get_mode().name

        g.node(name, label=label)

        # collapse edges over multiple times into the same outgoing edge
        enabled_transitions = [] # contains tuples (child_name, step_list)

        for op in self.op_list:
            if isinstance(op, OpTransition):
                if op.child_node is None:

                    # invisible outgoing edge
                    invis_name = "out_{}".format(id(op))
                    g.node(invis_name, style="invis")

                    g.edge(name, invis_name)
                else:
                    child_name = "node_{}".format(id(op.child_node))
                    found = False

                    for pair in enabled_transitions:
                        if pair[0] == child_name:
                            pair[1].append(op.step)
                            found = True

                    if not found:
                        enabled_transitions.append((child_name, [op.step]))
            elif isinstance(op, OpLeftInvariant):
                # flush remaining enabled_transitions
                for child_name, step_list in enabled_transitions:
                    label = str(step_list[0]) if len(step_list) == 1 else "[{}, {}]".format(step_list[0], step_list[-1])
                    g.edge(name, child_name, label=label)
                
                # print left-nvariant edge
                invis_name = "out_{}".format(id(op))
                g.node(invis_name, style="invis")

                label = str(op.step)
                g.edge(name, invis_name, label=label, arrowhead='dot', style='dotted')

            ###############################
            # remove enabled transitions that are no longer enabled
            new_enabled_transitions = []

            for pair in enabled_transitions:
                child_name = pair[0]
                step_list = pair[1]

                if step_list[-1] < op.step:
                    # print it
                    label = str(step_list[0]) if len(step_list) == 1 else "[{}, {}]".format( \
                        step_list[0], step_list[-1])
                    g.edge(name, child_name, label=label)
                else:
                    # keep it
                    new_enabled_transitions.append(pair)

            enabled_transitions = new_enabled_transitions

        # print the children after the current node
        for op in [op for op in self.op_list if isinstance(op, OpTransition) and op.child_node is not None]:
            if not op.child_node in already_drawn_nodes:
                op.child_node.viz(g, already_drawn_nodes)

        # ['step', 'parent_node', 'child_node', 'transition', 'premode_center', 'postmode_state']
