'''
Stanley Bak
Aggregation Directed Acyclic Graph (DAG) implementation
'''

from collections import namedtuple

from termcolor import cprint

from graphviz import Digraph

from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.util import Freezable
from hylaa.stateset import StateSet
from hylaa.timerutil import Timers
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

    def __str__(self):
        return f"[OpTransition({self.step}, {self.transition})]"

class AggDag(Freezable):
    'Aggregation directed acyclic graph (DAG) used to manage the deaggregation process'

    def __init__(self, settings, core):
        self.settings = settings
        self.core = core
        
        self.roots = [] # list of root AggDagNode where the computation begins
        self.cur_node = None # the aggdag_node currently under a continuous post operation

        self.waiting_list = [] # a list of tuples: (StateSet, OpTransition)

        self.viz_count = 0
        
        self.freeze_attrs()

    def get_cur_state(self):
        '''get the current state being propagated

        This may be None, or a StateSet
        '''

        rv = None

        if self.cur_node is not None:
            rv = self.cur_node.get_cur_state()

        return rv

    def cur_state_left_invariant(self):
        '''called when the current state left the invariant (or exceeds reach time bound)'''

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

    def add_transition_successor(self, t, t_lpi, cur_state=None, cur_node=None):
        '''take the passed-in transition from the current state (add to the waiting list)

        t is the AutomatonTransition onject
        t_lpi is an LpInstance consisting of the states in the premode where the current set and guard intersect
        '''

        if cur_state is None:
            cur_state = self.get_cur_state()

        if cur_node is None:
            cur_node = self.cur_node
            
        successor_has_inputs = t.to_mode.b_csr is not None

        op = OpTransition(cur_state.cur_step_in_mode, cur_node, None, t, None)

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

        cur_node.op_list.append(op)
        
        self.waiting_list.append((state, op))

    def pop_waiting_list(self):
        'pop a state off the waiting list, possibly doing state-set aggregation'

        agg_pair_list = self.settings.aggstrat.pop_waiting_list(self.waiting_list)

        assert agg_pair_list, "pop_waiting_list should return non-empty list"
        assert len(agg_pair_list[0]) == 2, "expected pop_waiting_list to return list of pairs: (StateSet, OpTransition)"
        assert isinstance(agg_pair_list[0][0], StateSet)
        assert isinstance(agg_pair_list[0][1], (OpTransition, type(None)))

        # if we're popping error mode, pop them one at a time
        # this is to prevent aggregation, which may use sucessor mode dynamics information
        if agg_pair_list[0][0].mode.a_csr is None: 
            agg_pair_list = [agg_pair_list[0]]

        # remove each element of agg_pair_list from the waiting_list
        waiting_list_presize = len(self.waiting_list)
        self.waiting_list = [pair for pair in self.waiting_list if not pair in agg_pair_list]
        assert waiting_list_presize == len(self.waiting_list) + len(agg_pair_list), "pop_waiting_list returned " + \
                                                                            "elements not orignally in waiting_list"

        agg_list, op_list = zip(*agg_pair_list)
        agg_type = None

        if len(agg_list) > 1:
            agg_type = self.settings.aggstrat.get_agg_type(agg_list, op_list)

        # create a new AggDagNode for the current computation
        self.cur_node = AggDagNode(agg_list, op_list, agg_type, self)

        return self.cur_node.get_cur_state()

    def save_viz(self):
        'save the viz to a sequentially-named file'

        self.viz(filename=f"viz{self.viz_count}")

        self.viz_count += 1

    def viz(self, lr=True, filename=None):
        'visualize the aggdag using graphviz'

        if filename is not None:
            g = Digraph(name='aggdag', format='png')
        else:
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

        if filename is not None:
            g.render(filename, cleanup=True)
        else:
            g.view(cleanup=True)

class AggDagNode(Freezable):
    'A node of the Aggregation DAG'

    def __init__(self, state_list, parent_op_list, agg_type, aggdag):
        self.aggdag = aggdag
        self.op_list = [] # list of Op* objects

        self.stateset = None # StateSet

        # parent information
        self.parent_ops = parent_op_list
        self.agg_type_from_parents = agg_type

        # make the stateset
        state = None
        
        if len(state_list) == 1:
            state = state_list[0]
        else:
            self.aggdag.core.print_verbose(f"Aggregating {len(state_list)} states")
            state = self.aggregate_from_state_op_list(state_list, parent_op_list, agg_type)

            print(f". aggregated state in mode {state.mode.name}")

        print(f". created AggDagNode from {len(parent_op_list)} ops in state {state.mode.name} with agg_type {agg_type}")
                
        # update the OpTransition objects with the child information
        all_none = True

        for op in parent_op_list:
            if op:
                all_none = False
                op.child_node = self

        # if all the parent transitions are None, this is a root node and must be stored as such
        if all_none:
            self.aggdag.roots.append(self)

        self.stateset = state
        
        self.freeze_attrs()

    def aggregate_from_state_op_list(self, agg_list, op_list, agg_type):
        '''aggregate states into a single state from a list of states and operatons

        agg_list - a list of states
        op_list - a list of operations
        agg_type the AggType tuple, the aggregation type to use

        returns a single StateSet which is the desired aggregation
        '''

        at = agg_type

        if at.is_chull:
            assert not at.is_box and not at.is_arnoldi_box

            rv = aggregate.aggregate_chull(agg_list, op_list, self.aggdag.core.print_debug)

        elif at.is_box or at.is_arnoldi_box:
            rv = aggregate.aggregate_box_arnoldi(agg_list, op_list, at.is_box, at.is_arnoldi_box, at.add_guard,
                                                 self.aggdag.core.print_debug)
        else:
            raise RuntimeError(f"Unsupported aggregation type: {at}")

        return rv

    def refine_split(self, agg_type):
        '''refine this aggdag node by splitting its aggregated set in two

        this returns an action list used for updating the plot frame-by-frame (if plotting is enabled)
        for action list documentaton see aggstrat.pre_pop_waiting_list
        '''

        self.aggdag.save_viz()

        actions = []

        mid_index = len(self.parent_ops) // 2
        parent_op_lists = [self.parent_ops[:mid_index], self.parent_ops[mid_index:]]

        # we only support splitting on leaf nodes currently (need to implement recursive version)
        for op in self.op_list:
            if isinstance(op, OpTransition):
                assert op.child_node is None, "refine_split currently only implemented for leaf nodes"

                # remove this entry from waiting list
                removed = False
                
                for i, pair in enumerate(self.aggdag.waiting_list):
                    _, op_trans = pair
                    if op_trans is op:
                        del self.aggdag.waiting_list[i]
                        removed = True
                        break

                assert removed, "op_transition not found in waiting list?"

        split_nodes = []
        plot_states = []

        for parent_op_list in parent_op_lists:
            # for each of the two split sets

            agg_list = [op.poststate for op in parent_op_list]
            node = AggDagNode(agg_list, parent_op_list, agg_type, self.aggdag)
            split_nodes.append(node)

            # plot node and split nodes
            plot_states.append(node.stateset)

            #self.aggdag.core.plotman.add_reachable_poly(node.stateset)
            actions.append((self._replay_split_op_list, (split_nodes, 0)))

        if self.aggdag.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            # plot a copy of self at step zero
            full_parent_op_list = self.parent_ops
            full_agg_list = [op.poststate for op in full_parent_op_list]

            self_copy = AggDagNode(full_agg_list, full_parent_op_list, self.agg_type_from_parents, self.aggdag)
            self.aggdag.core.plotman.highlight_states_gray([self_copy.stateset])
            
        # plot plot_states
        self.aggdag.core.plotman.highlight_states(plot_states)

        return actions

    def _replay_split_op_list(self, split_nodes, replay_start_step):
        '''replay the op list when splitting, up to the next transition

        returns new_action_list for the next call
        '''

        actions = []

        for i in range(replay_start_step, len(self.op_list)):
            op = self.op_list[i]
            
            for node in split_nodes:
                if not node.op_list or not isinstance(node.op_list[-1], OpLeftInvariant):
                    node.replay_op(i, op, self.op_list)

                    if node is split_nodes[0]:
                        print(f". new basis matrix:\n{node.stateset.basis_matrix}")

            if isinstance(op, OpTransition) and self.stateset.step_to_paths: # step_to_paths is None if plotting is off
                # draw the spliting and use action_list to delay further processing

                # first clear the old plotted state
                verts = self.stateset.del_plot_path(op.step)

                # plot the verts of the deleted old plotted state
                self.aggdag.core.plotman.highlight_states_gray([verts])
                
                state_list = []

                # also plot the new states (if they're feasible)
                for node in split_nodes:
                    if not node.op_list or not isinstance(node.op_list[-1], OpLeftInvariant):
                        state_list.append(node.stateset)

                        self.aggdag.core.plotman.add_reachable_poly(node.stateset)

                self.aggdag.core.plotman.highlight_states(state_list)

                # delay further processing the op list so we can draw a frame
                actions.append((self._replay_split_op_list, (split_nodes, i+1)))
                self.aggdag.save_viz()
                        
                break

        print("returning actions len {}".format(len(actions)))
        
        return actions

    def replay_op(self, i, op, op_list):
        '''
        replay a single operation in the current node
        this is used when nodes are split, to leverage parent information
        '''
        
        Timers.tic('replay_op')

        cur_state = self.get_cur_state()
        assert cur_state is not None

        self.aggdag.core.print_verbose(f"replaying {i}: {op}")

        if isinstance(op, OpLeftInvariant):
            self.op_list.append(op)
        elif isinstance(op, OpTransition):
            self.replay_op_transition(cur_state, op)
        elif isinstance(op, OpInvIntersect):
            # if there is a later invariant intersection with the same hyperplane and it's stronger, skip this one

            skip = False

            for future_i in range(i+1, len(op_list)):
                future_op = op_list[future_i]

                if not isinstance(future_op, OpInvIntersect):
                    # another op (such as a transition, can't skip current one)
                    break

                if future_op.i_index == op.i_index:
                    if future_op.is_stronger:
                        skip = True

                    break

            if not skip:                   
                is_feasible = self.replay_op_intersect_invariant(cur_state, op)

                if not is_feasible:
                    op = OpLeftInvariant(op.step, self)
                    self.op_list.append(op)

        Timers.toc('replay_op')

    def replay_op_transition(self, state, op):
        '''replay a single operation of type OpTransition
        '''

        assert op.child_node is None, "replay op_transition currently only implemented for leaf nodes"
        print_verbose = self.aggdag.core.print_verbose

        state.step(op.step)
        # check if the transition is still enabled
        
        t = op.transition

        t_lpi = t.get_guard_intersection(state.lpi)

        if t_lpi:
            if t.to_mode.is_error():
                self.aggdag.core.error_reached(state, t, t_lpi)

            self.aggdag.add_transition_successor(t, t_lpi, state, self)
            
            print_verbose("Replay Transition Added Discrete Successor to '{}' at step {}".format( \
                          t.to_mode.name, state.cur_steps_since_start))
        else:
            print_verbose(f"Replay skipped transition at step {state.cur_steps_since_start}")

    def replay_op_intersect_invariant(self, state, op):
        '''replay a single operation of type OpInvIntersect

        This returns a boolean: is the current state set is still feasible?
        '''

        step, _, invariant_index, is_stronger = op

        state.step(step)

        lc = state.mode.inv_list[invariant_index]
        
        if lputil.check_intersection(state.lpi, lc.negate()):
            old_row = state.invariant_constraint_rows[invariant_index]
            vec = lc.csr.toarray()[0]
            rhs = lc.rhs

            if old_row is None:
                # new constraint
                row = lputil.add_init_constraint(state.lpi, vec, rhs, state.basis_matrix,
                                                 state.input_effects_list)
                state.invariant_constraint_rows[invariant_index] = row
                is_stronger = False
            else:
                # strengthen existing constraint possibly
                row, is_stronger = lputil.try_replace_init_constraint(state.lpi, old_row, vec, rhs, \
                    state.basis_matrix, state.input_effects_list)
                state.invariant_constraint_rows[invariant_index] = row

            # ad the op to the aggdag
            op = OpInvIntersect(step, self, invariant_index, is_stronger)
            self.op_list.append(op)

        return state.lpi.is_feasible()

    def get_cur_state(self):
        'get the current state for this node (None if it has left the invariant)'

        if self.op_list and isinstance(self.op_list[-1], OpLeftInvariant):
            rv = None
        else:
            rv = self.stateset

        return rv

    def viz(self, g, already_drawn_nodes):
        '''draw the aggdag node using the graphiz library

        g is the DiGraph object to draw to
        '''

        already_drawn_nodes.append(self)
        name = "node_{}".format(id(self))
        label = self.stateset.mode.name

        if self.op_list and isinstance(self.op_list[-1], OpLeftInvariant):
            steps = self.op_list[-1].step
            label += f" ({steps})"

        print(f"vizing node {name} ({label})")
        g.node(name, label=label)

        # collapse edges over multiple times into the same outgoing edge
        enabled_transitions = [] # contains tuples (child_name, op_list)

        for op in self.op_list:
            if isinstance(op, OpTransition):
                if op.child_node is None:
                    # unprocessed transition child node
                    child_name = f"out_{id(self)}_{id(op.transition)}"
                else:
                    child_name = "node_{}".format(id(op.child_node))
                    
                found = False

                for pair in enabled_transitions:
                    if pair[0] == child_name:
                        pair[1].append(op)
                        found = True

                if not found:
                    enabled_transitions.append((child_name, [op]))
            elif isinstance(op, OpLeftInvariant):
                # flush remaining enabled_transitions
                for child_name, op_list in enabled_transitions:
                    if len(op_list) == 1:
                        label = str(op_list[0].step)
                    else:
                        label = "[{}, {}]".format(op_list[0].step, op_list[-1].step)

                    if child_name.startswith('out_'): # outgoing edges to unprocessed nodes
                        to_mode_name = op_list[0].transition.to_mode.name
                        g.node(child_name, label=to_mode_name, style="dashed")
                    
                    g.edge(name, child_name, label=label)

                enabled_transitions = []
                # print left-invariant edge
                #invis_name = "out_{}".format(id(op))
                #g.node(invis_name, style="invis")

                #label = str(op.step)
                #g.edge(name, invis_name, label=label, arrowhead='dot', style='dotted')

            ###############################
            # remove enabled transitions that are no longer enabled
            new_enabled_transitions = []

            for pair in enabled_transitions:
                child_name = pair[0]
                op_list = pair[1]

                if op_list[-1].step < op.step - 1:
                    # print it
                    if len(op_list) == 1:
                        label = str(op_list[0].step)
                    else:
                        label = "[{}, {}]".format(op_list[0].step, op_list[-1].step)

                    if child_name.startswith('out_'): # outgoing edges to unprocessed nodes
                        to_mode_name = op_list[0].transition.to_mode.name
                        g.node(child_name, label=to_mode_name, style="dashed")
                        
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
