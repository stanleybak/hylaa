'''
Stanley Bak
Aggregation Directed Acyclic Graph (DAG) implementation
'''

from termcolor import cprint

from graphviz import Digraph

from hylaa.settings import HylaaSettings
from hylaa.util import Freezable
from hylaa.stateset import StateSet
from hylaa.timerutil import Timers
from hylaa import lputil, aggregate
from hylaa.deaggregation import DeaggregationManager, OpInvIntersect, OpLeftInvariant, OpTransition

class AggDag(Freezable):
    'Aggregation directed acyclic graph (DAG) used to manage the deaggregation process'

    def __init__(self, settings, core):
        self.settings = settings
        self.core = core
        
        self.roots = [] # list of root AggDagNode where the computation begins
        self.cur_node = None # the aggdag_node currently under a continuous post operation

        self.waiting_list = [] # a list of OpTransition (with child_node = None). StateSet is in op.poststate

        self.deagg_man = DeaggregationManager(self)

        self.viz_count = 0
        
        self.freeze_attrs()

    @staticmethod
    def is_op_transition(op):
        'is the passed in op a OpTransition?'

        return isinstance(op, OpTransition)

    def get_cur_state(self):
        '''get the current state being propagated

        This may be None, or a StateSet
        '''

        rv = None

        if self.cur_node is not None:
            rv = self.cur_node.get_cur_state()

        return rv

    def cur_state_left_invariant(self, reached_time_bound=False):
        '''called when the current state left the invariant (or exceeds reach time bound)'''

        state = self.get_cur_state()

        op = OpLeftInvariant(state.cur_step_in_mode, self.cur_node, reached_time_bound)
        self.cur_node.op_list.append(op)

        self.cur_node = None

    def add_init_state(self, state):
        'add an initial state'

        op = OpTransition(0, None, None, None, state)

        self.waiting_list.append(op)

    def print_waiting_list(self):
        'print out the waiting list'

        if self.settings.stdout >= HylaaSettings.STDOUT_VERBOSE:
            col = self.settings.stdout_colors[HylaaSettings.STDOUT_VERBOSE]
            
            cprint("Waiting list has {} states".format(len(self.waiting_list)), col)

            if len(self.waiting_list) < 20:
                for op in self.waiting_list:
                    assert isinstance(op, OpTransition), f"expected OpTransition, but instead had: {op}"
                            
                    state = op.poststate
                    trans = ''

                    if isinstance(op, OpTransition):
                        trans = op.transition
                    
                    cprint(" [Mode: {} (from transition {}) at steps {}]".format(state.mode.name, trans,
                                                                                 state.cur_steps_since_start), col)

    def add_invariant_op(self, step, i_index, is_stronger):
        '''
        an invariant intersection occured, add the invariant intersection op to the current node's op list
        '''

        # OpInvIntersect = namedtuple('OpInvIntersect', ['step', 'node', 'i_index', 'is_stronger'])

        op = OpInvIntersect(step, self.cur_node, i_index, is_stronger)
        self.cur_node.op_list.append(op)

    def make_op_transition(self, t, t_lpi, state, parent_node):
        'make an OpTransition object, can return null if lp solving fails'

        step_in_mode = state.cur_step_in_mode
        steps_since_start = state.cur_steps_since_start
        is_concrete = state.is_concrete

        successor_has_inputs = t.to_mode.b_csr is not None

        op = OpTransition(step_in_mode, parent_node, None, t, None)

        succeeded = self.settings.aggstrat.pretransition(t, t_lpi, op)

        if not succeeded:
            self.core.print_verbose(f"Warning: aggstrat.pretransition returned None (LP solving failed)")
            op = None
        else:
            lputil.add_reset_variables(t_lpi, t.to_mode.mode_id, t.transition_index, \
                reset_csr=t.reset_csr, minkowski_csr=t.reset_minkowski_csr, \
                minkowski_constraints_csr=t.reset_minkowski_constraints_csr, \
                minkowski_constraints_rhs=t.reset_minkowski_constraints_rhs, successor_has_inputs=successor_has_inputs)

            if not t_lpi.is_feasible():
                raise RuntimeError("cur_state became infeasible after reset was applied")

            op_list = [op]
            state = StateSet(t_lpi, t.to_mode, steps_since_start, op_list, is_concrete)
            op.poststate = state

        return op

    def add_transition_successor(self, t, t_lpi, cur_state=None, cur_node=None):
        '''take the passed-in transition from the current state (add to the waiting list)

        t is the AutomatonTransition onject
        t_lpi is an LpInstance consisting of the states in the premode where the current set and guard intersect
        '''

        if cur_state is None:
            cur_state = self.get_cur_state()

        if cur_node is None:
            cur_node = self.cur_node

        op = self.make_op_transition(t, t_lpi, cur_state, cur_node)

        if op:
            cur_node.op_list.append(op)
            self.waiting_list.append(op)

    def _get_node_leaf_ops(self, node):
        'recursively get all the leaf ops originating from the given node'

        rv = []

        child_nodes = []

        for op in node.op_list:
            if isinstance(op, OpTransition):
                if op.child_node is None:
                    rv.append(op)
                elif not op.child_node in child_nodes: # don't insert the same node twice
                    child_nodes.append(op.child_node)

        for child_node in child_nodes:
            rv += self._get_node_leaf_ops(child_node)

        return rv

    def remove_node_decendants_from_waiting_list(self, node):
        'remove all waiting list states originating from the passed-in node'

        to_remove_ops = self._get_node_leaf_ops(node)

        new_waiting_list = []

        for op in self.waiting_list:
            if not op in to_remove_ops:
                new_waiting_list.append(op)

        self.waiting_list = new_waiting_list

    def pop_waiting_list(self):
        'pop a state off the waiting list, possibly doing state-set aggregation'

        op_list = self.settings.aggstrat.pop_waiting_list(self.waiting_list)

        assert op_list, "pop_waiting_list should return non-empty list"
        assert isinstance(op_list[0], OpTransition), f"expected OpTransition, but instead popped: {op_list[0]}"

        # if we're popping error mode, pop them one at a time
        # this is to prevent aggregation, which may use successor mode dynamics information
        if op_list[0].poststate.mode.a_csr is None: 
            op_list = [op_list[0]]

        # if we're popping a mode with no predecessor, an initial mode, pop one at a time
        for op in op_list:
            if op.parent_node is None:
                op_list = [op]
                break

        # remove each element of opr_list from the waiting_list
        waiting_list_presize = len(self.waiting_list)
        self.waiting_list = [op for op in self.waiting_list if not op in op_list]
        assert waiting_list_presize == len(self.waiting_list) + len(op_list), "pop_waiting_list returned " + \
                                                                            "elements not orignally in waiting_list"

        agg_type = None

        if len(op_list) > 1:
            agg_type = self.settings.aggstrat.get_agg_type(op_list)

        # create a new AggDagNode for the current computation
        self.cur_node = self.make_node(op_list, agg_type, 'full')

        return self.cur_node.get_cur_state()

    def make_node(self, ops, agg_type, aggstring):
        '''make an aggdag node

        aggstring is a string that describes the current aggregation from the previous transition
        'full' -> full aggergation
        'init' -> from initial state
        '0100' -> transition was split four times, this is the state from the first, second, first, first partitions
        '''

        node = AggDagNode(ops, agg_type, self, aggstring)

        #if plot:
        #    print(".aggdag make_node() plotting aggdagnode")
        #    self.core.plotman.add_plotted_states([node.stateset])
            
        return node

    def save_viz(self):
        '''save the viz to a sequentially-named file, returns the filename'''

        filename = f"aggdag_{self.viz_count:02d}"
        self.viz_count += 1
                
        self.viz(filename=filename)

        return filename

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
            print(f"vizzed to filename: {filename}")
            g.render(filename, cleanup=True)
        else:
            g.view(cleanup=True)

        # sanity check: all nodes in waiting list were drawn
        for op in self.waiting_list:
            if op.parent_node is not None:
                assert op.parent_node in already_drawn_nodes, f"parent node {op.parent_node} not found in drawn nodes"

class AggDagNode(Freezable):
    'A node of the Aggregation DAG'

    def __init__(self, parent_op_list, agg_type, aggdag, aggstring):
        self.aggdag = aggdag
        self.op_list = [] # list of Op* objects
        self.stateset = None # StateSet

        # parent information
        self.parent_ops = parent_op_list
        self.agg_type_from_parents = agg_type

        state_list = [op.poststate for op in parent_op_list]

        # make the stateset
        state = None
        
        if len(state_list) == 1:
            state = state_list[0]
        else:
            self.aggdag.core.print_verbose(f"Aggregating {len(state_list)} states")
            state = self._aggregate_from_state_op_list(state_list, parent_op_list, agg_type)

        state.aggstring = aggstring

        add_root = False

        for op in parent_op_list:
            # child node here may be None, or an existing child node that we override
            op.child_node = self
            
            if op.parent_node is None:
                add_root = True
            
        if add_root:
            self.aggdag.roots.append(self)

        self.stateset = state
        
        self.freeze_attrs()

    # override __hash__ and __eq__ so nodes can be keys in a dict
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return f"[AggDagNode in mode {self.stateset.mode.name} w/{len(self.parent_ops)} parent_ops, id={id(self)}]"

    def split(self):
        'for deaggreagtion, split the current (aggregated) node into two, returns a pair of nodes'

        assert len(self.parent_ops) > 1, "attempted to split() unaggregated aggdag node"

        parent_aggstring = self.stateset.aggstring

        if parent_aggstring == 'full':
            parent_aggstring = ''

        rv = []
        
        mid_index = len(self.parent_ops) // 2
        parent_op_lists = [self.parent_ops[:mid_index], self.parent_ops[mid_index:]]

        for agg_suffix, parent_op_list in zip(['0', '1'], parent_op_lists):
            aggstring = parent_aggstring + agg_suffix
            node = self.aggdag.make_node(parent_op_list, self.agg_type_from_parents, aggstring)

            rv.append(node)

        return rv

    def _aggregate_from_state_op_list(self, agg_list, op_list, agg_type):
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

    def node_left_invariant(self):
        '''
        has the stateset represented by the node become infeasible (left the invariant?)
        '''

        rv = False

        if self.op_list:
            rv = isinstance(self.op_list[-1], OpLeftInvariant)

        return rv

    def replay_op(self, op_list, i):
        '''
        replay a single operation in the current node
        this is used when nodes are split, to leverage parent information
        '''

        Timers.tic('replay_op')

        assert not self.node_left_invariant()

        op = op_list[i]

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

            #print(".aggdag %%%% debug never skipping invariant intersection")
            # hmm, skipping invariant intersections is only valid if deaggregation is a subset of aggregated set...
            # for our krylov aggregation, this isn't really the case though... so in general you need to invaraint
            # intersect at every step... plus this is really an optimization rather than the main algorithm
            
            if True or not skip:
                self.aggdag.core.print_verbose(
                    f"doing invariant intersection in replay at step {cur_state.cur_step_in_mode}")
                
                is_feasible = self.replay_op_intersect_invariant(cur_state, op)

                if not is_feasible:
                    op = OpLeftInvariant(op.step, self, False)
                    self.op_list.append(op)
            else:
                self.aggdag.core.print_verbose("skipping invariant intersection because stronger one is coming up")

        Timers.toc('replay_op')

    def replay_op_transition(self, state, op):
        '''replay a single operation of type OpTransition
        '''

        print_verbose = self.aggdag.core.print_verbose

        state.step(op.step) # advance the state first

        # check if the transition is still enabled
        
        t = op.transition

        t_lpi = t.get_guard_intersection(state.lpi)

        if t_lpi:
            if t.to_mode.is_error():
                self.aggdag.core.error_reached(state, t, t_lpi)

            new_op = self.aggdag.make_op_transition(t, t_lpi, state, self)

            if not new_op: # make_op_transition can fail due to numerical issues
                print_verbose("Replay Transition {} became infeasible after changing opt direction, skipping")
            else:
                self.op_list.append(new_op)

                if op.child_node is None:
                    self.aggdag.waiting_list.append(new_op)
                    print_verbose("Replay Transition {} when deaggreaged to steps {}".format( \
                                  t, state.cur_steps_since_start))
                else:
                    self.aggdag.deagg_man.update_transition_successors(op, new_op)

                    print_verbose("Replay Transition refined transition {} when deaggregated at steps {}".format( \
                                  t, state.cur_steps_since_start))
        else:
            print_verbose(f"Replay skipped transition {t} when deaggregated to steps {state.cur_steps_since_start}")

    def replay_op_intersect_invariant(self, state, op):
        '''replay a single operation of type OpInvIntersect

        This returns a boolean: is the current state set is still feasible?
        '''

        step, _, invariant_index, is_stronger = op

        state.step(step)

        lc = state.mode.inv_list[invariant_index]

        has_intersection = lputil.check_intersection(state.lpi, lc.negate())
        
        if has_intersection is None:
            rv = False # not feasible
        elif not has_intersection:
            rv = True # no intersection and still feasible
        else: 
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

            rv = state.lpi.is_feasible()

        return rv

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

        # sanity check that all parent nodes match
        for parent_op in self.parent_ops:
            assert parent_op.child_node is self

        already_drawn_nodes.append(self)
        name = "node_{}".format(id(self))
        label = self.stateset.mode.name

        if self.op_list and isinstance(self.op_list[-1], OpLeftInvariant):
            steps = self.op_list[-1].step

            if self.op_list[-1].reached_time_bound:
                label += f" ({steps}*)"
            else:
                label += f" ({steps})"
        else:
            steps = self.stateset.cur_step_in_mode
            label += f" (incomplete, {steps} so far)"

        print(f"vizing node {name} ({label}) with {len(self.parent_ops)} parent ops")
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
