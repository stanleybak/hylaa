'''
Deaggregation Management implementation

Stanley Bak
Nov 2018
'''

class DeaggregationManager(Freezable):
    'manager for deaggregation data'

    def __init__(self, aggdag):
        self.aggdag = aggdag
        
        self.waiting_nodes = [] # a list of 2-tuples (parent, children_list) of AggDagNodes awaiting deaggregation

        # associated with the current computaiton
        self.deagg_parent = None
        self.deagg_children = None

        # during replay, transition sucessors may have common children nodes that got aggregated
        # this maps the overapproximation node to the new unaggregated components (list of stateset and list of ops)
        # str(AggDagNode.id()) -> (list_of_StateSet, list_of_TransitionOps) 
        self.overapprox_to_states_ops = None # maps old parent aggregated node to list of new children

        self.replay_step = None # current step during a replay

    def doing_replay(self):
        'are we in the middle of a refinement replay?'

        return self.replay_step is not None or self.waiting_nodes

    def do_step(self):
        'do a refinement replay step'

        assert self.doing_replay()
