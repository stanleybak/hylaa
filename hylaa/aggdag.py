'''
Stanley Bak
Aggregation Directed Acyclic Graph (DAG)
'''

from collections import namedtuple

from hylaa.util import Freezable

class AggDag(Freezable):
    'Aggregation directed acyclic graph (DAG) used to manage the deaggregation process'

    def __init__(self):
        self.root = [] # list of root AggDagNode where the computation begins
        
        self.freeze_attrs()

class AggDagNode(Freezable):
    'A node of the Aggregation DAG'

    def __init__(self):
        self.op_list = [] # list of Op* objects

        self.aggregated_set = None # StateSet, or None if this is a non-aggergated set
        self.concrete_set = None # StateSet
        
        self.freeze_attrs()

# Operation types
OpInvIntersect = namedtuple('OpInvIntersect', ['step', 'i_index', 'stronger'])
OpTransition = namedtuple('OpTransition', ['step', 't_index'])
        
