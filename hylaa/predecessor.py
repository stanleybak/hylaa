'''
Stanley Bak
Predecessor types (including aggergation)

These instances are stored inside StateSets
'''

from collections import namedtuple
from hylaa.util import Freezable

TransitionPredecessor = namedtuple('TransitionPredecessor', ['state', 'transition', 'premode_lpi'])

class AggregationPredecessor(Freezable):
    'a predecessor which is an aggregation of many states'

    def __init__(self, states=None):

        self.states = states

        self.freeze_attrs()
