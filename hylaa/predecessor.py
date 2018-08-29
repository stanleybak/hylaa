'''
Stanley Bak
Predecessor types (including aggergation)

These instances are stored inside StateSets
'''

from collections import namedtuple
from hylaa.util import Freezable

TransitionPredecessor = namedtuple('TransitionPredecessor', ['state', 'transition', 'premode_lpi'])

class TransitionPredecessor(Freezable):
    'a predecessor which goes through a discrete transition'

    def __init__(self, state, transition, premode_lpi):

        self.state = state
        self.transition = transition
        self.premode_lpi = premode_lpi

        self.freeze_attrs()

    def __str__(self):
        return "[TransitionPredecessor with pathid = {}]".format(self.state.computation_path_id)

class AggregationPredecessor(Freezable):
    'a predecessor which is an aggregation of many states'

    def __init__(self, states=None):

        self.states = states

        self.freeze_attrs()
