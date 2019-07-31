'''
Stanley Bak

Code assocaited with counterexamples.
'''

from collections import defaultdict
from collections import deque

import numpy as np

from sympy import Polygon, Point, Point2D

from scipy.integrate import odeint
from scipy.sparse import csr_matrix

from hylaa.aggstrat import get_ancestors
from hylaa.util import Freezable

class HylaaResult(Freezable): # pylint: disable=too-few-public-methods
    'result object returned by core.run()'

    def __init__(self):
        self.top_level_timer = None # TimerData for total time

        # verification result:
        self.has_aggregated_error = False
        self.has_concrete_error = False

        self.counterexample = [] # if unsafe, a list of CounterExampleSegment objects

        # assigned if setting.plot.store_plot_result is True, an instance of PlotData
        self.plot_data = None

        # the last core.cur_state object... used for unit testing
        self.last_cur_state = None

        # lines plotted for simulations
        # nested lists. first index is plot number, second is segment number,
        # third is simulation number, last is list of 2-d points 
        self.sim_lines = None 

        self.freeze_attrs()

class PlotData(Freezable):
    'used if setting.plot.store_plot_result is True, stores data about the plots'

    def __init__(self, num_plots):
        # first index is plot number, second is mode name, result is a list of obj
        # each obj is a tuple: (verts, state, cur_step_in_mode, str_description)
        self.mode_to_obj_list = [defaultdict(list) for _ in range(num_plots)]

        self.freeze_attrs()

    def get_verts_list(self, mode_name, plot_index=0):
        'get a list of lists for the passed-in mode'

        return self.mode_to_obj_list[plot_index][mode_name][0]

    def add_state(self, state, verts, plot_index):
        'add a plotted state'

        mode_name = state.mode.name

        obj = (verts, state, state.cur_step_in_mode, f"{mode_name} at step {state.cur_step_in_mode}")

        self.mode_to_obj_list[plot_index][mode_name].append(obj)

    def remove_state(self, state, step):
        'remove a state that was previously added'

        found = False

        for mode_to_obj in self.mode_to_obj_list:
            obj_list = mode_to_obj[state.mode.name]

            for index, obj in enumerate(obj_list):
                _, istate, istep, desc = obj

                if istate is state and step == istep:
                    found = True

                    obj_list.pop(index)
                    break

            if found:
                break

        assert found

    def get_plot_data(self, x, y, subplot=0):
        'get the plot data at x, y, or None if not found'

        rv = None

        mode_to_obj = self.mode_to_obj_list[subplot]
        clicked = Point(x, y)

        for mode, obj_list in mode_to_obj.items():
            
            for obj in obj_list:
                verts = obj[0]

                if len(verts) < 4: # need at least 3 points (4 with wrap) to be clicked inside
                    continue

                verts_2dp = [Point2D(x, y) for x, y in verts[1:]]
                                
                poly = Polygon(*verts_2dp)

                if poly.encloses_point(clicked):
                    rv = obj
                    break

            if rv is not None:
                break

        return rv

class CounterExampleSegment(Freezable):
    'a part of a counter-example trace'

    def __init__(self):
        self.mode = None # Mode object
        self.start = []
        self.end = []
        self.steps = 0
        self.outgoing_transition = None # Transition object
        self.reset_minkowski_vars = [] # a list of minkowski variables in the outgoing reset

        self.inputs = deque() # inputs at each step (a deque of m-tuples, where m is the number of inputs)
        
        self.freeze_attrs()

    def __str__(self):
        return "[CE_Segment: {} -> {} in '{}']".format( \
            self.start, self.end, "<None>" if self.mode is None else self.mode.name)

    def __repr__(self):
        return str(self)

def make_counterexample(ha, state, transition_to_error, lpi):
    '''make and return the result counter-example from the lp solution'''

    lp_solution = lpi.minimize() # resolves the LP to get the full unsafe solution
    names = lpi.get_names()

    # first get the number of steps in each mode
    num_steps = []

    if state.aggdag_op_list[0] is not None:
        for node in get_ancestors(state.aggdag_op_list[0].child_node)[1:]:
            assert len(node.parent_ops) == 1
            num_steps.append(node.parent_ops[0].step)

    num_steps.append(state.cur_step_in_mode) # add the last mode

    counterexample = []

    for name, value in zip(names, lp_solution):

        # if first initial variable of mode then assign the segment.mode variable
        if name.startswith('m') and '_i0' in name:
            seg = CounterExampleSegment()
            seg.steps = num_steps[len(counterexample)]
            counterexample.append(seg)

            parts = name.split('_')

            if len(parts) == 2:
                assert len(counterexample) == 1, "only the initial mode should have no predecessor transition"
            else:
                assert len(parts) == 3

                # assign outgoing transition of previous counterexample segment
                transition_index = int(parts[2][1:])
                t = counterexample[-2].mode.transitions[transition_index]
                counterexample[-2].outgoing_transition = t

            mode_id = int(parts[0][1:])

            for mode in ha.modes.values():
                if mode.mode_id == mode_id:
                    seg.mode = mode
                    break

            assert seg.mode is not None, "mode id {} not found in automaton".format(mode_id)

        if name.startswith('m'): # mode variable
            if '_i' in name:
                seg.start.append(value)
            elif '_c' in name:
                seg.end.append(value)
            elif '_I' in name:
                if '_I0' in name:
                    seg.inputs.appendleft([])
                    
                # inputs are in backwards order due to how the LP is constructed, prepend it
                seg.inputs[0].append(value)

        elif name.startswith('reset'):
            seg.reset_minkowski_vars.append(value)

    # add the final transition which is not encoded in the names of the variables
    seg.outgoing_transition = transition_to_error

    return counterexample


def replay_counterexample(ce_segment_list, ha, settings):
    '''replay the counterexample

    returns a list of points and a list of times
    '''

    rv = []
    all_times = []

    epsilon = 1e-7
    step_size = settings.step_size
    
    for i, segment in enumerate(ce_segment_list):
        inv_list = segment.mode.inv_list
        a_mat = segment.mode.a_csr
        b_mat = segment.mode.b_csr

        assert b_mat is None, "todo: implement counter-example generation with inputs"

        der_func = make_der_func(a_mat, b_mat, [])

        start = segment.start
        times = np.linspace(0, segment.steps * step_size, num=segment.steps)
        tol = 1e-9
        states = odeint(der_func, start, times, col_deriv=True, rtol=tol, atol=tol, mxstep=int(1e7))

        rv += [s for s in states]
        t_offset = 0
        
        if all_times:
            t_offset = all_times[-1]
            
        all_times += [t + t_offset for t in times]

        # check if in invariant up to the last step
        for state in states[:-1]:
            for lc in inv_list:
                lhs = lc.csr * state

                assert lhs <= lc.rhs + epsilon, "invariant became false during replay counterexample"

        assert np.allclose(states[-1], segment.end)
                
        # check that last state -> reset -> first state is correct
        t = segment.outgoing_transition
        lhs = t.guard_csr * states[-1]

        for left, right in zip(lhs, t.guard_rhs):
            assert left <= right + epsilon, "guard was not enabled during replay counterexample"

        assert not t.reset_minkowski_csr, "unimplemented: resets with minkowski sums"

        if t.reset_csr is not None:
            poststate = t.reset_csr * states[-1]

            if i + 1 < len(ce_segment_list):
                next_prestate = ce_segment_list[i+1].start

                assert np.allclose(poststate, next_prestate)
            else:
                rv.append(poststate)
                all_times.append(all_times[-1])

    return rv, all_times

def make_der_func(a_matrix, b_matrix, input_vec):
    'make the derivative function with the given paremeters'

    assert isinstance(a_matrix, csr_matrix)

    if b_matrix is not None:
        assert isinstance(b_matrix, csr_matrix)

    input_vec = np.array(input_vec, dtype=float)

    def der_func(state, _):
        'the constructed derivative function'

        no_input = a_matrix * state
        no_input.shape = (state.shape[0],)

        if b_matrix is None:
            rv = no_input
        else:
            input_effects = b_matrix * input_vec
            input_effects.shape = (state.shape[0],)

            rv = np.add(no_input, input_effects)

        return rv

    return der_func
