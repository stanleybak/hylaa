'''
Generate concrete traces from counter-examples found by HyLAA.

The check() function performs a concrete simulation to find check close
a violation found by HyLAA is to an actual simulation.

Stanley Bak
December 2016
'''

import time
import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
from scipy.sparse import csr_matrix

from hylaa.util import Freezable

class CheckTraceData(Freezable):
    'class containing data about the counter-example trace'

    def __init__(self):
        self.sim_time = None
        self.abs_error = None
        self.rel_error = None

        self.freeze_attrs()

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

def check(a_matrix, b_matrix, step, max_time, start_point, inputs, normal_vec, end_val, \
          quick=False, stdout=True, approx_samples=2000):
    '''Run a simulation with the given dynamics and inputs to see how close it matches with HyLAA's prediction

    The comparison is printed showing the differences between the simulation's projection onto normal_vec
    and expected result (passed in as end_val), both in terms of abs_error and rel_error

    Returns a tuple, (states, times, CheckTraceData) where states is the projected simulation state at each time step
    '''

    a_matrix = csr_matrix(a_matrix)

    if b_matrix is not None:
        b_matrix = csr_matrix(b_matrix)

    num_dims = a_matrix.shape[0]
    assert num_dims > 0
    assert a_matrix.shape[0] == a_matrix.shape[1], "expected A matrix to be a square"
    assert len(start_point) == num_dims
    total_steps = int(round(max_time / step))

    assert abs(total_steps * step - max_time) < 1e-9, "Rounding issue with number of steps"

    if inputs is None:
        inputs = [0] * (total_steps)

    assert len(inputs) <= total_steps, "more inputs({}) than steps({})?".format(len(inputs), total_steps)

    data = CheckTraceData()
    data.sim_time = max_time

    # we want to roughly get the desired number of sample points, so we may need to do multiple
    # samples per input
    samples_per_input = approx_samples / max(1, len(inputs))
    if samples_per_input < 1:
        samples_per_input = 1

    start = time.time()
    sim_states = [start_point]
    sim_times = [0.0]

    index = 0

    while index < total_steps and index < len(inputs):
        num_steps = 1

        # try doing multiple steps at a time (better performance)
        while index + num_steps < total_steps and index + num_steps < len(inputs):
            if np.allclose(inputs[index], inputs[index + num_steps]):
                num_steps += 1
            else:
                break

        elapsed = time.time() - start
        remaining = elapsed / ((1.0+index) / len(inputs)) - elapsed

        if stdout and num_steps > 1 and remaining > 2.0:
            print("Combining {} steps (identical inputs)".format(num_steps))

        if stdout and remaining > 2.0:
            print("{} / {} (ETA: {:.2f} sec)".format(index, len(inputs), remaining))

        der_func = make_der_func(a_matrix, b_matrix, inputs[index])
        new_states, new_times = sim(sim_states[-1], der_func, step * num_steps, samples_per_input * num_steps, quick)

        sim_states += [s for s in new_states]
        time_offset = index * step
        sim_times += [t + time_offset for t in new_times]

        assert len(sim_states) == len(sim_times), "sim_states len = {}, sim_times len = {}".format(
            len(sim_states), len(sim_times))

        index += num_steps

    last_sim_point = sim_states[-1].copy()

    sim_val = np.dot(last_sim_point, normal_vec)
    diff = sim_val - end_val

    # normal_vec is -1.0

    data.abs_error = numerator = abs(diff)
    denominator = abs(sim_val)

    if denominator == 0:
        data.rel_error = 0.0
    else:
        data.rel_error = numerator / denominator

    if stdout:
        print("Final Time: {}".format(index * step))
        print("Absolute Error (l-2 norm): {}".format(numerator))
        print("Relative Error (l-2 norm): {}".format(data.rel_error))
        print("Runtime: {:.2f} seconds".format(time.time() - start))

    return (sim_states, sim_times, data)

def sim(start, der_func, time_amount, num_steps, quick):
    'simulate for some fixed time, and return the resultant (states, times) tuple'

    tol = 1.49012e-8

    if not quick:
        tol /= 1e5 # more accurate simulation

    times = np.linspace(0, time_amount, num=1 + num_steps)
    states = odeint(der_func, start, times, col_deriv=True, rtol=tol, atol=tol, mxstep=100000)

    states = states[1:]
    times = times[1:]
    assert len(states) == num_steps
    assert len(times) == num_steps

    return (states, times)

def plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step, xdim=None, ydim=None):
    '''plot the simulation trace in the normal direction

    if xdim and dim are not none, then a 2-d phase portrait plot will be given, otherwise a time-history plot
    is provided
    '''

    do_2d = xdim is not None and ydim is not None

    total_steps = int(math.ceil(max_time / step))
    end_point = sim_states[-1]
    end_val = np.dot(end_point, normal_vec)
    tol = 1e-6

    if len(end_point) < 10:
        print("End Point: {}".format(end_point))

    if end_val - tol <= normal_val:
        print("End Point is a violation (within tolerance): {} - {} <= {}".format(end_val, tol, normal_val))
    else:
        print("End point is NOT a violation: {} - {} (tolerance) > {}".format(end_val, tol, normal_val))

    epsilon = step / 8.0 # to prevent round-off error on the end range
    input_times = np.arange(0.0, max_time + epsilon, step)

    if inputs is not None:
        _, ax = plt.subplots(2, sharex=not do_2d, figsize=(14, 8))
    else:
        _, ax = plt.subplots(1, figsize=(14, 5))
        ax = [ax]

    if do_2d:
        xs = [state[xdim] for state in sim_states]
        ys = [state[ydim] for state in sim_states]

        ax[0].plot(xs, ys, 'k-', lw=2, label='Simulation')

        ax[0].plot(xs[-1], ys[-1], 'o', ms=10, mew=3, mec='red', mfc='none', label='End')
        ax[0].plot(xs[0], ys[0], '*', ms=10, mew=3, mec='blue', mfc='none', label='Start')

        ax[0].set_ylabel('Dim {}'.format(ydim), fontsize=22)
        ax[0].set_xlabel('Dim {}'.format(xdim), fontsize=22)
    else:
        normal_trace = np.dot(sim_states, normal_vec)
        ax[0].plot(sim_times, normal_trace, 'k-', lw=2, label='Simulation')
        ax[0].plot(sim_times[-1], normal_trace[-1], 'o', ms=10, mew=3, mec='red', mfc='none')
        ax[0].plot([sim_times[0], sim_times[-1]], [normal_val, normal_val], 'k--', lw=2, label='Violation')

        if inputs is None:
            ax[0].set_xlabel('Time', fontsize=22)

        ax[0].set_ylabel('Projected State', fontsize=22)
        ax[0].set_title('Counter-Example Trace', fontsize=28)

    if inputs is not None and len(inputs) > 0:
        inputs.append(inputs[-1]) # there is one less input than time instants
        inputs = np.array(inputs, dtype=float)

        flat_inputs = []
        flat_times = []

        for i in xrange(total_steps-1):
            flat_times += [input_times[i], input_times[i+1]]
            flat_inputs += [inputs[i, :], inputs[i, :]]

        # single step edge case
        if len(flat_inputs) == 0:
            flat_times += [input_times[0], input_times[0]]
            flat_inputs += [inputs[0, :], inputs[0, :]]

        for row in xrange(len(flat_inputs[0])):
            ax[1].plot(flat_times, [single_input[row] for single_input in flat_inputs], label="$u_{}$".format(row+1))

    ##################
    # visual

    ax[0].set_title('Simulation', fontsize=28)
    ax[0].tick_params(axis='both', which='major', labelsize=18)

    if inputs is not None:
        ax[1].set_ylabel('Input', fontsize=22)
        ax[1].set_xlabel('Time', fontsize=22)
        ax[1].tick_params(axis='both', which='major', labelsize=18)

    ##################
    # legend
    for i in xrange(2 if inputs is not None else 1):
        legend = ax[i].legend(loc='best', numpoints=1)

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

    lim = ax[1 if inputs is not None else 0].get_ylim()
    dy = lim[1] - lim[0]
    plt.ylim(lim[0] - dy * .1, lim[1] + dy * .1)

    if not do_2d:
        plt.xlim(0, max_time * 1.02)
    elif len(ax) > 1:
        ax[1].set_xlim(0, max_time * 1.02)

    plt.tight_layout()

    plt.show()
