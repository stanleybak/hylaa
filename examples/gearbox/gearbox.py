'''
Gearbox benchmark from:
"Motor-Transmission Drive System: a Benchmark Example for Safety Verification"
by Hongxu Chen, Sayan Mitra and Guangyu Tian in ARCH 2015

This model was also used in ARCHCOMP-17 and 18.

This model shows how to use hylaa.symbolic to construct the dynamics and reset / guard conditions from 
string expressions.

Stanley Bak, Nov 2018
'''

import math


from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings, LabelSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil, aggstrat, symbolic
from hylaa.aggstrat import Aggregated

def make_automaton():
    'make the hybrid automaton'

    ha = HybridAutomaton('Gearbox')

    # variables
    variables = ["px", "py", "vx", "vy", "I", "affine"]
    derivatives = ["vx", "vy", "Fs/ms", "-Rs*Tf/Jg2", "0"]

    constant_dict = {
        "zeta": 0.9,
        "ms": 3.4,
        "mg2": 18.1,
        "Jg2": 0.7,
        "Rs": 0.08,
        "theta": 0.628318530717959,
        "deltap": -0.003,
        "Fs": 70,
        "Tf": 1
        }

    ############## Modes ##############
    move_free = ha.new_mode('move_free')
    #
    a_mat = symbolic.make_dynamics(variables[:-1], derivatives, constant_dict, has_affine_variable=True)
    move_free.set_dynamics(a_mat)

    invariant = "px<=deltap & py<=-px*0.726542528005361 & py>=px*0.726542528005361"
    mat, rhs = symbolic.make_condition(variables, invariant.split('&'), constant_dict)
    move_free.set_invariant(mat, rhs)


    ############## Cyclic Transitions ##############
    t1 = ha.new_transition(move_free, move_free)
    guard = "py>=-px*0.726542528005361 & vx*0.587785252292473+vy*0.809016994374947>=0"
    mat, rhs = symbolic.make_condition(variables, guard.split('&'), constant_dict)
    t1.set_guard(mat, rhs)

    resets = ["px", "py", "(vx*(ms*0.809016994374947*0.809016994374947-mg2*zeta*0.587785252292473*0.587785252292473)+vy*(-(zeta+1)*mg2*0.587785252292473*0.809016994374947))/(ms*(0.809016994374947*0.809016994374947)+mg2*(0.587785252292473*0.587785252292473))", 
                "(vx*(-(zeta+1)*ms*0.587785252292473*0.809016994374947)+vy*(mg2*0.587785252292473*0.587785252292473-ms*zeta*0.809016994374947*0.809016994374947))/(ms*(0.809016994374947*0.809016994374947)+mg2*(0.587785252292473*0.587785252292473))", 
                "I+(vx*0.587785252292473+vy*0.809016994374947)*(zeta+1)*ms*mg2/(ms*(0.809016994374947*0.809016994374947)+mg2*(0.587785252292473*0.587785252292473))"]
    reset_mat = symbolic.make_dynamics(variables, resets, constant_dict, has_affine_variable=True)
    print(reset_mat)
    sys.exit(1)
    # reset

     
    return ha

def make_init(ha):
    'make the initial states'

    mode = ha.modes['move_free']
    # px==-0.0165 & py==0.003 & vx==0 & vy==0 & I==0 & affine==1.0
    init_lpi = lputil.from_box([(-0.0165, -0.0165), (0.003, 0.003), (0, 0), (0, 0), (0, 0), (1.0, 1.0)], mode)
    init_list = [StateSet(init_lpi, mode)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(0.001, 0.1) # step: 0.001, bound: 0.1

    #settings.stop_on_aggregated_error = False

    #settings.aggstrat = MyAggergated()
    settings.aggstrat.deaggregate = True # use deaggregation

    settings.stdout = HylaaSettings.STDOUT_VERBOSE

    settings.plot.plot_mode = PlotSettings.PLOT_INTERACTIVE
    settings.plot.filename = "gearbox.png"
    settings.plot.plot_size = (8, 9)

    settings.plot.xdim_dir = 0
    settings.plot.ydim_dir = 1
    
    settings.plot.label.axes_limits = [-0.02, 0, -0.008, 0.004]

    return settings

def main():
    'main entry point'

    ha = make_automaton()

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    main()
