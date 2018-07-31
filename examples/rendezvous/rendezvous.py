'''
Spacecraft Rendezvous System described in
"Verifying safety of an autonomous spacecraft rendezvous mission" by Nicole Chan and Sayan Mitra

This model was used in ARCHCOMP-18, specifically the variant with the abort maneuver.

Stanley Bak, July 2018
'''

import math

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def make_automaton(safe=True):
    'make the hybrid automaton'

    ha = HybridAutomaton('Spacecraft Rendezvous with Abort')
    
    ############## Modes ##############
    p2 = ha.new_mode('P2')
    a_mat = [\
        [0.0, 0.0, 1.0, 0.0, 0.0, 0], \
        [0.0, 0.0, 0.0, 1.0, 0.0, 0], \
        [-0.057599765881773, 0.000200959896519766, -2.89995083970656, 0.00877200894463775, 0.0, 0], \
        [-0.000174031357370456, -0.0665123984901026, -0.00875351105536225, -2.90300269286856, 0.0, 0.0], \
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], \
        [0, 0, 0, 0, 0, 0]]
    inv_mat = [[0.0, 0.0, 0.0, 0.0, 1.0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0]]
    inv_rhs = [125.0, -100.0]
    p2.set_dynamics(a_mat)
    p2.set_invariant(inv_mat, inv_rhs)


    p3 = ha.new_mode('P3')
    a_mat = [\
        [0.0, 0.0, 1.0, 0.0, 0.0, 0], \
        [0.0, 0.0, 0.0, 1.0, 0.0, 0], \
        [-0.575999943070835, 0.000262486079431672, -19.2299795908647, 0.00876275931760007, 0.0, 0], \
        [-0.000262486080737868, -0.575999940191886, -0.00876276068239993, -19.2299765959399, 0.0, 0], \
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.],\
        [0, 0, 0, 0, 0, 0]]
    inv_mat = [\
        [-1.0, 0., 0., 0., 0., 0], \
        [1.0, 0., 0., 0., 0., 0], \
        [0., -1.0, 0., 0., 0., 0], \
        [0., 1.0, 0., 0., 0., 0], \
        [-1.0, -1.0, 0.0, 0.0, 0.0, 0], \
        [1.0, 1.0, 0.0, 0.0, 0.0, 0], \
        [-1.0, 1.0, 0., 0., 0., 0], \
        [1.0, -1.0, 0., 0., 0., 0], \
        [0., 0., 0., 0., 1., 0]]
    inv_rhs = [100, 100, 100, 100, 141.1, 141.1, 141.1, 141.1, 120.0]
    p3.set_dynamics(a_mat)
    p3.set_invariant(inv_mat, inv_rhs)


    passive = ha.new_mode('passive')
    a_mat = [\
         [0, 0, 1, 0, 0, 0], \
         [0, 0, 0, 1, 0, 0], \
         [0.0000575894721132000, 0, 0, 0.00876276, 0, 0], \
         [0, 0, -0.00876276, 0, 0, 0], \
         [0, 0, 0, 0, 0, 1.], \
         [0, 0, 0, 0, 0, 0]]
    passive.set_dynamics(a_mat)


    error = ha.new_mode('error')

    ############## Normal Transitions ##############
    t1 = ha.new_transition(p2, p3)
    guard_mat = [\
        [-1.0, 0., 0., 0., 0., 0], \
        [1.0, 0., 0., 0., 0., 0], \
        [0., -1.0, 0., 0., 0., 0], \
        [0., 1.0, 0., 0., 0., 0], \
        [-1.0, -1.0, 0.0, 0.0, 0.0, 0], \
        [1.0, 1.0, 0.0, 0.0, 0.0, 0], \
        [-1.0, 1.0, 0., 0., 0., 0], \
        [1.0, -1.0, 0., 0., 0., 0]]
    guard_rhs = [100, 100, 100, 100, 141.1, 141.1, 141.1, 141.1]
    t1.set_guard(guard_mat, guard_rhs)

    ha.new_transition(p2, passive).set_guard([[0.0, 0.0, 0.0, 0.0, -1.0, 0]], [-120])

    ha.new_transition(p3, passive).set_guard([[0.0, 0.0, 0.0, 0.0, -1.0, 0]], [-120])

    ############## Error Transitions ##############
    # In the aborting mode, the vehicle must avoid the target, which is modeled as a box B with
    # 0.2m edge length and the center placed as the origin
    rad = 0.2
    t = ha.new_transition(passive, error)
    guard_mat = [ \
        [1, 0, 0., 0., 0., 0], \
        [-1, 0, 0., 0., 0., 0], \
        [0, 1., 0., 0., 0., 0], \
        [0, -1., 0., 0., 0., 0]]
    guard_rhs = [rad, rad, rad, rad]
    t.set_guard(guard_mat, guard_rhs)

    #In the rendezvous attempt the spacecraft must remain within the lineof-sight
    #cone L = {[x, y]^T | (x >= -100m) AND (y >= x*tan(30)) AND (-y >= x*tan(30))}
    ha.new_transition(p3, error).set_guard([[1, 0, 0., 0., 0., 0]], [-100])
    ha.new_transition(p3, error).set_guard([[-0.57735, 1, 0, 0., 0., 0]], [0])
    ha.new_transition(p3, error).set_guard([[-0.57735, -1, 0., 0., 0., 0]], [0])

    # sqrt(vx^2 + vy^2) should stay below 0.055 m/SECOND (time in model is in MINUTES)
    # to make the model unsafe, try changing this to 0.05
    meters_per_sec_limit = 0.055 if safe else 0.05
    meters_per_min_limit = meters_per_sec_limit * 60
    h = meters_per_min_limit * math.cos(math.pi / 8.0)
    w = meters_per_min_limit * math.sin(math.pi / 8.0)
    
    ha.new_transition(p3, error).set_guard([[0, 0, 1., 0., 0., 0]], [-h])
    ha.new_transition(p3, error).set_guard([[0, 0, -1., 0., 0., 0]], [-h])
    ha.new_transition(p3, error).set_guard([[0, 0, 0., 1., 0., 0]], [-h])
    ha.new_transition(p3, error).set_guard([[0, 0, 0., -1., 0., 0]], [-h])

    ha.new_transition(p3, error).set_guard([[0, 0, 1., 1., 0., 0]], [-(w + h)])
    ha.new_transition(p3, error).set_guard([[0, 0, -1., 1., 0., 0]], [-(w + h)])
    ha.new_transition(p3, error).set_guard([[0, 0, -1., -1., 0., 0]], [-(w + h)])
    ha.new_transition(p3, error).set_guard([[0, 0, 1., -1., 0., 0]], [-(w + h)])
    
    return ha

def make_init(ha):
    'make the initial states'

    p2 = ha.modes['P2']
    init_lpi = lputil.from_box([(-925.0, -875.0), (-425.0, -375.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)], p2)
    init_list = [StateSet(init_lpi, p2)]

    return init_list

def make_settings():
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(0.1, 200.0) # step: 0.1, bound: 200.0
    settings.plot.plot_mode = PlotSettings.PLOT_NONE
    settings.stdout = HylaaSettings.STDOUT_VERBOSE


    settings.plot.xdim_dir = 0
    settings.plot.ydim_dir = 1
    settings.plot.plot_size = (10, 10)
    settings.plot.label.big(size=32)
    settings.plot.filename = "rendezvous.png"
    settings.plot.label.x_label = '$x$'
    settings.plot.label.y_label = '$y$'

    return settings

def main():
    'main entry point'

    ha = make_automaton(safe=True)

    init_states = make_init(ha)

    settings = make_settings()

    Core(ha, settings).run(init_states)

if __name__ == "__main__":
    main()
