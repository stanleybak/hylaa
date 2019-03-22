'''
Spacecraft Rendezvous System described in
"Verifying safety of an autonomous spacecraft rendezvous mission" by Nicole Chan and Sayan Mitra

This model was used in ARCHCOMP-18, specifically the variant with the abort maneuver.

Stanley Bak, July 2018
'''

import math

from matplotlib import collections
from matplotlib.patches import Circle

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings, LabelSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil, aggstrat

def make_automaton(safe, passive_time=120):
    'make the hybrid automaton'

    ha = HybridAutomaton(f'Spacecraft Rendezvous (Abort at t={passive_time})')

    passive_time += 1e-4 # ensures switch to passive at a single step intead of at two steps
    
    ############## Modes ##############
    p2 = ha.new_mode('P2')
    a_mat = [\
        [0.0, 0.0, 1.0, 0.0, 0.0, 0], \
        [0.0, 0.0, 0.0, 1.0, 0.0, 0], \
        [-0.057599765881773, 0.000200959896519766, -2.89995083970656, 0.00877200894463775, 0.0, 0], \
        [-0.000174031357370456, -0.0665123984901026, -0.00875351105536225, -2.90300269286856, 0.0, 0.0], \
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], \
        [0, 0, 0, 0, 0, 0]]
    inv_mat = [[0.0, 0.0, 0.0, 0.0, 1.0, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0], [0., 0., 0., 0., 1., 0]]
    inv_rhs = [125.0, -100.0, passive_time]
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
    inv_rhs = [100, 100, 100, 100, 141.1, 141.1, 141.1, 141.1, passive_time]
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

    ha.new_transition(p2, passive).set_guard([[0.0, 0.0, 0.0, 0.0, -1.0, 0]], [-passive_time])

    ha.new_transition(p3, passive).set_guard([[0.0, 0.0, 0.0, 0.0, -1.0, 0]], [-passive_time])

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
    ha.new_transition(p3, error, 'los1').set_guard([[1, 0, 0., 0., 0., 0]], [-100])
    ha.new_transition(p3, error, 'los2').set_guard([[-0.57735, 1, 0, 0., 0., 0]], [0])
    ha.new_transition(p3, error, 'los3').set_guard([[-0.57735, -1, 0., 0., 0., 0]], [0])

    # sqrt(vx^2 + vy^2) should stay below 0.055 m/SECOND (time in model is in MINUTES)
    # to make the model unsafe, try changing this to 0.05
    meters_per_sec_limit = 0.055 if safe else 0.05
    meters_per_min_limit = meters_per_sec_limit * 60
    h = meters_per_min_limit * math.cos(math.pi / 8.0)
    w = meters_per_min_limit * math.sin(math.pi / 8.0)
    
    ha.new_transition(p3, error, 'g1').set_guard([[0, 0, 1., 0., 0., 0]], [-h])
    ha.new_transition(p3, error, 'g2').set_guard([[0, 0, -1., 0., 0., 0]], [-h])
    ha.new_transition(p3, error, 'g3').set_guard([[0, 0, 0., 1., 0., 0]], [-h])
    ha.new_transition(p3, error, 'g4').set_guard([[0, 0, 0., -1., 0., 0]], [-h])

    ha.new_transition(p3, error, 'g5').set_guard([[0, 0, 1., 1., 0., 0]], [-(w + h)])
    ha.new_transition(p3, error, 'g6').set_guard([[0, 0, -1., 1., 0., 0]], [-(w + h)])
    ha.new_transition(p3, error, 'g7').set_guard([[0, 0, -1., -1., 0., 0]], [-(w + h)])
    ha.new_transition(p3, error, 'g8').set_guard([[0, 0, 1., -1., 0., 0]], [-(w + h)])
    
    return ha

def make_init(ha):
    'make the initial states'

    p2 = ha.modes['P2']

    box = [(-925.0, -875.0), (-425.0, -375.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
    
    init_lpi = lputil.from_box(box, p2)
    init_list = [StateSet(init_lpi, p2)]

    #corners = [[(box[0][0], box[0][0]), (box[1][0], box[1][0]), (0, 0), (0, 0), (0, 0), (1, 1)], \
    #           [(box[0][1], box[0][1]), (box[1][0], box[1][0]), (0, 0), (0, 0), (0, 0), (1, 1)], \
    #           [(box[0][1], box[0][1]), (box[1][1], box[1][1]), (0, 0), (0, 0), (0, 0), (1, 1)], \
    #           [(box[0][0], box[0][0]), (box[1][0], box[1][0]), (0, 0), (0, 0), (0, 0), (1, 1)]]

    #init_list = [StateSet(lputil.from_box(c, p2), p2) for c in corners]

    return init_list

def make_settings(safe, passive_time=120):
    'make the reachability settings object'

    # see hylaa.settings for a list of reachability settings
    settings = HylaaSettings(0.1, 300.0) # step: 0.1, bound: 200.0
    settings.plot.plot_mode = PlotSettings.PLOT_NONE #IMAGE
    settings.stdout = HylaaSettings.STDOUT_VERBOSE

    #settings.aggstrat = aggstrat.Unaggregated()
    #settings.stop_on_concrete_error = False
    
    settings.stop_on_aggregated_error = False
    #settings.process_urgent_guards = True

    settings.aggstrat.deaggregate = True # use deaggregation
    settings.aggstrat.deagg_preference = aggstrat.Aggregated.DEAGG_LEAVES_FIRST

    settings.plot.filename = f"rendezvous_t{passive_time:03}.png"
    settings.plot.plot_size = (8, 10)
        
    settings.plot.xdim_dir = [0, 0, 2]
    settings.plot.ydim_dir = [1, 1, 3]
    settings.plot.label = [LabelSettings(), LabelSettings(), LabelSettings()]

    settings.plot.label[0].big(size=26)
    settings.plot.label[0].title_size = 22
    settings.plot.label[1].big(size=26)
    settings.plot.label[2].big(size=26)

    settings.plot.label[0].x_label = '$x$'
    settings.plot.label[0].y_label = '$y$'

    settings.plot.label[1].x_label = '$x$'
    settings.plot.label[1].y_label = '$y$'

    settings.plot.label[2].x_label = '$x_{vel}$'
    settings.plot.label[2].y_label = '$y_{vel}$'

    settings.plot.label[0].axes_limits = [-950, 200, -450, 200]
    #settings.plot.label[1].axes_limits = [-10, 10, -10, 10]
    settings.plot.label[1].axes_limits = [-110, 10, -110, 110]
    settings.plot.label[2].axes_limits = [-5, 5, -5, 5]

    y = 57.735
    line = [(-100, y), (-100, -y), (0, 0), (-100, y)]
    c1 = collections.LineCollection([line], animated=True, colors=('gray'), linewidths=(2), linestyle='dashed')
    c1_copy = collections.LineCollection([line], animated=True, colors=('gray'), linewidths=(2), linestyle='dashed')

    rad = 0.2
    line = [(-rad, -rad), (-rad, rad), (rad, rad), (rad, -rad), (-rad, -rad)]
    c2 = collections.LineCollection([line], animated=True, colors=('red'), linewidths=(2))
    c2_copy = collections.LineCollection([line], animated=True, colors=('red'), linewidths=(2))

    meters_per_sec_limit = 0.055 if safe else 0.05
    meters_per_min_limit = meters_per_sec_limit * 60
    x = meters_per_min_limit * math.cos(math.pi / 8.0)
    y = meters_per_min_limit * math.sin(math.pi / 8.0)

    line = [(-x, y), (-y, x), (y, x), (x, y), (x, -y), (y, -x), (-y, -x), (-x, -y), (-x, y)]
    vel_oct = collections.LineCollection([line], animated=True, colors=('gray'), linewidths=(2), linestyle='dashed')

    r = meters_per_min_limit
    patch_col = collections.PatchCollection([Circle((0, 0), r)], alpha=0.25)

    settings.plot.extra_collections = [[c1_copy, c2_copy], [c1, c2], [vel_oct, patch_col]]

    return settings

def run_hylaa():
    'main entry point'

    safe = True

    for passive_time in [120]: #range(5, 265, 5):
        print(passive_time)
        ha = make_automaton(safe, passive_time)
        init_states = make_init(ha)
        settings = make_settings(safe, passive_time)
        Core(ha, settings).run(init_states)

if __name__ == "__main__":
    run_hylaa()

