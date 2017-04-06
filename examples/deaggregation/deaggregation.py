'''
Manually created hybrid automaton for hylaa testing.
'''

from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint
from hylaa.hybrid_automaton import HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings

from numpy import array as nparray
import math

def define_ha():
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton('Trimmed Harmonic Oscillator w/ successor')
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('green')
    loc1.a_matrix = nparray([[0, 1], [-1, 0]])
    loc1.c_vector = nparray([0, 0])
    
    inv1 = LinearConstraint([0., -1.], 0.0) # y >= 0
    loc1.inv_list = [inv1]
    
    loc2 = ha.new_mode('cyan')
    loc2.a_matrix = nparray([[0, 0], [0, 0]])
    loc2.c_vector = nparray([0, -2])
    
    inv2 = LinearConstraint([0., -1.], 2.5) # y >= -2.5
    loc2.inv_list = [inv2]
    
    loc3 = ha.new_mode('orange')
    loc3.a_matrix = nparray([[0, 0], [0, 0]])
    loc3.c_vector = nparray([0, -2])
    inv3 = LinearConstraint([0., -1.], 4.0) # y >= -4
    loc3.inv_list = [inv3]
    
    guard = LinearConstraint([0., -1.], 0.0) # y >= 0
    trans = ha.new_transition(loc1, loc2)
    trans.condition_list = [guard]
    
    guard1 = LinearConstraint([0., 1.], -0) # y <= -0
    guard2 = LinearConstraint([1., 0], 0.5) # x <= 0.5
    guard3 = LinearConstraint([-1., 0], 0.5) # x >= -0.5
    trans = ha.new_transition(loc2, loc3)
    trans.condition_list = [guard1, guard2, guard3]

    return ha

def define_init_states(ha):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []

    r = HyperRectangle([(-5.5, -4.5), (0, 1)])
    rv.append((ha.modes['green'], r))

    return rv

def define_settings():
    'get the hylaa settings object'
                            
    plot_settings = PlotSettings()

    # save to a video file
    plot_settings.make_video("deaggregation.mp4", frames=150, fps=5)
    
    plot_settings.xdim = 0
    plot_settings.ydim = 1
    plot_settings.extra_lines = [[(-0.5, -4), (-0.5, -0), (0.5, -0), (0.5, -4)]] 
    
    settings = HylaaSettings(step=0.25, max_time=6.0, plot_settings=plot_settings)
    settings.process_urgent_guards = True
    
    settings.simulation.threads=1

    return settings
    
def run_hylaa(settings):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha = define_ha()
    init = define_init_states(ha)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

    return engine.result

if __name__ == '__main__':
    run_hylaa(define_settings())

