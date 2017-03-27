'''
This is a harmonic oscillator with an invariant condition to trim the reachable set of states.
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

    ha = LinearHybridAutomaton('Trimmed Harmonic Oscillator')
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('loc1')
    loc1.a_matrix = nparray([[-0.2, 1], [-1, -0.2]])
    loc1.c_vector = nparray([0, 0])
    
    inv1 = LinearConstraint([0., 1.], 4.0) # y <= 4
    loc1.inv_list = [inv1]
    
    return ha

def define_init_states(ha):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []

    r = HyperRectangle([(-5.9, -4.9), (0, 1)])
    rv.append((ha.modes['loc1'], r))

    return rv

def main():
    'runs hylaa on the model'
    ha = define_ha()
    init = define_init_states(ha)
                            
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_INTERACTIVE
    #plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    #plot_settings.video_frames = 220
    #plot_settings.video_filename = "out.wma"
    #plot_settings.video_codec = None
    
    plot_settings.xdim = 0
    plot_settings.ydim = 1
    
    settings = HylaaSettings(step=0.1, max_time=10.0, plot_settings=plot_settings)

    engine = HylaaEngine(ha, settings)
    engine.run(init)

if __name__ == '__main__':
    main()

