'''Script for generating NAV benchmark and running with pysim'''

# make sure hybridpy is on your PYTHONPATH: hyst/src/hybridpy
import hybridpy.hypy as hypy

def main():
    '''main entry point'''
    
    noise = 0.0

    nav_fig1b_matrix = [[-1.2, 0.1], [0.1, -1.2]]
    nav_fig1b_list = ['B', 2, 4, 4, 3, 4, 2, 2, 'A']
    gen_nav("nav_fig1b", nav_fig1b_matrix, nav_fig1b_list, width=3, start_point=(0.5, 1.5), time=5, noise=noise)

    nav_fig2a_matrix = [[-0.8, -0.2], [-0.1, -0.8]]
    nav_fig2a_list = [2, 4, 6, 6, 6, 2, 4, 7, 7, 4, 2, 4, 'B', 3, 4, 2, 4, 6, 6, 6, 2, 'A', 0, 0, 0]
    gen_nav("nav_fig2a", nav_fig2a_matrix, nav_fig2a_list, width=5, start_point=(3.5, 3.5), time=10, noise=noise)

    nav_fig2b_matrix = [[-1.2, 0.1], [0.2, -1.2]]
    nav_fig2b_list = [4, 6, 6, 6, 4, 4, 'A', 1, 1, 4, 4, 2, 1, 1, 4, 3, 0, 0, 'B', 4, 3, 0, 0, 6, 6]
    gen_nav("nav_fig2b", nav_fig2b_matrix, nav_fig2b_list, width=5, start_point=(3.5, 3.5), time=20, noise=noise)

# -generate nav "-matrix -1.2 0.1 0.1 -1.2 -i_list 2 2 A 4 3 4 B 2 4 -width 3 -startx 0.5 
# -starty 1.5 -time 3.0 -noise 0.25" -pysim -tp legend=False:corners=True:rand=100
def gen_nav(name, a_matrix, i_list, width, start_point, time, noise):
    'generate a navigation benchmark instance and plot a simulation'

    gen_param = '-matrix ' + str(a_matrix[0][0]) + ' ' + str(a_matrix[0][1]) + \
        ' ' + str(a_matrix[1][0]) + ' ' + str(a_matrix[1][1]) + ' -i_list '
    
    for i in i_list:
        gen_param += str(i) + ' '

    (start_x, start_y) = start_point        
    gen_param += '-width ' +  str(width) + ' -startx ' + str(start_x) + ' -starty ' + str(start_y) + \
        ' -noise ' + str(noise)

    # run pysim simulation
    pysim_param = "-corners True -legend False -rand 100 -time {} -title {}".format(time, name)
    e = hypy.Engine('pysim', pysim_param)
    e.set_generator('nav', gen_param)
    print 'Simulating {} with pysim'.format(name)
    res = e.run(image_path="{}_sim.png".format(name))

    e = hypy.Engine('hylaa2', "")
    e.set_generator('nav', gen_param)
    e.set_output(name + '.py')
    
    e.add_pass("simplify", "-python")
    
    print 'Generating hylaa2 model ' + name
    res = e.run(run_tool=False, print_stdout=False)
    print 'Finished processing ' + name + "\n"

    if res['code'] != hypy.Engine.SUCCESS:
        raise RuntimeError('Error in ' + name + ': ' + str(res['code']))

if __name__ == '__main__':
    main()
