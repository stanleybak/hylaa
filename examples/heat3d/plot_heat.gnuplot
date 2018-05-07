set terminal pngcairo size 980,600 font 'Verdana,24' fontscale 1.0

# buffer is 200 on left, 80 on right, 230 total
set lmargin at screen 200.0/980
set rmargin at screen 900.0/980

set xrange [0:25]

set title 'Symmetric 3D Heat Diffusion' font 'Verdana,34' offset 0,-1
set xlabel "Time"
set ylabel "Temperature at Center"

set output "sym_heat_reach.png"
set style fill solid 1.0 noborder
#unset border
#unset xtics
#unset ytics

set key bottom right

stats 'reach_data.txt' nooutput
blocks = STATS_blocks


plot for [i=0:blocks-1] 'reach_data.txt' index i with lines lc rgb "#8ff00000" notitle, \
    'reach_data.txt' index 0 with filledcurves lc rgb "#8ff00000" title 'reach\_data.txt', 
    
