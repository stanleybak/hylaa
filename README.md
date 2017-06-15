# Hylaa #

<p align="center"> <img src="hylaa_logo_small.png" alt="Hylaa Logo"/> </p>

Hylaa (**HY**brid **L**inear **A**utomata **A**nalyzer) is a verification tool for system models with linear ODEs, time-varying inputs, and possibly hybrid dynamics. 

In this branch, we focus on safety verification of strictly linear models. The focus is to scale to very large systems, with sparse specifications (a small number of conjunctions of linear constraints).

The latest version of Hylaa is always available on our github repository at https://github.com/stanleybak/hylaa . A website for Hylaa is maintained at http://stanleybak.com/hylaa .

The code was mostly written by Stanley Bak (http://stanleybak.com) with input from Parasara Sridhar Duggirala (http://engr.uconn.edu/~psd). Hylaa is released under the GPL v3 license (see the LICENSE file). It has been approved for public release (DISTRIBUTION A: Approved for public release; distribution unlimited #88ABW-2016-5976, 22 NOV 2016).


Libraries: numpy, scipy, matplotlib, krypy (pip install krypy)

### Installation on Amazon Ec2 GPU Instance with Ubuntu ###
Installation libraries on amazon ec2 gpu instance (p2.xlarge) using image Ubuntu 16.04 LTS Server:

sudo apt-get update

sudo apt-get install make nvidia-cuda-toolkit python-numpy python-scipy

(optional) sudo apt-get install emacs24

### Profiling Results ###

## random million matrix (5 entires / col) ##

matrix-vector multiplication: 4.1980 ms
achieved megaflops = 2382.000000
parallel multiplcation elapsed time 12.0ms
sparse multiplcation elapsed time 29.3ms


## random diagonal million matrix (5 entires / col) ##

matrix-vector multiplication: 0.6640 ms
achieved megaflops = 15060.000000
parallel multiplcation elapsed time 13.2ms
sparse multiplcation elapsed time 7.7ms


## ISSS 4000 copies ##
matrix-vector multiplication: 0.4420 ms
achieved megaflops = 7330.000000
parallel multiplcation elapsed time 14.3ms
sparse multiplcation elapsed time 4.2ms



