# Hylaa #

<p align="center"> <img src="hylaa_logo_small.png" alt="Hylaa Logo"/> </p>

Hylaa (**HY**brid **L**inear **A**utomata **A**nalyzer) is a verification tool for system models with linear ODEs, time-varying inputs, and possibly hybrid dynamics. 

In this branch, we focus on safety verification of strictly linear models. The focus is to scale to very large systems, with sparse specifications (a small number of conjunctions of linear constraints).

The latest version of Hylaa is always available on our github repository at https://github.com/stanleybak/hylaa . A website for Hylaa is maintained at http://stanleybak.com/hylaa .

The code was mostly written by Stanley Bak (http://stanleybak.com) with input from Parasara Sridhar Duggirala (http://engr.uconn.edu/~psd). Hylaa is released under the GPL v3 license (see the LICENSE file). It has been approved for public release (DISTRIBUTION A: Approved for public release; distribution unlimited #88ABW-2016-5976, 22 NOV 2016).


Libraries: numpy, scipy, matplotlib, krypy (pip install krypy)

----------------------
Installation libraries on amazon ec2 gpu instance (p2.xlarge) using image Ubuntu 16.04 LTS Server:

sudo apt-get update

sudo apt-get install make nvidia-cuda-toolkit python-numpy python-scipy


you also may want:

sudo apt-get install emacs24

