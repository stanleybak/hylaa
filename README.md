# Hylaa #

<p align="center"> <img src="hylaa_logo_small.png" alt="Hylaa Logo"/> </p>

Hylaa (**HY**brid **L**inear **A**utomata **A**nalyzer) is a verification tool for system models with linear ODEs, time-varying inputs, and possibly hybrid dynamics. 

This is the continuous branch of Hylaa, where we focus on safety verification of strictly linear models. The goal is to scale to very large systems with dynamics given by sparse matrices A.

The latest version of Hylaa is always available on our github repository at https://github.com/stanleybak/hylaa . A website for Hylaa is maintained at http://stanleybak.com/hylaa .

The code was mostly written by Stanley Bak (http://stanleybak.com) with input from Parasara Sridhar Duggirala (http://engr.uconn.edu/~psd). Work on this branch was also done by Dung Hoang Tran. 

Hylaa is released under the GPL v3 license (see the LICENSE file). It has been approved for public release (DISTRIBUTION A: Approved for public release; distribution unlimited #88ABW-2016-5976, 22 NOV 2016).

### Installation ###

Hylaa is mostly written in Python, with a few C++ parts (linear programming solving, GPU interface). You'll need to get a few required libraries, compiles the C++ portions as shared libraries, and then setup the evnironment variables. Then, to run a model you simply do 'python modelname.py'. Even if you're not planning on using a GPU, you still need the install cuda to use their compiler. A

These instructions are made for an Ubuntu system, such as the Amazon EC2 GPU instance (p2.xlarge) using the Ubuntu 16.04 LTS Server image.

# Install Packages #

sudo apt-get update
sudo apt-get install python-numpy python-scipy python-matplotlib make nvidia-cuda-toolkit

Not sure (check): libblas-dev

# Compile GLPK Interface as Shared Library #

This a custom C++ interface to GLPK for use in Hylaa that you need to compile. See hylaa/glpk-interface/README for details on how to do this. Essentially, you need to get glpk-4.60 (which may be newer than what comes with Ubuntu), and then run make (the Makefile is in that folder). This will produce hylaa_glpk.so.

# Compile Arnoldi GPU / CPU code as Shared Library #

You also need to compile the arnoldi code, which include a GPU interface. This requires the nvcc compiler from n-videa for general gpu computation using cuda. This compiler turns .cu files into executables (or shared libraries in this instance). You should use at least version 7.5 of nvcc (use 'nvcc --version' to check). Some people have reported that right after installation they need to reboot their system for the nvcc compiler to work correctly, so if the 'nvidia-cuda-toolkit' installation seemed to work and you have compiling issues, try that first.

To compile, go into the hylaa/glpk-interface folder and run 'make' (the Makefile is in that folder). This should produce cusp_krylov_stan.so.

# Setup PYTHONPATH Environment Variable #

A Hylaa model is given in python code, which imports the hylaa classes, creates a model definition and settings objects, and then calls a function with these objects. The hylaa folder contains the python package. You should add the parent folder of the hylaa folder to your PYTHONPATH environment variable. On Linux, this can be done by updating your ~/.profile or ~/.bashrc to include:

export PYTHONPATH="${PYTHONPATH}:/path/to/parent/of/hylaa/folder"

After you do this, you may need to restart the terminal (for ~/.bashrc) or log out and log back in (for ~/.profile), or otherwise ensure the environment variable is updated (do echo $PYTHONPATH to see if it includes the correct folder). Once this is done, you should be able to run the example models.

# Video export (optional #
For .mp4 (and other format) video export, ffmpeg is used. Make sure you can run the command ffmpeg from a terminal for this to work.

### Getting Started + Example ###

The easiest way to get started with Hylaa is to run some of the examples. Models in Hylaa are defined in python code (more on the input format in the next section), and the tool is executed using python as well.

Go to `examples/harmonic_oscillator` and run `ha.py` from the command line (`python ha.py`). This should create `plot.png` in the same folder, which will be an 2-d plot of the reachable set. This plot is similar to the timed harmonic oscillator plot given in the paper "Affine Systems Verification with Double-Projected Reachability: 10^6 Dimensions and Beyond". 

The dynamics in Hylaa are given as x' = **A**x, where **A** is the sparse dynamics matrix. Initial states and unsafe states are given as conjunctions of linear constraints. You can see the 4-d system in the timed harmonic oscialltor case and the error states defined in the `define_ha` function in ha.py. Try changing the dynamics slightly and re-running the script to see the effect.

Computation settings are given in the `define_settings` function. To switch from plotting a static image to a live plot during the computation, for example, change `plot_settings.plot_mode` to be `PlotSettings.PLOT_FULL`. Lots of settings exist in Hylaa (plotting mode, verification options, ect.). The default settings are generally okay, as long as you provide the time bound and time step size. If you want to do more advanced things such as select between cpu and gpu, or fix the number of arnoldi iterations rather that auto-tune it, or select the auto-tuning relative error threshold, the settings is the way to do that. All of them, as well as comments describing them can be found in `hylaa/containers.py`.

The easiest way to use Hylaa on your example is to copy an example from the examples folder and edit that. Notice that it's python code that creates the sparse A matrix. This means you can create the model programatically using loops (as we do in the heat system) or by loading the dynamics from a .mat file (as we do for MNA5).

