# Hylaa #

<p align="center"> <img src="hylaa_logo_small.png" alt="Hylaa Logo"/> </p>

Hylaa (**HY**brid **L**inear **A**utomata **A**nalyzer) is a verification tool for system models with linear ODEs, time-varying inputs, and possibly hybrid dynamics. The latest version of Hylaa is always available on our github repository at https://github.com/stanleybak/hylaa . A website for Hylaa is maintained at http://stanleybak.com/hylaa .

The code was mostly written by Stanley Bak (http://stanleybak.com) with input from Parasara Sridhar Duggirala (http://engr.uconn.edu/~psd). Hylaa is released under the GPL v3 license (see the LICENSE file). It has been approved for public release (DISTRIBUTION A: Approved for public release; distribution unlimited #88ABW-2016-5976, 22 NOV 2016).

### Introduction ###

Hylaa computes *simulation-equivalent* reachability. That is, Hylaa computes the set of states that can be reached by any fixed-step simulation from any initial start state (given a bounded set of start states) under any possible input (given a bounded set of possible inputs). For systems with time-varying inputs, this corresponds to the case where inputs can change at each time step, but are fixed between steps. These restrictions allow Hylaa's analysis to be exact (subject to some restrictions discussed next), and allow Hylaa to be able to generate counter-example traces when an error is found.

Some considerations: although we are confident in the underlying theoretical techniques Hylaa uses, we do not claim the implementation and its source code is fully correct and the tool may contain bugs (please report these to us!). The tool does not account for floating-point errors which may accumulate during the computation. Simulation-equivalent reachability only looks at the system state at specific time instances, and so may miss error states that occur between time-steps (traditional reachability analysis would catch these cases). Further, our notion of time-varying inputs only considers inputs which change at multiples of the time-step, not at any point in time. Despite these limitations, we believe and hope Hylaa can be used improve the confidence in a system's correctness.

There are also expressiveness limitations with the current implementation. The current version of Hylaa can handle either time-varying inputs or hybrid dynamics, but not both at the same time (except when creating error modes for systems with inputs). Resets in discrete transitions are not yet implemented. Also, discrete transitions with hybrid dynamics may not always work, depending on subtle properties of the system (the tool will output a basis matrix error message if these conditions fail). We plan to add these features over time.

### Setup ###

You can setup Hylaa with a few steps. These instructions are for Ubuntu Linux, although they may also help on other systems.

1. This a custom C++ interface to GLPK for use in Hylaa that you need to compile. See `hylaa/glpk-interface/README` for details on how to do this. Essentially, you need to get glpk-4.60 (which may be newer than what comes with Ubuntu), and then run `make` (the Makefile is in that folder). This will produce `hylaa_glpk.so`.

2. Hylaa is python based, and uses a few libraries which you may or may not have. Run the following to ensure you get the required packages: 

   `sudo apt-get install python python-numpy python-scipy python-matplotlib`

3. Setup the `PYTHONPATH` environment variable. A Hylaa model is just python code, which imports the hylaa files, creates a model definition and settings objects, and then calls a function with these objects. The `hylaa` folder contains the python package. You should add the parent folder of the `hylaa` folder to your `PYTHONPATH` environment variable. On Linux, this can be done by updating your `~/.profile` or `~/.bashrc` to include: 

   `export PYTHONPATH="${PYTHONPATH}:/path/to/parent/of/hylaa/folder"`
   
   After you do this, you may need to restart the terminal (for `~/.bashrc`) or log out and log back in (for `~/.profile`), or otherwise ensure the environment variable is updated (do `echo $PYTHONPATH` to see if it includes the correct folder). Once this is done, you should be able to run the example models.

4. (Optional) For .mp4 video export, ffmpeg is used. Make sure you can run the command `ffmpeg` from a terminal for this to work.

5. (Optional) If you're dealing with large systems, you can speed up matrix multiplication by using OpenBLAS instead of the standard implementation of `numpy.dot` for matrix multiplication. See the comments at the top of `tests/np_dot_benchmark.py` for how to check if your implementation is optimized and how to connect OpenBLAS with python (on Linux). Hylaa will work without this, but performance may be degraded, especially for high-dimensional systems.

### Getting Started + Example ###

The easiest way to get started with Hylaa is to run some of the examples. Models in Hylaa are defined in python code (more on the input format in the next section), and the tool is executed using python as well.

Go to `examples/damped_oscillator` and run `damped_oscillator.py` from the command line (`python damped_oscillator.py`). This should create `plot.png` in the same folder, which will be an 2-d plot of the reachable set. Compare this to the SpaceEx output given in `spaceex_output.png`. The dynamics for this system are `x' = -0.1 * x + y` and `y' = -x - 0.1 * y`, with the initial states `x = [-6, -5]` and `y = [0, 1]`. 

The dynamics in Hylaa are given as x' = **A**x + **B**u + c, where x is a vector of variables, **A** is the dynamics matrix, **B** is optional and if given is a matrix of input effects, c is a vector (the affine term of the dynamics), and u is a vector of the input variables. Inputs, if used, are time-varying, with bounds given by linear constraints as **A_constraints** * u <= b_constraints, where **A_constraints** is a matrix, and b_constraints is a vector. Initial states, inputs, invariants and guards are given as conjunctions of linear constraints. The unsafe error state specification is created by marking certain modes of the hybrid automaton as error modes, and so, unsafe conditions have the same restrictions as guards (they are conjunctions of linear constraints).

In the damped_oscillator example, there are no inputs. The line `a_matrix = np.array([[-0.1, 1], [-1, -0.1]])` defines the dynamics **A** matrix, and the line `c_vector = np.array([0, 0], dtype=float)` defines the c_vector affine term. Try changing these slightly and re-running the script to see the effect.

Computation settings are given in the `define_settings` function. To switch from plotting a static image to a live plot during the computation, for example, change `plot_settings.plot_mode` to be `PlotSettings.PLOT_FULL`. Lots of settings exist in Hylaa (plotting mode, verification options, ect.). All of them, as well as comments describing them can be found in `hylaa/containers.py`.

Some of the other examples show how different features can be implemented:

* `examples/input_oscillator` - time-varying inputs
* `examples/building` - time-varying inputs (50-dimensions) with unsafe error states
* `examples/invariant_trim` - mode invariants
* `examples/ball_string` - discrete transitions
* `examples/deaggregation` - discrete successor aggregation and deaggregation across multiple guards with `.mp4` video output

### Input Format ###

Hylaa takes input python objects describing the hybrid automaton. One way to make these files is manually. However, there is also a Hyst [1] printer available, which allows one to convert SpaceEx models to the Hylaa input format. You can then also use the hypy [2] library to script together multiple runs of Hyst+Hylaa.

With this approach, you can create models in the SpaceEx [3] format using the SpaceEx model editor and convert the `.xml` and `.cfg` files SpaceEx would use into a runnable Hylaa python file. Many of the examples included with Hylaa were initially created with this approach, with some of the settings adjusted after generating the model. An example SpaceEx file you can try converting is in the `examples/damped_oscillator` folder (`osc_spaceex.xml` and `osc_spaceex.cfg`).

The Hylaa printer is included in version 1.4 of Hyst, which also includes hypy: https://github.com/stanleybak/hyst/releases/tag/v1.4

[1] "HYST: A Source Transformation and Translation Tool for Hybrid Automaton Models", S. Bak, S. Bogomolov, T. Johnson, Tools Paper, ACM/IEEE 18th International Conference on Hybrid Systems: Computation and Control (HSCC 2015)

[2] "High-level Hybrid Systems Analysis with Hypy", S. Bak, S. Bogomolov, C. Schiling, Applied Verification for Continuous and Hybrid Systems (ARCH 2016)

[3] "SpaceEx: Scalable verification of hybrid systems", G. Frehse, C. Le Guernic, A. Donzé, S. Cotton, R. Ray, O. Lebeltel, R. Ripado, A. Girard, T. Dang, and O. Maler, International Conference on Computer Aided Verification, (CAV 2011)

### Code Tests ###

A number of unit and regressions tests are included in the `tests` folder. Hylaa uses python's unit testing framework `pyunit`. To run all the tests, simply run `python -m unittest discover` in a terminal after changing to the `tests` folder. If you are debugging, you can run an individual test script directly using something line `python test_star.py`, An individual test method within a test script can also be run using something line `python -m unittest test_star.TestStar.test_hr_to_star`.

### Hylaa Publications ###

"Simulation-Equivalent Reachability of Large Linear Systems with Inputs", S. Bak, P. Duggirala, 29th International Conference on Computer-Aided Verification (CAV 2017)

"HyLAA: A Tool for Computing Simulation-Equivalent Reachability for Linear Systems", S. Bak, P. Duggirala, 20th International Conference on Hybrid Systems: Computation and Control (HSCC 2017)

"Rigorous Simulation-Based Analysis of Linear Hybrid Systems", S. Bak, P. Duggirala, 23rd International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2017)

"Direct Verification of Linear Systems with over 10000 Dimensions", S. Bak, P. Duggirala, Applied Verification for Continuous and Hybrid Systems (ARCH 2017)

### Code Contributions ###

We welcome external contributions to the code, although please submit high quality code with appropriate tests. You can contact us if you're planning to submit something and we can try to help. Also ensure the code passes all the existing tests before submitting it (and add your own tests for your new features). 

Hylaa's python code uses the pylint static analysis tool to ensure reasonable code cleanliness. Please use this and generally try to eliminate every warning it raises. There is an included `.pylintrc` file with our pylint settings. Please ensure your code generally passes pylint's checks prior to submitting it. This is much easier if you integrate pylint into your development environment and correct the code as you are developing it. For the C++ code, a `.clang-format` file is in the `hylaa/glpk_interface` folder which specifies the code format that should be used with the clang static checker tool.

Once your code is clean and passes all the tests, you can submit a pull request to our github repository: https://github.com/stanleybak/hylaa .
