# Hylaa #

Hylaa is a simulation-equivalent verification tool for linear hybrid systems. The code was mostly written by Stanley Bak with input from Parasara Sridhar Duggirala.

Hylaa is released under the GPL v3 license (see the LICENSE file). It has been approved for public release.

### Setup ###

Hylaa uses the following dependencies: python, glpk, python-glpk, cvxopt, python-matplotlib, python-numpy

On Ubuntu, this involves running: sudo apt-get install python-glpk <TODO: add other deps here>

The "hylaa" folder contains the python package. You should add the parent folder of the hylaa folder to your PYTHONPATH environment variable. On Linux, this can be done by updating your "~/.profile" or "~/.bashrc" to include:

export PYTHONPATH="${PYTHONPATH}:/path/to/parent/of/hylaa/folder"

### Example Model ###

You can try some of the examples by running the python scripts in the examples folder. Go to "examples/damped_oscillator" and run osc_hylaa.py from the command line ("python osc_hylaa.py"). Compare the plot hylaa creates to "spaceex_output.png" in the same folder (produced by SpaceEx on the same model).

### Input Format ###

Hylaa takes input python objects describing the hybrid automaton. You can see this in some of the example models (open the .py files in a text editor). The best way to produce these model files, if they are complicated at all, is using the Hyst model generator. With this approach, you can create models in the SpaceEx format using the SpaceEx model editor and convert the .xml and .cfg files SpaceEx would use into a runnable .py file. Many of the examples included with Hylaa were initially created with this approach (some of the settings may have been adjusted after generating the model). An example SpaceEx file you can try converting is in the "examples/damped_oscillator" folder (osc_spaceex.xml and osc_spaceex.cfg).

Hyst: https://github.com/stanleybak/hyst

SpaceEx Model Editor: http://spaceex.imag.fr/download-6

### Hylaa and Plot Settings ###

There are lots of settings that can be adjusted. The are listed in hylaa/containers.py in the HylaaSettings and PlotSettings objects. Most are self-explanatory or have comments in the source code describing what they do.

### Code Tests ###

A number of unit and regressions tests are included in the "tests" folder. To run all of these, simply run "make" in a terminal after changing to the "tests" directory. HyLAA uses pyunit, and "make" actually just runs "python -m unittest discover". If you are debugging, you can run an individual test script directly using something line "python test_star.py", An individual test method within a test script can also be run using something line "python -m unittest test_star.TestStar.test_hr_to_star".

### Code Contributions ###

We welcome external contributions to the code, although please submit high quality code with appropriate tests. Also ensure the code passes all the existing tests before submitting it. HyLAA uses the pylint static analysis tool to ensure reasonable code cleanliness, and generally try to eliminate every warning it raises. There is an included .pylintrc file with our pylint settings. Please ensure your code passes pylint's checks prior to submitting it. This is much easier if you integrate pylint into your development environment and correct the code as you are developing it.
