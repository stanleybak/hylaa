[![Build Status](https://travis-ci.org/stanleybak/python-sparse-glpk.svg?branch=deploy)](https://travis-ci.org/stanleybak/python-sparse-glpk)

This a custom lightweight C++ interface to linear programming with GLPK for Python code, using efficient scipy sparse matrices for setting the constraints. GLPK is a free library for solving linear programs. This interface connects GLPK with python code using python's ctypes library.


This was created for efficiency and various bugs encountered with other libraries. cvxopt did not support the warm-start LP interface to glpk, and python-glpk was excessively slow in setting up the LP.

To setup, first install the required packages (see .travis.yml for specific version numbers):

`sudo apt install python python-numpy python-scipy python-pytest cvxopt g++ make libglpk-dev`

To run the unit tests, first build the .so file using make, then use:

`py.test`


The code is based off of glpk-4.65 (newer than what's available in the ubuntu packages at the time of this writing), so you may need to install that from source: https://ftp.gnu.org/gnu/glpk/ . This should install using the standard "./configure", "make", "sudo make install" process.

Commands:

```
wget https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
tar -xvf glpk-4.65.tar.gz
cd glpk-4.65
./configure
make
sudo make install
```

or in one command:

`wget https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz && tar -xvf glpk-4.65.tar.gz && cd glpk-4.65 && ./configure && make && sudo make install`


Created by Stanley Bak, 2018
Licensed under GPL v3
