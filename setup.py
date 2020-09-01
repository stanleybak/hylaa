# Based on: 
# https://betterscientificsoftware.github.io/
# python-for-hpc/tutorials/python-pypi-packaging/

from setuptools import setup

setup(
    name='hylaa',
    version='2.0',    
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Stephen Hudson',
    author_email='shudson@anl.gov',
    license='BSD 2-clause',
    packages=['hylaa'],
    install_requires=['ffmpeg',
                      'pytest', 
                      'numpy', 
                      'scipy', 
                      'sympy', 
                      'matplotlib', 
                      'termcolor',
                      'swiglpk', 
                      'graphviz'
                      ]
)