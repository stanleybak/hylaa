# Based on:
# https://betterscientificsoftware.github.io/
# python-for-hpc/tutorials/python-pypi-packaging/

from setuptools import setup
import versioneer

setup(
    name="hylaa",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Hylaa",
    url="https://github.com/stanleybak/hylaa",
    author="Stanley Bak",
    author_email="bak2007-DONTSENDMEEMAIL@gmail.com",
    license="GPLv3",
    packages=["hylaa"],
    install_requires=[
        "ffmpeg",
        "pytest",
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "termcolor",
        "swiglpk",
        "graphviz",
    ],
)
