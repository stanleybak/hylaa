# Based on:
# https://betterscientificsoftware.github.io/
# python-for-hpc/tutorials/python-pypi-packaging/

from setuptools import setup
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hylaa",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Hylaa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanleybak/hylaa",
    author="Stanley Bak",
    author_email="bak2007-DONTSENDMEEMAIL@gmail.com",
    license="GPLv3",
    packages=["hylaa"],
    install_requires=[
        "ffmpeg-python",
        "pytest",
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "termcolor",
        "swiglpk",
        "graphviz",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
)
