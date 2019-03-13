# Dockerfile for Hylaa v2

FROM python:3.6

# ffmpeg is only needed if you want video (.mp4) export
RUN apt-get update && apt-get -qy install ffmpeg

# install other (required) dependencies
RUN pip3 install pytest numpy scipy sympy matplotlib termcolor swiglpk graphviz

# set environment variable
ENV PYTHONPATH=$PYTHONPATH:/hylaa

# copy current directory to docker
COPY . /hylaa

### As default command: run the tests ###
CMD python3 -m pytest /hylaa/tests

# USAGE:
# Build container and name it 'hylaa':
# docker build . -t hylaa

# # run tests (default command)
# docker run hylaa

# # get a shell:
# docker run -it hylaa bash
# hylaa is available in /hylaa
# to delete docker container use: docker rm hylaa
