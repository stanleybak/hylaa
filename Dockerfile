# Dockerfile building a docker container with Hylaa v2

FROM ubuntu:18.04
# apt should not ask any questions:
ENV DEBIAN_FRONTEND=noninteractive

# if you want, add the following line to use a mirror which is nearer to you than the default archive.ubuntu.com (example: ftp.fau.de):
# RUN sed 's@archive.ubuntu.com@ftp.fau.de@' -i /etc/apt/sources.list


################################

# install dependencies
RUN apt-get update && apt-get -qy install curl unzip python3 python3-pip python3-matplotlib
RUN pip3 install pytest numpy scipy sympy matplotlib termcolor swiglpk graphviz

# set environment variable
ENV PYTHONPATH=$PYTHONPATH:/hylaa

# copy current directory to docker
COPY . /hylaa

# only for CI testing: Switch matplotlib backend to from TkAgg (interactive) to Agg (noninteractive).
RUN sed -i 's/^backend *: *TkAgg$/backend: Agg/i' /etc/matplotlibrc

##################
# As default command: run the tests
##################

CMD python3 -m pytest /hylaa/tests

# # USAGE:
# # Build container and name it 'hylaa':
# docker build . -t hylaa
# # run tests (default command)
# docker run hylaa
# # get a shell:
# docker run -it hylaa bash
# hylaa is available in /hylaa
# to delete docker container use: docker rm hylaa
