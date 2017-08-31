# -*- indent-tabs-mode:t; -*-

all: test

test: test.cu
	nvcc -O3 test.cu -lcublas -lcurand -Wno-deprecated-gpu-targets -o test

clean:
	rm -fv test
