# -*- indent-tabs-mode:t; -*-

all: test

test: test.cu
	nvcc -D_FORCE_INLINES -O3 test.cu -lcublas -lcurand -Wno-deprecated-gpu-targets -o test

clean:
	rm -fv test
