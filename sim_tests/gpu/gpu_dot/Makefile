# -*- indent-tabs-mode:t; -*-

CC=nvcc

#CUSP_FOLDER=cusplibrary-release-v0.6.0-rc
CUSP_FOLDER=cusplibrary-0.5.1

CFLAGS = --compiler-options '-O2 -fPIC -Wall -Werror' -O2 -Wno-deprecated-gpu-targets -D_FORCE_INLINES -I $(CUSP_FOLDER)

all: test

test: test.cu
	$(CC) $(CFLAGS) -lcublas test.cu -o test

clean:
	rm -fv test

