# -*- indent-tabs-mode:t; -*-

CC=nvcc

#CUSP_FOLDER=cusplibrary-release-v0.6.0-rc
CUSP_FOLDER=cusplibrary-0.5.1

CFLAGS = --compiler-options '-O2 -fPIC -Wall -Werror' -O2 -Wno-deprecated-gpu-targets -D_FORCE_INLINES -I $(CUSP_FOLDER)
CFLAGS_SHARED = $(CFLAGS) -shared

all: cusp_krylov_stan.so

cusp_krylov_tran.so: cusp_krylov_tran.cu
	$(CC) $(CFLAGS_SHARED) cusp_krylov_tran.cu -o cusp_krylov_tran.so

cusp_krylov_stan.so: cusp_krylov_stan.cu gpu_util.h
	$(CC) $(CFLAGS_SHARED) -lcublas cusp_krylov_stan.cu -o cusp_krylov_stan.so

clean:
	rm -fv  cusp_krylov_tran.so cusp_krylov_stan.so test_cusp_krylov_stan

