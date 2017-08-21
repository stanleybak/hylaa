# -*- indent-tabs-mode:t; -*-

CC=nvcc

CFLAGS = --compiler-options '-O2 -fPIC -Wall -Werror' -O2 -Wno-deprecated-gpu-targets -D_FORCE_INLINES -Icusplibrary-0.5.1
CFLAGS_SHARED = $(CFLAGS) -shared

all: cusp_krylov_stan.so

cusp_krylov_tran.so: cusp_krylov_tran.cu
	$(CC) $(CFLAGS_SHARED) cusp_krylov_tran.cu -o cusp_krylov_tran.so

cusp_krylov_stan.so: cusp_krylov_stan.cu gpu_util.h
	$(CC) $(CFLAGS_SHARED) cusp_krylov_stan.cu -o cusp_krylov_stan.so

clean:
	rm -fv  cusp_krylov_tran.so cusp_krylov_stan.so test_cusp_krylov_stan

