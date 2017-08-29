# -*- indent-tabs-mode:t; -*-

CC=nvcc

CFLAGS_SHARED = --compiler-options '-fPIC -Wall -Werror' -O2 -D_FORCE_INLINES -Icusplibrary-0.5.1 -shared
CFLAGS = -O2 -D_FORCE_INLINES -Icusplibrary-0.5.1

all: gpu_mult.so gpu_arnoldi.so gpu_krylov_sim.so gpu_arnoldi_test gpu_flops

gpu_flops: gpu_flops.cu 
	$(CC) $(CFLAGS) gpu_flops.cu -o gpu_flops

gpu_mult.so: gpu_mult.cu 
	$(CC) $(CFLAGS_SHARED) gpu_mult.cu -o gpu_mult.so

gpu_arnoldi.so: gpu_arnoldi.cu
	$(CC) $(CFLAGS_SHARED) gpu_arnoldi.cu -o gpu_arnoldi.so

gpu_krylov_sim.so: gpu_krylov_sim.cu
	$(CC) $(CFLAGS_SHARED) gpu_krylov_sim.cu -o gpu_krylov_sim.so

gpu_arnoldi_test: gpu_arnoldi_test.cu
	$(CC) $(CFLAGS) gpu_arnoldi_test.cu -o gpu_arnoldi_test

clean:
	rm -fv gpu_mult.so gpu_arnoldi.so gpu_krylov_sim.so gpu_arnoldi_test flops

