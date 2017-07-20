# -*- indent-tabs-mode:t; -*-

CC=nvcc

CFLAGS_SHARED = --compiler-options '-fPIC -Wall -Werror' -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0 -shared
CFLAGS = -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0

all:  gpu_krylov_sim.so 

gpu_krylov_sim.so: gpu_krylov_sim.cu
	$(CC) $(CFLAGS_SHARED) gpu_krylov_sim.cu -o gpu_krylov_sim.so


clean:
	rm -fv gpu_krylov_sim.so 

