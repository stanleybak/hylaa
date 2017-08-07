# -*- indent-tabs-mode:t; -*-

CC=nvcc

#CFLAGS_SHARED = --compiler-options '-fPIC -Wall -Werror' -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0 -shared
CFLAGS_SHARED = --compiler-options '-fPIC -Wall' -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0 -shared
CFLAGS = -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0

all:  cusp_krylov_sim_CPU.so cusp_krylov_sim_GPU.so 

cusp_krylov_sim_CPU.so: cusp_krylov_sim_CPU.cu
	$(CC) $(CFLAGS_SHARED) cusp_krylov_sim_CPU.cu -o cusp_krylov_sim_CPU.so

cusp_krylov_sim_GPU.so: cusp_krylov_sim_GPU.cu
	$(CC) $(CFLAGS_SHARED) cusp_krylov_sim_GPU.cu -o cusp_krylov_sim_GPU.so

clean:
	rm -fv  cusp_krylov_sim_CPU.so cusp_krylov_sim_GPU.so 

