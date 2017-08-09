# -*- indent-tabs-mode:t; -*-

CC=nvcc

CFLAGS_SHARED = --compiler-options '-O2 -fPIC -Wall -Werror' -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0 -shared
CFLAGS = -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0

all:  cusp_krylov.so

cusp_krylov.so: cusp_krylov.cu
	$(CC) $(CFLAGS_SHARED) cusp_krylov.cu -o cusp_krylov.so

clean:
	rm -fv  cusp_krylov.so 

