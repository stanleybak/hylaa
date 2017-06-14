# -*- indent-tabs-mode:t; -*-

CC=nvcc

CFLAGS = --compiler-options '-fPIC' -O2 -D_FORCE_INLINES -Icusplibrary-0.4.0 

all: gpu_mult.so

gpu_mult.so: gpu_mult.cu 
	$(CC) -shared $(CFLAGS) gpu_mult.cu -o gpu_mult.so

clean:
	rm -fv gpu_mult.so

