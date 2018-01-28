# -*- indent-tabs-mode:t; -*-

# after building the .so file, you can run 'ldd hylaa_glpk.so' on it to
# make sure it's linked with the correct glpk.so file

CC=g++

CFLAGS = -Wall -Werror -fPIC -O3 -std=c++0x

LDFLAGS = -lpthread

all: test_performance_c.so

test_performance_c.so: test_performance_c.cpp
	$(CC) -shared $(CFLAGS) test_performance_c.cpp $(LDFLAGS) -o test_performance_c.so

clean:
	rm -fv test_performance_c.so

