# -*- indent-tabs-mode:t; -*-

# after building the .so file, you can run 'ldd hylaa_glpk.so' on it to
# make sure it's linked with the correct glpk.so file

CC=g++

CFLAGS = -Wall -Werror -fPIC -O2 -std=c++0x  

# use glpk-4.60 installed from source (in /usr/local/lib/)
LDFLAGS = -L/usr/local/lib -lglpk -Wl,-rpath=/usr/local/lib

all: hylaa_glpk.so

hylaa_glpk.so: hylaa_glpk.cpp hylaa_glpk.h hylaa_glpk_tests.cpp hylaa_glpk_tests.h
	$(CC) -shared $(CFLAGS) hylaa_glpk.cpp hylaa_glpk_tests.cpp $(LDFLAGS) -o hylaa_glpk.so

clean:
	rm -fv hylaa_glpk.so

