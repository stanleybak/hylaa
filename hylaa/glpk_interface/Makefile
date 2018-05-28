# -*- indent-tabs-mode:t; -*-

CC=g++

CFLAGS = -Wall -Werror -fPIC -O2 -std=c++11

# use a version of glpk installed from source (in /usr/local/lib/)
#LDFLAGS = -L/usr/local/lib -lglpk -Wl,-rpath=/usr/local/lib
# after building the .so file, you can run 'ldd python_glpk.so' on it to
# make sure it's linked with the correct glpk.so file


# use default glpk binary
LDFLAGS = -lglpk

all: python_glpk.so

python_glpk.so: python_glpk.c 
	$(CC) -shared $(CFLAGS) python_glpk.c $(LDFLAGS) -o python_glpk.so

test: python_glpk.c
	$(CC) $(CFLAGS) python_glpk.c $(LDFLAGS) -o test

clean:
	rm -fv python_glpk.so test

