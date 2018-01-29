# -*- indent-tabs-mode:t; -*-

NAME=fast_mult
CC=g++
CFLAGS = -Wall -Werror -fPIC -O3 -std=c++0x
LDFLAGS = -lpthread

all: $(NAME).so

$(NAME).so: $(NAME).cpp
	$(CC) -shared $(CFLAGS) $(NAME).cpp $(LDFLAGS) -o $(NAME).so

clean:
	rm -fv $(NAME).so

