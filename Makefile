CC= gcc-4.9
CFLAGS= -Wall -g -O3
OPENMPFLAG= -fopenmp
LIBS=
UNAME = $(shell uname)
ifneq ($(UNAME),Darwin)
  LIBS += -lrt -lm
endif
EXEC = main

main: main.o neural_net.o matrix_helpers.o randomizing_helpers.o
	$(CC) $(CFLAGS) $(OPENMPFLAG) main.o neural_net.o matrix_helpers.o randomizing_helpers.o -o $(EXEC) $(LIBS)

main.o: main.c
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c main.c $(LIBS)

neural_net.o: neural_net.c neural_net.h
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c neural_net.c $(LIBS)

matrix_helpers.o: matrix_helpers.c matrix_helpers.h
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c matrix_helpers.c $(LIBS)

randomizing_helpers.o: randomizing_helpers.c randomizing_helpers.c
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c randomizing_helpers.c $(LIBS)

clean:
	rm *.o $(EXEC)
