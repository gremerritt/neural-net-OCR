main: main.o neural_net.o matrix_helpers.o
	gcc main.o neural_net.o matrix_helpers.o -o main

neural_net.o: neural_net.c neural_net.h
	gcc -c neural_net.c

matrix_helpers.o: matrix_helpers.c matrix_helpers.h
	gcc -c matrix_helpers.c

clean:
	rm *.o main
