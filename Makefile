CC= gcc
NVCC= nvcc
CFLAGS= -Wall -g
CUDAFLAGS= -g
OPENMPFLAG= -fopenmp
LIBS= -lrt -lm

main: main.o neural_net.o matrix_helpers.o randomizing_helpers.o
	$(CC) $(CFLAGS) main.o neural_net.o matrix_helpers.o randomizing_helpers.o -o main $(LIBS)

osx: neural_net_osx.o matrix_helpers_osx.o randomizing_helpers_osx.o
	$(CC) $(CFLAGS) main.c neural_net_osx.o matrix_helpers_osx.o randomizing_helpers_osx.o -o main

cuda: neural_net_cuda.o matrix_helpers_cuda.o randomizing_helpers_cuda.o
	$(NVCC) $(CUDAFLAGS) main.c neural_net_cuda.o matrix_helpers_cuda.o randomizing_helpers_cuda.o -o main

neural_net.o: neural_net.c neural_net.h
	$(CC) $(CFLAGS) -c neural_net.c $(LIBS)

matrix_helpers.o: matrix_helpers.c matrix_helpers.h
	$(CC) $(CFLAGS) -c matrix_helpers.c -O3 $(LIBS)

randomizing_helpers.o: randomizing_helpers.c randomizing_helpers.c
	$(CC) $(CFLAGS) -c randomizing_helpers.c $(LIBS)

neural_net_osx.o: neural_net.c neural_net.h
	$(CC) $(CFLAGS) -c neural_net.c -o neural_net_osx.o

matrix_helpers_osx.o: matrix_helpers.c matrix_helpers.h
	$(CC) $(CFLAGS) -c matrix_helpers.c -O3 -o matrix_helpers_osx.o

randomizing_helpers_osx.o: randomizing_helpers.c randomizing_helpers.c
	$(CC) $(CFLAGS) -c randomizing_helpers.c -o randomizing_helpers_osx.o

neural_net_cuda.o: neural_net.c neural_net.h
	$(NVCC) $(CUDAFLAGS) -c neural_net.c -o neural_net_cuda.o

matrix_helpers_cuda.o: matrix_helpers.c matrix_helpers.h
	$(NVCC) $(CUDAFLAGS) -c matrix_helpers.c -O3 -o matrix_helpers_cuda.o

randomizing_helpers_cuda.o: randomizing_helpers.c randomizing_helpers.c
	$(NVCC) $(CUDAFLAGS) -c randomizing_helpers.c -o randomizing_helpers_cuda.o

clean:
	rm *.o main
