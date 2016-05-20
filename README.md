# neural-net-OCR

## General Information
This project implements OCR with a neural net.

For a great introduction to neural nets, I highly suggest reading Michael Nielsen's online book
_Neural Networks and Deep Learning_, which is available for free at: http://neuralnetworksanddeeplearning.com/  
The general approach in this project was guided by this book.

  - Training and test datasets can be downloaded from MNIST: http://yann.lecun.com/exdb/mnist/
  - The dataset loader here was borrow from: https://github.com/projectgalateia/mnist

## Compiling
Clone the repo and `cd` into the project directory. Open the Makefile and change the first line to
your favorite compiler. Note that (at least on OSX) this must be the actual compiler name and not
an `alias` as the aliases are not resolved using make.

Then open up main.c and change the `OMP_NUM_THREADS_TRAINING` to the number of cores you would like
to use.
On Linux you can determine this by typing at the command line:
```
nproc
```
Or on OSX:
```
sysctl -n hw.ncpu
```

If you oversubscribe the number of cores, the program will run but your results will not be good.

Note that because this project uses OpenMP, **this project will NOT compile with clang**. It will compile on OSX assuming you are using a true version of `gcc` (or some other OpenMP compatible compiler).

## Parallelism
This neural net uses training example parallelism (exemplar parallelism) to optimize performance.
Parallelism is implemented with OpenMP.
