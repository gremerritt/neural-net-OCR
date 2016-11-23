# neural-net-OCR

## General Information
This project implements OCR with a Neural Network.

You can find my project writeup [here](https://www.dropbox.com/s/wwxyf4n9zavz8xk/Neural_Network_Multicore_Project.pdf?dl=0).

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

Note that because this project uses OpenMP, **this project will NOT compile with clang**. It will compile on OSX assuming you are using a true version of `gcc` (or some other OpenMP compatible compiler). You will need to update the `CC` (compiler) in the `Makefile` to match the name of the compiler you are using.

Then in the project directory simply run

    $ make

to compile to project.

## Running

First, download the 4 MNIST training and testing data sets from [the MNIST website](http://yann.lecun.com/exdb/mnist/). Make sure they are in this project directory.

By default the Neural Net will run with the following configuration

    Total Layers:           4
    Hidden Layers:          2
    Inputs:                 784
    Outputs:                10
    Nodes in Hidden Layers: 60
    Batch Size:             5
    Learning Rate:          1.500000

The number of hidden layers, nodes in the hidden layers, batch size, and learning rate are all configurable on the command line.

    --hidden-layers=4 [or --hl=4]
    --hidden-nodes=100 [or --hn=100]
    --batch-size=10 [or --bs=10]
    --learning-rate=1.2 [or --lr=1.2]

To run the program in the project directory:

    $ ./main [options]

## Parallelism
This neural net uses training example parallelism (exemplar parallelism) to optimize performance. Parallelism is implemented with OpenMP.
