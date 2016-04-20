#ifndef NEURALNET_H
#define NEURALNET_H

#include "math.h"

typedef double nn_type;

struct neural_net {
  int number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers;
  int number_of_inputs;
  int number_of_outputs;
  int batch_size;
  nn_type eta;   // eta is the learning rate

  // Each entry in 'biases' is a pointer to an array -
  // one for each hidden layer and one for the output layer.
  // Note that the input layer doesn't have any biases.
  nn_type **bias;

  // Each entry in 'weights' is a pointer to a 'matrix' (really
  // an array that we'll treat as a matrix). Each matrix
  // represents the weights connecting two layers of neurons.
   nn_type **weight;

   // Each entry in 'z_vector' is a pointer to an array of the
   // z-values for the corresponding layer in the neural net.
   // The z-value is:
   //    weights * activation(previous level) + biases
   // This value is passes through the sigmoid function to get
   // the layer's activation.
    nn_type **z_vector;

    // Each entry in 'activation' is a pointer to an array of
    // the activations for the layer. The activation is the
    // z-vector passed through the sigmoid function.
    nn_type **activation;

    // Each entry in 'delta' is a pointer to an array of the
    // deltas for the layer. In the output layer delta has the
    //    Grad(Cost) . sigmoidPrime(z)
    //      = (Activation - y) . sigmoidPrime(z)
    // where the "." is the Hadamard product (element-wise
    // multiplication) and y is the expected output of the
    // neural net for a given input.
    nn_type **delta;
};

void destroy_nn(struct neural_net *nn);

void create_neural_net(struct neural_net *nn,
                       int number_of_hidden_layers,
                       int number_of_nodes_in_hidden_layers,
                       int number_of_inputs,
                       int number_of_outputs,
                       int batch_size,
                       nn_type eta);

void feed_forward(struct neural_net *nn,
                  nn_type *result,
                  nn_type *activation_initial,
                  int target_value,
                  char training,
                  int *count);

void backpropagate(struct neural_net *nn,
                   nn_type *activation_initial,
                   int target_value);

nn_type sigmoid(nn_type z);
nn_type sigmoidPrime(nn_type z);

#endif
