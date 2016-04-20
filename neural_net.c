#include "neural_net.h"
#include "time.h"
#include "matrix_helpers.h"
#include <stdio.h>
#include <stdlib.h>

// This function sets the various 'hyperparameters' (i.e. learning rate,
// number of layers, nodes per layer, etc.) It also intiializes the
// arrays and matrices used in the neural net. All arrays and matrices
// are initializes to a random value between -1 and 1 in order to give
// the neural net a place to start.
void create_neural_net(struct neural_net *nn,
                       int number_of_hidden_layers,
                       int number_of_nodes_in_hidden_layers,
                       int number_of_inputs,
                       int number_of_outputs,
                       nn_type eta)
{
  int i, j;
  // srand(time(NULL));
  srand(1461103727);

  (*nn).number_of_hidden_layers          = number_of_hidden_layers;
  (*nn).number_of_nodes_in_hidden_layers = number_of_nodes_in_hidden_layers;
  (*nn).number_of_inputs                 = number_of_inputs;
  (*nn).number_of_outputs                = number_of_outputs;
  (*nn).eta                              = eta;

  //---------------------------------------------------------------------------
  // allocate space for our 'bias', 'z_vector', 'activation' and 'delta' arrays and initialize them
  (*nn).bias       = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*nn).z_vector   = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*nn).activation = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*nn).delta      = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this loop does the hidden layers
  for (i=0; i<number_of_hidden_layers; i++) {
    (*nn).bias[i]       = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    (*nn).z_vector[i]   = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    (*nn).activation[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    (*nn).delta[i]      = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    for (j=0; j<number_of_nodes_in_hidden_layers; j++) {
      (*nn).bias[i][j]       = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
      (*nn).z_vector[i][j]   = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
      (*nn).activation[i][j] = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
      (*nn).delta[i][j]      = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
    }
  }
  //
  // this does the output layer
  (*nn).bias[number_of_hidden_layers]       = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
  (*nn).z_vector[number_of_hidden_layers]   = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
  (*nn).activation[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
  (*nn).delta[number_of_hidden_layers]      = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
  for (i=0; i<number_of_outputs; i++) {
    (*nn).bias[number_of_hidden_layers][i]       = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
    (*nn).z_vector[number_of_hidden_layers][i]   = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
    (*nn).activation[number_of_hidden_layers][i] = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
    (*nn).delta[number_of_hidden_layers][i]      = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // allocate space for our 'weight' array and initialize it
  (*nn).weight = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this does the first hidden layer to the input layer
  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  (*nn).weight[0] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
  for (i=0; i<number_of_matrix_elements; i++) {
    (*nn).weight[0][i] = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
  }
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  for (i=1; i<number_of_hidden_layers; i++) {
    (*nn).weight[i] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
    for (j=0; j<number_of_matrix_elements; j++) {
      (*nn).weight[i][j] = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
    }
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  (*nn).weight[number_of_hidden_layers] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
  for (j=0; j<number_of_matrix_elements; j++) {
    (*nn).weight[number_of_hidden_layers][j] = ((nn_type)rand() / ( (nn_type)RAND_MAX / 2.0 )) - 1.0;
  }
  //---------------------------------------------------------------------------
}

void destroy_nn(struct neural_net *nn)
{
  int i;
  int number_of_hidden_layers = (*nn).number_of_hidden_layers;

  // free 'bias'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).bias[i]);
  free ((*nn).bias);

  // free 'weight'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).weight[i]);
  free((*nn).weight);

  // free 'z-vector'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).z_vector[i]);
  free ((*nn).z_vector);

  // free 'activation'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).activation[i]);
  free ((*nn).activation);

  // free 'delta'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).delta[i]);
  free ((*nn).delta);
}

void feed_forward(struct neural_net *nn,
                  nn_type *result,
                  nn_type *activation_initial,
                  int target_value,
                  char training,
                  int *count)
{
  int i, j;
  int row, col;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_outputs                = (*nn).number_of_outputs;
  nn_type *tmp_activation              = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
  nn_type *last_activation             = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );

  //---------------------------------------------------------------------------
  // feed from input layer -> first hidden layer
  //  do dot product
  matrix_vector_multiply((*nn).z_vector[0],
                        (*nn).weight[0],
                        activation_initial,
                        number_of_nodes_in_hidden_layers,
                        number_of_inputs);
  //  compute the z-vector
  add_vectors((*nn).z_vector[0],
              (*nn).bias[0],
              number_of_nodes_in_hidden_layers);
  //  compute activation
  sigmoidify((*nn).activation[0],
             (*nn).z_vector[0],
             number_of_nodes_in_hidden_layers);
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // feed through the hidden layers
  for(i=1; i<number_of_hidden_layers; i++) {
    //  do dot product
    matrix_vector_multiply((*nn).z_vector[i],
                           (*nn).weight[i],
                           (*nn).activation[i-1],
                           number_of_nodes_in_hidden_layers,
                           number_of_nodes_in_hidden_layers);
    //  compute the z-vector
    add_vectors((*nn).z_vector[i],
                (*nn).bias[i],
                number_of_nodes_in_hidden_layers);
    //  compute activation
    sigmoidify((*nn).activation[i],
               (*nn).z_vector[i],
               number_of_nodes_in_hidden_layers);
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // feed from the last hidden layer -> output layer
  //  do dot product
  matrix_vector_multiply((*nn).z_vector[number_of_hidden_layers],
                         (*nn).weight[number_of_hidden_layers],
                         (*nn).activation[number_of_hidden_layers-1],
                         number_of_outputs,
                         number_of_nodes_in_hidden_layers);
  //  compute the z-vector
  add_vectors((*nn).z_vector[number_of_hidden_layers],
              (*nn).bias[number_of_hidden_layers],
              number_of_outputs);
  //  compute activation
  sigmoidify((*nn).activation[number_of_hidden_layers],
             (*nn).z_vector[number_of_hidden_layers],
             number_of_outputs);
  //---------------------------------------------------------------------------

  for (i=0; i<number_of_outputs; i++) result[i] = (*nn).activation[number_of_hidden_layers][i];

  if (training) backpropagate(nn,
                              activation_initial,
                              target_value);
  else {
    nn_type max = 0.0;
    int max_index = 0;
    for (i=0; i<number_of_outputs; i++) {
      if ((*nn).activation[number_of_hidden_layers][i] > max) {
        max = (*nn).activation[number_of_hidden_layers][i];
        max_index = i;
      }
    }
    if (max_index == target_value) (*count)++;
  }
}

void backpropagate(struct neural_net *nn,
                   nn_type *activation_initial,
                   int target_value)
{
  // printf("backpropagating with %i\n", target_value);
  int i;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_outputs                = (*nn).number_of_outputs;
  int eta                              = (*nn).eta;

  // find the delta value in the output layer
  delta_output_layer((*nn).delta[number_of_hidden_layers],
                     (*nn).activation[number_of_hidden_layers],
                     (*nn).z_vector[number_of_hidden_layers],
                     target_value,
                     number_of_outputs);

  // backpropagate delta -> last hidden layer
  //  Note that row, col dimensions here are for the matrix W
  //  NOT the transpose of W. The transpose will be taken care
  //  of in the function.
  delta_hidden_layers((*nn).delta[number_of_hidden_layers-1],
                      (*nn).weight[number_of_hidden_layers],
                      (*nn).delta[number_of_hidden_layers],
                      (*nn).z_vector[number_of_hidden_layers-1],
                      number_of_outputs,
                      number_of_nodes_in_hidden_layers);

  // backpropagate delta -> hidden layers
  for (i=number_of_hidden_layers-2; i>=0; i--) {
    delta_hidden_layers((*nn).delta[i],
                        (*nn).weight[i+1],
                        (*nn).delta[i+1],
                        (*nn).z_vector[i],
                        number_of_nodes_in_hidden_layers,
                        number_of_nodes_in_hidden_layers);
  }

  // -----------------------------------------------------------------
  // now that we have all of our deltas, adjust the weights and biases
  //  adjust the first hidden layer
  adjust_weight(activation_initial,
                (*nn).weight[0],
                (*nn).delta[0],
                number_of_nodes_in_hidden_layers,
                number_of_inputs,
                eta,
                1);
  adjust_bias((*nn).bias[0],
              (*nn).delta[0],
              number_of_nodes_in_hidden_layers,
              eta,
              1);
  //
  //  adjust the hidden layers
  for (i=1; i<number_of_hidden_layers; i++) {
    adjust_weight((*nn).activation[i-1],
                  (*nn).weight[i],
                  (*nn).delta[i],
                  number_of_nodes_in_hidden_layers,
                  number_of_nodes_in_hidden_layers,
                  eta,
                  1);
    adjust_bias((*nn).bias[i],
                (*nn).delta[i],
                number_of_nodes_in_hidden_layers,
                eta,
                1);
  }
  //
  //  adjust the output hidden layer
  adjust_weight((*nn).activation[number_of_hidden_layers-1],
                (*nn).weight[number_of_hidden_layers],
                (*nn).delta[number_of_hidden_layers],
                number_of_outputs,
                number_of_nodes_in_hidden_layers,
                eta,
                1);
  adjust_bias((*nn).bias[number_of_hidden_layers],
              (*nn).delta[number_of_hidden_layers],
              number_of_outputs,
              eta,
              1);
  // -----------------------------------------------------------------
}

nn_type sigmoid(nn_type z) {
  return 1.0 / (1.0 + exp(-z));
}

nn_type sigmoidPrime(nn_type z) {
  nn_type exponential = exp(z);
  return exponential / pow(exponential + 1, 2);
}
