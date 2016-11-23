#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "neural_net.h"
#include "matrix_helpers.h"
#include "randomizing_helpers.h"

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
                       int batch_size,
                       nn_type eta)
{
  int i, j, random_count;
  random_count = number_of_hidden_layers * number_of_nodes_in_hidden_layers;
  random_count += number_of_outputs;
  random_count += number_of_nodes_in_hidden_layers * (number_of_inputs +
                                                      (number_of_hidden_layers * number_of_nodes_in_hidden_layers ) +
                                                      number_of_outputs);
  double *random = malloc( random_count * sizeof(double) );
  generate_guassian_distribution(random, random_count);
  random_count = 0;

  (*nn).number_of_hidden_layers          = number_of_hidden_layers;
  (*nn).number_of_nodes_in_hidden_layers = number_of_nodes_in_hidden_layers;
  (*nn).number_of_inputs                 = number_of_inputs;
  (*nn).number_of_outputs                = number_of_outputs;
  (*nn).batch_size                       = batch_size;
  (*nn).eta                              = eta;

  //---------------------------------------------------------------------------
  // allocate space for our 'bias', 'z_vector', 'activation' and 'delta' arrays and initialize them
  (*nn).bias       = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*nn).z_matrix   = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*nn).activation = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*nn).delta      = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this loop does the hidden layers
  for (i=0; i<number_of_hidden_layers; i++) {
    (*nn).bias[i]       = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    (*nn).delta[i]      = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    (*nn).z_matrix[i]   = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    (*nn).activation[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    for (j=0; j<number_of_nodes_in_hidden_layers; j++)
      (*nn).bias[i][j] = random[random_count++];
  }
  //
  // this does the output layer
  (*nn).bias[number_of_hidden_layers]       = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
  (*nn).delta[number_of_hidden_layers]      = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
  (*nn).z_matrix[number_of_hidden_layers]   = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
  (*nn).activation[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
  for (i=0; i<number_of_outputs; i++)
    (*nn).bias[number_of_hidden_layers][i] = random[random_count++];
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // allocate space for our 'weight' array and initialize it
  (*nn).weight = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this does the first hidden layer to the input layer
  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  (*nn).weight[0] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
  for (i=0; i<number_of_matrix_elements; i++)
    (*nn).weight[0][i] = random[random_count++];
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  for (i=1; i<number_of_hidden_layers; i++) {
    (*nn).weight[i] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
    for (j=0; j<number_of_matrix_elements; j++)
      (*nn).weight[i][j] = random[random_count++];
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  (*nn).weight[number_of_hidden_layers] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
  for (j=0; j<number_of_matrix_elements; j++)
    (*nn).weight[number_of_hidden_layers][j] = random[random_count++];
  //---------------------------------------------------------------------------

  free(random);
}

void duplicate_nn(struct neural_net *new_nn, struct neural_net *nn)
{
  int i, j;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_outputs                = (*nn).number_of_outputs;
  int batch_size                       = (*nn).batch_size;

  (*new_nn).number_of_hidden_layers          = number_of_hidden_layers;
  (*new_nn).number_of_nodes_in_hidden_layers = number_of_nodes_in_hidden_layers;
  (*new_nn).number_of_inputs                 = number_of_inputs;
  (*new_nn).number_of_outputs                = number_of_outputs;
  (*new_nn).batch_size                       = batch_size;
  (*new_nn).eta                              = (*nn).eta;

  (*new_nn).bias       = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*new_nn).z_matrix   = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*new_nn).activation = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  (*new_nn).delta      = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );

  // this loop does the hidden layers
  for (i=0; i<number_of_hidden_layers; i++) {
    (*new_nn).bias[i]       = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    (*new_nn).delta[i]      = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    (*new_nn).z_matrix[i]   = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    (*new_nn).activation[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    for (j=0; j<number_of_nodes_in_hidden_layers; j++)
      (*new_nn).bias[i][j] = (*nn).bias[i][j];
  }
  //
  // this does the output layer
  (*new_nn).bias[number_of_hidden_layers]       = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
  (*new_nn).delta[number_of_hidden_layers]      = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
  (*new_nn).z_matrix[number_of_hidden_layers]   = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
  (*new_nn).activation[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
  for (i=0; i<number_of_outputs; i++)
    (*new_nn).bias[number_of_hidden_layers][i] = (*nn).bias[number_of_hidden_layers][i];
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // allocate space for our 'weight' array and initialize it
  (*new_nn).weight = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this does the first hidden layer to the input layer
  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  (*new_nn).weight[0] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
  for (i=0; i<number_of_matrix_elements; i++)
    (*new_nn).weight[0][i] = (*nn).weight[0][i];
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  for (i=1; i<number_of_hidden_layers; i++) {
    (*new_nn).weight[i] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
    for (j=0; j<number_of_matrix_elements; j++)
      (*new_nn).weight[i][j] = (*nn).weight[i][j];
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  (*new_nn).weight[number_of_hidden_layers] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
  for (j=0; j<number_of_matrix_elements; j++)
    (*new_nn).weight[number_of_hidden_layers][j] = (*nn).weight[number_of_hidden_layers][j];
  //---------------------------------------------------------------------------
}

void sync_nn(struct neural_net *target_nn, struct neural_net *source_nn)
{
  int i, j;
  int number_of_hidden_layers          = (*source_nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*source_nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*source_nn).number_of_inputs;
  int number_of_outputs                = (*source_nn).number_of_outputs;

  #pragma omp parallel for shared(number_of_hidden_layers, number_of_nodes_in_hidden_layers, target_nn, source_nn) \
                           private(i, j)
  for (i=0; i<number_of_hidden_layers; i++) {
    for (j=0; j<number_of_nodes_in_hidden_layers; j++)
      (*target_nn).bias[i][j] = (*source_nn).bias[i][j];
  }
  #pragma omp parallel for shared(number_of_outputs, number_of_hidden_layers, target_nn, source_nn) \
                           private(i)
  for (i=0; i<number_of_outputs; i++)
    (*target_nn).bias[number_of_hidden_layers][i] = (*source_nn).bias[number_of_hidden_layers][i];

  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_matrix_elements, target_nn, source_nn) \
                           private(i)
  for (i=0; i<number_of_matrix_elements; i++)
    (*target_nn).weight[0][i] = (*source_nn).weight[0][i];
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_hidden_layers, number_of_matrix_elements, target_nn, source_nn) \
                           private(i, j)
  for (i=1; i<number_of_hidden_layers; i++) {
    for (j=0; j<number_of_matrix_elements; j++)
      (*target_nn).weight[i][j] = (*source_nn).weight[i][j];
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_matrix_elements, number_of_hidden_layers, target_nn, source_nn) \
                           private(i)
  for (i=0; i<number_of_matrix_elements; i++)
    (*target_nn).weight[number_of_hidden_layers][i] = (*source_nn).weight[number_of_hidden_layers][i];
}

void destroy_nns(struct neural_net *nns, int num)
{
  int i;
  for (i=0; i<num; i++)
    destroy_nn(&(nns[i]));
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
    free((*nn).z_matrix[i]);
  free ((*nn).z_matrix);

  // free 'activation'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).activation[i]);
  free ((*nn).activation);

  // free 'delta'
  for (i=0; i<number_of_hidden_layers + 1; i++)
    free((*nn).delta[i]);
  free ((*nn).delta);
}

void initialize_change_matrices(struct change_matrices *cm, struct neural_net *nn)
{
  int i;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_outputs                = (*nn).number_of_outputs;

  (*cm).bias_change = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this loop does the hidden layers
  for (i=0; i<number_of_hidden_layers; i++) {
    (*cm).bias_change[i] = (nn_type *)calloc( number_of_nodes_in_hidden_layers, sizeof( nn_type ) );
  }
  //
  // this does the output layer
  (*cm).bias_change[number_of_hidden_layers] = (nn_type *)calloc( number_of_outputs, sizeof( nn_type ) );
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // allocate space for our 'weight' array and initialize it
  (*cm).weight_change = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
  //
  // this does the first hidden layer to the input layer
  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  (*cm).weight_change[0] = (nn_type *)calloc( number_of_matrix_elements, sizeof( nn_type ) );
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  for (i=1; i<number_of_hidden_layers; i++) {
    (*cm).weight_change[i] = (nn_type *)calloc( number_of_matrix_elements, sizeof( nn_type ) );
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  (*cm).weight_change[number_of_hidden_layers] = (nn_type *)calloc( number_of_matrix_elements, sizeof( nn_type ) );
  //---------------------------------------------------------------------------
}

void get_changes(struct change_matrices *cm, struct neural_net *new_nn, struct neural_net *original_nn)
{
  int i, j;
  int number_of_hidden_layers          = (*original_nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*original_nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*original_nn).number_of_inputs;
  int number_of_outputs                = (*original_nn).number_of_outputs;

  #pragma omp parallel for shared(number_of_hidden_layers, number_of_nodes_in_hidden_layers, cm, new_nn, original_nn) \
                           private(i, j)
  for (i=0; i<number_of_hidden_layers; i++) {
    for (j=0; j<number_of_nodes_in_hidden_layers; j++)
      (*cm).bias_change[i][j] += (*new_nn).bias[i][j] - (*original_nn).bias[i][j];
  }
  #pragma omp parallel for shared(number_of_outputs, number_of_hidden_layers, cm, new_nn, original_nn) \
                           private(i)
  for (i=0; i<number_of_outputs; i++)
    (*cm).bias_change[number_of_hidden_layers][i] += (*new_nn).bias[number_of_hidden_layers][i] -
                                                     (*original_nn).bias[number_of_hidden_layers][i];

  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_matrix_elements, cm, new_nn, original_nn) \
                           private(i)
  for (i=0; i<number_of_matrix_elements; i++)
    (*cm).weight_change[0][i] += (*new_nn).weight[0][i] - (*original_nn).weight[0][i];
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_hidden_layers, number_of_matrix_elements, cm, new_nn, original_nn) \
                           private(i, j)
  for (i=1; i<number_of_hidden_layers; i++) {
    for (j=0; j<number_of_matrix_elements; j++)
      (*cm).weight_change[i][j] += (*new_nn).weight[i][j] - (*original_nn).weight[i][j];
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_hidden_layers, number_of_matrix_elements, cm, new_nn, original_nn) \
                           private(i, j)
  for (i=0; i<number_of_matrix_elements; i++)
    (*cm).weight_change[number_of_hidden_layers][i] += (*new_nn).weight[number_of_hidden_layers][i] -
                                                      (*original_nn).weight[number_of_hidden_layers][i];
}

void apply_changes(struct change_matrices *cm, struct neural_net *nn, int threads)
{
  int i, j;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_outputs                = (*nn).number_of_outputs;
  double dividor = (double)threads;

  #pragma omp parallel for shared(number_of_hidden_layers, number_of_nodes_in_hidden_layers, cm, nn) \
                           private(i, j)
  for (i=0; i<number_of_hidden_layers; i++) {
    for (j=0; j<number_of_nodes_in_hidden_layers; j++)
      (*nn).bias[i][j] += (*cm).bias_change[i][j] / dividor;
  }
  #pragma omp parallel for shared(number_of_outputs, number_of_hidden_layers, cm, nn) \
                           private(i)
  for (i=0; i<number_of_outputs; i++)
    (*nn).bias[number_of_hidden_layers][i] += (*cm).bias_change[number_of_hidden_layers][i] / dividor;

  int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_matrix_elements, cm, nn) \
                           private(i)
  for (i=0; i<number_of_matrix_elements; i++)
    (*nn).weight[0][i] += (*cm).weight_change[0][i] / dividor;
  //
  // this does all the hidden layers
  number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_hidden_layers, number_of_matrix_elements, cm, nn) \
                           private(i, j)
  for (i=1; i<number_of_hidden_layers; i++) {
    for (j=0; j<number_of_matrix_elements; j++)
      (*nn).weight[i][j] += (*cm).weight_change[i][j] / dividor;
  }
  //
  // this does the output layer to the last hidden layer
  number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
  #pragma omp parallel for shared(number_of_matrix_elements, number_of_hidden_layers, cm, nn) \
                           private(i, j)
  for (i=0; i<number_of_matrix_elements; i++)
    (*nn).weight[number_of_hidden_layers][i] += (*cm).weight_change[number_of_hidden_layers][i] / dividor;
}

void feed_forward(struct neural_net *nn,
                  nn_type *result,
                  nn_type *activation_initial,
                  int *target_value,
                  char training,
                  int *count,
                  char *correct)
{
  int i, j;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_outputs                = (*nn).number_of_outputs;
  int batch_size                       = (*nn).batch_size;

  //---------------------------------------------------------------------------
  // feed from input layer -> first hidden layer
  //  do matrix multiply
  //    Dimensions:
  //      Weight Matrix:      'Nodes in target layer' rows X 'Nodes in source layer' columns
  //      Activation Matric:  'Nodes in source layer' rows X 'Batch size' columns
  //  Get the z-matrix
  calculate_z_matrix((*nn).z_matrix[0],
                     (*nn).weight[0],
                     activation_initial,
                     (*nn).bias[0],
                     number_of_nodes_in_hidden_layers,
                     number_of_inputs,
                     batch_size);
  //  Compute activation
  sigmoidify((*nn).activation[0],
             (*nn).z_matrix[0],
             number_of_nodes_in_hidden_layers,
             batch_size);
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // feed through the hidden layers
  for(i=1; i<number_of_hidden_layers; i++) {
    //  Get the z-matrix
    calculate_z_matrix((*nn).z_matrix[i],
                       (*nn).weight[i],
                       (*nn).activation[i-1],
                       (*nn).bias[i],
                       number_of_nodes_in_hidden_layers,
                       number_of_nodes_in_hidden_layers,
                       batch_size);
    //  Compute activation
    sigmoidify((*nn).activation[i],
               (*nn).z_matrix[i],
               number_of_nodes_in_hidden_layers,
               batch_size);
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // feed from the last hidden layer -> output layer
  //  Get the z-matrix
  calculate_z_matrix((*nn).z_matrix[number_of_hidden_layers],
                     (*nn).weight[number_of_hidden_layers],
                     (*nn).activation[number_of_hidden_layers-1],
                     (*nn).bias[number_of_hidden_layers],
                     number_of_outputs,
                     number_of_nodes_in_hidden_layers,
                     batch_size);
  //  compute activation
  sigmoidify((*nn).activation[number_of_hidden_layers],
             (*nn).z_matrix[number_of_hidden_layers],
             number_of_outputs,
             batch_size);
  //---------------------------------------------------------------------------

  int num_outputs = number_of_outputs * batch_size;
  for (i=0; i<num_outputs; i++) result[i] = (*nn).activation[number_of_hidden_layers][i];

  if (training) backpropagate(nn,
                              activation_initial,
                              target_value);
  else {
    for (i=0; i<batch_size; i++) {
      nn_type max = 0.0;
      int max_index = 0;
      for (j=0; j<number_of_outputs; j++) {
        if ((*nn).activation[number_of_hidden_layers][(j*batch_size)+i] > max) {
          max = (*nn).activation[number_of_hidden_layers][(j*batch_size)+i];
          max_index = j;
        }
      }

      if (max_index == target_value[i]) {
        (*count)++;
        correct[i] = max_index;
      }
      else correct[i] = max_index + number_of_outputs;
    }
  }
}

void backpropagate(struct neural_net *nn,
                   nn_type *activation_initial,
                   int *target_value)
{
  // printf("backpropagating with %i\n", target_value);
  int i;
  int number_of_hidden_layers          = (*nn).number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = (*nn).number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = (*nn).number_of_inputs;
  int number_of_outputs                = (*nn).number_of_outputs;
  int batch_size                       = (*nn).batch_size;
  int eta                              = (*nn).eta;

  // find the delta value in the output layer
  delta_output_layer((*nn).delta[number_of_hidden_layers],
                     (*nn).activation[number_of_hidden_layers],
                     (*nn).z_matrix[number_of_hidden_layers],
                     target_value,
                     number_of_outputs,
                     batch_size);

  // backpropagate delta -> last hidden layer
  //  Note that row, col dimensions here are for the matrix W
  //  NOT the transpose of W. The transpose will be taken care
  //  of in the function.
  delta_hidden_layers((*nn).delta[number_of_hidden_layers-1],
                      (*nn).weight[number_of_hidden_layers],
                      (*nn).delta[number_of_hidden_layers],
                      (*nn).z_matrix[number_of_hidden_layers-1],
                      number_of_outputs,
                      number_of_nodes_in_hidden_layers,
                      batch_size);

  // backpropagate delta -> hidden layers
  for (i=number_of_hidden_layers-2; i>=0; i--) {
    delta_hidden_layers((*nn).delta[i],
                        (*nn).weight[i+1],
                        (*nn).delta[i+1],
                        (*nn).z_matrix[i],
                        number_of_nodes_in_hidden_layers,
                        number_of_nodes_in_hidden_layers,
                        batch_size);
  }

  // -----------------------------------------------------------------
  // now that we have all of our deltas, adjust the weights and biases
  //  adjust the first hidden layer
  adjust_weight(activation_initial,
                (*nn).weight[0],
                (*nn).delta[0],
                number_of_nodes_in_hidden_layers,
                number_of_inputs,
                batch_size,
                eta);
  adjust_bias((*nn).bias[0],
              (*nn).delta[0],
              number_of_nodes_in_hidden_layers,
              batch_size,
              eta);
  //
  //  adjust the hidden layers
  for (i=1; i<number_of_hidden_layers; i++) {
    adjust_weight((*nn).activation[i-1],
                  (*nn).weight[i],
                  (*nn).delta[i],
                  number_of_nodes_in_hidden_layers,
                  number_of_nodes_in_hidden_layers,
                  batch_size,
                  eta);
    adjust_bias((*nn).bias[i],
                (*nn).delta[i],
                number_of_nodes_in_hidden_layers,
                batch_size,
                eta);
  }
  //
  //  adjust the output hidden layer
  adjust_weight((*nn).activation[number_of_hidden_layers-1],
                (*nn).weight[number_of_hidden_layers],
                (*nn).delta[number_of_hidden_layers],
                number_of_outputs,
                number_of_nodes_in_hidden_layers,
                batch_size,
                eta);
  adjust_bias((*nn).bias[number_of_hidden_layers],
              (*nn).delta[number_of_hidden_layers],
              number_of_outputs,
              batch_size,
              eta);
  // -----------------------------------------------------------------
}
