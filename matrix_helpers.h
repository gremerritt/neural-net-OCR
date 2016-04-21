#ifndef MATRIXHELPERS_H
#define MATRIXHELPERS_H

#include "neural_net.h"

// This method multiplies the weight matrix by the
// preivous layers activation matrix, and adds the
// bias matrix to the result. The result is the
// layers so-called 'z_vector'
void calculate_z_matrix(nn_type *z_matrix,
                        nn_type *weight,
                        nn_type *activation,
                        nn_type *bias,
                        int weight_rows,
                        int weight_cols,
                        int activation_cols);

void sigmoidify(nn_type *activation,
                nn_type *z_vector,
                int rows,
                int cols);

void delta_output_layer(nn_type *delta,
                        nn_type *activation,
                        nn_type *z_vector,
                        int target_value,
                        int dim);

void delta_hidden_layers(nn_type *delta,
                         nn_type *weight_downstream,
                         nn_type *delta_downstream,
                         nn_type *z_vector,
                         int weight_rows,
                         int weight_cols);

void adjust_weight(nn_type *activation_initial,
                   nn_type *weight,
                   nn_type *delta,
                   int weight_rows,
                   int weight_cols,
                   nn_type eta,
                   int batch_size);

void adjust_bias(nn_type *bias,
                 nn_type *delta,
                 int dim,
                 nn_type eta,
                 int batch_size);

#endif
