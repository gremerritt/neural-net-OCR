#ifndef MATRIXHELPERS_H
#define MATRIXHELPERS_H

#include "neural_net.h"

void matrix_vector_multiply(nn_type *output,
                            nn_type *M,
                            nn_type *V,
                            int M_row_dim,
                            int M_col_dim);

void add_vectors(nn_type *V1,
                 nn_type *V2,
                 int dim);

void sigmoidify(nn_type *activation,
                nn_type *z_vector,
                int dim);

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
