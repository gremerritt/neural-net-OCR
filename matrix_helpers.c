#include <stdio.h>
#include "neural_net.h"

void calculate_z_matrix(nn_type *z_matrix,
                        nn_type *weight,
                        nn_type *activation,
                        nn_type *bias,
                        int weight_rows,
                        int weight_cols,
                        int activation_cols)
{
  // this is a boring simple matrix multiply
  int Act_col, W_row, W_col;
  for (Act_col=0; Act_col<activation_cols; Act_col++) {
    for (W_row=0; W_row<weight_rows; W_row++) {
      nn_type accum = 0.0;
      int weight_offset = W_row*weight_cols;
      for (W_col=0; W_col<weight_cols; W_col++) {
        accum += weight[weight_offset + W_col] * activation[(W_col * activation_cols) + Act_col];
      }
      accum += bias[W_row];
      z_matrix[(W_row*activation_cols) + Act_col] = accum;
    }
  }
}

void sigmoidify(nn_type *activation,
                nn_type *z_vector,
                int rows,
                int cols)
{
  int i;

  // printf("\nz_vector:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", z_vector[i]);

  for (i=0; i<rows*cols; i++) activation[i] = sigmoid(z_vector[i]);

  // printf("\nactivation:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", activation[i]);
}

void delta_output_layer(nn_type *delta,
                        nn_type *activation,
                        nn_type *z_matrix,
                        int *target_value,
                        int outputs,
                        int batch_size)
{
  int row, col;
  for (col=0; col<batch_size; col++) {
    int target = target_value[col];
    for (row=0; row<outputs; row++) {
      int index = (row*batch_size) + col;
      delta[index] = (activation[index] - ((row == target) ? 1.0 : 0.0)) * sigmoidPrime(z_matrix[index]);
    }
  }
}

void delta_hidden_layers(nn_type *delta,
                         nn_type *weight_downstream,
                         nn_type *delta_downstream,
                         nn_type *z_matrix,
                         int weight_rows,
                         int weight_cols,
                         int batch_size)
{
  int weight_row, weight_col, delta_col;

  // int row, col;
  // printf("\n\nWeight downstream:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   for (col=0; col<weight_cols; col++)
  //     printf("%f ", weight_downstream[(row*weight_cols) + col]);
  //   if (row==weight_rows-1) printf("]");
  //   else printf(";\n");
  // }
  // //
  // printf("\n\nDelta downstream:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   for (col=0; col<batch_size; col++)
  //     printf("%f ", delta_downstream[(row*batch_size) + col]);
  //   if (row==weight_rows-1) printf("]");
  //   else printf(";\n");
  // }

  for (delta_col=0; delta_col<batch_size; delta_col++) {
    for (weight_col=0; weight_col<weight_cols; weight_col++) {
      nn_type accum = 0.0;
      for (weight_row=0; weight_row<weight_rows; weight_row++) {
        accum += weight_downstream[(weight_cols*weight_row) + weight_col] *
                 delta_downstream[(weight_row*batch_size) + delta_col];
      }
      accum *= sigmoidPrime(z_matrix[(weight_col*batch_size)+delta_col]);
      delta[(weight_col*batch_size)+delta_col] = accum;
    }
  }

  // printf("\n\nDelta:\n");
  // printf("[");
  // for (row=0; row<weight_cols; row++) {
  //   for (col=0; col<batch_size; col++)
  //     printf("%f ", delta[(row*batch_size) + col]);
  //   if (row==weight_rows-1) printf("]");
  //   else printf(";\n");
  // }
}

void adjust_weight(nn_type *activation,
                   nn_type *weight,
                   nn_type *delta,
                   int weight_rows,
                   int weight_cols,
                   nn_type eta,
                   int batch_size)
{
  int row, col;

  // printf("\n\nWeight:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   for (col=0; col<weight_cols; col++)
  //     printf("%f ", weight[(row*weight_cols) + col]);
  //   if (row==weight_rows-1) printf("]");
  //   else printf(";\n");
  // }
  //
  // printf("\n\nActivation:\n");
  // printf("[");
  // for (col=0; col<weight_cols; col++) {
  //   if (col==weight_cols-1) printf("%f]", activation[col]);
  //   else printf("%f;\n", activation[col]);
  // }
  //
  // printf("\n\nDelta:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   if (row==weight_rows-1) printf("%f]", delta[row]);
  //   else printf("%f;\n", delta[row]);
  // }

  for (row=0; row<weight_rows; row++) {
    int offset = row*weight_cols;
    for (col=0; col<weight_cols; col++) {
      weight[offset + col] -= (eta / batch_size) * activation[col] * delta[row];
    }
  }

  // printf("\n\nWeight:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   for (col=0; col<weight_cols; col++)
  //     printf("%f ", weight[(row*weight_cols) + col]);
  //   if (row==weight_rows-1) printf("]\n\n");
  //   else printf(";\n");
  // }
}

void adjust_bias(nn_type *bias,
                 nn_type *delta,
                 int dim,
                 nn_type eta,
                 int batch_size)
{
  int i;

  // printf("\n\nBias:\n");
  // printf("[");
  // for (i=0; i<dim; i++) {
  //   if (i==dim-1) printf("%f]", bias[i]);
  //   else printf("%f;\n", bias[i]);
  // }
  //
  // printf("\n\nDelta:\n");
  // printf("[");
  // for (i=0; i<dim; i++) {
  //   if (i==dim-1) printf("%f]", delta[i]);
  //   else printf("%f;\n", delta[i]);
  // }

  for (i=0; i<dim; i++) {
    bias[i] -= (eta / batch_size) * delta[i];
  }

  // printf("\n\nBias:\n");
  // printf("[");
  // for (i=0; i<dim; i++) {
  //   if (i==dim-1) printf("%f]\n\n", bias[i]);
  //   else printf("%f;\n", bias[i]);
  // }
}
