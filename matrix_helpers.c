#include <stdio.h>
#include "neural_net.h"

void matrix_vector_multiply(nn_type *output,
                            nn_type *M,
                            nn_type *V,
                            int M_row_dim,
                            int M_col_dim)
{
  int row, col;

  // printf("\nM:\n");
  // for (row=0; row<M_row_dim; row++) {
  //   for (col=0; col<M_col_dim; col++)
  //     printf("%f ", M[(row*M_col_dim) + col]);
  //   printf("\n");
  // }
  // printf("\nV:\n");
  // for (col=0; col<M_col_dim; col++)
  //   printf("%f\n", V[col]);

  for (row=0; row<M_row_dim; row++) {
    int col_offset = row*M_col_dim;
    nn_type accum = 0.0;
    for (col=0; col<M_col_dim; col++) {
      accum += M[col_offset + col] * V[col];
    }
    output[row] = accum;
  }

  // printf("\nResult:\n");
  // for (row=0; row<M_row_dim; row++)
  //   printf("%f\n", output[row]);
}

void add_vectors(nn_type *V1,
                 nn_type *V2,
                 int dim)
{
  int i;

  // printf("\nV1:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", V1[i]);
  // printf("\nV2:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", V2[i]);

  for (i=0; i<dim; i++)
    V1[i] = V1[i] + V2[i];

  // printf("\nVout:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", V1[i]);
}

void sigmoidify(nn_type *activation,
                nn_type *z_vector,
                int dim)
{
  int i;

  // printf("\nz_vector:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", z_vector[i]);

  for (i=0; i<dim; i++) activation[i] = sigmoid(z_vector[i]);

  // printf("\nactivation:\n");
  // for (i=0; i<dim; i++)
  //   printf("%f\n", activation[i]);
}

void delta_output_layer(nn_type *delta,
                        nn_type *activation,
                        nn_type *z_vector,
                        int target_value,
                        int dim)
{
  int i;
  for (i=0; i<dim; i++) {
    delta[i] = (activation[i] - ((i == target_value) ? 1.0 : 0.0)) * sigmoidPrime(z_vector[i]);
  }
}

void delta_hidden_layers(nn_type *delta,
                         nn_type *weight_downstream,
                         nn_type *delta_downstream,
                         nn_type *z_vector,
                         int weight_rows,
                         int weight_cols)
{
  int row, col;

  // printf("\n\nWeight downstream:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   for (col=0; col<weight_cols; col++)
  //     printf("%f ", weight_downstream[(row*weight_cols) + col]);
  //   if (row==weight_rows-1) printf("]");
  //   else printf(";\n");
  // }
  //
  // printf("\n\nDelta downstream:\n");
  // printf("[");
  // for (row=0; row<weight_rows; row++) {
  //   if (row==weight_rows-1) printf("%f]", delta_downstream[row]);
  //   else printf("%f;\n", delta_downstream[row]);
  // }
  //
  // printf("\n\nZ vector:\n");
  // printf("[");
  // for (col=0; col<weight_cols; col++) {
  //   if (col==weight_cols-1) printf("%f]", z_vector[col]);
  //   else printf("%f;\n", z_vector[col]);
  // }
  //
  // printf("\n\nZ vector - sigmoid prime:\n");
  // printf("[");
  // for (col=0; col<weight_cols; col++) {
  //   if (col==weight_cols-1) printf("%f]", sigmoidPrime(z_vector[col]));
  //   else printf("%f;\n", sigmoidPrime(z_vector[col]));
  // }

  // We do the transpose of the 'weight' matrix by just
  // handling the indices differently.
  for (col=0; col<weight_cols; col++) {
    nn_type accum = 0.0;
    for (row=0; row<weight_rows; row++) {
      accum += weight_downstream[(weight_cols*row) + col] * delta_downstream[row];
    }
    accum *= sigmoidPrime(z_vector[col]);
    delta[col] = accum;
  }

  // printf("\n\nDelta:\n");
  // printf("[");
  // for (col=0; col<weight_cols; col++) {
  //   if (col==weight_cols-1) printf("%f]", delta[col]);
  //   else printf("%f;\n", delta[col]);
  // }
  // printf("\n---\n");
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
