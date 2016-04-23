#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#include "neural_net.h"
#include "randomizing_helpers.h"

#define DIM 28
#define NUM_OUTPUTS 10
#define NUM_NODES_IN_HIDDEN_LAYERS 50
#define NUM_HIDDEN_LAYERS 2
#define LEARNING_RATE 1.5
#define BATCH_SIZE 5
#define EPOCHS 20
#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000
#define TRAINING_PRINT_RESULTS_EVERY 30000
#define TEST_PRINT_RESULTS_EVERY 400

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"

void create_batch(nn_type *batch,
	                int *label,
									mnist_data *data,
									int *sequence,
									int batch_size,
									int iteration);
void print_result(int iter, int *label, nn_type *result, char *correct);
int main(int argc, char **argv) {
	mnist_data *training_data;
	mnist_data *test_data;
	unsigned int cnt;
	int ret;

	printf("Loading training image set... ");
	ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &training_data, &cnt);
	if (ret) printf("An error occured: %d\n", ret);
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", cnt);
	}

	printf("\nLoading test image set... ");
	ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_data, &cnt);
	if (ret) printf("An error occured: %d\n", ret);
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", cnt);
	}

	int image_idx;
	int i, j;
// 	printf("label: %u \n", data[image].label);
// 	printf("{");
// 	for (i=0; i<28; i++)
// 	{
// 		printf("{");
// 		for (j=0; j<28; j++)
// 		{
// 			//mnist_data d = *(*data)[0];
// 			if (j<27) printf("%f,", 1.0-data[image].data[i][j]);
// 			else printf("%f", 1.0-data[image].data[i][j]);
// 		}
// 		if (i<27) printf("},");
// 		else printf("}");
// 	}
// 	printf("}\n");

	// for (image_idx=1000; image_idx<1001; image_idx++)
	// {
	// 	printf("\n\n\nlabel with: %u \n", data[image_idx].label);
	// 	for (i=0; i<28; i++)
	// 	{
	// 		for (j=0; j<28; j++)
	// 		{
	// 			if (data[image_idx].data[28*i + j] > 0.0100) printf("%i", data[image_idx].label);
	// 			else printf(".");
	// 		}
	// 		printf("\n");
	// 	}
	// }

	nn_type result[NUM_OUTPUTS*BATCH_SIZE];
	int number_of_hidden_layers          = NUM_HIDDEN_LAYERS;
	int number_of_nodes_in_hidden_layers = NUM_NODES_IN_HIDDEN_LAYERS;
	int number_of_inputs                 = DIM*DIM;
	int number_of_outputs                = NUM_OUTPUTS;
	int batch_size                       = BATCH_SIZE;
	nn_type learning_rate                = LEARNING_RATE;

	printf("\nInitializing neural net:");
	struct neural_net nn;
	create_neural_net(&nn, number_of_hidden_layers,
                         number_of_nodes_in_hidden_layers,
                         number_of_inputs,
                         number_of_outputs,
                         batch_size,
                         learning_rate);

	printf("\n  Total Layers:           %i", nn.number_of_hidden_layers+2);
	printf("\n  Hidden Layers:          %i", nn.number_of_hidden_layers);
	printf("\n  Inputs:                 %i", nn.number_of_inputs);
	printf("\n  Outputs:                %i", nn.number_of_outputs);
	printf("\n  Nodes in Hidden Layers: %i", nn.number_of_nodes_in_hidden_layers);
	printf("\n  Batch Size:             %i", nn.batch_size);
	printf("\n  Learning Rate:          %f", nn.eta);
	printf("\n------------------\n");

	int count = 0;
	int epoch;

	int *sequence  = malloc( TRAINING_SAMPLES * sizeof(int) );
	nn_type *batch = malloc( DIM * DIM * BATCH_SIZE * sizeof(nn_type) );
	int *label     = malloc( BATCH_SIZE * sizeof(int) );
	char *correct  = malloc( BATCH_SIZE * sizeof(char) );

	// TRAINING
	printf("\nTraining...\n");
	for (epoch=0; epoch<EPOCHS; epoch++) {
		printf("\n  Epoch %i\n", epoch);

		for (i=0; i<TRAINING_SAMPLES; i++) sequence[i] = i;
		shuffle(sequence, TRAINING_SAMPLES);

		for (i=0; i<TRAINING_SAMPLES / BATCH_SIZE; i++) {
			create_batch(batch, label, training_data, sequence, BATCH_SIZE, i);
			feed_forward(&nn, result, batch, label, 1, &count, correct);
			if (i%TRAINING_PRINT_RESULTS_EVERY == 0 && i != 0)
				print_result(i, label, result, correct);
		}

		int previous_count = 0;
		printf("    Running tests...\n");
		for (i=0; i<TEST_SAMPLES / BATCH_SIZE; i++) {
			for (j=0; j<TEST_SAMPLES; j++) sequence[j] = j;
			create_batch(batch, label, test_data, sequence, BATCH_SIZE, i);
			feed_forward(&nn, result, batch, label, 0, &count, correct);
			if (i%TEST_PRINT_RESULTS_EVERY == 0 && i != 0)
				print_result(i, label, result, correct);
		}
		printf("\n      Count: %i\n", count);
		count = 0;
	}

	destroy_nn(&nn);
	free(training_data);
	free(test_data);

	return 0;
}

void create_batch(nn_type *batch,
	                int *label,
									mnist_data *data,
									int *sequence,
									int batch_size,
									int iteration)
{
	int i, j;
	int mod = batch_size * iteration;
	int max = batch_size * (iteration + 1);
	for (i=mod; i<max; i++) {
		int offset = (iteration > 0) ? (i % mod) : i;
		label[offset] = data[sequence[i]].label;
		for (j=0; j<DIM*DIM; j++) {
			batch[(j*batch_size) + offset] = data[sequence[i]].data[j];
		}
	}
}

void print_result(int iter, int *label, nn_type *result, char *correct)
{
	int row, col;
	printf("\n    ITERATION %i\n    ", iter);
	for (row=0; row<BATCH_SIZE; row++) printf("       %i  ", label[row]);
	printf("\n    ");
	for (row=0; row<NUM_OUTPUTS; row++) {
		for (col=0; col<BATCH_SIZE; col++) {
			if (correct[col] == row+10)   printf(KRED "%f  ", result[(row*BATCH_SIZE)+col]);
			else if (correct[col] == row) printf(KGRN "%f  ", result[(row*BATCH_SIZE)+col]);
			else                          printf(KNRM "%f  ", result[(row*BATCH_SIZE)+col]);
		}
		printf(KNRM "\n    ");
	}
}
