#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include "mnist.h"
#include "neural_net.h"
#include <stdio.h>
#include <stdlib.h>

#define DIM 28
#define NUM_OUTPUTS 10
#define NUM_NODES_IN_HIDDEN_LAYERS 30
#define NUM_HIDDEN_LAYERS 1
#define LEARNING_RATE 3.0
#define EPOCHS 3
#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000
#define TRAINING_PRINT_RESULTS_EVERY 100000
#define TEST_PRINT_RESULTS_EVERY 1000

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

	nn_type result[NUM_OUTPUTS];
	int number_of_hidden_layers          = NUM_HIDDEN_LAYERS;
	int number_of_nodes_in_hidden_layers = NUM_NODES_IN_HIDDEN_LAYERS;
	int number_of_inputs                 = DIM*DIM;
	int number_of_outputs                = NUM_OUTPUTS;
	nn_type learning_rate                = LEARNING_RATE;

	printf("\nInitializing neural net:");
	struct neural_net nn;
	create_neural_net(&nn, number_of_hidden_layers,
		                     number_of_nodes_in_hidden_layers,
												 number_of_inputs,
												 number_of_outputs,
											   learning_rate);

	printf("\n  Total Layers:           %i", nn.number_of_hidden_layers+2);
	printf("\n  Hidden Layers:          %i", nn.number_of_hidden_layers);
	printf("\n  Inputs:                 %i", nn.number_of_inputs);
	printf("\n  Outputs:                %i", nn.number_of_outputs);
	printf("\n  Nodes in Hidden Layers: %i", nn.number_of_nodes_in_hidden_layers);
	printf("\n  Learning Rate:          %f", nn.eta);
	printf("\n------------------\n");

	int count = 0;
	int epoch;

	// TRAINING
	printf("\nTraining...\n");
	for (epoch=0; epoch<EPOCHS; epoch++) {
		printf("  Epoch %i\n", epoch);
		for (i=0; i<TRAINING_SAMPLES; i++) {
			feed_forward(&nn, result, training_data[i].data, training_data[i].label, 1, &count);
			// if (i%TRAINING_PRINT_RESULTS_EVERY == 0) {
			// 	printf("\n------------------\nlabel: %i\n", training_data[i].label);
			// 	for (j=0; j<NUM_OUTPUTS; j++) printf("  %f\n", result[j]);
			// }
		}

		printf("    Running tests...\n");
		for (i=0; i<TEST_SAMPLES; i++) {
			feed_forward(&nn, result, test_data[i].data, test_data[i].label, 0, &count);
			// if (i%TEST_PRINT_RESULTS_EVERY == 0) {
			// 	printf("\n------------------\nlabel: %i\n", test_data[i].label);
			// 	for (j=0; j<NUM_OUTPUTS; j++) printf("  %f\n", result[j]);
			// }
		}
		printf("      Count: %i\n", count);
		count = 0;
	}

	printf("\nRunning tests...\n");
	for (i=0; i<TEST_SAMPLES; i++) {
		feed_forward(&nn, result, test_data[i].data, test_data[i].label, 0, &count);
		// if (i%TEST_PRINT_RESULTS_EVERY == 0) {
		// 	printf("\n------------------\nlabel: %i\n", test_data[i].label);
		// 	for (j=0; j<NUM_OUTPUTS; j++) printf("  %f\n", result[j]);
		// }
	}

	printf("\nCount: %i\n", count);

	destroy_nn(&nn);
	free(training_data);
	free(test_data);

	return 0;
}
