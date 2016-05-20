#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mnist.h"
#include "neural_net.h"
#include "randomizing_helpers.h"
#include "matrix_helpers.h"
#include <omp.h>

#define DIM 28
#define NUM_OUTPUTS 10
#define NUM_NODES_IN_HIDDEN_LAYERS 60
#define NUM_HIDDEN_LAYERS 2
#define LEARNING_RATE 1.5
#define BATCH_SIZE 5
#define EPOCHS 25
#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000
#define TRAINING_PRINT_RESULTS_EVERY 60000
#define TEST_PRINT_RESULTS_EVERY 10000
#define OMP_NUM_THREADS_TRAINING 4

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"

void create_batch_with_sequence(nn_type *batch,
	                              int *label,
									              mnist_data *data,
									              int batch_size,
									              int iteration,
								                int *sequence);
void create_batch_no_sequence(nn_type *batch,
	                            int *label,
									            mnist_data *data,
									            int batch_size,
									            int iteration);
void print_result(int iter,
	                int *label, nn_type *result,
									char *correct);
void process_command_line(int argc, char **argv,
                          int *number_of_hidden_layers,
                          int *number_of_nodes_in_hidden_layers,
                          int *batch_size,
                          nn_type *learning_rate);
int main(int argc, char **argv) {
	mnist_data *training_data;
	mnist_data *test_data;
	unsigned int cnt;
	int ret;
	double epoch_t1, epoch_t2, epoch_duration;
	double training_t1, training_t2, training_duration;
	double syncing_t1, syncing_t2, syncing_duration;
	double testing_t1, testing_t2, testing_duration;
	int counts[EPOCHS];
	double epoch_times[EPOCHS];
	double training_times[EPOCHS];
	double syncing_times[EPOCHS];
	double testing_times[EPOCHS];

	printf("Loading training image set... ");
	ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &training_data, &cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return 0;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", cnt);
	}

	printf("\nLoading test image set... ");
	ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_data, &cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return 0;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", cnt);
	}

	int number_of_hidden_layers          = NUM_HIDDEN_LAYERS;
	int number_of_nodes_in_hidden_layers = NUM_NODES_IN_HIDDEN_LAYERS;
	int number_of_inputs                 = DIM*DIM;
	int number_of_outputs                = NUM_OUTPUTS;
	int batch_size                       = BATCH_SIZE;
	nn_type learning_rate                = LEARNING_RATE;

	process_command_line(argc, argv,
		                   &number_of_hidden_layers,
	                     &number_of_nodes_in_hidden_layers,
										   &batch_size,
										   &learning_rate);

	printf("\nInitializing neural net:");
	struct neural_net neural_nets[OMP_NUM_THREADS_TRAINING+1];
	create_neural_net(&(neural_nets[0]), number_of_hidden_layers,						// nn[0] is master
                             number_of_nodes_in_hidden_layers,
                             number_of_inputs,
                             number_of_outputs,
                             batch_size,
                             learning_rate);
	int k;
	for (k=1; k<OMP_NUM_THREADS_TRAINING+1; k++) duplicate_nn(&(neural_nets[k]), &(neural_nets[0]));

	printf("\n  Total Layers:           %i", neural_nets[0].number_of_hidden_layers+2);
	printf("\n  Hidden Layers:          %i", neural_nets[0].number_of_hidden_layers);
	printf("\n  Inputs:                 %i", neural_nets[0].number_of_inputs);
	printf("\n  Outputs:                %i", neural_nets[0].number_of_outputs);
	printf("\n  Nodes in Hidden Layers: %i", neural_nets[0].number_of_nodes_in_hidden_layers);
	printf("\n  Batch Size:             %i", neural_nets[0].batch_size);
	printf("\n  Learning Rate:          %f", neural_nets[0].eta);
	printf("\n------------------\n");

	int count = 0;
	int epoch;
	int i;
	int tid;

	int *sequence  = malloc( TRAINING_SAMPLES * sizeof(int) );

	// TRAINING
	omp_set_dynamic(0);
	printf("\nTraining...\n");
	for (epoch=0; epoch<EPOCHS; epoch++) {
		printf("\n  Epoch %i\n", epoch);
		epoch_t1 = omp_get_wtime();

		#pragma omp parallel for shared(sequence) private(i)
		for (i=0; i<TRAINING_SAMPLES; i++) sequence[i] = i;

		shuffle(sequence, TRAINING_SAMPLES);

		training_t1 = omp_get_wtime();
		#pragma omp parallel for shared(sequence, training_data, neural_nets) \
		                         private(i, tid) \
														 reduction(+:count) \
														 num_threads(OMP_NUM_THREADS_TRAINING)
		for (i=0; i<TRAINING_SAMPLES / BATCH_SIZE; i++) {
			tid = omp_get_thread_num();

			nn_type result [NUM_OUTPUTS * BATCH_SIZE];
			nn_type batch  [DIM * DIM * BATCH_SIZE];
			int     label  [BATCH_SIZE];
			char    correct[BATCH_SIZE];

			create_batch_with_sequence(batch, label, training_data, BATCH_SIZE, i, sequence);
			feed_forward(&(neural_nets[tid+1]), result, batch, label, 1, &count, correct);
			if (i%TRAINING_PRINT_RESULTS_EVERY == 0 && i != 0) {
				// there's a bug with printing from multithreaded run...
				print_result(i, label, result, correct);
			}
		}
		training_t2 = omp_get_wtime();
		training_duration = (training_t2 - training_t1);

		// determine weight change matrix
		syncing_t1 = omp_get_wtime();
		struct change_matrices cm;
		initialize_change_matrices(&cm, &(neural_nets[0]));
		for (i=1; i<OMP_NUM_THREADS_TRAINING+1; i++) {
			get_changes(&cm, &(neural_nets[i]), &(neural_nets[0]));
		}
		apply_changes(&cm, &(neural_nets[0]), OMP_NUM_THREADS_TRAINING);
		for (i=1; i<OMP_NUM_THREADS_TRAINING+1; i++) {
			sync_nn(&(neural_nets[i]), &(neural_nets[0]));
		}
		syncing_t2 = omp_get_wtime();
		syncing_duration = (syncing_t2 - syncing_t1);

		printf("    Running tests...\n");
		testing_t1 = omp_get_wtime();
		#pragma omp parallel for shared(test_data, neural_nets) \
		                         private(i, tid) \
														 reduction(+:count) \
														 num_threads(OMP_NUM_THREADS_TRAINING)
		for (i=0; i<TEST_SAMPLES / BATCH_SIZE; i++) {
			tid = omp_get_thread_num();

			nn_type result [NUM_OUTPUTS * BATCH_SIZE];
			nn_type batch  [DIM * DIM * BATCH_SIZE];
			int     label  [BATCH_SIZE];
			char    correct[BATCH_SIZE];

			create_batch_no_sequence(batch, label, test_data, BATCH_SIZE, i);
			feed_forward(&(neural_nets[tid+1]), result, batch, label, 0, &count, correct);
			if (i%TEST_PRINT_RESULTS_EVERY == 0 && i != 0) {
				// there's a bug with printing from multithreaded run...
				print_result(i, label, result, correct);
			}
		}
		testing_t2 = omp_get_wtime();
		testing_duration = (testing_t2 - testing_t1);

		epoch_t2 = omp_get_wtime();
		epoch_duration = (epoch_t2 - epoch_t1);
	  printf("\n      Epoch Duration:    %f\n", epoch_duration);
		printf("      Training Duration: %f\n", training_duration);
		printf("      Syncing Duration:  %f\n", syncing_duration);
		printf("      Testing Duration:  %f\n", testing_duration);
		printf("      Count: %i\n", count);
		epoch_times[epoch]    = epoch_duration;
		training_times[epoch] = training_duration;
		syncing_times[epoch]  = syncing_duration;
		testing_times[epoch]  = testing_duration;
		counts[epoch]         = count;
		count = 0;
	}

	printf("Count, Epoch Time (s), Training Time (s), Syncing Time (s), Testing Time (s)\n");
	for (epoch=0; epoch<EPOCHS; epoch++)
	{
		printf("%i, %f, %f, %f, %f\n", counts[epoch],
		                               epoch_times[epoch],
															     training_times[epoch],
																	 syncing_times[epoch],
															     testing_times[epoch]);
	}

	// destroy_nn(&nn);
	free(training_data);
	free(test_data);

	return 0;
}

void create_batch_with_sequence(nn_type *batch,
	                              int *label,
									              mnist_data *data,
									              int batch_size,
									              int iteration,
								                int *sequence)
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

void create_batch_no_sequence(nn_type *batch,
	                            int *label,
									            mnist_data *data,
									            int batch_size,
									            int iteration)
{
	int i, j;
	int mod = batch_size * iteration;
	int max = batch_size * (iteration + 1);
	for (i=mod; i<max; i++) {
		int offset = (iteration > 0) ? (i % mod) : i;
		label[offset] = data[i].label;
		for (j=0; j<DIM*DIM; j++) {
			batch[(j*batch_size) + offset] = data[i].data[j];
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

void process_command_line(int argc, char **argv,
                          int *number_of_hidden_layers,
                          int *number_of_nodes_in_hidden_layers,
                          int *batch_size,
                          nn_type *learning_rate)
{
	int i;
	for (i=1; i<argc; i++) {
		char *str   = argv[i];
		char *param = strtok(str, "=");
		char *val   = strtok(NULL, "=");

		if ((strcmp(param, "--hidden-layers") == 0) ||
				(strcmp(param, "--hl") == 0)) {
			*number_of_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strcmp(param, "--hidden-nodes") == 0) ||
						 (strcmp(param, "--hn") == 0)) {
			*number_of_nodes_in_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strcmp(param, "--batch-size") == 0) ||
						 (strcmp(param, "--bs") == 0)) {
			*batch_size = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strcmp(param, "--learning-rate") == 0) ||
						 (strcmp(param, "--lr") == 0)) {
			*learning_rate = (nn_type)strtod( strtok(val, " "), NULL);
		}
	}
}
