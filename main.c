// TODO: re-structure the directories, delete input, output dirs


#define _CRT_SECURE_NO_WARNINGS             //   TODO: remove CTR_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#include "main.h"
#include "cpu_funcs.h"
#include "def.h"
#include "mutant.h"
#include "program_data.h"

void clear()
{
    printf("\e[2J\e[H"); // magic!
}

int main(int argc, char* argv[])
{
    int  my_rank;			//	rank of process
	int  num_processes;     //	number of processes
    MPI_Init(&argc, &argv);	//	start MPI

    /* create a type for data struct */
	MPI_Datatype 	mpi_data_type;
	
					//	number of blocks for each parameter
	int          	blocklengths[NUM_OF_PARAMS] = { SEQ1_MAX_LEN, SEQ2_MAX_LEN, WEIGHTS_COUNT, 1, 1, 1 };
	
					//	offset of each parameter, calculated by size of previous parameters
	MPI_Aint 		displacements[NUM_OF_PARAMS] = {    0,                              //  _data.seq1 offset
                                                        sizeof(char) * SEQ1_MAX_LEN,    //  _data.seq2 offset
                                                        sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN,      //  _data.weights offset
                                                        sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN + sizeof(double) * WEIGHTS_COUNT,     //  _data.is_max offset
                                                        sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN + sizeof(double) * WEIGHTS_COUNT + sizeof(int),   //  _data.tasks offset
                                                        sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN + sizeof(double) * WEIGHTS_COUNT + sizeof(int) }; //  _data.offset_add offset

	MPI_Datatype 	types[NUM_OF_PARAMS] = { MPI_CHAR, MPI_CHAR, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT };

	MPI_Type_create_struct(NUM_OF_PARAMS, blocklengths, displacements, types, &mpi_data_type);
	MPI_Type_commit(&mpi_data_type);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);		//	get process rank
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);	//	get number of processes

    ProgramData data;

    if (my_rank == ROOT)
    {
        FILE* input_file;

        input_file = fopen(INPUT_FILE, "r");
        if (!input_file)
        {
            printf("Error open input file `%s`\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 2);
            exit(1);
        }
        
        // if (!read_seq_and_weights_from_file(input_file, seq1, seq2, weights, &is_max))
        if (!read_seq_and_weights_from_file(input_file, &data))
        {
            printf("Error reading input file `%s`\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 2);
            exit(1);
        }
        fclose(input_file);

        int iterations = strlen(data.seq1) - strlen(data.seq2) + 1;
        data.num_tasks = iterations / num_processes;
        data.offset_add = iterations % num_processes;   //  if amount of offset does not divide by amount of processes, the root process will take the additional tasks

        //  send data to other process
    }

    MPI_Bcast(&data, 1, mpi_data_type, ROOT, MPI_COMM_WORLD);

    int start_offset = data.num_tasks * num_processes;  //  each process will handle the same amount of tasks, the offset will be multiply by the process index + the additional offset

    if (my_rank == ROOT)
    {
        data.num_tasks += data.offset_add;
    }

    // int offset = 0;
    // compare_evaluate_seq(seq1, seq2, weights, offset, NULL);

    // print_seq(seq1, seq2, weights, offset);

    // Mutant best_mutant;
    // best_mutant.score = data.is_max ? __DBL_MIN__ : __DBL_MAX__;
    // best_mutant.offset = 0;

    char mutant[SEQ2_MAX_LEN] = { '\0' };

    // data.start_offset = 0;
    int iterations = strlen(data.seq1) - strlen(data.seq2) + 1 - data.num_tasks;
    // int best_offset = 0;
    // double best_score = 0;

    // struct timespec t;
    // t.tv_sec = 0;
    // t.tv_nsec = 1000000000 * 0.5;

    for (int i = 0; i < iterations; i++)
    {
        double score = find_mutant(data.seq1, data.seq2, data.weights, i, mutant, data.is_max);
        // double score = compare_evaluate_seq(data.seq1, data.seq2, data.weights, i, NULL);
        // clear();
        // print_seq(data.seq1, data.seq2, data.weights, i);
        // nanosleep(&t, &t);

        printf("offset: %4d, score: %g\n", i, score);
        print_seq(data.seq1, mutant, data.weights, i);
        // print_seq(data.seq1, data.seq2, data.weights, i);

        break;
    }
    printf("iterations: %d, process: %d\n", iterations, data.num_tasks);


    // print_seq(seq1, seq2, weights, best_offset);

}