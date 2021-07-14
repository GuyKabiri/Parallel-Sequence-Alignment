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

    MPI_Status status;
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

//         if (!read_seq_and_weights_from_file(input_file, seq1, seq2, weights, &is_max))
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

//    MPI_Bcast(&data, 1, mpi_data_type, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(data.seq1, SEQ1_MAX_LEN, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(data.seq2, SEQ2_MAX_LEN, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(data.weights, WEIGHTS_COUNT, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&data.num_tasks, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&data.offset_add, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&data.is_max, 1, MPI_INT, ROOT, MPI_COMM_WORLD);


//    printf("proc %d: tasks: %d, off: %d\n", my_rank, data.num_tasks, data.offset_add);

    int start_offset = data.num_tasks * my_rank;  //  each process will handle the same amount of tasks, the offset will be multiply by the process index + the additional offset
    if (my_rank == ROOT)
    {
        data.num_tasks += data.offset_add;
    }
    else
    {
    	start_offset += data.offset_add;
    }
//    printf("pro %d: tasks: %d, start: %d, end: %d\n", my_rank, data.num_tasks, start_offset, data.num_tasks + start_offset);


    char mutant[SEQ2_MAX_LEN] = { '\0' };

     int best_offset = 0;
     double best_score = 0;


    for (int i = start_offset; i < data.num_tasks + start_offset; i++)
    {
        double score = find_mutant(data.seq1, data.seq2, data.weights, i, mutant, data.is_max);
        if (score > best_score)
        {
        	best_score = score;
        	best_offset = i;
        }
    }

//    printf("proc %2d: offset: %3d, score: %g\n", my_rank, best_offset, best_score);
//    double score = find_mutant(data.seq1, data.seq2, data.weights, best_offset, mutant, data.is_max);




    double mymax[2] = { 0 };
    mymax[0] = best_score;
    mymax[1] = my_rank;
    double globalmax[2] = { 0 };

    MPI_Allreduce(mymax, globalmax, 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_COMM_WORLD);
    int sender_rank = globalmax[1];

//    printf("me: %d, global[0]: %g, global[1]: %g\n", my_rank, globalmax[0], globalmax[1]);

    if (my_rank == ROOT)
    {
    	best_score = globalmax[0];
    	MPI_Recv(&best_offset, 1, MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, &status);
    	MPI_Recv(mutant, SEQ2_MAX_LEN, MPI_CHAR, sender_rank, 0, MPI_COMM_WORLD, &status);
    	printf("best offset: %3d, by procs: %2d, score: %g\n%s\n", best_offset, sender_rank, best_score, mutant);
    	FILE* out_file = fopen(OUTPUT_FILE, "w");
    	if (!out_file)
    	{
    		printf("Error open or write to the output file %s\n", OUTPUT_FILE);
    		MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	if (!write_results_to_file(out_file, mutant, best_offset, best_score))
    	{
    		printf("Error write to the output file %s\n", OUTPUT_FILE);
			MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	fclose(out_file);
    }
    else if (my_rank == sender_rank)
    {
    	MPI_Send(&best_offset, 1, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
    	MPI_Send(mutant, SEQ2_MAX_LEN, MPI_CHAR, ROOT, 0, MPI_COMM_WORLD);
    }


	MPI_Finalize();

}
