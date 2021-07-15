#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "main.h"
#include "cpu_funcs.h"

int main(int argc, char* argv[])
{
    int  pid;			//	rank of process
	int  num_processes;     //	number of processes
    double time = 0;
    printf("sd\n");

    // MPI_Status status;
    /* create a type for data struct */
	// MPI_Datatype 	mpi_data_type;
	
	// 				//	number of blocks for each parameter
	// int          	blocklengths[NUM_OF_PARAMS] = { SEQ1_MAX_LEN, SEQ2_MAX_LEN, WEIGHTS_COUNT, 1, 1, 1 };
	
	// 				//	offset of each parameter, calculated by size of previous parameters
	// MPI_Aint 		displacements[NUM_OF_PARAMS] = {    0,                              //  _data.seq1 offset
    //                                                     sizeof(char) * SEQ1_MAX_LEN,    //  _data.seq2 offset
    //                                                     sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN,      //  _data.weights offset
    //                                                     sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN + sizeof(double) * WEIGHTS_COUNT,     //  _data.is_max offset
    //                                                     sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN + sizeof(double) * WEIGHTS_COUNT + sizeof(int),   //  _data.tasks offset
    //                                                     sizeof(char) * SEQ1_MAX_LEN + sizeof(char) * SEQ2_MAX_LEN + sizeof(double) * WEIGHTS_COUNT + sizeof(int) }; //  _data.offset_add offset

	// MPI_Datatype 	types[NUM_OF_PARAMS] = { MPI_CHAR, MPI_CHAR, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT };

	// MPI_Type_create_struct(NUM_OF_PARAMS, blocklengths, displacements, types, &mpi_data_type);
	// MPI_Type_commit(&mpi_data_type);




    MPI_Init(&argc, &argv);	//	start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);		//	get process rank
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);	//	get number of processes

    if (argc == 2)
    {
        cuda_percentage = atoi(argv[1]);
        if (cuda_percentage < 0 || cuda_percentage > 100)
        {
            if (pid == ROOT)
                printf("Cuda percentage invalid (%d), set cude_percentage=0\n", cuda_percentage);
            cuda_percentage = 0;
        }
    }

    time -= MPI_Wtime();    //  substract the mpi initiation time
    cpu_run_program(pid, num_processes);
    time += MPI_Wtime();    //  get program time

    if (pid == ROOT)
        printf("total time: %g\n", time);

	MPI_Finalize();
    return 0;
}
