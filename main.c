#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "main.h"
#include "cpu_funcs.h"

int main(int argc, char* argv[])
{
    int  pid;			//	rank of process
	int  num_processes;     //	number of processes
    double time = MPI_Wtime();
    
    MPI_Init(&argc, &argv);	//	start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);		//	get process rank
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);	//	get number of processes

    MPI_Status status;
    /* create a type for data struct */

    				//	number of blocks for each parameter
	int          	blocklengths[NUM_OF_PARAMS] = { 1, 1, SEQ1_MAX_LEN, SEQ2_MAX_LEN, WEIGHTS_COUNT };
	
					//	offset of each parameter, calculated by size of previous parameters
	MPI_Aint 		displacements[NUM_OF_PARAMS] = {    0,                              //  _data.seq1 offset

                                                        sizeof(int),
                                                        sizeof(int) * 2,
                                                        sizeof(int) * 2 + sizeof(char) * SEQ1_MAX_LEN,
                                                        sizeof(int) * 2 + sizeof(char) * (SEQ1_MAX_LEN + SEQ2_MAX_LEN) };

	MPI_Datatype 	types[NUM_OF_PARAMS] = { MPI_INT, MPI_INT, MPI_CHAR, MPI_CHAR, MPI_DOUBLE };

	MPI_Type_create_struct(NUM_OF_PARAMS, blocklengths, displacements, types, &program_data_type);
	MPI_Type_commit(&program_data_type);


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
    else if (argc == 1)
    {
        if (pid == ROOT)
                printf("Cuda percentage did not set, set cude_percentage=0\n");
            cuda_percentage = 0;
    }

    if (num_processes == 1)
        omp_set_num_threads(1);
    else
        // omp_set_num_threads(MAX_THREADS / num_processes);
        omp_set_num_threads(4);

    // time -= MPI_Wtime();    //  substract the mpi initiation time
    cpu_run_program(pid, num_processes);
    // time += MPI_Wtime();    //  get program time
    time = MPI_Wtime() - time;    //  get program time

    if (pid == ROOT)
        printf("total time: %g\n", time);

    MPI_Type_free(&program_data_type);
	MPI_Finalize();

    return 0;
}
