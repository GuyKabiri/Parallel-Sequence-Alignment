#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "main.h"
#include "cpu_funcs.h"
#include "mpi_funcs.h"

extern MPI_Datatype mutant_type;
extern MPI_Datatype program_data_type;

int main(int argc, char* argv[])
{
    int  pid;			//	rank of process
	int  num_processes;     //	number of processes
    float time = MPI_Wtime();
    
    MPI_Init(&argc, &argv);	//	start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);		//	get process rank
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);	//	get number of processes
    
    /* create a type for data struct */
    program_data_type_initiate();
    mutant_type_initiate();

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
    initiate_program(pid, num_processes);
    // time += MPI_Wtime();    //  get program time
    time = MPI_Wtime() - time;    //  get program time

    if (pid == ROOT)
        printf("total time: %g\n", time);

    mpi_free_type(&program_data_type);
    mpi_free_type(&mutant_type);

	MPI_Finalize();

    return 0;
}
