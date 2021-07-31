#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "main.h"
#include "cpu_funcs.h"
#include "mpi_funcs.h"

extern MPI_Datatype mutant_type;
extern MPI_Datatype program_data_type;

void g()
{
    time_t t;
    FILE* f = fopen("t.txt", "w");

    srand((unsigned) time(&t));
    int first = rand() % 10000;
    int second;
    do {
        second = rand() % 5000;
    } while (second >= first);
    int count = 0;
    char c;
    while (count< first)
    {
        c = FIRST_CHAR + rand() % NUM_CHARS;
        fprintf(f, "%c", c);
        count++;
    }
    fprintf(f, "\n");
    count=0;
    while (count< second)
    {
        c = FIRST_CHAR + rand() % NUM_CHARS;
        fprintf(f, "%c", c);
        count++;
    }
    fclose(f);
}

int main(int argc, char* argv[])
{
    int  pid;			//	rank of process
	int  num_processes;     //	number of processes
    double time = 0;
    
    MPI_Init(&argc, &argv);	//	start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);		//	get process rank
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);	//	get number of processes
    
    //  create mpi datatypes
    program_data_type_initiate();
    mutant_type_initiate();

    omp_set_num_threads(THREADS_COUNT);

    if (argc == 2)
    {
        cuda_percentage = atoi(argv[1]);
        if (cuda_percentage == -100)
            omp_set_num_threads(1);
        if (cuda_percentage < 0 || cuda_percentage > 100)
        {
            if (pid == ROOT)
                printf("Set cude_percentage=0\n");
            cuda_percentage = 0;
        }
    }
    else if (argc == 1)
    {
        if (pid == ROOT)
                printf("CUDA percentage did not set, set cude_percentage=0\n");
            cuda_percentage = 0;
    }

    if (pid == ROOT)
        printf("threads=%2d, processes=%2d\n", omp_get_max_threads(), num_processes);

    time -= MPI_Wtime();    //  substract the mpi initiation time
    initiate_program(pid, num_processes);
    time += MPI_Wtime();    //  get program time

    if (pid == ROOT)
        printf("total time: %g\n", time);

    //  free mpi's datatypes
    mpi_free_type(&program_data_type);  
    mpi_free_type(&mutant_type);

	MPI_Finalize();

    return 0;
}
