#include "mpi_funcs.h"
#include "def.h"
#include "program_data.h"
#include "mutant.h"

MPI_Datatype mutant_type;
MPI_Datatype program_data_type;

void program_data_type_initiate()
{
        				//	number of blocks for each parameter
	int          	blocklengths_program[NUM_OF_PARAMS_DATA] = { 1, 1, SEQ1_MAX_LEN, SEQ2_MAX_LEN, WEIGHTS_COUNT };
	
					//	offset of each parameter, calculated by size of previous parameters
	MPI_Aint 		displacements_program[NUM_OF_PARAMS_DATA] = {    0,                              //  _data.seq1 offset
                                                                sizeof(int),
                                                                sizeof(int) * 2,
                                                                sizeof(int) * 2 + sizeof(double) * WEIGHTS_COUNT,
                                                                sizeof(int) * 2 + sizeof(double) * WEIGHTS_COUNT + sizeof(char) * SEQ1_MAX_LEN };

	MPI_Datatype 	types_program[NUM_OF_PARAMS_DATA] = { MPI_INT, MPI_INT, MPI_CHAR, MPI_CHAR, MPI_DOUBLE };

	MPI_Type_create_struct(NUM_OF_PARAMS_DATA, blocklengths_program, displacements_program, types_program, &program_data_type);
	MPI_Type_commit(&program_data_type);
}

void mutant_type_initiate()
{
    int          	blocklengths_mutant[NUM_OF_PARAMS_MUTANT] = { 1, 1, 1 };
	
					//	offset of each parameter, calculated by size of previous parameters
	MPI_Aint 		displacements_mutant[NUM_OF_PARAMS_MUTANT] = {    0,                              //  _data.seq1 offset
                                                                    sizeof(int),
                                                                    sizeof(int) * 2  };

	MPI_Datatype 	types_mutant[NUM_OF_PARAMS_MUTANT] = { MPI_INT, MPI_INT, MPI_CHAR };

	MPI_Type_create_struct(NUM_OF_PARAMS_MUTANT, blocklengths_mutant, displacements_mutant, types_mutant, &mutant_type);
	MPI_Type_commit(&mutant_type);
}

void mpi_free_type(MPI_Datatype* type)
{
	MPI_Type_free(type);
}