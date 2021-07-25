#include "mpi_funcs.h"
#include "def.h"
#include "program_data.h"
#include "mutant.h"

MPI_Datatype mutant_type;
MPI_Datatype program_data_type;

void program_data_type_initiate()
{
	MPI_Datatype struct_type;

        				//	number of blocks for each parameter
	int          	blocklengths[NUM_OF_PARAMS_DATA] = { 1, WEIGHTS_COUNT, SEQ1_MAX_LEN, SEQ2_MAX_LEN };
	
					//	offset of each parameter, calculated by size of previous parameters
	MPI_Aint 		displacements[NUM_OF_PARAMS_DATA] = {    0,			//  _data.is_max
														sizeof(int),	//  _data.weights
														sizeof(int) + sizeof(double) * WEIGHTS_COUNT,	//  _data.seq1
														sizeof(int) + sizeof(double) * WEIGHTS_COUNT + sizeof(char) * SEQ1_MAX_LEN };	//  _data.seq2

	MPI_Datatype 	types[NUM_OF_PARAMS_DATA] = { MPI_INT, MPI_DOUBLE, MPI_CHAR, MPI_CHAR };

	MPI_Type_create_struct(NUM_OF_PARAMS_DATA, blocklengths, displacements, types, &struct_type);
	MPI_Type_create_resized(struct_type, 0, sizeof(ProgramData), &program_data_type);
	MPI_Type_commit(&program_data_type);
}

void mutant_type_initiate()
{
	MPI_Datatype struct_type;

    int          	blocklengths[NUM_OF_PARAMS_MUTANT] = { 1, 1, 1 };
	
					//	offset of each parameter, calculated by size of previous parameters
	MPI_Aint 		displacements[NUM_OF_PARAMS_MUTANT] = {    0,					//	_mutant.offset
															sizeof(int),			//	_mutant.char_offset
															sizeof(int) * 2  };		//	_mutant.ch

	MPI_Datatype 	types[NUM_OF_PARAMS_MUTANT] = { MPI_INT, MPI_INT, MPI_CHAR };

	MPI_Type_create_struct(NUM_OF_PARAMS_MUTANT, blocklengths, displacements, types, &struct_type);
	MPI_Type_create_resized(struct_type, 0, sizeof(Mutant), &mutant_type);
	MPI_Type_commit(&mutant_type);

	
}

void mpi_free_type(MPI_Datatype* type)
{
	MPI_Type_free(type);
}