#include <stddef.h>
#include "mpi_funcs.h"
#include "def.h"
#include "program_data.h"
#include "mutant.h"

MPI_Datatype mutant_type;
MPI_Datatype program_data_type;


//	initiating MPI's program data datatype
void program_data_type_initiate()
{
					//	each parameter count
	int          	blocklengths[NUM_OF_PARAMS_DATA] = { 1, WEIGHTS_COUNT, SEQ1_MAX_LEN, SEQ2_MAX_LEN };

					//	each parameter position
	MPI_Aint 		displacements[NUM_OF_PARAMS_DATA] = {   offsetof(struct _data, is_max),
															offsetof(struct _data, weights),
															offsetof(struct _data, seq1),
															offsetof(struct _data, seq2) };

					//	each parameter type
	MPI_Datatype 	types[NUM_OF_PARAMS_DATA] = { MPI_INT, MPI_DOUBLE, MPI_CHAR, MPI_CHAR };

	MPI_Type_create_struct(NUM_OF_PARAMS_DATA, blocklengths, displacements, types, &program_data_type);
	MPI_Type_commit(&program_data_type);
}


//	initiating MPI's mutation datatype
void mutant_type_initiate()
{
					//	each parameter count
    int          	blocklengths[NUM_OF_PARAMS_MUTANT] = { 1, 1, 1 };
					//	each parameter position
	MPI_Aint 		displacements[NUM_OF_PARAMS_MUTANT] = {	offsetof(struct _mutant, offset),
															offsetof(struct _mutant, char_offset),
															offsetof(struct _mutant, ch)  };
					//	each parameter type
	MPI_Datatype 	types[NUM_OF_PARAMS_MUTANT] = { MPI_INT, MPI_INT, MPI_CHAR };
	
	MPI_Type_create_struct(NUM_OF_PARAMS_MUTANT, blocklengths, displacements, types, &mutant_type);
	MPI_Type_commit(&mutant_type);
}


//	freeing MPI datatype
void mpi_free_type(MPI_Datatype* type)
{
	MPI_Type_free(type);
}