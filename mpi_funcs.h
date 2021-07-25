#ifndef __MPI_FUNCS_H__
#define __MPI_FUNCS_H__

#include <mpi.h>

void mutant_type_initiate();
void program_data_type_initiate();
void mpi_free_type(MPI_Datatype* type);

#endif //   __MPI_FUNCS_H__