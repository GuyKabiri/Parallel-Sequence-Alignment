#ifndef __CUDA_FUNCS_H__
#define __CUDA_FUNCS_H__

// #include <cuda_runtime.h>
// #include <helper_cuda.h>

#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

#include "def.h"
#include "program_data.h"
#include "mutant.h"

__constant__  char conservatives_arr_cuda[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
__constant__  char semi_conservatives_arr_cuda[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
__constant__  char char_hash_cuda[NUM_CHARS][NUM_CHARS];

double gpu_run_program(ProgramData* data, Mutant* my_mutant, int first_offset, int last_offset);

#endif //   __CUDA_FUNCS_H__