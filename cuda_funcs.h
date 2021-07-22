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
__device__ char char_hash_cuda[NUM_CHARS][NUM_CHARS];


double gpu_run_program(ProgramData* data, Mutant* returned_mutant, int first_offset, int last_offset);

__global__ void get_best_mutant_gpu(ProgramData* data, Mutant* mutants, double* scores, int first_offset, int last_offset);
__device__ double find_best_mutant_offset_gpu(ProgramData* data, int offset, Mutant* mt);
__device__ char find_char_gpu(char c1, char c2, double* w, int is_max);
__device__ char find_max_char_gpu(char c1, char c2, char sign, double* w);
__device__ char find_min_char_gpu(char c1, char c2, char sign, double* w);
__device__ char get_char_by_sign_with_restrictions_gpu(char by, char sign, char rest);

__device__ char get_hash_sign_gpu(char c1, char c2);
__device__ double get_weight_gpu(char sign, double* w);

__device__ int strlen_gpu(char* str);
__device__ int is_contain_gpu(char* str, char c);
__device__ int is_conservative_gpu(char c1, char c2);
__device__ int is_semi_conservative_gpu(char c1, char c2);
__device__ char evaluate_chars_gpu(char a, char b);
__global__ void fill_hashtable_gpu();



#endif //   __CUDA_FUNCS_H__