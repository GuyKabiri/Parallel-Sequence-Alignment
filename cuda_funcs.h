#ifndef __CUDA_FUNCS_H__
#define __CUDA_FUNCS_H__

// #include <cuda_runtime.h>
// #include <helper_cuda.h>

#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

#include "def.h"
#include "program_data.h"
#include "mutant.h"

#define BLOCK_SIZE  256

__constant__  char conservatives_gpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
__constant__  char semi_conservatives_gpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
__device__ char hashtable_gpu[NUM_CHARS][NUM_CHARS];

#if (!(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0)))

extern char conservatives_cpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN];
extern char semi_conservatives_cpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN];
extern char hashtable_cpu[NUM_CHARS][NUM_CHARS];

#endif

double gpu_run_program(ProgramData* data, Mutant* returned_mutant, int first_offset, int last_offset);

__global__ void find_best_mutant_gpu(ProgramData* data, Mutant* mutants, double* scores, int first_offset, int last_offset);
__device__ double find_best_mutant_offset_gpu(ProgramData* data, int offset, Mutant* mt);
__host__ __device__ char find_char(char c1, char c2, double* w, int is_max);
__host__ __device__ char find_max_char(char c1, char c2, char sign, double* w);
__host__ __device__ char find_min_char(char c1, char c2, char sign, double* w);
__host__ __device__ char find_optimal_char(int is_max, double diff1, char sub1, double diff2, char sub2, char def_char);
__host__ __device__ char get_char_by_sign_with_restrictions(char by, char sign, char rest);

__host__ __device__ char get_hash_sign(char c1, char c2);
__host__ __device__ double get_weight(char sign, double* w);

__host__ __device__ int is_contain(char* str, char c);
__host__ __device__ int is_conservative(char c1, char c2);
__host__ __device__ int is_semi_conservative(char c1, char c2);
__host__ __device__ char evaluate_chars(char a, char b);

__device__ int strlen_gpu(char* str);
__global__ void fill_hashtable_gpu();



#endif //   __CUDA_FUNCS_H__