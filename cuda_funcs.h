#ifndef __CUDA_FUNCS_H__
#define __CUDA_FUNCS_H__

// #include <cuda_runtime.h>
// #include <helper_cuda.h>

#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

#include "def.h"
#include "program_data.h"
#include "mutant.h"

#ifdef PRINT_DEBUG_CUDA
#define PRINT_DEBUG_CUDA_CHARS
#define PRINT_DEBUG_CUDA_OFFSETS
#endif

#define PRINT_DEBUG_CUDA_CHARS
// #define PRINT_DEBUG_CUDA_OFFSETS


// #define BLOCK_SIZE  256
#define BLOCK_MAX_SIZE 1024

__constant__  char conservatives_gpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
__constant__  char semi_conservatives_gpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
__device__ char hashtable_gpu[NUM_CHARS][NUM_CHARS];

#if (!(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0)))

extern char conservatives_cpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN];
extern char semi_conservatives_cpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN];
extern char hashtable_cpu[NUM_CHARS][NUM_CHARS];

#endif

float gpu_run_program(ProgramData* cpu_data, Mutant* returned_mutant, int first_offset, int last_offset);

__global__ void max_reduction_chars2(float* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars);
__global__ void max_reduction_offsets2(float* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars);


__global__ void max_reduction_chars(float* scores, Mutant_GPU* mutants, int is_max, int offsets, int chars);
__global__ void max_reduction_offsets(float* scores, Mutant_GPU* mutants, int is_max, int offsets, int chars);
__global__ void find_best_mutant_gpu(ProgramData* data, Mutant_GPU* mutants, float* scores, int offsets, int chars, int start_offset);
__device__ int is_swapable(float* scores, Mutant_GPU* mutants, int idx1, int idx2, int is_max, int diff_only);
__host__ __device__ int floor_highest_power_of2(int n);
__host__ __device__ int ceil_highest_power_of2(int n);

__host__ __device__ float find_best_mutant_offset(ProgramData* data, int offset, Mutant* mt);
__host__ __device__ char find_char(char c1, char c2, float* w, int is_max);
__host__ __device__ char find_max_char(char c1, char c2, char sign, float* w);
__host__ __device__ char find_min_char(char c1, char c2, char sign, float* w);
__host__ __device__ char find_optimal_char(int is_max, float diff1, char sub1, float diff2, char sub2);
__host__ __device__ char get_char_by_sign_with_restrictions(char by, char sign, char rest);

__host__ __device__ char get_hash_sign(char c1, char c2);
__host__ __device__ float get_weight(char sign, float* w);

__host__ __device__ int is_contain(char* str, char c);
__host__ __device__ int is_conservative(char c1, char c2);
__host__ __device__ int is_semi_conservative(char c1, char c2);
__host__ __device__ char evaluate_chars(char a, char b);

__host__ __device__ int my_strlen(char* str);
__global__ void fill_hashtable_gpu();



#endif //   __CUDA_FUNCS_H__