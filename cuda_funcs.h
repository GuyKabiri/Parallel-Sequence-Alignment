#ifndef __CUDA_FUNCS_H__
#define __CUDA_FUNCS_H__

#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

#include "def.h"
#include "program_data.h"
#include "mutant.h"

#ifdef PRINT_DEBUG_CUDA
#define PRINT_DEBUG_CUDA_CHARS
#define PRINT_DEBUG_CUDA_OFFSETS
#endif

// #define PRINT_DEBUG_CUDA_CHARS
// #define PRINT_DEBUG_CUDA_OFFSETS

#define MAX_BLOCK_SIZE 1024

__constant__  char conservatives_gpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
__constant__  char semi_conservatives_gpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
__device__ char hashtable_gpu[NUM_CHARS][NUM_CHARS];

#if (!(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0)))

extern char conservatives_cpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN];
extern char semi_conservatives_cpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN];
extern char hashtable_cpu[NUM_CHARS][NUM_CHARS];

#endif

double gpu_run_program(ProgramData* cpu_data, Mutant* returned_mutant, int first_offset, int last_offset);

__global__ void calc_mutants_scores(ProgramData* data, Mutant_GPU* mutants, double* scores, int offsets, int chars, int start_offset);
__global__ void reduction(double* scores, Mutant_GPU* mutants, int is_max, int num_elements, int stride, int to_aggregate);
__global__ void max_reduction_chars(double* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars);
__global__ void max_reduction_offsets(double* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars);
__global__ void fill_hashtable_gpu();

__device__ int my_ceil(double num);
__device__ int is_swapable(Mutant_GPU* m1, Mutant_GPU* m2, double s1, double s2, int is_max);
__device__ double reduce_last_results(double* scores, Mutant_GPU* mutants, int is_max, int stride);

__host__ __device__ char get_substitute(char c1, char c2, double* w, int is_max);
__host__ __device__ char get_max_substitute(char c1, char c2, char sign, double* w);
__host__ __device__ char get_min_substitute(char c1, char c2, char sign, double* w);
__host__ __device__ char get_optimal_substitute(int is_max, double diff1, char sub1, double diff2, char sub2);
__host__ __device__ char get_substitute_by_sign_with_restrictions(char by, char sign, char rest);

__host__ __device__ char get_hashtable_sign(char c1, char c2);
__host__ __device__ double get_weight(char sign, double* w);
__host__ __device__ int is_contain(char* str, char c);
__host__ __device__ int is_conservative(char c1, char c2);
__host__ __device__ int is_semi_conservative(char c1, char c2);
__host__ __device__ char get_pair_sign(char a, char b);

__host__ __device__ int floor_highest_power_of2(int n);
__host__ __device__ int ceil_highest_power_of2(int n);
__host__ __device__ int is_power2(int n);
__host__ __device__ int my_strlen(char* str);


#endif //   __CUDA_FUNCS_H__