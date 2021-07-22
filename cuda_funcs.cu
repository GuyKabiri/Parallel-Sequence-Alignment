#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>
#include "cuda_funcs.h"

__global__ void get_max_value_GPU(ProgramData* data, Mutant* mutants, double* scores, int first_offset, int last_offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays
    // printf("%2d, %2d, %2d\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("%2d, %2d, %2d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // printf("%2d, %2d, %2d\n", threadIdx.x, threadIdx.y, threadIdx.z);

    // printf("seq off=%3d, char off=%3d index=%3d val=%f\n", blockIdx.x, threadIdx.x, idx, scores[idx]);

    int offsets = last_offset - first_offset;
    int chars = gpustrlen(data->seq2);
    int num_tasks = chars * offsets;

    scores[idx] = -900;

    // printf("seq off=%3d, char off=%3d index=%3d val=%f\n", blockIdx.x, threadIdx.x, idx, scores[idx]);

    // int sum = 0;
    // for (int i = idx; i < num_tasks; i += blockDim.x)
    //     sum += scores[i];

    // __shared__ int r[31];
    // r[idx] = sum;
    // __syncthreads();
    // for (int size = blockDim.x/2; size>0; size/=2) { //uniform
    //     if (idx<size)
    //     r[idx] += r[idx+size];
    //     __syncthreads();
    // }
    // if (idx == 0)   printf("---------------\nscore=%g\n", scores[idx]);

    // int chars = strlen(data->seq2);
    // int offsets = last_offset - first_offset;
    // int num_tasks = offsets + chars;

    // if (i >= num_tasks)
    //     return;

    // int seq_off = 0;
    // int char_off = 0;
    // printf("gpu tid %4d, offset %3d\n", i, my_offset);

    
}

__device__ int gpustrlen(char* str)
{
    int count = 0;
    char* t = str;
    while (*t)
    {
        ++count;
        ++t;
    }
    return count;
}


/*
    an array of mutants will be allocated, the size of array will be amount of offsets X amount of characters in seq2
    that means each thread will have to evaluate only one pair of characters and returned their original score, the subtitution
    and the difference in the score with the subtitution, scores need to be summed up, and optimal mutant difference need to be found (MAX / MIN)
*/
double gpu_run_program(ProgramData* data, Mutant* returned_mutant, int first_offset, int last_offset)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate memory on GPU to copy the data from the host
    ProgramData* gpu_data;
    Mutant* gpu_mutant;
    double* scores;
    double returned_score = -999;

    int offsets = last_offset - first_offset;
    int chars = strlen(data->seq2);
    int num_tasks = offsets * chars;

    err = cudaMalloc(&gpu_data, sizeof(ProgramData));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpu_data, data, sizeof(ProgramData), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&gpu_mutant, num_tasks * sizeof(Mutant));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&scores, num_tasks * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    // Launch the Kernel
    int threadsPerBlock = chars;
    int blocksPerGrid = offsets;
    get_max_value_GPU<<<blocksPerGrid, threadsPerBlock, 0>>>(gpu_data, gpu_mutant, scores, first_offset, last_offset);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  the best mutant is in index 0 in mutants array
    err = cudaMemcpy(returned_mutant, &gpu_mutant[0], sizeof(Mutant), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // it's score in index 0 in scores array
    err = cudaMemcpy(&returned_score, &scores[0], sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(gpu_data);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(gpu_mutant);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(scores);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // printf("gpu finih\n");
    return returned_score;
}
