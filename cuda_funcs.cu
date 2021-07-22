#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cuda_funcs.h"
#include "def.h"

__global__ void get_best_mutant_gpu(ProgramData* data, Mutant* mutants, double* scores, int first_offset, int last_offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays
    // printf("%2d, %2d, %2d\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("%2d, %2d, %2d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // printf("%2d, %2d, %2d\n", threadIdx.x, threadIdx.y, threadIdx.z);

    // printf("index=%3d val=%f\n", idx, scores[idx]);
    // printf("%c\n", char_hash_cuda[0][0]);
    // fill_hashtable_gpu();

    


    int offsets = last_offset - first_offset;

    if (idx >= offsets)
    {
        scores[idx] = 0;
        return;
    }


  
}

__device__ double find_best_mutant_offset_gpu(ProgramData* data, int offset, Mutant* mt)
{
    double total_score = 0;
    int idx1, idx2;

    int chars = strlen_gpu(data->seq2);

    for (int i = 0; i < chars; i++)
    {
        idx1 = offset + i;
        idx2 = i;


    }
    return -990;
}



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
    // int chars = strlen(data->seq2);
    // int num_tasks = offsets * chars;

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

    err = cudaMalloc(&gpu_mutant, offsets * sizeof(Mutant));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&scores, offsets * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 threadsPerBlockHash(NUM_CHARS, NUM_CHARS);
    dim3 numBlocksHash(1, 1);
    fill_hashtable_gpu<<<numBlocksHash, threadsPerBlockHash>>>();



    // Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (offsets + threadsPerBlock - 1) / threadsPerBlock;//offsets;
    printf("blocks=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
    get_best_mutant_gpu<<<blocksPerGrid, threadsPerBlock, 0>>>(gpu_data, gpu_mutant, scores, first_offset, last_offset);
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
        fprintf(stderr, "Failed to copy result mutant from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // it's score in index 0 in scores array
    err = cudaMemcpy(&returned_score, &scores[0], sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result score from device to host -%s\n", cudaGetErrorString(err));
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











__device__ int strlen_gpu(char* str)
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

__device__ int is_contain_gpu(char* str, char c)
{
    char* t = str;

    while (*t)
    {
        if (*t == c)
            return 1;
        
        ++t;
    }
    return 0;
}

//  check if both characters present in the same conservative group
__device__ int is_conservative_gpu(char c1, char c2)
{
    for (int i = 0; i < CONSERVATIVE_COUNT; i++)    //  iterate over the conservative groups
        if (is_contain_gpu(conservatives_arr_cuda[i], c1) && is_contain_gpu(conservatives_arr_cuda[i], c2))   //  if both characters present
            return 1;
    return 0;
}

//  check if both characters present in the same semi-conservative group
__device__ int is_semi_conservative_gpu(char c1, char c2)
{
    for (int i = 0; i < SEMI_CONSERVATIVE_COUNT; i++)   //  iterate over the semi-conservative groups
            if (is_contain_gpu(semi_conservatives_arr_cuda[i], c1) && is_contain_gpu(semi_conservatives_arr_cuda[i], c2))   //  if both characters present
                return 1;
    return 0;
}

__device__ char evaluate_chars_gpu(char a, char b)
{
    if      (a == b)                        return STAR;
    else if (is_conservative_gpu(a, b))         return COLON;
    else if (is_semi_conservative_gpu(a, b))    return DOT;

    return SPACE;
}

__global__ void fill_hashtable_gpu()
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    char c1 = FIRST_CHAR + row;
    char c2 = FIRST_CHAR + col;
    char_hash_cuda[row][col] = evaluate_chars_gpu(c1, c2);
}
