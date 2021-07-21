#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cuda_funcs.h"

__global__ void get_max_value_GPU(ProgramData* data, Mutant* my_mutant, int first_offset, int last_offset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int num_elements = last_offset - first_offset;
    if (i >= num_elements)
        return;

    int my_offset = first_offset + i;
    // printf("gpu tid %4d, offset %3d\n", i, my_offset);
}


double gpu_run_program(ProgramData* data, Mutant* my_mutant, int first_offset, int last_offset)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate memory on GPU to copy the data from the host
    ProgramData* gpu_data;
    Mutant* gpu_mutant;

    err = cudaMalloc((void **)&gpu_data, sizeof(ProgramData));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy data from host to the GPU memory
    err = cudaMemcpy(gpu_data, data, sizeof(ProgramData), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&gpu_mutant, sizeof(Mutant));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int num_elements = last_offset - first_offset;

    // Launch the Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    get_max_value_GPU<<<blocksPerGrid, threadsPerBlock, 0>>>(gpu_data, gpu_mutant, first_offset, last_offset);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the  result from GPU to the host memory so the host could sum it up
    err = cudaMemcpy(my_mutant, gpu_mutant, sizeof(Mutant), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free allocated memory on GPU
    if (cudaFree(gpu_data) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(gpu_mutant) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
