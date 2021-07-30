#include <helper_cuda.h>
#include <math.h>
#include "cuda_funcs.h"
#include "def.h"

double gpu_run_program(ProgramData* cpu_data, Mutant* returned_mutant, int first_offset, int last_offset)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate memory on GPU to copy the data from the host
    ProgramData* gpu_data;
    Mutant_GPU* gpu_mutant;
    // int* offset_scores;
    double* scores;
    double returned_score = cpu_data->is_max ? -INFINITY : INFINITY;


    int offsets = last_offset - first_offset;
    int chars = my_strlen(cpu_data->seq2);

    int block_size = floor_highest_power_of2(chars);
    if (block_size > MAX_BLOCK_SIZE)
        block_size = MAX_BLOCK_SIZE;

    int array_size = block_size * offsets;
    int grid_size = offsets; //(array_size + threadsPerBlock - 1) / threadsPerBlock;

    printf("threads=%d, blocks=%d, array size=%d, bytes=%lu\n", block_size, grid_size, array_size, (sizeof(Mutant_GPU) + sizeof(double)) * array_size);


    err = cudaMalloc(&gpu_data, sizeof(ProgramData));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpu_data, cpu_data, sizeof(ProgramData), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&gpu_mutant, array_size * sizeof(Mutant_GPU));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (mutant array) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&scores, array_size * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (score array) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int hashtable_block_size = ceil_highest_power_of2(NUM_CHARS);
    int hashtable_grid_size = (NUM_CHARS + hashtable_block_size - 1) / hashtable_block_size;

    dim3 hashtable_block(hashtable_block_size, hashtable_block_size);                                                                                                             //////////////////////////////////////// fix size to power of 2
    dim3 hashtable_grid(hashtable_grid_size, hashtable_grid_size);
    fill_hashtable_gpu<<<hashtable_grid, hashtable_block>>>();
    err = cudaGetLastError();   
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel for hashtable filling -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
    // printf("blocks=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
    calc_mutants_scores<<<grid_size, block_size, 0>>>(gpu_data, gpu_mutant, scores, offsets, chars, first_offset);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    max_reduction_chars<<<grid_size, block_size>>>(scores, gpu_mutant, cpu_data->is_max, offsets, chars);

    block_size = floor_highest_power_of2(offsets);
    if (block_size > MAX_BLOCK_SIZE)
        block_size = MAX_BLOCK_SIZE;
    

    grid_size = (offsets + block_size - 1) / block_size;
    printf("offsets threads=%d, blocks=%d\n", block_size, grid_size);

    max_reduction_offsets<<<grid_size, block_size>>>(scores, gpu_mutant, cpu_data->is_max, offsets, chars);

    //  the best mutant is in index 0 in mutants array
    err = cudaMemcpy(returned_mutant, &gpu_mutant[0].mutant, sizeof(Mutant), cudaMemcpyDeviceToHost);
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

    if (returned_mutant->ch == NOT_FOUND_CHAR)
        return cpu_data->is_max ? -INFINITY : INFINITY;

    return returned_score;
}

__global__ void calc_mutants_scores(ProgramData* data, Mutant_GPU* mutants, double* scores, int offsets, int chars, int start_offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays

    if (idx >= offsets * blockDim.x)
        return;

    scores[idx] = 0;
    mutants[idx].mutant.ch = NOT_FOUND_CHAR;
    mutants[idx].mutant.offset = -1;
    mutants[idx].mutant.char_offset = -1;
    mutants[idx].diff = data->is_max ? -INFINITY : INFINITY;

    int thrd_offset = blockIdx.x + start_offset;
    int thrd_charoffset;
    int iterations_per_thrd = my_ceil((double)chars / (double)blockDim.x);

    double _pair_score;
    Mutant_GPU temp_mutant;
    int idx1, idx2;
    char c1, c2;
    for (int i = 0; i < iterations_per_thrd; i++)
    {
        thrd_charoffset = threadIdx.x + i * blockDim.x;

        if (thrd_charoffset >= chars)   //  this thread does not have more tasks
            break;

        idx1 = thrd_offset + thrd_charoffset;
        idx2 = thrd_charoffset;

        c1 = data->seq1[idx1];
        c2 = data->seq2[idx2];

        _pair_score = get_weight(get_hashtable_sign(c1, c2), data->weights);
        scores[idx] += _pair_score;

        temp_mutant.mutant.ch = get_substitute(c1, c2, data->weights, data->is_max);
        temp_mutant.mutant.offset = thrd_offset;
        temp_mutant.mutant.char_offset = thrd_charoffset;
        if (temp_mutant.mutant.ch == NOT_FOUND_CHAR)
            temp_mutant.diff = data->is_max ? -INFINITY : INFINITY;
        else
            temp_mutant.diff = get_weight(get_hashtable_sign(c1, temp_mutant.mutant.ch), data->weights) - _pair_score;

        if (i == 0 || is_swapable(&mutants[idx], &temp_mutant, 0, 0, data->is_max))
            mutants[idx] = temp_mutant;   
    }
}

__global__ void max_reduction_chars(double* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int array_size = num_offsets * blockDim.x;
    if (idx >= array_size)
        return;

    int char_pow2 = floor_highest_power_of2(num_chars);

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (idx % blockDim.x  < i)
        {
            int other_idx = idx + i;
            scores[idx] += scores[other_idx];
            if (is_swapable(&mutants[idx], &mutants[other_idx], 0, 0, is_max))
                mutants[idx] = mutants[other_idx];
        }
        __syncthreads();
    }
}

__global__ void max_reduction_offsets(double* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx >= num_offsets)
        return;

    int char_pow2 = floor_highest_power_of2(num_chars);     //  char_pow2 is the previous block size
    if (char_pow2 > MAX_BLOCK_SIZE)
        char_pow2 = MAX_BLOCK_SIZE;

    int idx_global = tidx * char_pow2;  //  id for the global arrays

    int offset_pow2 = floor_highest_power_of2(num_offsets);
    if (offset_pow2 > MAX_BLOCK_SIZE)
        offset_pow2 = MAX_BLOCK_SIZE;

    int last_block_num_threads = num_offsets - blockDim.x * (gridDim.x - 1);

    if (blockIdx.x == gridDim.x - 1 && is_power2(last_block_num_threads))
    {
        int last_block_pow2 = floor_highest_power_of2(last_block_num_threads);

        if (threadIdx.x == last_block_pow2 - 1)   //  last thread that its ID is power of 2
        {
            int other_global_idx;
            __syncthreads();
            for (int i = 1; i <= last_block_num_threads - last_block_pow2; i++)
            {
                other_global_idx = idx_global + i * char_pow2;

                if (is_swapable(&mutants[idx_global], &mutants[other_global_idx], scores[idx_global], scores[other_global_idx], is_max))
                {
                    scores[idx_global] = scores[other_global_idx];
                    mutants[idx_global] = mutants[other_global_idx];
                }
            }
        }
    }

    __syncthreads();
    for (int i = offset_pow2 / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {
            int other_global_idx = idx_global + i * char_pow2;

            if (other_global_idx < num_offsets * char_pow2 &&
                is_swapable(&mutants[idx_global], &mutants[other_global_idx], scores[idx_global], scores[other_global_idx], is_max))
            {
                scores[idx_global] = scores[other_global_idx];
                mutants[idx_global] = mutants[other_global_idx];
            }
        }
        __syncthreads();
    }

    if (tidx == 0)
        scores[0] = index_best_mutant(scores, mutants, is_max, blockDim.x, char_pow2);
}

__global__ void fill_hashtable_gpu()
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= NUM_CHARS || col >= NUM_CHARS)
        return;

    char c1 = FIRST_CHAR + row;
    char c2 = FIRST_CHAR + col;
    hashtable_gpu[row][col] = get_pair_sign(c1, c2);
}


__device__ int my_ceil(double num)
{
    int inum = (int)num;
    if (num == (double)inum)
        return inum;
    return inum + 1;
}

__device__ double index_best_mutant(double* scores, Mutant_GPU* mutants, int is_max, int offsets_block_size, int chars_block_size)
{
    Mutant_GPU best_mutant = mutants[0];
    double best_score = scores[0];
    int idx;
    for (int i = 1; i < gridDim.x; i++)
    {
        idx = i * offsets_block_size * chars_block_size;
        if (is_swapable(&best_mutant, &best_mutant, best_score, scores[idx], is_max))
        {
            best_mutant = mutants[idx];
            best_score = scores[idx];
        }
    }
    mutants[0] = best_mutant;
    return best_score + best_mutant.diff;
}

__device__ int is_swapable(Mutant_GPU* m1, Mutant_GPU* m2, double s1, double s2, int is_max)
{
    double total1 = m1->diff + s1;
    double total2 = m2->diff + s2;

    if ((is_max && total2 > total1) || (!is_max && total2 < total1))
        return TRUE;

    if (total2 == total1)
    {
        if (m2->mutant.offset < m1->mutant.offset)
            return TRUE;

        if (m2->mutant.offset == m1->mutant.offset)
        {
            if (m2->mutant.char_offset < m1->mutant.char_offset)
                return TRUE;
        }
    }
    return FALSE;
}

__host__ __device__ char get_substitute(char c1, char c2, double* w, int is_max)
{
    char sign = get_hashtable_sign(c1, c2);

    return  is_max ?
            get_max_substitute(c1, c2, sign, w)   :
            get_min_substitute(c1, c2, sign, w);
}

__host__  __device__ char get_max_substitute(char c1, char c2, char sign, double* w)
{
    double dot_diff, space_diff;

    switch (sign)
    {
    case DOT:                   //  if there is DOT between two characters, an ASTERISK substitution is possible
    case SPACE:                 //  if there is SPACE between two characters, an ASTERISK substitution is possible
        return c1;

    case ASTERISK:
        dot_diff = - w[ASTERISK_W] - w[DOT_W];
        space_diff = - w[ASTERISK_W] - w[SPACE_W];
        break;

    case COLON:
        dot_diff = w[COLON_W] - w[DOT_W];
        space_diff = w[COLON_W] - w[SPACE_W];
        break;
    }

    char dot_sub = get_substitute_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_substitute_by_sign_with_restrictions(c1, SPACE, c2);

    return get_optimal_substitute(TRUE, dot_diff, dot_sub, space_diff, space_sub);
}

__host__ __device__ char get_min_substitute(char c1, char c2, char sign, double* w)
{   
    char colon_sub = get_substitute_by_sign_with_restrictions(c1, COLON, c2);
    char dot_sub = get_substitute_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_substitute_by_sign_with_restrictions(c1, SPACE, c2);
    char substitue;
    
    double diff1, diff2;
    char sub1, sub2;

    switch (sign)
    {
    case ASTERISK:
        diff1 = - w[ASTERISK_W] - w[DOT_W];     sub1 = dot_sub;     //  DOT differences
        diff2 = - w[ASTERISK_W] - w[SPACE_W];   sub2 = space_sub;   //  SPACE differences
        break;
    
    case COLON:
        diff1 = w[COLON_W] - w[DOT_W];      sub1 = dot_sub;      //  DOT differences
        diff2 = w[COLON_W] - w[SPACE_W];    sub2 = space_sub;    //  SPACE differences
        break;

    case DOT:
        diff1 = w[DOT_W] - w[COLON_W];      sub1 = colon_sub;      //  COLON differences
        diff2 = w[DOT_W] - w[SPACE_W];      sub2 = space_sub;      //  SPACE differences
        break;

    case SPACE:
        diff1 = w[SPACE_W] - w[COLON_W];    sub1 = colon_sub;    //  COLON differences
        diff2 = w[SPACE_W] - w[DOT_W];      sub2 = dot_sub;      //  DOT differences
        break;
    }
    
    if (sign == ASTERISK || sign == COLON)
        return get_optimal_substitute(FALSE, diff1, sub1, diff2, sub2);


    //  if sign is SPACE or DOT, and a substitution would not be possible
    //  C1 will returned because ASTERISK subtitution will always be possible
    substitue = get_optimal_substitute(FALSE, diff1, sub1, diff2, sub2);

    if ((sign == DOT || sign == SPACE) && substitue == NOT_FOUND_CHAR)
        return c1;

    return substitue;
}

__host__ __device__ char get_optimal_substitute(int is_max, double diff1, char sub1, double diff2, char sub2)
{
    //  if first different is better, and such substitue exists
    if ((is_max && diff1 >= diff2) || (!is_max && diff1 <= diff2))
        if (sub1 != NOT_FOUND_CHAR)
            return sub1;

    //  diff1 is not better than diff2, or first substitue is not possible
    //  therefore examination if diff2 is better than 0 is necessary
    if ((is_max && diff2 > 0) || (!is_max && diff2 < 0))
    {
        //  if second substitue is possible, return it
        if (sub2 != NOT_FOUND_CHAR)
            return sub2;

        //  second subtitue is not possible, but the first one might be better than nothing
        if ((is_max && diff1 > 0) || (!is_max && diff1 < 0))
            if (sub1 != NOT_FOUND_CHAR)
                return sub1;
    }

    return NOT_FOUND_CHAR;  //  could not find any substitution
}

__host__ __device__ char get_substitute_by_sign_with_restrictions(char by, char sign, char rest)
{
    char last_char = FIRST_CHAR + NUM_CHARS;
    for (char ch = FIRST_CHAR; ch < last_char; ch++)   //  iterate over alphabet (A-Z)
    {
        if (get_hashtable_sign(by, ch) == sign && get_hashtable_sign(rest, ch) != COLON)  //  if found character which is not in the same conservative group with the previous one
            return ch;
    }
    return NOT_FOUND_CHAR;
}

__host__ __device__ char get_hashtable_sign(char c1, char c2)
{
    if (c1 == HYPHEN && c2 == HYPHEN)   return ASTERISK;
    if (c1 == HYPHEN || c2 == HYPHEN)   return SPACE;
    if (c1 >= FIRST_CHAR + NUM_CHARS || c2 >= FIRST_CHAR + NUM_CHARS)   return NOT_FOUND_CHAR;
	if (c1 < FIRST_CHAR || c2 < FIRST_CHAR)   return NOT_FOUND_CHAR;	

    if (c1 >= c2)       //  only the bottom triangle of the hash table is full -> (hash[x][y] = hash[y][x])
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        return hashtable_gpu[c1 - FIRST_CHAR][c2 - FIRST_CHAR];
    return hashtable_gpu[c2 - FIRST_CHAR][c1 - FIRST_CHAR];
#else
        return hashtable_cpu[c1 - FIRST_CHAR][c2 - FIRST_CHAR];
    return hashtable_cpu[c2 - FIRST_CHAR][c1 - FIRST_CHAR];
#endif
}

__host__ __device__ double get_weight(char sign, double* w)
{
    switch (sign)
    {
    case ASTERISK:  return w[ASTERISK_W];
    case COLON:     return -w[COLON_W];
    case DOT:       return -w[DOT_W];
    case SPACE:     return -w[SPACE_W];
    }
    return 0;
}

__host__ __device__ int is_contain(char* str, char c)
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
__host__ __device__ int is_conservative(char c1, char c2)
{
    for (int i = 0; i < CONSERVATIVE_COUNT; i++)    //  iterate over the conservative groups
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        if (is_contain(conservatives_gpu[i], c1) && is_contain(conservatives_gpu[i], c2))   //  if both characters present
#else
        if (is_contain(conservatives_cpu[i], c1) && is_contain(conservatives_cpu[i], c2))   //  if both characters present
#endif
            return 1;
    return 0;
}

//  check if both characters present in the same semi-conservative group
__host__ __device__ int is_semi_conservative(char c1, char c2)
{
    for (int i = 0; i < SEMI_CONSERVATIVE_COUNT; i++)   //  iterate over the semi-conservative groups
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            if (is_contain(semi_conservatives_gpu[i], c1) && is_contain(semi_conservatives_gpu[i], c2))   //  if both characters present
#else
            if (is_contain(semi_conservatives_cpu[i], c1) && is_contain(semi_conservatives_cpu[i], c2))   //  if both characters present
#endif
                return 1;
    return 0;
}

__host__ __device__ char get_pair_sign(char a, char b)
{
    if      (a == b)                        return ASTERISK;
    else if (is_conservative(a, b))         return COLON;
    else if (is_semi_conservative(a, b))    return DOT;

    return SPACE;
}


__host__ __device__ int floor_highest_power_of2(int n)
{
    for (int i = n; i >= 1; i--)
    {
        if (is_power2(i))
            return i;
    }
    return 0;
}

__host__ __device__ int ceil_highest_power_of2(int n)
{
    for (int i = n; i <= INT_MAX; i++)
    {
        if (is_power2(i))
            return i;
    }
    return 0;
}

__host__ __device__ int is_power2(int n)
{
    if ((n & (n - 1)) == 0)     //  8 = (1000), 7 = (0111)  ->  (8 & 7) = 0000 (num of 1's is 0)
        return TRUE;
    return FALSE;
}

__host__ __device__ int my_strlen(char* str)
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