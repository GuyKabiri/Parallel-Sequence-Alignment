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
    double* scores;
    double returned_score = cpu_data->is_max ? INT_MIN : INT_MAX;

    int offsets = last_offset - first_offset;
    int chars = my_strlen(cpu_data->seq2);
    int array_size = chars * offsets;

    int threadsPerBlock = 64;    //////////////////////////////////////////////s/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int best = array_size % threadsPerBlock;
    for (int i = threadsPerBlock; i <= BLOCK_MAX_SIZE; i *= 2)
    {
        if (array_size % i < best)
        {
            best = array_size % i;
            threadsPerBlock = i;
        }
    }

    int blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;

    printf("threads=%d, blocks=%d\n", threadsPerBlock, blocksPerGrid);


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
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&scores, array_size * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 threadsPerBlockHash(NUM_CHARS, NUM_CHARS);                                                                                                             //////////////////////////////////////// fix size to power of 2
    dim3 numBlocksHash(1, 1);
    fill_hashtable_gpu<<<numBlocksHash, threadsPerBlockHash>>>();
    err = cudaGetLastError();   
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel for hashtable filling -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    // Launch the Kernel
    // printf("blocks=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
    find_best_mutant_gpu<<<blocksPerGrid, threadsPerBlock, 0>>>(gpu_data, gpu_mutant, scores, offsets, chars);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    cudaDeviceSynchronize();
    // printf("---------------------------(((((((((((((((((((((()))))))))))))))))))))))))))))))-------------------------\n");
    max_reduction_chars<<<blocksPerGrid, threadsPerBlock>>>(scores, gpu_mutant, cpu_data->is_max, offsets, chars);
    // max_reduction<<<1, threadsPerBlock>>>(scores, gpu_mutant, array_size, cpu_data->is_max, offsets, chars, FALSE);
    cudaDeviceSynchronize();


    threadsPerBlock = 64;    //////////////////////////////////////////////s/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    best = offsets % threadsPerBlock;
    for (int i = threadsPerBlock; i <= BLOCK_MAX_SIZE; i *= 2)
    {
        if (offsets % i < best)
        {
            best = offsets % i;
            threadsPerBlock = i;
        }
    }

    blocksPerGrid = (offsets + threadsPerBlock - 1) / threadsPerBlock;
    max_reduction_offsets<<<blocksPerGrid, threadsPerBlock>>>(scores, gpu_mutant, cpu_data->is_max, offsets, chars);

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
        return cpu_data->is_max ? INT_MIN : INT_MAX;

    return returned_score;
}

__global__ void find_best_mutant_gpu(ProgramData* data, Mutant_GPU* mutants, double* scores, int offsets, int chars)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays

    if (idx >= offsets * chars)
        return;

    int idx1 = idx / chars + idx % chars;
    int idx2 = idx % chars;

    char c1 = data->seq1[idx1];
    char c2 = data->seq2[idx2];

    scores[idx] = get_weight(get_hash_sign(c1, c2), data->weights);
    char sub = find_char(c1, c2, data->weights, data->is_max);

    mutants[idx].mutant.ch = find_char(c1, c2, data->weights, data->is_max);
    mutants[idx].mutant.char_offset = idx2 % chars;
    mutants[idx].mutant.offset = idx / chars;
    if (mutants[idx].mutant.ch == NOT_FOUND_CHAR)
        mutants[idx].diff = data->is_max ? INT_MIN : INT_MAX;
    else
        mutants[idx].diff = get_weight(get_hash_sign(c1, sub), data->weights) - scores[idx];

    // printf("offset=%3d, char=%3d, c1=%c, c2=%c, sign=%c, score=%g, diff=%g\n",
    //         mutants[idx].mutant.offset,
    //         mutants[idx].mutant.char_offset,
    //         c1, c2, sub,
    //         scores[idx],
    //         mutants[idx].diff);

    //  each thread holds a mutant for specifict pair in a specific offset
    //  reduction is needed now to sum the offset total score
    //  and to decide which mutant is the most optimal in each offset
    //  later, another reduction is needed to find max of all offsets   
}

__device__ int highest_power_of2(int n)
{
    for (int i = n; i >= 1; i--)
    {
        // if i is a power of 2
        if ((i & (i - 1)) == 0)     //  8 = (1000), 7 = (0111)  ->  (8 & 7) = 0000 (num of 1's is 0)
        {
            return i;
        }
    }
    return 0;
}

__global__ void max_reduction_chars(double* scores, Mutant_GPU* mutants, int is_max, int offsets, int chars)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays

    int array_size = offsets * chars;
    if (idx >= array_size)
        return;

    int t_charoffset = idx % chars;     //  thread's char offset

    //  if the array amount of chars is not a power of 2, an aggregate from the last chars
    //  should applied to the highest power of 2

    int chars_pow2 = highest_power_of2(chars);



    if (chars_pow2 != chars)
    {
        if (t_charoffset >= chars_pow2)
            return;     //  these threads are usless here

        if (t_charoffset == chars_pow2 - 1) //  the new last thread will aggregate the rest of threads
        {
            for (int i = chars_pow2; i < chars; i++)
            {
                scores[idx] += scores[i];       //  aggregate the scores of the rest of array

                //  if better mutant found in the higher index, save it
                if ((mutants[i].diff == mutants[idx].diff && mutants[i].mutant.offset < mutants[idx].mutant.offset)   ||
                    (is_max && mutants[i].diff > mutants[idx].diff) || (!is_max && mutants[i].diff < mutants[idx].diff))
                {
                    mutants[idx] = mutants[i];
                }
            }
        }
    }


    //  here, each threads has offset of Seq1 and char offset of a pair in Seq1 and Seq2,
    //  the amount of characters here is a power of 2 and can be reduct
    //  the reduction will perform between all the char in each offsets
    //  which would lead eventually that the thread with the char offset of 0
    //  in each offset will save the optimal mutant of this offset, and the total score of
    //  this offset (without the mutation)
    //  in global terms index=(i * chars) will hold the above (where i from 0 to offsets-1)

    for (int size = chars_pow2 / 2; size > 0; size /= 2)
    {
        if (t_charoffset >= size)
            break;

        // printf("%d, %d  %f, %f\n", idx, idx+size, mutants[idx + 0].diff, mutants[idx + size].diff);

        scores[idx] += scores[idx + size];

        if ((mutants[idx + size].diff == mutants[idx].diff && mutants[idx + size].mutant.offset < mutants[idx].mutant.offset) ||
            (is_max && mutants[idx + size].diff > mutants[idx].diff) || (!is_max && mutants[idx + size].diff < mutants[idx].diff))
        {
            mutants[idx] = mutants[idx + size];
        }
        __syncthreads();
    }

    // printf("id=%3d, offset=%3d, char=%3d, score=%g, diff=%g\n",
    // idx,
    //         mutants[idx].mutant.offset,
    //         mutants[idx].mutant.char_offset,
    //         scores[idx],
    //         mutants[idx].diff);
}

__global__ void max_reduction_offsets(double* scores, Mutant_GPU* mutants, int is_max, int offsets, int chars)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays

    if (idx >= offsets)
        return;

    //  here amount off threads equals to amount of offsets
    //  each thread should look at the global index of array idx * chars

    int offsets_pow2 = highest_power_of2(offsets);

    if (offsets_pow2 != offsets)
    {
        if (idx >= offsets_pow2)
            return;
        
        if (idx == offsets_pow2 - 1)
        {
            for (int i = offsets_pow2; i < offsets; i++)
            {
                int index = i * chars;
                double my_total = scores[idx] + mutants[index].diff;
                double other_total = scores[index] + mutants[index].diff;
                if ((other_total == my_total && mutants[index].mutant.offset < mutants[idx].mutant.offset)   ||
                    (is_max && other_total > my_total) || (!is_max && mutants[index].diff < my_total))
                {
                    mutants[idx] = mutants[index];
                    scores[idx] = scores[index];
                }
            }
        }
    }

    for (int size = offsets_pow2 / 2; size > 0; size /= 2)
    {
        int index = size * chars;
        if (idx >= size)
            break;

        // printf("%d, %d\n", idx, idx + index);

        double my_total = scores[idx] + mutants[index].diff;
        double other_total = scores[index] + mutants[index].diff;
        if ((other_total == my_total && mutants[index].mutant.offset < mutants[idx].mutant.offset)   ||
            (is_max && other_total > my_total) || (!is_max && mutants[index].diff < my_total))
        {
            mutants[idx] = mutants[index];
            scores[idx] = scores[index];
        }
        __syncthreads();
    }
}

// __global__ void multiblock_max_reduction(double* scores, Mutant_GPU* mutants, int array_size, int is_max)
// {
//     int thIdx = threadIdx.x;
//     int gthIdx = thIdx + blockIdx.x*BLOCK_SIZE;
//     const int gridSize = BLOCK_SIZE*gridDim.x;
//     double s = is_max ? INT_MIN : INT_MAX;
//     Mutant m;
//     for (int i = gthIdx; i < array_size; i += gridSize)
//     {   
//         if ((is_max && scores[i] > s) || (!is_max && scores[i] < s))
//         {
//             s = scores[i];
//             m = mutants[i];
//         }
//     }
//     scores[thIdx] = s;
//     mutants[thIdx] = m;
//     __syncthreads();
//     for (int size = BLOCK_SIZE / 2; size > 0; size /= 2)
//     { //uniform
//         if (thIdx<size && thIdx + size < array_size)
//         {
//             if ((is_max && scores[thIdx + size] >= scores[thIdx]) ||
//                 (!is_max && scores[thIdx + size] <= scores[thIdx]))   //  include '==' to make sure the smaller offset is saved
//             {
//                 if (scores[thIdx + size] == scores[thIdx] && !(mutants[thIdx + size].offset < shMutant[thIdx].offset))   //  if scores equal and the smaller offset is save -> continue
//                     continue;
//                 //  otherwise, the scores are not equal, or they are equal, but the greater one is saved -> save the new score
//                 scores[thIdx] = scores[thIdx + size];
//                 shMutant[thIdx] = shMutant[thIdx + size];
//             }

//         }
//         __syncthreads();
//     }
//     if (thIdx == 0)
//     {
//         scores[blockIdx.x] = scores[0];
//         mutants[blockIdx.x] = shMutant[0];
//     }
// }

__host__ __device__ double find_best_mutant_offset(ProgramData* data, int offset, Mutant* mt)
{
    int idx1, idx2;
    double total_score = 0;
    double pair_score, mutant_diff;
    double best_mutant_diff = data->is_max ? INT_MIN : INT_MAX;
    int iterations = my_strlen(data->seq2);
    char c1, c2, sub;

    mt->offset = -1;
    mt->char_offset = -1;
    mt->ch = NOT_FOUND_CHAR;

    for (int i = 0; i < iterations; i++)            //  iterate over all the characters
    {
        idx1 = offset + i;                      //  index of seq1
        idx2 = i;                               //  index of seq2
        c1 = data->seq1[idx1];                  //  current char in seq1
        c2 = data->seq2[idx2];                  //  current char in seq2
        pair_score = get_weight(get_hash_sign(c1, c2), data->weights);    //  get weight before substitution
        total_score += pair_score;

        sub = find_char(c1, c2, data->weights, data->is_max);

        // printf("c1=%c, c2=%c, s=(%c)\n", c1, c2, sub);                           ///////////////////////////////////////////////////////////////////////////////////

        if (sub == NOT_FOUND_CHAR)   //  if did not find any substitution
            continue;

        mutant_diff = get_weight(get_hash_sign(c1, sub), data->weights) - pair_score;    //  difference between original and mutation weights


        if ((data->is_max && mutant_diff > best_mutant_diff) || 
            (!data->is_max && mutant_diff < best_mutant_diff))
        {
            best_mutant_diff = mutant_diff;
            mt->ch = sub;
            mt->char_offset = i;        //  offset of char inside seq2
            mt->offset = offset;
        }
    }

    if (mt->ch == NOT_FOUND_CHAR)       //  mutation is not possible in this offset
        return best_mutant_diff;
    return total_score + best_mutant_diff;     //  best mutant is returned in struct mt
}

__host__ __device__ char find_char(char c1, char c2, double* w, int is_max)
{
    char sign = get_hash_sign(c1, c2);

    return  is_max ?
            find_max_char(c1, c2, sign, w)   :
            find_min_char(c1, c2, sign, w);
}

__host__  __device__ char find_max_char(char c1, char c2, char sign, double* w)
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

    char dot_sub = get_char_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_char_by_sign_with_restrictions(c1, SPACE, c2);

    return find_optimal_char(TRUE, dot_diff, dot_sub, space_diff, space_sub);
}

__host__ __device__ char find_min_char(char c1, char c2, char sign, double* w)
{   
    char colon_sub = get_char_by_sign_with_restrictions(c1, COLON, c2);
    char dot_sub = get_char_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_char_by_sign_with_restrictions(c1, SPACE, c2);
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
        return find_optimal_char(FALSE, diff1, sub1, diff2, sub2);


    //  if sign is SPACE or DOT, and a substitution would not be possible
    //  C1 will returned because ASTERISK subtitution will always be possible
    substitue = find_optimal_char(FALSE, diff1, sub1, diff2, sub2);

    if ((sign == DOT || sign == SPACE) && substitue == NOT_FOUND_CHAR)
        return c1;

    return substitue;
}

__host__ __device__ char find_optimal_char(int is_max, double diff1, char sub1, double diff2, char sub2)
{
    //  if first different is better, and such substitue exists
    if ((is_max && diff1 >= diff2) || (!is_max && diff1 <= diff2))
        if (sub1 != NOT_FOUND_CHAR)
            return sub1;

    return sub2;

    // //  diff1 is not better than diff2, or first substitue is not possible
    // //  therefore examination if diff2 is better than 0 is necessary
    // if ((is_max && diff2 > 0) || (!is_max && diff2 < 0))
    // {
    //     //  if second substitue is possible, return it
    //     if (sub2 != NOT_FOUND_CHAR)
    //         return sub2;

    //     //  second subtitue is not possible, but the first one might be better than nothing
    //     if ((is_max && diff1 > 0) || (!is_max && diff1 < 0))
    //         if (sub1 != NOT_FOUND_CHAR)
    //             return sub1;
    // }

    // return NOT_FOUND_CHAR;  //  could not find any substitution
}

__host__ __device__ char get_char_by_sign_with_restrictions(char by, char sign, char rest)
{
    char last_char = FIRST_CHAR + NUM_CHARS;
    for (char ch = FIRST_CHAR; ch < last_char; ch++)   //  iterate over alphabet (A-Z)
    {
        if (get_hash_sign(by, ch) == sign && get_hash_sign(rest, ch) != COLON)  //  if found character which is not in the same conservative group with the previous one
            return ch;
    }
    return NOT_FOUND_CHAR;
}

__host__ __device__ char get_hash_sign(char c1, char c2)
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

__host__ __device__ char evaluate_chars(char a, char b)
{
    if      (a == b)                        return ASTERISK;
    else if (is_conservative(a, b))         return COLON;
    else if (is_semi_conservative(a, b))    return DOT;

    return SPACE;
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

__global__ void fill_hashtable_gpu()
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    char c1 = FIRST_CHAR + row;
    char c2 = FIRST_CHAR + col;
    hashtable_gpu[row][col] = evaluate_chars(c1, c2);
}
