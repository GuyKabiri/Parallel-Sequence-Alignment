#include <helper_cuda.h>
#include <math.h>
#include "cuda_funcs.h"
#include "def.h"

float gpu_run_program(ProgramData* cpu_data, Mutant* returned_mutant, int first_offset, int last_offset)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate memory on GPU to copy the data from the host
    ProgramData* gpu_data;
    Mutant_GPU* gpu_mutant;
    // int* offset_scores;
    float* scores;
    float returned_score = cpu_data->is_max ? -INFINITY : INFINITY;


    int offsets = last_offset - first_offset;
    int chars = my_strlen(cpu_data->seq2);

    int threadsPerBlock = floor_highest_power_of2(chars);
    if (threadsPerBlock > MAX_BLOCK_SIZE)
        threadsPerBlock = MAX_BLOCK_SIZE;

    int array_size = threadsPerBlock * offsets;
    int blocksPerGrid = offsets; //(array_size + threadsPerBlock - 1) / threadsPerBlock;

    printf("threads=%d, blocks=%d, array size=%d, bytes=%lu\n", threadsPerBlock, blocksPerGrid, array_size, (sizeof(Mutant_GPU) + sizeof(float)) * array_size);


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

    err = cudaMalloc(&scores, array_size * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (score array) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int hashtable_threadsPerBlock = ceil_highest_power_of2(NUM_CHARS);
    int hashtable_Blocks = (NUM_CHARS + hashtable_threadsPerBlock - 1) / hashtable_threadsPerBlock;

    dim3 threadsPerBlockHash(hashtable_threadsPerBlock, hashtable_threadsPerBlock);                                                                                                             //////////////////////////////////////// fix size to power of 2
    dim3 numBlocksHash(hashtable_Blocks, hashtable_Blocks);
    fill_hashtable_gpu<<<numBlocksHash, threadsPerBlockHash>>>();
    err = cudaGetLastError();   
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel for hashtable filling -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Kernel
    // printf("blocks=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
    find_best_mutant_gpu<<<blocksPerGrid, threadsPerBlock, 0>>>(gpu_data, gpu_mutant, scores, offsets, chars, first_offset);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    max_reduction_chars2<<<blocksPerGrid, threadsPerBlock>>>(scores, gpu_mutant, cpu_data->is_max, offsets, chars);
    // max_reduction<<<1, threadsPerBlock>>>(scores, gpu_mutant, array_size, cpu_data->is_max, offsets, chars, FALSE);

    
    threadsPerBlock = floor_highest_power_of2(offsets);
    printf("%d , %d    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",offsets, threadsPerBlock);
    if (threadsPerBlock > MAX_BLOCK_SIZE)
        threadsPerBlock = MAX_BLOCK_SIZE;
    

    blocksPerGrid = (offsets + threadsPerBlock - 1) / threadsPerBlock;
    printf("offsets threads=%d, blocks=%d\n", threadsPerBlock, blocksPerGrid);

    max_reduction_offsets2<<<blocksPerGrid, threadsPerBlock>>>(scores, gpu_mutant, cpu_data->is_max, offsets, chars);

    //  the best mutant is in index 0 in mutants array
    err = cudaMemcpy(returned_mutant, &gpu_mutant[0].mutant, sizeof(Mutant), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result mutant from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // it's score in index 0 in scores array
    err = cudaMemcpy(&returned_score, &scores[0], sizeof(float), cudaMemcpyDeviceToHost);
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

__global__ void find_best_mutant_gpu(ProgramData* data, Mutant_GPU* mutants, float* scores, int offsets, int chars, int start_offset)
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

    float _pair_score;
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

        _pair_score = get_weight(get_hash_sign(c1, c2), data->weights);
        scores[idx] += _pair_score;

        temp_mutant.mutant.ch = find_char(c1, c2, data->weights, data->is_max);
        temp_mutant.mutant.offset = thrd_offset;
        temp_mutant.mutant.char_offset = thrd_charoffset;
        if (temp_mutant.mutant.ch == NOT_FOUND_CHAR)
            temp_mutant.diff = data->is_max ? -INFINITY : INFINITY;
        else
            temp_mutant.diff = get_weight(get_hash_sign(c1, temp_mutant.mutant.ch), data->weights) - _pair_score;

        // printf("%d, %d, %d, id=%3d, off=%3d, char=%3d, c1=%c, c2=%c, s=%c, score=%f, diff=%f\n",
        //         chars, blockDim.x, iterations_per_thrd,
        //         idx,
        //         thrd_offset, 
        //         thrd_charoffset, 
        //         c1, c2,
        //         temp_mutant.mutant.ch,
        //         _pair_score, 
        //         temp_mutant.diff);

        if (i == 0 || is_swapable(&mutants[idx], &temp_mutant, 0, 0, data->is_max))
            mutants[idx] = temp_mutant;   
    }

    // printf("id=%3d, off=%3d, char=%3d, s=%c, score=%f, diff=%f\n",
    //             idx,
    //             mutants[idx].mutant.offset, 
    //             mutants[idx].mutant.char_offset, 
    //             mutants[idx].mutant.ch,
    //             scores[idx], 
    //             mutants[idx].diff);
}



__global__ void max_reduction_chars2(float* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int array_size = num_offsets * blockDim.x;
    if (idx >= array_size)
        return;

    int char_pow2 = floor_highest_power_of2(num_chars);

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        // if (idx % char_pow2 < i)
        if (idx % blockDim.x  < i)
        {
            int _idx = idx + i;

            // printf("%3d, %3d, score=%g, %g, %g diff=%g, %g swap? %s\n",
            //         idx, 
            //         _idx, 
            //         scores[idx], 
            //         scores[_idx],
            //         scores[idx]+ scores[_idx],
            //         mutants[idx].diff,
            //         mutants[_idx].diff,
            //         is_swapable(&mutants[idx], &mutants[_idx], 0, 0, is_max)?"true":"false");

            scores[idx] += scores[_idx];
            if (is_swapable(&mutants[idx], &mutants[_idx], 0, 0, is_max))
                mutants[idx] = mutants[_idx];
        }
        __syncthreads();
    }
    
    // if (threadIdx.x == 0)
    //     printf("----------------******************** %3d off=%3d, score=%g, diff=%g\n", 
    //             idx,
    //             mutants[idx].mutant.offset,
    //             scores[idx], 
    //             mutants[idx].diff);
}

__global__ void max_reduction_offsets2(float* scores, Mutant_GPU* mutants, int is_max, int num_offsets, int num_chars)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx >= num_offsets)
        return;

    // printf("++++++++++++ %d offsets=%d ++++++++++++++++\n", tidx, num_offsets);

    int char_pow2 = floor_highest_power_of2(num_chars);     //  char_pow2 is the previous block size
    if (char_pow2 > MAX_BLOCK_SIZE)
        char_pow2 = MAX_BLOCK_SIZE;

    int idx_global = tidx * char_pow2;  //  id for the global arrays

    // printf("++++++++++++++++ id=%3d, offset=%3d, %g %g\n", 
    //         idx_global, 
    //         mutants[idx_global].mutant.offset,
    //         scores[idx_global],  
    //         mutants[idx_global].diff);

    int offset_pow2 = floor_highest_power_of2(num_offsets);
    if (offset_pow2 > MAX_BLOCK_SIZE)
        offset_pow2 = MAX_BLOCK_SIZE;

    int runabble = offset_pow2;

    if (blockIdx.x == gridDim.x - 1 && num_offsets != offset_pow2)
    {
        // if (tidx % num_offsets >= offset_pow2)                      ////////////////////////////////////////////////////////////////////////  
        // if (threadIdx.x >= offset_pow2)                      ////////////////////////////////////////////////////////////////////////    
        // {
        //     printf("bye %d\n", tidx);
        //     return;
        // }
        int last_block_num_threads = num_offsets - offset_pow2;
        runabble = floor_highest_power_of2(last_block_num_threads);

        if (threadIdx.x == runabble - 1)   //  last thread that its ID is power of 2
        {
            __syncthreads();
            // int i = 1; i < num_chars - char_pow2 + 1
            for (int i = 1; i < last_block_num_threads - runabble + 1; i++)
            {
                int other_global_idx = idx_global + i * char_pow2;

                // printf("in id=%2d,  glob id=%3d, %3d, score=%g, %g | %g %g swap? (%d)\n",
                //         threadIdx.x,
                //         idx_global, 
                //         other_global_idx, 

                //         scores[idx_global], 
                //         mutants[idx_global].diff,
                //         scores[other_global_idx],
                //         mutants[other_global_idx].diff,
                //         is_swapable(&mutants[idx_global], &mutants[other_global_idx], scores[idx_global], scores[other_global_idx], is_max));

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
            // printf("%4d, %4d\n", idx_global, other_global_idx);


            // printf("in= %3d, %3d, global=%3d, %3d, offsets=%3d %3d, score=%g, %g | %g %g swap? (%d) block=%d\n",
            //             tidx, tidx + i,
            //             idx_global, 
            //             other_global_idx, 
            //             mutants[idx_global].mutant.offset,
            //             mutants[other_global_idx].mutant.offset,
            //             scores[idx_global], 
            //             mutants[idx_global].diff,
            //             scores[other_global_idx],
            //             mutants[other_global_idx].diff,
            //             is_swapable(&mutants[idx_global], &mutants[other_global_idx], scores[idx_global], scores[other_global_idx], is_max),
            //             blockIdx.x);

            if (other_global_idx < num_offsets * char_pow2)
            if (is_swapable(&mutants[idx_global], &mutants[other_global_idx], scores[idx_global], scores[other_global_idx], is_max))
            {
                scores[idx_global] = scores[other_global_idx];
                mutants[idx_global] = mutants[other_global_idx];
            }

            // if (idx_global == 0 && other_global_idx == 5)
            // {
            //     printf("o score=%g, 5 score=%g\n", scores[idx_global] + mutants[idx_global].diff, scores[other_global_idx] + mutants[other_global_idx].diff);
            // }
        }
        __syncthreads();
    }
    // printf("%3d -> %g\n", idx_global, scores[idx_global]);

    // if (tidx == 0 && mutants[0].mutant.ch != NOT_FOUND_CHAR)
    //     scores[0] += mutants[0].diff;






    // if (tidx == 0)
    //     scores[0] += mutants[0].diff;

    // if (tidx % blockDim.x == 0)
    //     printf("))))))))))))))))))))&&&&&&&&&&&&&&&&&&&& tid=%3d, score=%g diff=%g total=%g offset=%d pos=%d\n",
    //             tidx, scores[idx_global], mutants[idx_global].diff, scores[idx_global] + mutants[idx_global].diff, mutants[idx_global].mutant.offset, mutants[idx_global].mutant.char_offset);

    if (tidx == 0)
        scores[0] = index_best_mutant(scores, mutants, is_max, blockDim.x, char_pow2);
}

__device__ float index_best_mutant(float* scores, Mutant_GPU* mutants, int is_max, int offsets_block_size, int chars_block_size)
{
    Mutant_GPU best_mutant = mutants[0];
    float best_score = scores[0];
    int idx;

    for (int i = 1; i < gridDim.x; i++)
    {
        idx = i * offsets_block_size * chars_block_size;
        if (is_swapable(&best_mutant, &mutants[idx], best_score, scores[idx], is_max))
        {
            best_mutant = mutants[idx];
            best_score = scores[idx];
        }
    }

    return best_score + best_mutant.diff;
}


__host__ __device__ int floor_highest_power_of2(int n)
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

__host__ __device__ int ceil_highest_power_of2(int n)
{
    for (int i = n; i <= INT_MAX; i++)
    {
        if ((i & (i - 1)) == 0)     //  8 = (1000), 7 = (0111)  ->  (8 & 7) = 0000 (num of 1's is 0)
        {
            return i;
        }
    }
    return 0;
}

__device__ int is_swapable(Mutant_GPU* m1, Mutant_GPU* m2, float s1, float s2, int is_max)
{
    float total1 = m1->diff + s1;
    float total2 = m2->diff + s2;

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


// __global__ void multiblock_max_reduction(float* scores, Mutant_GPU* mutants, int array_size, int is_max)
// {
//     int thIdx = threadIdx.x;
//     int gthIdx = thIdx + blockIdx.x*BLOCK_SIZE;
//     const int gridSize = BLOCK_SIZE*gridDim.x;
//     float s = is_max ? INT_MIN : INT_MAX;
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

__host__ __device__ float find_best_mutant_offset(ProgramData* data, int offset, Mutant* mt)
{
    int idx1, idx2;
    float total_score = 0;
    float pair_score, mutant_diff;
    float best_mutant_diff = data->is_max ? -INFINITY : INFINITY;

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

__host__ __device__ char find_char(char c1, char c2, float* w, int is_max)
{
    char sign = get_hash_sign(c1, c2);

    return  is_max ?
            find_max_char(c1, c2, sign, w)   :
            find_min_char(c1, c2, sign, w);
}

__host__  __device__ char find_max_char(char c1, char c2, char sign, float* w)
{
    float dot_diff, space_diff;

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

__host__ __device__ char find_min_char(char c1, char c2, char sign, float* w)
{   
    char colon_sub = get_char_by_sign_with_restrictions(c1, COLON, c2);
    char dot_sub = get_char_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_char_by_sign_with_restrictions(c1, SPACE, c2);
    char substitue;
    
    float diff1, diff2;
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

__host__ __device__ char find_optimal_char(int is_max, float diff1, char sub1, float diff2, char sub2)
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

__host__ __device__ float get_weight(char sign, float* w)
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

__device__ int my_ceil(double num)
{
    int inum = (int)num;
    if (num == (double)inum)
        return inum;
    return inum + 1;
}

__global__ void fill_hashtable_gpu()
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= NUM_CHARS || col >= NUM_CHARS)
        return;

    char c1 = FIRST_CHAR + row;
    char c2 = FIRST_CHAR + col;
    hashtable_gpu[row][col] = evaluate_chars(c1, c2);
}
