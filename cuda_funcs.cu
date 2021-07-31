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
    double* gpu_scores;
    double returned_score = cpu_data->is_max ? -INFINITY : INFINITY;


    int offsets = last_offset - first_offset;
    int chars = strlen_gpu(cpu_data->seq2);

    int chars_block_size = floor_highest_power_of2(chars);
    if (chars_block_size > MAX_BLOCK_SIZE)
        chars_block_size = MAX_BLOCK_SIZE;

    int array_size = chars_block_size * offsets;
    int chars_grid_size = offsets;

    int offsets_block_size = floor_highest_power_of2(offsets);
    if (offsets_block_size > MAX_BLOCK_SIZE)
        offsets_block_size = MAX_BLOCK_SIZE;
    
    int offsets_grid_size = (offsets + offsets_block_size - 1) / offsets_block_size;

    int hashtable_block_size = ceil_highest_power_of2(NUM_CHARS);
    int hashtable_grid_size = (NUM_CHARS + hashtable_block_size - 1) / hashtable_block_size;

#ifdef DEBUG_PRINT
    printf("chars threads=%d, blocks=%d, array size=%d, bytes=%lu\n", chars_block_size, chars_grid_size, array_size, (sizeof(Mutant_GPU) + sizeof(double)) * array_size);
    printf("offsets threads=%d, blocks=%d\n", offsets_block_size, offsets_grid_size);
#endif


    err = cudaMalloc(&gpu_data, sizeof(ProgramData));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (gpu program data) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpu_data, cpu_data, sizeof(ProgramData), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device (gpu program data) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&gpu_mutant, array_size * sizeof(Mutant_GPU));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (mutants array) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&gpu_scores, array_size * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (scores array) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 hashtable_block(hashtable_block_size, hashtable_block_size);
    dim3 hashtable_grid(hashtable_grid_size, hashtable_grid_size);
    fill_hashtable_gpu<<<hashtable_grid, hashtable_block>>>();
    err = cudaGetLastError();   
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (hashtable) -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    calc_mutants_scores<<<chars_grid_size, chars_block_size>>>(gpu_data, gpu_mutant, gpu_scores, offsets, chars, first_offset);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (calculations) -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    reduction<<<chars_grid_size, chars_block_size>>>(gpu_scores, gpu_mutant, cpu_data->is_max, array_size, 1, TRUE);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (chars reduction) -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    reduction<<<offsets_grid_size, offsets_block_size>>>(gpu_scores, gpu_mutant, cpu_data->is_max, array_size, chars_block_size, FALSE);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (offsets reduction) -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  the best mutant is in index 0 in mutants array
    err = cudaMemcpy(returned_mutant, &gpu_mutant[0].mutant, sizeof(Mutant), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result mutant from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  the best score is in index 0 in scores array
    err = cudaMemcpy(&returned_score, &gpu_scores[0], sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result score from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  free all GPU memory
    err = cudaFree(gpu_data);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data (gpu program data) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(gpu_mutant);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data (gpu mutants) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(gpu_scores);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data (gpu scores) - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (returned_mutant->ch == NOT_FOUND_CHAR)
        return cpu_data->is_max ? -INFINITY : INFINITY;
    return returned_score;
}

//  calculate mutations for each pair of characters in each offset
__global__ void calc_mutants_scores(ProgramData* data, Mutant_GPU* mutants, double* scores, int offsets, int chars, int start_offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //  calculate thread index in the arrays

    if (idx >= offsets * blockDim.x)
        return;

    //  initiate thread's data
    scores[idx] = 0;
    mutants[idx].mutant.ch = NOT_FOUND_CHAR;
    mutants[idx].mutant.offset = -1;
    mutants[idx].mutant.char_offset = -1;
    mutants[idx].diff = data->is_max ? -INFINITY : INFINITY;

    int thrd_offset = blockIdx.x + start_offset;        //  thread's mutant offset
    int thrd_charoffset;                                //  thread's mutant char offset
    int iterations_per_thrd = my_ceil((double)chars / (double)blockDim.x);      //  in case some threads will have more than one mutation to calculate

    double _pair_score;
    Mutant_GPU temp_mutant;
    int idx1, idx2;
    char c1, c2;
    for (int i = 0; i < iterations_per_thrd; i++)
    {
        thrd_charoffset = threadIdx.x + i * blockDim.x;

        if (thrd_charoffset >= chars)   //  this thread does not have more tasks
            break;

        idx1 = thrd_offset + thrd_charoffset;       //  index of seq1
        idx2 = thrd_charoffset;                     //  index of seq2
        c1 = data->seq1[idx1];
        c2 = data->seq2[idx2];

        _pair_score = get_weight(get_hashtable_sign(c1, c2), data->weights);    //  score the characters
        scores[idx] += _pair_score;

        temp_mutant.mutant.ch = get_substitute(c1, c2, data->weights, data->is_max);    //  find subtitute for this pair
        temp_mutant.mutant.offset = thrd_offset;
        temp_mutant.mutant.char_offset = thrd_charoffset;

        if (temp_mutant.mutant.ch == NOT_FOUND_CHAR)                //  if there is no subtitution
            temp_mutant.diff = data->is_max ? -INFINITY : INFINITY;
        else
            temp_mutant.diff = get_weight(get_hashtable_sign(c1, temp_mutant.mutant.ch), data->weights) - _pair_score;

        //  whether to save this mutation or the previous one
        if (i == 0 || is_swapable(&mutants[idx].mutant, &temp_mutant.mutant, mutants[idx].diff, temp_mutant.diff, data->is_max))
            mutants[idx] = temp_mutant;   
    }
}

//  max and sum reduction for the mutations and scores
__global__ void reduction(double* scores, Mutant_GPU* mutants, int is_max, int num_elements, int stride, int to_aggregate)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= num_elements)
        return;

    int elmnt_idx = idx * stride;
    int other_elmnt_idx;

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        other_elmnt_idx = elmnt_idx + i * stride;               //  calculate the other index to compare with
        if (threadIdx.x < i && other_elmnt_idx < num_elements)  //  if the index exceeds the array boundaries
        {
            if (to_aggregate)   //  whether to aggregate the scores or just comparing
            {
                scores[elmnt_idx] += scores[other_elmnt_idx];
                if (is_swapable(&mutants[elmnt_idx].mutant, &mutants[other_elmnt_idx].mutant, mutants[elmnt_idx].diff, mutants[other_elmnt_idx].diff, is_max))
                    mutants[elmnt_idx] = mutants[other_elmnt_idx];
            }
            else
            {
                double score1 = mutants[elmnt_idx].diff + scores[elmnt_idx];
                double score2 = mutants[other_elmnt_idx].diff + scores[other_elmnt_idx];
                if (is_swapable(&mutants[elmnt_idx].mutant, &mutants[other_elmnt_idx].mutant, score1, score2, is_max))
                {
                    mutants[elmnt_idx] = mutants[other_elmnt_idx];
                    scores[elmnt_idx] = scores[other_elmnt_idx];
                }
            }
        }
        __syncthreads();
    }

    //  last reduce between first thread in each block
    if (idx == 0 && !to_aggregate)
        scores[0] = reduce_last_results(scores, mutants, is_max, stride * blockDim.x);
}

//  reduce the first thread in each block
__device__ double reduce_last_results(double* scores, Mutant_GPU* mutants, int is_max, int stride)
{
    Mutant_GPU best_mutant = mutants[0];
    double best_score = scores[0];
    int idx;
    double score1;
    double score2;
    for (int i = 1; i < gridDim.x; i++)
    {
        idx = i * stride;
        score1 = best_mutant.diff + best_score;
        score2 = mutants[idx].diff + scores[idx];
        if (is_swapable(&best_mutant.mutant, &mutants[idx].mutant, score1, score2, is_max))
        {
            best_mutant = mutants[idx];
            best_score = scores[idx];
        }
    }
    mutants[0] = best_mutant;
    return best_score + best_mutant.diff;
}

//  filling GPU's hashtable
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

//  calculate ceil of a number
__device__ int my_ceil(double num)
{
    int inum = (int)num;
    if (num == (double)inum)
        return inum;
    return inum + 1;
}

//  whether to swap between 2 mutants
__host__ __device__ int is_swapable(Mutant* m1, Mutant* m2, double score1, double score2, int is_max)
{
    if ((is_max && score2 > score1) || (!is_max && score2 < score1))
        return TRUE;

    if (score2 == score1)
    {
        if (m2->offset < m1->offset)
            return TRUE;

        if (m2->offset == m1->offset)
        {
            if (m2->char_offset < m1->char_offset)
                return TRUE;
        }
    }
    return FALSE;
}

//  returns the optimal subtitution for a pair of characters
__host__ __device__ char get_substitute(char c1, char c2, double* w, int is_max)
{
    char sign = get_hashtable_sign(c1, c2);

    return  is_max ?
            get_max_substitute(c1, c2, sign, w)   :
            get_min_substitute(c1, c2, sign, w);
}

//  returns the max optimal substitution for a pair of characters
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

//  returns the min optimal substitution for a pair of characters
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

//  returns the optimal substitution between 2 possible substitution
__host__ __device__ char get_optimal_substitute(int is_max, double diff1, char sub1, double diff2, char sub2)
{
    //  if first different is better, and such substitue exists
    if ((is_max && diff1 >= diff2) || (!is_max && diff1 <= diff2))
        if (sub1 != NOT_FOUND_CHAR)
            return sub1;

    //  diff1 is not better than diff2, or first substitue is not possible
    //  therefore, check if diff2 is possible, if not return diff1 even if it is not possible (NOT_FOUND_CHAR)
    if (sub2 != NOT_FOUND_CHAR)
        return sub2;

    return sub1;
}

//  get a character substitution with a restriction that the substitution would not produce a COLON sign
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

//  get sign of a pair of characters
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

//  get the equivalent weight for a sign
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

//  strchr function for GPU
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

//  get the sign of a pair of characters
__host__ __device__ char get_pair_sign(char a, char b)
{
    if      (a == b)                        return ASTERISK;
    else if (is_conservative(a, b))         return COLON;
    else if (is_semi_conservative(a, b))    return DOT;

    return SPACE;
}

//  get the max power of 2 number that is smaller than n
__host__ __device__ int floor_highest_power_of2(int n)
{
    for (int i = n; i >= 1; i--)
    {
        if (is_power2(i))
            return i;
    }
    return 0;
}

//  get the min power of 2 number that is bigger than n
__host__ __device__ int ceil_highest_power_of2(int n)
{
    for (int i = n; i <= INT_MAX; i++)
    {
        if (is_power2(i))
            return i;
    }
    return 0;
}

//  return true if n is a power of 2, otherwise false
__host__ __device__ int is_power2(int n)
{
    if ((n & (n - 1)) == 0)     //  8 = (1000), 7 = (0111)  ->  (8 & 7) = 0000 (num of 1's is 0)
        return TRUE;
    return FALSE;
}

//  strln for GPU
__host__ __device__ int strlen_gpu(char* str)
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