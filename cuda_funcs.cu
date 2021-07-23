#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cuda_funcs.h"
#include "def.h"

#define BLOCK_SIZE  256

// #if (!(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0)))

// extern char conservatives_arr[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN];
// extern char semi_conservatives_arr[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN];
extern char char_hash[NUM_CHARS][NUM_CHARS];

// #endif

__global__ void sumCommMultiBlock(double* scores, Mutant* mutants, int array_size, int is_max)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*BLOCK_SIZE;
    const int gridSize = BLOCK_SIZE*gridDim.x;
    double s = is_max ? INT_MIN : INT_MAX;
    Mutant m;
    for (int i = gthIdx; i < array_size; i += gridSize)
    {   
        if ((is_max && scores[i] > s) || (!is_max && scores[i] < s))
        {
            s = scores[i];
            m = mutants[i];
        }
    }
    __shared__ double shScore[BLOCK_SIZE];
    __shared__ Mutant shMutant[BLOCK_SIZE];
    shScore[thIdx] = s;
    shMutant[thIdx] = m;
    __syncthreads();
    for (int size = BLOCK_SIZE/2; size>0; size/=2)
    { //uniform
        if (thIdx<size && thIdx + size < array_size)
        {
            if ((is_max && shScore[thIdx + size] >= shScore[thIdx]) ||
                (!is_max && shScore[thIdx + size] <= shScore[thIdx]))   //  include '==' to make sure the smaller offset is saved
            {
                if (shScore[thIdx + size] == shScore[thIdx] && !(shMutant[thIdx + size].offset < shMutant[thIdx].offset))   //  if scores equal and the smaller offset is save -> continue
                    continue;
                //  otherwise, the scores are not equal, or they are equal, but the greater one is saved -> save the new score
                shScore[thIdx] = shScore[thIdx + size];
                shMutant[thIdx] = shMutant[thIdx + size];
            }

        }
        __syncthreads();
    }
    if (thIdx == 0)
    {
        scores[blockIdx.x] = shScore[0];
        mutants[blockIdx.x] = shMutant[0];
    }
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
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (offsets + threadsPerBlock - 1) / threadsPerBlock;//offsets;
    printf("blocks=%d, threads=%d\n", blocksPerGrid, threadsPerBlock);
    get_best_mutant_gpu<<<blocksPerGrid, threadsPerBlock, 0>>>(gpu_data, gpu_mutant, scores, first_offset, last_offset);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    sumCommMultiBlock<<<blocksPerGrid, threadsPerBlock>>>(scores, gpu_mutant, last_offset - first_offset, data->is_max);
    sumCommMultiBlock<<<1, threadsPerBlock>>>(scores, gpu_mutant, blocksPerGrid, data->is_max);
    cudaDeviceSynchronize();


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // gpu_mutant[0] = gpu_mutant[10];
    // scores[0] = scores[10];

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    // printf("%2d, %2d, %f\n", returned_mutant->offset, returned_mutant->char_offset);


    return returned_score;
}


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

    // mutants[idx]
    // scores[idx]

    scores[idx] = find_best_mutant_offset_gpu(data, first_offset + idx, &mutants[idx]);
    // printf("off=%2d charoff=%2d, %f, %f\n", mutants[idx].offset, mutants[idx].char_offset, scores[idx]);

    __syncthreads();
}

__device__ double find_best_mutant_offset_gpu(ProgramData* data, int offset, Mutant* mt)
{
    int idx1, idx2;
    double total_score = 0;
    double pair_score, mutant_diff, best_mutant_diff;
    int iterations = strlen_gpu(data->seq2);
    char c1, c2, subtitue;

    for (int i = 0; i < iterations; i++)            //  iterate over all the characters
    {
        idx1 = offset + i;                      //  index of seq1
        idx2 = i;                               //  index of seq2
        c1 = data->seq1[idx1];                  //  current char in seq1
        c2 = data->seq2[idx2];                  //  current char in seq2
        pair_score = get_weight(get_hash_sign(c1, c2), data->weights);    //  get weight before substitution
        total_score += pair_score;

        subtitue = find_char(c1, c2, data->weights, data->is_max);
        mutant_diff = get_weight(get_hash_sign(c1, subtitue), data->weights) - pair_score;    //  difference between original and mutation weights
        mutant_diff = abs(mutant_diff);


        if (mutant_diff > best_mutant_diff || i == 0)
        {
            best_mutant_diff = mutant_diff;
            mt->ch = subtitue;
            mt->char_offset = i;        //  offset of char inside seq2
            mt->offset = offset;
        }
    }
    if (data->is_max)
        return total_score + best_mutant_diff;
    return total_score - best_mutant_diff;     //  best mutant is returned in struct mt
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
    char ch;
    switch (sign)
    {
    case STAR:
        return c2;

    case DOT:                   //  if there is DOT between two characters, a START subtitution is possible
    case SPACE:                 //  if there is SPACE between two characters, a START subtitution is possible
        return c1;

    case COLON:
        double dot_diff = w[COLON_W] - w[DOT_W];
        double space_diff = w[COLON_W] - w[SPACE_W];

        if (!(dot_diff > 0 || space_diff > 0))      //  if both not greater than 0 (negative change or no change at all)
        {                                           //  then, no score change and return the same character
            return c2;
        }

        if (space_diff > dot_diff)                 //  if SPACE subtitution is better than DOT
        {
            ch = get_char_by_sign_with_restrictions(c1, SPACE, c2);
            if (ch != NOT_FOUND_CHAR)       //  if found SPACE subtitution
                return ch;
            
            //  if could not find SPACE subtitution, and DOT is better than no subtitution
            if (dot_diff > 0)
            {
                ch = get_char_by_sign_with_restrictions(c1, DOT, c2);
                if (ch != NOT_FOUND_CHAR)       //  if found DOT subtitution
                    return ch;
            }

            //  otherwise, no subtitution found
            return c2;
        }

        //  otherwise, it will try to find DOT subtitution
        ch = get_char_by_sign_with_restrictions(c1, DOT, c2);
        if (ch != NOT_FOUND_CHAR)       //  if found DOT subtitution
            return ch;

        //  if could not find DOT subtitution, and SPACE is better than no subtitution
        if (space_diff > 0)
        {
            ch = get_char_by_sign_with_restrictions(c1, SPACE, c2);
            if (ch != NOT_FOUND_CHAR)       //  if found SPACE subtitution
                return ch;
        }

        //  otherwise, no subtitution found
        return c2;
    }
    return c2;
}

__host__ __device__ char find_min_char(char c1, char c2, char sign, double* w)
{   
    char colon_sub = get_char_by_sign_with_restrictions(c1, COLON, c2);
    char dot_sub = get_char_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_char_by_sign_with_restrictions(c1, SPACE, c2);

    double colon_diff, dot_diff, space_diff;

    switch (sign)
    {
    case STAR:
        dot_diff = - w[STAR_W] - w[DOT_W];
        space_diff = - w[STAR_W] - w[SPACE_W];

        if (!(dot_diff < 0 || space_diff < 0))    //  if any subtitution will not decrease the score
            return c2;                                              //  than return the same letter and score

        if (dot_diff < space_diff)
        {
            if (dot_sub != NOT_FOUND_CHAR)
                return dot_sub;
        }

        //  could not find DOT subtitution
        if (space_diff < 0)
        {
            if (space_sub != NOT_FOUND_CHAR)
                return space_sub;

            //  could not find SPACE subtitution, but DOT might be better than nothing
            if (dot_diff < 0 && dot_sub != NOT_FOUND_CHAR)
                return dot_sub;
        }

        return c2;  //  could not find any subtitution
    
    case COLON:
        dot_diff = w[COLON_W] - w[DOT_W];
        space_diff = w[COLON_W] - w[SPACE_W];

        if (!(dot_diff < 0 || space_diff < 0))      //  if any subtitution will not decrease the score
            return c2;                              //  than return the same letter and score

        if (dot_diff < space_diff)                  //  if DOT subtitution is better than SPACE
        {
            if (dot_sub != NOT_FOUND_CHAR)          //  if found DOT subtitution
                return dot_sub;
        }

        if (space_diff < 0)
        {
            if (space_sub != NOT_FOUND_CHAR)
                return space_sub;

            //  could not find SPACE subtitution, but DOT might be better than nothing
            if (dot_diff < 0 && dot_sub != NOT_FOUND_CHAR)
                return dot_sub;
        }
        
        return c2;  // could not find any subtitution

    case DOT:
        colon_diff = w[DOT_W] - w[COLON_W];
        space_diff = w[DOT_W] - w[SPACE_W];

        if (!(colon_diff < 0 && space_diff < 0))    //  if any subtitution will not decrease the score
            return c2;                              //  than return the same letter and score

        if (colon_diff < space_diff)                //  if COLON subtitution is better than SPACE   
        {
            if (colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        if (space_diff < 0)
        {
            if (space_sub != NOT_FOUND_CHAR)
                return space_sub;
            
            //  could not find SPACE subtitution, but COLON might still be better than nothing
            if (colon_diff < 0 && colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        return c2;  // could not find any subtitution

    case SPACE:
        colon_diff = w[SPACE_W] - w[COLON_W];
        dot_diff = w[SPACE_W] - w[DOT_W];

        if (!(colon_diff < 0 && dot_diff < 0))      //  if any subtitution will not decrease the score
            return c2;                              //  than return the same letter and score

        if (colon_diff < dot_diff)                  //  if COLON subtitution is better than DOT
        {
            if (colon_sub != NOT_FOUND_CHAR)        //  if found COLON subtitution
                return colon_sub;
        }

        if (dot_diff < 0)
        {
            if (dot_sub != NOT_FOUND_CHAR)          //  if found DOT subtitution
                return dot_sub;

            //  could not find DOT subtitution, but COLON might still be better than nothing
            if (colon_diff < 0 && colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        return c2;  // could not find any subtitution
    }
    return c2;      //  sign was not any of the legal signs
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
    if (c1 > FIRST_CHAR + NUM_CHARS || c2 > FIRST_CHAR + NUM_CHARS)   return DASH;
    if (c1 == DASH && c2 == DASH)   return STAR;
    if (c1 == DASH || c2 == DASH)   return SPACE;

    if (c1 >= c2)       //  only the bottom triangle of the hash table is full -> (hash[x][y] = hash[y][x])
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        return char_hash_cuda[c1 - FIRST_CHAR][c2 - FIRST_CHAR];
    return char_hash_cuda[c2 - FIRST_CHAR][c1 - FIRST_CHAR];
#else
        return char_hash[c1 - FIRST_CHAR][c2 - FIRST_CHAR];
    return char_hash[c2 - FIRST_CHAR][c1 - FIRST_CHAR];
#endif
}

__host__ __device__ double get_weight(char sign, double* w)
{
    switch (sign)
    {
    case STAR:  return w[STAR_W];
    case COLON: return -w[COLON_W];
    case DOT:   return -w[DOT_W];
    case SPACE: return -w[SPACE_W];
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
        if (is_contain(conservatives_arr_cuda[i], c1) && is_contain(conservatives_arr_cuda[i], c2))   //  if both characters present
#else
        if (is_contain(conservatives_arr[i], c1) && is_contain(conservatives_arr[i], c2))   //  if both characters present
#endif
            return 1;
    return 0;
}

//  check if both characters present in the same semi-conservative group
__host__ __device__ int is_semi_conservative(char c1, char c2)
{
    for (int i = 0; i < SEMI_CONSERVATIVE_COUNT; i++)   //  iterate over the semi-conservative groups
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            if (is_contain(semi_conservatives_arr_cuda[i], c1) && is_contain(semi_conservatives_arr_cuda[i], c2))   //  if both characters present
#else
            if (is_contain(semi_conservatives_arr[i], c1) && is_contain(semi_conservatives_arr[i], c2))   //  if both characters present
#endif
                return 1;
    return 0;
}

__host__ __device__ char evaluate_chars(char a, char b)
{
    if      (a == b)                        return STAR;
    else if (is_conservative(a, b))         return COLON;
    else if (is_semi_conservative(a, b))    return DOT;

    return SPACE;
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

__global__ void fill_hashtable_gpu()
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    char c1 = FIRST_CHAR + row;
    char c2 = FIRST_CHAR + col;
    char_hash_cuda[row][col] = evaluate_chars(c1, c2);
}
