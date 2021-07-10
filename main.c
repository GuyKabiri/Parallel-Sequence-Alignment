#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "main.h"
#include "cpu_funcs.h"
#include "def.h"
#include "mutant.h"
#include "program_data.h"

void clear()
{
    printf("\e[2J\e[H"); // magic!
}

int main(int argc, char* argv[])
{
    FILE* input_file;

    ProgramData data;

    // double weights[WEIGHTS_COUNT] = { 1 };
    // char seq1[SEQ1_MAX_LEN] = { '\0' };
    // char seq2[SEQ2_MAX_LEN] = { '\0' };
    // int is_max;

    input_file = fopen(INPUT_FILE, "r");
    if (!input_file)
    {
        printf("Error open input file `%s`\n", INPUT_FILE);
        exit(1);
    }
    
    // if (!read_seq_and_weights_from_file(input_file, seq1, seq2, weights, &is_max))
    if (!read_seq_and_weights_from_file(input_file, &data))
    {
        printf("Error reading input file `%s`\n", INPUT_FILE);
        exit(1);
    }
    fclose(input_file);
    printf("%s problem\n\n", data.is_max ? MAXIMUM_FUNC : MINIMUM_FUNC);

    int offset = 0;
    // compare_evaluate_seq(seq1, seq2, weights, offset, NULL);

    // print_seq(seq1, seq2, weights, offset);

    Mutant best_mutant;
    best_mutant.score = data.is_max ? __DBL_MIN__ : __DBL_MAX__;
    best_mutant.offset = 0;

    char mutant[SEQ2_MAX_LEN];

    // data.start_offset = 0;
    int iterations = strlen(data.seq1) - strlen(data.seq2) + 1 - data.start_offset;
    // int best_offset = 0;
    // double best_score = 0;

    struct timespec t;
    t.tv_sec = 0;
    t.tv_nsec = 1000000000 * 0.2;

    for (int i = 0; i < iterations; i++)
    {
        find_mutant(data.seq1, data.seq2, data.weights, i, mutant, data.is_max);
        double score = compare_evaluate_seq(data.seq1, data.seq2, data.weights, i, NULL);
        // if (score > best_score)
        // {
        //     best_score = score;
        //     best_offset = i;
        // }
        clear();
        print_seq(data.seq1, data.seq2, data.weights, i);
        nanosleep(&t, &t);
    }
    printf("iterations: %d, process: %d\n", iterations, data.start_offset);


    // print_seq(seq1, seq2, weights, best_offset);

}

int find_mutant(char* seq1, char* seq2, double* weights, int offset, char* mutant, int is_max)
{
    int seq1_idx = offset;
    int seq2_idx = 0;
    double total_score = 0;
    int iterations = fmin(strlen(seq2), strlen(seq1) - offset);        // TODO: change fmin with min
    char ch;

    for ( ; seq2_idx < iterations; seq1_idx++, seq2_idx++)
    {
        ch = is_max ?
            maximize(seq1[seq1_idx], seq2[seq2_idx], weights) :
            minimize(seq1[seq1_idx], seq2[seq2_idx], weights);

        mutant[seq2_idx] = ch;
    }

    return 0;
}

/*  
    s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
    therefore, if c1 and c2 are NOT conservative, miximize the equation should be by max(w1, -w3, -w4)
*/
//  TODO:   finding a char that will evaluated as the sign SPACE and do not return SPACE
char maximize(char c1, char c2, double* weights)
{
    char max_letter;                        
    if (is_conservative(c1, c2))    return c2;      //  if the characters are conservative, substitute is not allowed   ->  return the same letter

    if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
    {
        if      (weights[0] > -weights[2] && weights[0] > -weights[3])   return c1;     //  if w1 > w3, w4 then return START
        else if (weights[3] > -weights[2] && weights[3] > -weights[0])   return DASH;  //  if w4 > w1, w3 then return SPACE
        return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
    }

    //  otherwise, the characters are neither conservative, not semi conservative
    //  then, maximize by max(w1, w4) which means to return c1 or SPACE

    return weights[0] > -weights[3] ? c1 : DASH;
}

/*  
    s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
    therefore, if c1 and c2 are NOT conservative, minimize the equation should be by min(w1, -w3, -w4)
*/
char minimize(char c1, char c2, double* weights)
{
    char max_letter;                        
    if (is_conservative(c1, c2))    return c2;      //  if the characters are conservative, substitute is not allowed   ->  return the same letter

    if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
    {
        if      (weights[0] < -weights[2] && weights[0] < -weights[3])   return c1;     //  if w1 > w3, w4 then return START
        else if (weights[3] < -weights[2] && weights[3] < -weights[0])   return DASH;  //  if w4 > w1, w3 then return SPACE
        return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
    }

    //  otherwise, the characters are neither conservative, not semi conservative
    //  then, minimize by min(w1, w4) which means to return c1 or SPACE

    return weights[0] < -weights[3] ? c1 : DASH;
}





// /*  
//     s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
//     therefore, if c1 and c2 are NOT conservative, miximize the equation should be by max(w1, -w3, -w4)
// */
// char find_letter(char c1, char c2, double* weights, void (*eval_func)(void*, void*))
// {
//     char max_letter;                        
//     if (is_conservative(c1, c2))    return c2;      //  if the characters are conservative, substitute is not allowed   ->  return the same letter

//     if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
//     {
//         if      (eval_func(weights[0], -weights[2]) && eval_func(weights[0], -weights[3]))   return c1;     //  if w1 > w3, w4 then return START
//         else if (eval_func(weights[3], -weights[2]) && eval_func(weights[3], -weights[0]))   return SPACE;  //  if w4 > w1, w3 then return SPACE
//         return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
//     }

//     //  otherwise, the characters are neither conservative, not semi conservative
//     //  then, maximize by max(w1, w4), or minimize by min(w1, w4) which means to return c1 or SPACE

//     return eval_func(weights[0], weights[3]) ? c1 : SPACE;
// }