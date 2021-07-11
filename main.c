// TODO: re-structure the directories, delete input, output dirs


#define _CRT_SECURE_NO_WARNINGS             //   TODO: remove CTR_NO_WARNINGS
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

    // int offset = 0;
    // compare_evaluate_seq(seq1, seq2, weights, offset, NULL);

    // print_seq(seq1, seq2, weights, offset);

    // Mutant best_mutant;
    // best_mutant.score = data.is_max ? __DBL_MIN__ : __DBL_MAX__;
    // best_mutant.offset = 0;

    char mutant[SEQ2_MAX_LEN] = { '\0' };

    // data.start_offset = 0;
    int iterations = strlen(data.seq1) - strlen(data.seq2) + 1 - data.start_offset;
    // int best_offset = 0;
    // double best_score = 0;

    // struct timespec t;
    // t.tv_sec = 0;
    // t.tv_nsec = 1000000000 * 0.5;

    for (int i = 0; i < iterations; i++)
    {
        double score = find_mutant(data.seq1, data.seq2, data.weights, i, mutant, data.is_max);
        // double score = compare_evaluate_seq(data.seq1, data.seq2, data.weights, i, NULL);
        // clear();
        // print_seq(data.seq1, data.seq2, data.weights, i);
        // nanosleep(&t, &t);

        printf("offset: %4d, score: %g\n", i, score);
        print_seq(data.seq1, mutant, data.weights, i);
        // print_seq(data.seq1, data.seq2, data.weights, i);

        break;
    }
    printf("iterations: %d, process: %d\n", iterations, data.start_offset);


    // print_seq(seq1, seq2, weights, best_offset);

}