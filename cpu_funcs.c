#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include "cpu_funcs.h"
#include "def.h"
#include "program_data.h"


char conservatives_arr[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
char semi_conservatives_arr[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };


int compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs)      //  signs array is for debugging and pretty printing
{
    if (!seq1 || !seq2 || !weights)  return 0;

    int seq1_idx = offset;
    int seq2_idx = 0;
    double total_score = 0;
    int iterations = fmin(strlen(seq2), strlen(seq1) - offset);        // TODO: change fmin with min
    char _sign;
    // double t = MPI_Wtime();
#pragma omp parallel for
    for ( ; seq2_idx < iterations; seq1_idx++, seq2_idx++)
    {
        if      (seq1[seq1_idx] == seq2[seq2_idx])                      { _sign = STAR;  total_score += weights[0]; }
        else if (is_conservative(seq1[seq1_idx], seq2[seq2_idx]))       { _sign = COLON; total_score -= weights[1]; }
        else if (is_semi_conservative(seq1[seq1_idx], seq2[seq2_idx]))  { _sign = POINT; total_score -= weights[2]; }
        else                                                            { _sign = SPACE; total_score -= weights[3]; }

        if (signs)  signs[seq2_idx] = _sign;
    }
    // printf("%g\n", MPI_Wtime() - t);
    return total_score;
}

char* is_contain(const char* s, const char c)
{
    return strchr(s, c);
}

int is_conservative(const char c1, const char c2)
{
    for (int i = 0; i < CONSERVATIVE_COUNT; i++)
        if (is_contain(conservatives_arr[i], c1) && is_contain(conservatives_arr[i], c2))
            return 1;
    return 0;
}

int is_semi_conservative(const char c1, const char c2)
{
    for (int i = 0; i < SEMI_CONSERVATIVE_COUNT; i++)
            if (is_contain(semi_conservatives_arr[i], c1) && is_contain(semi_conservatives_arr[i], c2))
                return 1;
        return 0;
}

// int read_seq_and_weights_from_file(FILE* file, char* seq1, char* seq2, double* weights, int* type)
// {
//     if (!file || !seq1 || !seq2 || !weights)  return 0;

//     if (fscanf(file, "%lf %lf %lf %lf", &weights[0], &weights[1], &weights[2], &weights[3]) != 4)   return 0;
//     if (fscanf(file, "%s", seq1) != 1)   return 0;
//     if (fscanf(file, "%s", seq2) != 1)   return 0;

//     char func_type[FUNC_NAME_LEN];
//     if (fscanf(file, "%s", func_type) != 1)   return 0;
//     *type = strcmp(func_type, MAXIMUM_FUNC) == 0 ? 1 : 0;

//     return 1;
// }

ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data)
{
    if (!file || !data)  return NULL;

    if (fscanf(file, "%lf %lf %lf %lf", &data->weights[0], &data->weights[1], &data->weights[2], &data->weights[3]) != 4)   return NULL;
    if (fscanf(file, "%s", data->seq1) != 1)   return NULL;
    if (fscanf(file, "%s", data->seq2) != 1)   return NULL;

    char func_type[FUNC_NAME_LEN];
    if (fscanf(file, "%s", func_type) != 1)   return NULL;
    data->is_max = strcmp(func_type, MAXIMUM_FUNC) == 0 ? 1 : 0;
    data->start_offset = (strlen(data->seq1) - strlen(data->seq2) + 1) / 2;

    return data;
}

void print_seq(char* seq1, char* seq2, double* weights, int offset)
{
        if (!seq1 || !seq2 || !weights)  return;

    printf("%s\n", seq1);
    for (int i = 0; i < offset; i++)    printf(" ");
    printf("%s\n", seq2);

    char signs[SEQ2_MAX_LEN] = { '\0' };
    double score = compare_evaluate_seq(seq1, seq2, weights, offset, signs);
    for (int i = 0; i < offset; i++)    printf(" ");
    printf("%s\n", signs);

    printf("Score: %g\n", score);
}