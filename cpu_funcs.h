#ifndef __CPU_FUNCS_H__
#define __CPU_FUNCS_H__

#include "def.h"
#include "program_data.h"

// typedef struct _pair{
//     char seq1[SEQ1_MAX_LEN];
//     char seq2[SEQ2_MAX_LEN];
//     char signs[SEQ2_MAX_LEN];
//     int offset;
//     double score; 
// } Pair;

#define STAR    '*'     //  w1
#define COLON   ':'     //  w2
#define POINT   '.'     //  w3
#define SPACE   ' '     //  w4
#define DASH    '-'

int compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs);      //  signs array is for debugging and pretty printing
char* is_contain(const char* s, const char c);
int is_conservative(const char c1, const char c2);
int is_semi_conservative(const char c1, const char c2);
// int read_seq_and_weights_from_file(FILE* file, char* seq1, char* seq2, double* weights, int* type);
ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data);
void print_seq(char* seq1, char* seq2, double* weights, int offset);

#endif //__CPU_FUNCS_H__