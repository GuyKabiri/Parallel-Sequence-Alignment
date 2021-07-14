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

int is_greater(double a, double b);
int is_smaller(double a, double b);

char* is_contain(const char* s, const char c);
int is_conservative(const char c1, const char c2);
int is_semi_conservative(const char c1, const char c2);

double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, char* mutant, int is_max);
char find_char(char c1, char c2, double* weights, double* score, int (*eval_func)(double, double));
char find_diff_char(char c);


ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data);
int write_results_to_file(FILE* file, char* mutant, int offset, double score);

void print_seq(char* seq1, char* seq2, double* weights, int offset);
double evaluate_chars(char a, char b, double* weights, char* s);
double compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs);

#endif //__CPU_FUNCS_H__
