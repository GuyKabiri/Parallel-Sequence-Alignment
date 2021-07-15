#ifndef __CPU_FUNCS_H__
#define __CPU_FUNCS_H__

#include "program_data.h"

void cpu_run_program(int pid, int num_processes);

int is_greater(double a, double b);
int is_smaller(double a, double b);

char* is_contain(char* s, char c);
int is_conservative(char c1, char c2);
int is_semi_conservative(char c1, char c2);

double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, char* mutant, int is_max);
char find_char(char c1, char c2, double* weights, double* score, int (*eval_func)(double, double));
double find_max_char(char c1, char c2, double* weights, char* return_ch);
char find_char_to_space(char c);
char find_char_to_dot(char c);


ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data);
int write_results_to_file(FILE* file, char* mutant, int offset, double score);

void print_seq(char* seq1, char* seq2, double* weights, int offset);
double evaluate_chars(char a, char b, double* weights, char* s);
double compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs);

#endif //__CPU_FUNCS_H__
