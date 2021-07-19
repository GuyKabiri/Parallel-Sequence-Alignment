#ifndef __CPU_FUNCS_H__
#define __CPU_FUNCS_H__

#include "program_data.h"
#include "mutant.h"

void cpu_run_program(int pid, int num_processes);

int is_greater(double a, double b);
int is_smaller(double a, double b);

char* is_contain(char* s, char c);
int is_conservative(char c1, char c2);
int is_semi_conservative(char c1, char c2);

double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, int is_max, Mutant* mt);
void fill_hash(double* weights);
void print_hash();
char get_hash_sign(char c1, char c2);
double get_weight(char sign, double* weights);
char find_char(char c1, char c2, double* weights, int is_max);
char find_min_char(char c1, char c2, char sign, double* weights);
char find_max_char(char c1, char c2, char sign, double* weights);

char get_char_by_sign_with_restrictions(char by, char sign, char rest);


ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data);
int write_results_to_file(FILE* file, char* mutant, int offset, double score);

void print_seq(char* seq1, char* seq2, double* weights, int offset);
double evaluate_chars(char a, char b, double* weights, char* s);
double compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs);

#endif //__CPU_FUNCS_H__
