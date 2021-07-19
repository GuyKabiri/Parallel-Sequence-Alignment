#ifndef __CPU_FUNCS_H__
#define __CPU_FUNCS_H__

#include "program_data.h"
#include "mutant.h"

void cpu_run_program(int pid, int num_processes);

char* is_contain(char* s, char c);
int is_conservative(char c1, char c2);
int is_semi_conservative(char c1, char c2);
char evaluate_chars(char a, char b, double* weights);
void fill_hash(double* weights);
void print_hash();
char get_hash_sign(char c1, char c2);
double get_weight(char sign, double* weights);

double find_best_mutant(int pid, ProgramData* data, Mutant* return_mutant);
double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, int is_max, Mutant* mt);
char find_char(char c1, char c2, double* weights, int is_max);
char find_min_char(char c1, char c2, char sign, double* weights);
char find_max_char(char c1, char c2, char sign, double* weights);
char get_char_by_sign_with_restrictions(char by, char sign, char rest);

ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data);
int write_results_to_file(FILE* file, char* mutant, int offset, double score);

void pretty_print_seq_mut(char* seq1, char* seq2, char* mut, double* weights, int offset, int char_offset);
double get_score_and_signs(char* seq1, char* seq2, double* weights, int offset, char* signs);
void print_with_offset(char* chrs, int offset, int char_offset);

#endif //__CPU_FUNCS_H__
