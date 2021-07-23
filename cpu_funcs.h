#ifndef __CPU_FUNCS_H__
#define __CPU_FUNCS_H__

#include "program_data.h"
#include "mutant.h"

void cpu_run_program(int pid, int num_processes);

void fill_hash(double* weights, int pid);
void print_hash();

double find_best_mutant_cpu(int pid, ProgramData* data, Mutant* return_mutant, int first_offset, int last_offset);
double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, int is_max, Mutant* mt);

ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data);
int write_results_to_file(FILE* file, char* mutant, int offset, double score);

void pretty_print(ProgramData* data, char* mut, int offset, int char_offset);
double get_score_and_signs(char* seq1, char* seq2, double* weights, int offset, char* signs);
void print_with_offset(char* chrs, int offset, int char_offset);

#endif //__CPU_FUNCS_H__
