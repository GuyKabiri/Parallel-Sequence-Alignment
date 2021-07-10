#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include "cpu_funcs.h"

int main(int argc, char* argv[]);
int find_mutant(char* seq1, char* seq2, double* weights, int offset, char* mutant, int is_max);
char maximize(char c1, char c2, double* weights);
char minimize(char c1, char c2, double* weights);

#endif //__MAIN_H__