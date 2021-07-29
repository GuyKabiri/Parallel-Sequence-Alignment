#ifndef __MUTANT_H__
#define __MUTANT_H__

#include "def.h"

typedef struct _mutant {
    int offset;
    int char_offset;
    char ch;
} Mutant;

#define NUM_OF_PARAMS_MUTANT 3

typedef struct _gpu_mutant {
    Mutant mutant;
    float diff;
} Mutant_GPU;

#endif //   __MUTANT_H__