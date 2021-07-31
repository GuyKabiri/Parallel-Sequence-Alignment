#ifndef __MUTANT_H__
#define __MUTANT_H__

#include "def.h"

typedef struct _mutant {
    int offset;
    int char_offset;
    char ch;
} Mutant;

#define NUM_OF_PARAMS_MUTANT 3

//  GPU algorithm is needed an extra double variable for each mutant in order to save and compare mutations
typedef struct _gpu_mutant {
    Mutant mutant;
    double diff;
} Mutant_GPU;

#endif //   __MUTANT_H__