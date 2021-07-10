#ifndef __MUTANT_H__
#define __MUTANT_H__

#include "def.h"

typedef struct _mutant {
    char mut[SEQ2_MAX_LEN];
    int offset;
    double score;
} Mutant;

#endif //   __MUTANT_H__