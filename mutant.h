#ifndef __MUTANT_H__
#define __MUTANT_H__

#include "def.h"

typedef struct _mutant {
    int offset;
    int char_offset;
    char ch;
    double score;
} Mutant;

#endif //   __MUTANT_H__