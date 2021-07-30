#ifndef __PROGRAM_DATA_H__
#define __PROGRAM_DATA_H__

#include "def.h"

typedef struct _data {
    int is_max;
    double weights[WEIGHTS_COUNT];
    char seq1[SEQ1_MAX_LEN];
    char seq2[SEQ2_MAX_LEN];
} ProgramData;

#define NUM_OF_PARAMS_DATA 4

#endif //__PROGRAM_DATA_H__