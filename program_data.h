#ifndef __PROGRAM_DATA_H__
#define __PROGRAM_DATA_H__

#include "def.h"

typedef struct _data {
    int proc_count;
    int is_max;
    char seq1[SEQ1_MAX_LEN];
    char seq2[SEQ2_MAX_LEN];
    double weights[WEIGHTS_COUNT];
} ProgramData;

#define NUM_OF_PARAMS 5

#endif //__PROGRAM_DATA_H__