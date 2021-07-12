#ifndef __PROGRAM_DATA_H__
#define __PROGRAM_DATA_H__

#include "def.h"

typedef struct _data {
    char seq1[SEQ1_MAX_LEN];
    char seq2[SEQ2_MAX_LEN];
    double weights[WEIGHTS_COUNT];
    int is_max;
    int num_tasks;
    int offset_add;
} ProgramData;

#define NUM_OF_PARAMS 6

#endif //__PROGRAM_DATA_H__