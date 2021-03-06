#ifndef __DEF_H__
#define __DEF_H__

#define ASTERISK    '*'     //  w1
#define COLON       ':'     //  w2
#define DOT         '.'     //  w3
#define SPACE       '_'     //  w4
#define HYPHEN      '-'

#define NOT_FOUND_CHAR '\0'

#define ASTERISK_W  0
#define COLON_W     1
#define DOT_W       2
#define SPACE_W     3

#define ROOT 0
#define THREADS_COUNT 4

#define INPUT_FILE "./input.txt"
#define OUTPUT_FILE "./output.txt"

#define TRUE 1
#define FALSE 0

#define FIRST_CHAR 'A'
#define NUM_CHARS 26

#define CONSERVATIVE_COUNT 9
#define SEMI_CONSERVATIVE_COUNT 11

#define CONSERVATIVE_MAX_LEN 5
#define SEMI_CONSERVATIVE_MAX_LEN 7

#define SEQ1_MAX_LEN 10000 + 1
#define SEQ2_MAX_LEN 5000 + 1

#define MAX_OFFSETS SEQ1_MAX_LEN - SEQ2_MAX_LEN + 1

#define WEIGHTS_COUNT 4

#define FUNC_NAME_LEN 8         //  len(maximum) = len(minimum) = 7 + '\0' = 8
#define MAXIMUM_STR "maximum"
#define MINIMUM_STR "minimum"
#define MAXIMUM_FUNC 1
#define MINIMUM_FUNC 0

typedef unsigned int uint;

#endif // __DEF_H__
