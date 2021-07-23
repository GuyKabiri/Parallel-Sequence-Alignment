#ifndef __DEF_H__
#define __DEF_H__

#define STAR    '*'     //  w1
#define COLON   ':'     //  w2
#define DOT     '.'     //  w3
#define SPACE   ' '     //  w4
#define DASH    '-'

#define STAR_W  0
#define COLON_W 1
#define DOT_W   2
#define SPACE_W 3

#define ROOT 0

#define INPUT_FILE "./input/input.dat"
#define OUTPUT_FILE "./output/output.dat"

#define TRUE 1
#define FALSE 0

#define NOT_FOUND_CHAR '\0'

#define NUM_CHARS 26
#define FIRST_CHAR 'A'

#define CONSERVATIVE_COUNT 9
#define SEMI_CONSERVATIVE_COUNT 11

#define CONSERVATIVE_MAX_LEN 5
#define SEMI_CONSERVATIVE_MAX_LEN 7

#define SEQ1_MAX_LEN 10000
#define SEQ2_MAX_LEN 5000

#define WEIGHTS_COUNT 4

#define FUNC_NAME_LEN 8         //  len(maximum) = len(minimum) = 7 + '\0' = 8
#define MAXIMUM_STR "maximum"
#define MINIMUM_STR "minimum"
#define MAXIMUM_FUNC 1
#define MINIMUM_FUNC 0

// enum eSign { eStar, eColon, ePoint, eSpace };

static char conservatives_arr[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
static char semi_conservatives_arr[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };

#endif // __DEF_H__
