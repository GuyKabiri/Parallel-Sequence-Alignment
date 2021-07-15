#ifndef __DEF_H__
#define __DEF_H__

#define STAR    '*'     //  w1
#define COLON   ':'     //  w2
#define DOT     '.'     //  w3
#define SPACE   ' '     //  w4
#define DASH    '-'

#define ROOT 0

#define INPUT_FILE "./input/input2.dat"
#define OUTPUT_FILE "./output/output.dat"


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
#define MAXIMUM_FUNC "maximum"
#define MINIMUM_FUNC "minimum"



#define NOT_EVALUATED -999.5


// enum eSign { eStar, eColon, ePoint, eSpace };


#endif // __DEF_H__
