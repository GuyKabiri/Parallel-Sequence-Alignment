#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include "cpu_funcs.h"
#include "def.h"
#include "program_data.h"


char conservatives_arr[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
char semi_conservatives_arr[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };


double compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs)      //  signs array is for debugging and pretty printing
{
    if (!seq1 || !seq2 || !weights)  return 0;

    int seq1_idx = offset;
    int seq2_idx = 0;
    double total_score = 0;
    int iterations = fmin(strlen(seq2), strlen(seq1) - offset);        // TODO: change fmin with min
    char _sign;
    // double t = MPI_Wtime();
    for ( ; seq2_idx < iterations; seq2_idx++, seq1_idx++)
    {
        // if      (seq1[seq1_idx] == seq2[seq2_idx])                      { _sign = STAR;  total_score += weights[0]; }
        // else if (is_conservative(seq1[seq1_idx], seq2[seq2_idx]))       { _sign = COLON; total_score -= weights[1]; }
        // else if (is_semi_conservative(seq1[seq1_idx], seq2[seq2_idx]))  { _sign = POINT; total_score -= weights[2]; }
        // else                                                            { _sign = SPACE; total_score -= weights[3]; }
        total_score += evaluate_chars(seq1[seq1_idx], seq2[seq2_idx], weights, &_sign);

        if (signs)  signs[seq2_idx] = _sign;
        
    }
    // printf("%g\n", MPI_Wtime() - t);
    return total_score;
}

//  evaluate pair of characters, return their score, and suitable sign
double evaluate_chars(char a, char b, double* weights, char* s)
{
    char temp;
    if (s == NULL)
        s = &temp;  //  in case the returned char is not required
    if      (a == b)                        { *s = STAR;  return weights[0]; }
    else if (is_conservative(a, b))         { *s = COLON; return -weights[1]; }
    else if (is_semi_conservative(a, b))    { *s = POINT; return -weights[2]; }
    
    *s = SPACE;
    return -weights[3];
}

//  check for the character c in the string s
//  returns NULL if c is not presented in s, otherwise returns the address of the occurrence of c
char* is_contain(const char* s, const char c)
{
    return strchr(s, c);
}

//  cheack if both characters present in the same conservative group
int is_conservative(const char c1, const char c2)
{
    for (int i = 0; i < CONSERVATIVE_COUNT; i++)    //  iterate over the conservative groups
        if (is_contain(conservatives_arr[i], c1) && is_contain(conservatives_arr[i], c2))   //  if both characters present
            return 1;
    return 0;
}

//  cheack if both characters present in the same semi-conservative group
int is_semi_conservative(const char c1, const char c2)
{
    for (int i = 0; i < SEMI_CONSERVATIVE_COUNT; i++)   //  iterate over the semi-conservative groups
            if (is_contain(semi_conservatives_arr[i], c1) && is_contain(semi_conservatives_arr[i], c2))   //  if both characters present
                return 1;
    return 0;
}

// int read_seq_and_weights_from_file(FILE* file, char* seq1, char* seq2, double* weights, int* type)
// {
//     if (!file || !seq1 || !seq2 || !weights)  return 0;

//     if (fscanf(file, "%lf %lf %lf %lf", &weights[0], &weights[1], &weights[2], &weights[3]) != 4)   return 0;
//     if (fscanf(file, "%s", seq1) != 1)   return 0;
//     if (fscanf(file, "%s", seq2) != 1)   return 0;

//     char func_type[FUNC_NAME_LEN];
//     if (fscanf(file, "%s", func_type) != 1)   return 0;
//     *type = strcmp(func_type, MAXIMUM_FUNC) == 0 ? 1 : 0;

//     return 1;
// }

//  reads two sequences, weights, and the assignment type (maximum / minimum) from a input file
ProgramData* read_seq_and_weights_from_file(FILE* file, ProgramData* data)
{
    if (!file || !data)  return NULL;   //  if file or structure did not allocated

    //  first, read the 4 weights, after read 2 sequences of characters
    if (fscanf(file, "%lf %lf %lf %lf", &data->weights[0], &data->weights[1], &data->weights[2], &data->weights[3]) != 4)   return NULL;
    if (fscanf(file, "%s", data->seq1) != 1)   return NULL;
    if (fscanf(file, "%s", data->seq2) != 1)   return NULL;

    //  allocated space to read the assignment type
    char func_type[FUNC_NAME_LEN];
    if (fscanf(file, "%s", func_type) != 1)   return NULL;
    data->is_max = strcmp(func_type, MAXIMUM_FUNC) == 0 ? 1 : 0;    //  saves '1' if it is a maximum, otherwise, saves '0'

    //  divide the amount of offsets by 2 (each computers' tasks)
    // data->start_offset = (strlen(data->seq1) - strlen(data->seq2) + 1) / 2;
    data->num_tasks = 0;
    data->offset_add = 0;

    return data;
}

//  pretty printing the sequences and the character-wise comparation between them
void print_seq(char* seq1, char* seq2, double* weights, int offset)
{
    if (!seq1 || !seq2 || !weights)  return;

    printf("%s\n", seq1);       //  print 1st sequence
    for (int i = 0; i < offset; i++)    printf(" ");    //  print spaces to align the offset of sequences
    printf("%s\n", seq2);       //  print 2nd sequence

    char signs[SEQ2_MAX_LEN] = { '\0' };
    double score = compare_evaluate_seq(seq1, seq2, weights, offset, signs);    //  evaluate the score of the sequences by the wanted offset, and create the signs sequence
    for (int i = 0; i < offset; i++)    printf(" ");    //  print spaces to align the offset of sequences
    printf("%s\n", signs);       //  print signs sequence

    printf("Offset: %4d, Score: %g\n", offset, score);
}

//  find the best mutant for a given offset
int find_mutant(char* seq1, char* seq2, double* weights, int offset, char* mutant, int is_max)
{
    int seq1_idx, seq2_idx;
    double total_score = 0;
    int iterations = fmin(strlen(seq2), strlen(seq1) - offset);        // TODO: change fmin with min
    char ch;

#pragma omp parallel for
    for (int i = 0; i < iterations; i++)          //  iterate over all the characters
    {
        seq1_idx = offset + i;
        seq2_idx = i;
        ch = is_max ?                                               //  if it is a maximus assignment
            maximize(seq1[seq1_idx], seq2[seq2_idx], weights, &total_score) :     //  find the maximum suitable character
            minimize(seq1[seq1_idx], seq2[seq2_idx], weights, &total_score);      //  find the minimum suitable character

        mutant[seq2_idx] = ch;
    }

    return total_score;           //  TODO: return new score
}

/*  
    s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
    therefore, if c1 and c2 are NOT conservative, miximize the equation should be by max(w1, -w3, -w4)
*/
//  TODO:   finding a char that will evaluated as the sign SPACE and do not return SPACE
char maximize(char c1, char c2, double* weights, double* score)
{
    if (is_conservative(c1, c2))    //  if the characters are conservative, substitute is not allowed   ->  return the same letter
    {    
        if (c1 == c2)               //  if same letters -> add the suitable weight
            *score += weights[0];
        else                        //  not same letters, but conservative ones -> substruct the suitable weight
            *score -= weights[1];
        return c2;      
    }

    if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
    {
        if      (weights[0] > -weights[2] && weights[0] > -weights[3])  { *score += weights[0]; return c1; }    //  if w1 > w3, w4 then return STAR
        else if (-weights[3] > -weights[2] && -weights[3] > weights[0]) { *score -= weights[3]; return find_different_char(c1); }  //  if w4 > w1, w3 then return SPACE
        return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
    }

    //  otherwise, the characters are neither conservative, nor semi conservative
    //  then, maximize by max(w1, w4) which means to return c1 or SPACE
    if (weights[0] > -weights[3])
    {
        *score += weights[0];
        return c1;
    }
    
    *score -= weights[3];
    return find_different_char(c1);
}

char find_different_char(char c)
{
	char other = c;
	do
	{
		other = (other + 1) % NUM_CHARS + FIRST_CHAR;	//	get next letter cyclically
	} while(is_conservative(c, other) || is_semi_conservative(c, other));
	return other;
}

/*  
    s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
    therefore, if c1 and c2 are NOT conservative, minimize the equation should be by min(w1, -w3, -w4)
*/
char minimize(char c1, char c2, double* weights, double* score)
{                       
    if (is_conservative(c1, c2))    return c2;      //  if the characters are conservative, substitute is not allowed   ->  return the same letter

    if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
    {
        if      (weights[0] < -weights[2] && weights[0] < -weights[3])      return c1;     //  if w1 > w3, w4 then return STAR
        else if (-weights[3] < -weights[2] && -weights[3] < weights[0])   return DASH;   //  if w4 > w1, w3 then return SPACE
        return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
    }

    //  otherwise, the characters are neither conservative, nor semi conservative
    //  then, minimize by min(w1, w4) which means to return c1 or SPACE

    return weights[0] < -weights[3] ? c1 : DASH;
}





// /*  
//     s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
//     therefore, if c1 and c2 are NOT conservative, miximize the equation should be by max(w1, -w3, -w4)
// */
// char find_letter(char c1, char c2, double* weights, void (*eval_func)(void*, void*))
// {
//     char max_letter;                        
//     if (is_conservative(c1, c2))    return c2;      //  if the characters are conservative, substitute is not allowed   ->  return the same letter

//     if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
//     {
//         if      (eval_func(weights[0], -weights[2]) && eval_func(weights[0], -weights[3]))   return c1;     //  if w1 > w3, w4 then return START
//         else if (eval_func(weights[3], -weights[2]) && eval_func(weights[3], -weights[0]))   return SPACE;  //  if w4 > w1, w3 then return SPACE
//         return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
//     }

//     //  otherwise, the characters are neither conservative, not semi conservative
//     //  then, maximize by max(w1, w4), or minimize by min(w1, w4) which means to return c1 or SPACE

//     return eval_func(weights[0], weights[3]) ? c1 : SPACE;
// }
