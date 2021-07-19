#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include "cpu_funcs.h"
#include "def.h"
#include "program_data.h"
#include "mutant.h"


char conservatives_arr[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
char semi_conservatives_arr[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
char char_hash[NUM_CHARS][NUM_CHARS];

extern int cuda_percentage;

void cpu_run_program(int pid, int num_processes)
{
    ProgramData data;

    if (pid == ROOT)
    {
        FILE* input_file;

        input_file = fopen(INPUT_FILE, "r");
        if (!input_file)
        {
            printf("Error open input file `%s`\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

//         if (!read_seq_and_weights_from_file(input_file, seq1, seq2, weights, &is_max))
        if (!read_seq_and_weights_from_file(input_file, &data))
        {
            printf("Error reading input file `%s`\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        fclose(input_file);

        int iterations = strlen(data.seq1) - strlen(data.seq2) + 1;
        data.num_tasks = iterations / num_processes;
        data.offset_add = iterations % num_processes;   //  if amount of offset does not divide by amount of processes, the root process will take the additional tasks

        printf("%s\n", data.is_max ? "maximum" : "minimum");
    }

//    MPI_Bcast(&data, 1, mpi_data_type, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(data.seq1, SEQ1_MAX_LEN, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(data.seq2, SEQ2_MAX_LEN, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(data.weights, WEIGHTS_COUNT, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&data.num_tasks, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&data.offset_add, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&data.is_max, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    fill_hash(data.weights);
    if (pid == ROOT)
        print_hash();

//    printf("proc %d: tasks: %d, off: %d\n", pid, data.num_tasks, data.offset_add);

    int start_offset = data.num_tasks * pid;  //  each process will handle the same amount of tasks, the offset will be multiply by the process index + the additional offset
    if (pid == ROOT)
    {
        data.num_tasks += data.offset_add;
    }
    else
    {
    	start_offset += data.offset_add;
    }
//    printf("pro %d: tasks: %d, start: %d, end: %d\n", pid, data.num_tasks, start_offset, data.num_tasks + start_offset);


    Mutant my_mutant;

    int best_offset = 0;
    double best_score = 0;
    Mutant best_mutant;

    double curr_score, mutant_score;

    for (int i = start_offset; i < data.num_tasks + start_offset; i++)
    {
        curr_score = find_best_mutant_offset(data.seq1, data.seq2, data.weights, i, data.is_max, &my_mutant);
        mutant_score = curr_score + my_mutant.mutant_diff;

        if (mutant_score > best_score)
        {
            best_mutant.ch = my_mutant.ch;
            best_mutant.char_offset = my_mutant.char_offset;
            best_mutant.mutant_diff = my_mutant.mutant_diff;

            best_score = mutant_score;
        	best_offset = i;
        }
    }

//    printf("proc %2d: offset: %3d, score: %g\n", my_rank, best_offset, best_score);
//    double score = find_mutant(data.seq1, data.seq2, data.weights, best_offset, mutant, data.is_max);




    double mymax[2] = { 0 };
    mymax[0] = best_score;
    mymax[1] = pid;
    double globalmax[2] = { 0 };

    MPI_Allreduce(mymax, globalmax, 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_COMM_WORLD);
    int sender_rank = globalmax[1];

//    printf("me: %d, global[0]: %g, global[1]: %g\n", pid, globalmax[0], globalmax[1]);

    MPI_Status status;
    if (pid == sender_rank)    //  send only from the process with max score
    {
        data.seq2[my_mutant.char_offset] = my_mutant.ch;
    	MPI_Send(&best_offset, 1, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
    	MPI_Send(data.seq2, SEQ2_MAX_LEN, MPI_CHAR, ROOT, 0, MPI_COMM_WORLD);
    }
    if (pid == ROOT)
    {
    	best_score = globalmax[0];
    	MPI_Recv(&best_offset, 1, MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, &status);
    	MPI_Recv(data.seq2, SEQ2_MAX_LEN, MPI_CHAR, sender_rank, 0, MPI_COMM_WORLD, &status);
    	printf("best offset: %3d, by procs: %2d, score: %g\n%s\n", best_offset, sender_rank, best_score, data.seq2);
    	FILE* out_file = fopen(OUTPUT_FILE, "w");
    	if (!out_file)
    	{
    		printf("Error open or write to the output file %s\n", OUTPUT_FILE);
    		MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	if (!write_results_to_file(out_file, data.seq2, best_offset, best_score))
    	{
    		printf("Error write to the output file %s\n", OUTPUT_FILE);
			MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	fclose(out_file);
        
        print_seq(data.seq1, data.seq2, data.weights, best_offset);
    }
}


void fill_hash(double* weights)
{
// #pragma omp parallel for
    for (int i = 0; i < NUM_CHARS; i++)
    {
        char c1 = FIRST_CHAR + i;               //  FIRST_CHAR = A -> therefore FIRST_CHAR + i will represent all characters from A to Z
        for (int j = 0; j < NUM_CHARS; j++)
        {
            char c2 = FIRST_CHAR + j;
            evaluate_chars(c1, c2, weights, &char_hash[i][j]);
        }
    }
}

void print_hash()
{
    printf("   ");
    for (int i = 0; i < NUM_CHARS; i++)
        printf("%c ", i + FIRST_CHAR);
    printf("\n");
    printf("   ");
    for (int i = 0; i < NUM_CHARS; i++)
        printf("__");
    printf("\n");
    for (int i = 0; i < NUM_CHARS; i++)
    {
        char c1 = FIRST_CHAR + i;               //  FIRST_CHAR = A -> therefore FIRST_CHAR + i will represent all characters from A to Z
        printf("%c |", c1);
        for (int j = 0; j < NUM_CHARS; j++)
        {
            printf("%c ", char_hash[i][j]);
        }
        printf("\n");
    }
}

//  check for the character c in the string s
//  returns NULL if c is not presented in s, otherwise returns the address of the occurrence of c
char* is_contain(char* s, char c)
{
    return strchr(s, c);
}

//  check if both characters present in the same conservative group
int is_conservative(char c1, char c2)
{
    for (int i = 0; i < CONSERVATIVE_COUNT; i++)    //  iterate over the conservative groups
        if (is_contain(conservatives_arr[i], c1) && is_contain(conservatives_arr[i], c2))   //  if both characters present
            return 1;
    return 0;
}

//  check if both characters present in the same semi-conservative group
int is_semi_conservative(char c1, char c2)
{
    for (int i = 0; i < SEMI_CONSERVATIVE_COUNT; i++)   //  iterate over the semi-conservative groups
            if (is_contain(semi_conservatives_arr[i], c1) && is_contain(semi_conservatives_arr[i], c2))   //  if both characters present
                return 1;
    return 0;
}

double get_weight(char sign, double* weights)
{
    double w;
    switch (sign)
    {
    case STAR:  w = weights[STAR_W];    break;
    case COLON: w = -weights[COLON_W];    break;
    case DOT:   w = -weights[DOT_W];    break;
    case SPACE: w = -weights[SPACE_W];    break;
    default:    w = -1;    break;
    }
    return w;
}

char get_hash_sign(char c1, char c2)
{
    return char_hash[c1 - FIRST_CHAR][c2 - FIRST_CHAR];
}

//  find the best mutant for a given offset
double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, int is_max, Mutant* mt)
{
    int seq1_idx, seq2_idx;
    double total_score = 0;
    double pair_score, mutant_diff, best_mutant_diff;
    int iterations = strlen(seq2);
    char ch;

    for (int i = 0; i < iterations; i++)          //  iterate over all the characters
    {
        seq1_idx = offset + i;
        seq2_idx = i;
        char c1 = seq1[seq1_idx];
        char c2 = seq2[seq2_idx];
        pair_score = get_weight(get_hash_sign(c1, c2), weights);
        total_score += pair_score;

        ch = find_char(c1, c2, weights, is_max);
        mutant_diff = get_weight(get_hash_sign(c1, ch), weights) - pair_score;

        if (mutant_diff > best_mutant_diff)
        {
            best_mutant_diff = mutant_diff;
            mt->ch = ch;
            mt->char_offset = i;        //  offset of char inside seq2
            mt->mutant_diff = mutant_diff;
        }
    }
    
    return total_score;     //  best mutant is returned in struct mt
}

char find_char(char c1, char c2, double* weights, int is_max)
{
    char sign = char_hash[c1 - FIRST_CHAR][c2 - FIRST_CHAR];

    return  is_max ?
            find_max_char(c1, c2, sign, weights)   :
            find_min_char(c1, c2, sign, weights);
}



char find_max_char(char c1, char c2, char sign, double* weights)
{
    //  TODO: check maybe calculate signs array in parallel while decrease the computation time, over calculate each sign seperate
    char ch;
    switch (sign)
    {
    case STAR:
        return c2;

    case DOT:                   //  if there is DOT between two characters, a START subtitution is possible
        return c1;

    case SPACE:                 //  if there is SPACE between two characters, a START subtitution is possible
        return c1;

    case COLON:
        double dot_diff = weights[COLON_W] - weights[DOT_W];
        double space_diff = weights[COLON_W] - weights[SPACE_W];

        if (!(dot_diff > 0 || space_diff > 0))      //  if both not greater than 0 (negative change or no change at all)
        {                                           //  then, no score change and return the same character
            return c2;
        }

        if (space_diff > dot_diff)                 //  if SPACE subtitution is better than DOT
        {
            ch = get_char_by_sign_with_restrictions(c1, SPACE, c2);
            if (ch != NOT_FOUND_CHAR)       //  if found SPACE subtitution
                return ch;
            
            //  if could not find SPACE subtitution, and DOT is better than no subtitution
            if (dot_diff > 0)
            {
                ch = get_char_by_sign_with_restrictions(c1, DOT, c2);
                if (ch != NOT_FOUND_CHAR)       //  if found DOT subtitution
                    return ch;
            }

            //  otherwise, no subtitution found
            return c2;
        }

        //  otherwise, it will try to find DOT subtitution
        ch = get_char_by_sign_with_restrictions(c1, DOT, c2);
        if (ch != NOT_FOUND_CHAR)       //  if found DOT subtitution
            return ch;

        //  if could not find DOT subtitution, and SPACE is better than no subtitution
        if (space_diff > 0)
        {
            ch = get_char_by_sign_with_restrictions(c1, SPACE, c2);
            if (ch != NOT_FOUND_CHAR)       //  if found SPACE subtitution
                return ch;
        }

        //  otherwise, no subtitution found
        return c2;
    // default:
    //     ch = 'd';
    //     return ch;
    }
    return c2;
}


char find_min_char(char c1, char c2, char sign, double* weights)
{
    char ch;

    char colon_sub = get_char_by_sign_with_restrictions(c1, COLON, c2);
    char dot_sub = get_char_by_sign_with_restrictions(c1, DOT, c2);
    char space_sub = get_char_by_sign_with_restrictions(c1, SPACE, c2);

    double colon_diff, dot_diff, space_diff;

    switch (sign)
    {
    case STAR:
        colon_diff = - weights[STAR_W] - weights[COLON_W];
        dot_diff = - weights[STAR_W] - weights[DOT_W];
        space_diff = - weights[STAR_W] - weights[SPACE_W];

        if (!(colon_diff < 0 || dot_diff < 0 || space_diff < 0))    //  if any subtitution will not decrease the score
            return c2;                                              //  than return the same letter and score

        if (colon_diff < dot_diff && colon_diff < space_diff)
        {
            if (colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        //  could not find COLON subtitution
        if (dot_diff < space_diff)
        {
            if (dot_sub != NOT_FOUND_CHAR)
                return dot_sub;
            
            // could not find DOT subtitution and COLON is better than space
            if (colon_diff < space_sub && colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        //  could not dinf DOT subtitution
        if (space_diff < 0)
        {
            if (space_sub != NOT_FOUND_CHAR)
                return space_sub;

            // could not find SPACE subtitution, but DOT or COLON might still be better than nothing
            if (colon_diff < dot_diff && colon_sub != NOT_FOUND_CHAR)
                return colon_sub;

            //  could not find neither SPACE, not COLON subtitution
            if (dot_diff < 0 && dot_sub != NOT_FOUND_CHAR)
                return dot_sub;
        }

        return c2;  //  could not find any subtitution
    
    case COLON:
        dot_diff = weights[COLON_W] - weights[DOT_W];
        space_diff = weights[COLON_W] - weights[SPACE_W];

        if (!(dot_diff < 0 || space_diff < 0))      //  if any subtitution will not decrease the score
            return c2;                              //  than return the same letter and score

        if (dot_diff < space_diff)                  //  if DOT subtitution is better than SPACE
        {
            if (dot_sub != NOT_FOUND_CHAR)          //  if found DOT subtitution
                return dot_sub;
        }

        if (space_diff < 0)
        {
            if (space_sub != NOT_FOUND_CHAR)
                return space_sub;

            //  could not find SPACE subtitution, but DOT might be better than nothing
            if (dot_diff < 0 && dot_sub != NOT_FOUND_CHAR)
                return dot_sub;
        }
        
        return c2;  // could not find any subtitution

    case DOT:
        colon_diff = weights[DOT_W] - weights[COLON_W];
        space_diff = weights[DOT_W] - weights[SPACE_W];

        if (!(colon_diff < 0 && space_diff < 0))    //  if any subtitution will not decrease the score
            return c2;                              //  than return the same letter and score

        if (colon_diff < space_diff)                //  if COLON subtitution is better than SPACE   
        {
            if (colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        if (space_diff < 0)
        {
            if (space_sub != NOT_FOUND_CHAR)
                return space_sub;
            
            //  could not find SPACE subtitution, but COLON might still be better than nothing
            if (colon_diff < 0 && colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        return c2;  // could not find any subtitution

    case SPACE:
        colon_diff = weights[SPACE_W] - weights[COLON_W];
        dot_diff = weights[SPACE_W] - weights[DOT_W];

        if (!(colon_diff < 0 && dot_diff < 0))      //  if any subtitution will not decrease the score
            return c2;                              //  than return the same letter and score

        if (colon_diff < dot_diff)                  //  if COLON subtitution is better than DOT
        {
            if (colon_sub != NOT_FOUND_CHAR)        //  if found COLON subtitution
                return colon_sub;
        }

        if (dot_diff < 0)
        {
            if (dot_sub != NOT_FOUND_CHAR)          //  if found DOT subtitution
                return dot_sub;

            //  could not find DOT subtitution, but COLON might still be better than nothing
            if (colon_diff < 0 && colon_sub != NOT_FOUND_CHAR)
                return colon_sub;
        }

        return c2;  // could not find any subtitution
    }
    return c2;      //  sign was not any of the legal signs
}

//  evaluate pair of characters, return their score, and suitable sign
double evaluate_chars(char a, char b, double* weights, char* s)
{
    char temp;
    if (s == NULL)
        s = &temp;  //  in case the returned char is not required
    if      (a == b)                        { *s = STAR;  return weights[STAR_W]; }
    else if (is_conservative(a, b))         { *s = COLON; return -weights[COLON_W]; }
    else if (is_semi_conservative(a, b))    { *s = DOT; return -weights[DOT_W]; }

    *s = SPACE;
    return -weights[SPACE_W];
}

char get_char_by_sign_with_restrictions(char by, char sign, char rest)
{
    int hash_idx = by - FIRST_CHAR;
    for (int i = 0; i < NUM_CHARS; i++)
    {
        if (char_hash[hash_idx][i] == sign && char_hash[rest][i] != COLON)
            return i + FIRST_CHAR;
    }
    return NOT_FOUND_CHAR;
}

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

//	write the results into a file
//	return 0 on error, otherwise 1
int write_results_to_file(FILE* file, char* mutant, int offset, double score)
{
	if (!file || !mutant)	return 0;

	return fprintf(file, "%s\n%d %g", mutant, offset, score) > 0;	//	fprintf will return negative value if error occurred while writing to the file
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

double compare_evaluate_seq(char* seq1, char* seq2, double* weights, int offset, char* signs)      //  signs array is for debugging and pretty printing
{
    if (!seq1 || !seq2 || !weights)  return 0;

    int seq1_idx = offset;
    int seq2_idx = 0;
    double total_score = 0;
    int iterations = fmin(strlen(seq2), strlen(seq1) - offset);        // TODO: change fmin with min
    char _sign;
    for ( ; seq2_idx < iterations; seq2_idx++, seq1_idx++)
    {
        total_score += evaluate_chars(seq1[seq1_idx], seq2[seq2_idx], weights, &_sign);

        if (signs)  signs[seq2_idx] = _sign;

    }
    return total_score;
}


// /*  
//     s = w1 * n_start - w2 * n_colon - w3 * n_point - w4 * n_space
//     therefore, if c1 and c2 are NOT conservative, maximize the equation should be by max(w1, -w3, -w4)
//     in case -w4 is greater than the others, a random character has to be found for the mutant, that is
//     different than c1 and is neither conservative, nor semi-conservative with c1

//     returns the char for the mutant and the new score in the score pointer
// */
// char find_char(char c1, char c2, double* weights, double* score, int (*eval_func)(double, double))
// {
//     if (is_conservative(c1, c2))    //  if the characters are conservative, substitute is not allowed   ->  return the same letter
//     {    
//         if (c1 == c2)               //  if same letters -> add the suitable weight
//             *score += weights[0];
//         else                        //  not same letters, but conservative ones -> substruct the suitable weight
//             *score -= weights[1];
//         return c2;      
//     }

//     if (is_semi_conservative(c1, c2))   //  if the characters are semi conservative, then
//     {
//         if      (eval_func(weights[0], -weights[2]) && eval_func(weights[0], -weights[3]))  { *score += weights[0]; return c1; }    //  if w1 > w3, w4 then return STAR
//         else if (eval_func(-weights[3], -weights[2]) && eval_func(-weights[3], weights[0])) { *score -= weights[3]; return find_char_to_space(c1); }  //  if w4 > w1, w3 then return SPACE
//         return c2;                                                                      //  otherwise, return COLON (same letter, changing to other semi conservative will have no effect)
//     }

//     //  otherwise, the characters are neither conservative, nor semi conservative
//     //  then, maximize by max(w1, w4) which means to return c1 or SPACE
//     if (eval_func(weights[0], -weights[3]))
//     {
//         *score += weights[0];
//         return c1;
//     }
    
//     *score -= weights[3];
//     return find_char_to_space(c1);
// }