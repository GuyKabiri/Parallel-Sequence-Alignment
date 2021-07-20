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



#define PRINT_SIGN_MAT

extern int cuda_percentage;
extern MPI_Datatype program_data_type;

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

        if (!read_seq_and_weights_from_file(input_file, &data))
        {
            printf("Error reading input file `%s`\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        fclose(input_file);

        data.proc_count = num_processes;
        printf("%s\n", data.is_max ? "maximum" : "minimum");
    }

    if (num_processes > 1)      //  broadcast with MPI only if there are more processes
    {
        MPI_Bcast(&data, 1, program_data_type, ROOT, MPI_COMM_WORLD);
    }

    Mutant my_mutant;
    double best_score = find_best_mutant(pid, &data, &my_mutant);

    double my_best[2] = { 0 };
    my_best[0] = best_score;
    my_best[1] = pid;
    double gloabl_best[2] = { 0 };

    //  MPI_Allreduce will find the MAX or MIN value that sent from all the processes
    //  and send it to all processes with the process id that holds that value
    if (num_processes > 1)
    {
        if (data.is_max)
            MPI_Allreduce(my_best, gloabl_best, 1, MPI_2DOUBLE_PRECISION, MPI_MAXLOC, MPI_COMM_WORLD);
        else
            MPI_Allreduce(my_best, gloabl_best, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_WORLD);
    }
    
    int sender = gloabl_best[1];    //  the id of the process with the best value

    //  if the sender is not the ROOT (ROOT does not need to send the best value to himself)
    if (sender != ROOT && pid == sender)
    {
        MPI_Send(&my_mutant.offset, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
        MPI_Send(&my_mutant.char_offset, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
        MPI_Send(&my_mutant.ch, 1, MPI_CHAR, ROOT, 0, MPI_COMM_WORLD);
    }
    
    if (pid == ROOT)
    {   
        MPI_Status status;
        if (sender != ROOT)     //  if ROOT process does not have the best score -> retrieve it from the process that does
        {
            best_score = gloabl_best[0];        //  best score already sent to all processes by MPI_Allreduce
    	    MPI_Recv(&my_mutant.offset, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &status);
    	    MPI_Recv(&my_mutant.char_offset, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &status);
    	    MPI_Recv(&my_mutant.ch, 1, MPI_CHAR, sender, 0, MPI_COMM_WORLD, &status);
        }
        
        char mut[SEQ2_MAX_LEN];
        strcpy(mut, data.seq2);
        mut[my_mutant.char_offset] = my_mutant.ch;

    	FILE* out_file = fopen(OUTPUT_FILE, "w");
    	if (!out_file)
    	{
    		printf("Error open or write to the output file %s\n", OUTPUT_FILE);
    		MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	if (!write_results_to_file(out_file, data.seq2, my_mutant.offset, best_score))
    	{
    		printf("Error write to the output file %s\n", OUTPUT_FILE);
			MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	fclose(out_file);
        
        pretty_print_seq_mut(data.seq1, data.seq2, mut, data.weights, my_mutant.offset, my_mutant.char_offset);
    }
}

double find_best_mutant(int pid, ProgramData* data, Mutant* return_mutant)
{
        fill_hash(data->weights, pid);
#ifdef PRINT_SIGN_MAT
    if (pid == ROOT)
        print_hash();
#endif

    int total_tasks = strlen(data->seq1) - strlen(data->seq2) + 1;
    int my_tasks = total_tasks / data->proc_count;

    int first_offset = my_tasks * pid;  //  each process will handle the same amount of tasks, therefore, offset will be multiply by the process index
    int last_offset = my_tasks + first_offset;
    if (pid == data->proc_count - 1)    //  if the tasks do not divide by the number of processes, the last process will handle any additional tasks
        last_offset += total_tasks % data->proc_count;
    
    double gloabl_score = 0;    //  global variable for the best score among all threads
#pragma omp parallel
{
    double best_score = 0;      //  private variable for thread's best score
    double curr_score;          //  private variable for thread's specific offset score
    Mutant best_mutant;         //  private variable for thread's best mutant
    Mutant temp_mutant;         //  private variable for thread's specific offset mutant
    int to_save;                //  private variable whether to save the current score or not

#pragma omp for nowait//schedule(dynamic, 2)  //  each thread will calculate some of the process tasks and save it's best mutant
    for (int curr_offset = first_offset; curr_offset < last_offset; curr_offset++)    //  iterate for amount of tasks
    {     
        //  clculate this offset score, and find the best mutant in that offset
        curr_score = find_best_mutant_offset(data->seq1, data->seq2, data->weights, curr_offset, data->is_max, &temp_mutant);

        to_save = (data->is_max) ?                  //  if this is a maximum problem
                    (curr_score > best_score) :     //  save if the current score is greater than the best
                    (curr_score < best_score);      //  otherwise, save if the current score is smaller than the best

        if (to_save)              //  if found better mutation, or it is the first iteration
        {
            best_mutant = temp_mutant;
            best_mutant.offset = curr_offset;
            best_score = curr_score;
        }
    }

    //  synchronize writing to the global score
    #pragma omp critical
    {
        to_save = (data->is_max) ?                  //  if this is a maximum problem
                    (best_score > gloabl_score) :     //  save if the current score is greater than the best
                    (best_score < gloabl_score);      //  otherwise, save if the current score is smaller than the best

        if (to_save)
        {
            gloabl_score = best_score;
            *return_mutant = best_mutant;
        }
    }
}
    return gloabl_score;
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

//  evaluate pair of characters, return their score, and suitable sign
char evaluate_chars(char a, char b, double* weights)
{
    if      (a == b)                        return STAR;
    else if (is_conservative(a, b))         return COLON;
    else if (is_semi_conservative(a, b))    return DOT;

    return SPACE;
}


void fill_hash(double* weights, int pid)
{
#pragma omp parallel for
    for (int i = 0; i < NUM_CHARS; i++)
    {
        char c1 = FIRST_CHAR + i;               //  FIRST_CHAR = A -> therefore (FIRST_CHAR + i) will represent all characters from A to Z
        for (int j = 0; j <= i; j++)            //  it would be time-consuming to fill the top triangle of a hash table, because it is cyclic (hash[x][y] = hash[y][x])
        {
            char c2 = FIRST_CHAR + j;
            char_hash[i][j] = evaluate_chars(c1, c2, weights);
        }
    }
}

void print_hash()
{
    char last_char = FIRST_CHAR + NUM_CHARS;
    printf("   ");
    for (int i = FIRST_CHAR; i < last_char; i++)
        printf("%c ", i);
    printf("\n");
    printf("   ");
    for (int i = FIRST_CHAR; i < last_char; i++)
        printf("__");
    printf("\n");
    for (int i = FIRST_CHAR; i < last_char; i++)
    {
        printf("%c |", i);
        for (int j = FIRST_CHAR; j < last_char; j++)
        {
            printf("%c ", get_hash_sign(i, j));
        }
        printf("\n");
    }
}

char get_hash_sign(char c1, char c2)
{
    if (c1 >= c2)       //  only the bottom triangle of the hash table is full -> (hash[x][y] = hash[y][x])
        return char_hash[c1 - FIRST_CHAR][c2 - FIRST_CHAR];
    return char_hash[c2 - FIRST_CHAR][c1 - FIRST_CHAR];
}

double get_weight(char sign, double* weights)
{
    double w;
    switch (sign)
    {
    case STAR:  return weights[STAR_W];
    case COLON: return -weights[COLON_W];
    case DOT:   return -weights[DOT_W];
    case SPACE: return -weights[SPACE_W];
    }
    return 0;
}

//  find the best mutant for a given offset
double find_best_mutant_offset(char* seq1, char* seq2, double* weights, int offset, int is_max, Mutant* mt)
{
    int seq1_idx, seq2_idx;
    double total_score = 0;
    double pair_score, mutant_diff, best_mutant_diff;
    int iterations = strlen(seq2);
    char subtitue;

    for (int i = 0; i < iterations; i++)            //  iterate over all the characters
    {
        seq1_idx = offset + i;                      //  index of seq1
        seq2_idx = i;                               //  index of seq2
        char c1 = seq1[seq1_idx];                   //  current char in seq1
        char c2 = seq2[seq2_idx];                   //  current char in seq2
        pair_score = get_weight(get_hash_sign(c1, c2), weights);    //  get weight before substitution
        total_score += pair_score;

        subtitue = find_char(c1, c2, weights, is_max);
        mutant_diff = get_weight(get_hash_sign(c1, subtitue), weights) - pair_score;    //  difference between original and mutation weights
        mutant_diff = abs(mutant_diff);

        if (mutant_diff > best_mutant_diff || i == 0)
        {
            best_mutant_diff = mutant_diff;
            mt->ch = subtitue;
            mt->char_offset = i;        //  offset of char inside seq2
        }
    }
    if (is_max)
        return total_score + best_mutant_diff;
    return total_score - best_mutant_diff;     //  best mutant is returned in struct mt
}

char find_char(char c1, char c2, double* weights, int is_max)
{
    char sign = get_hash_sign(c1, c2);

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

char get_char_by_sign_with_restrictions(char by, char sign, char rest)
{
    char last_char = FIRST_CHAR + NUM_CHARS;
    for (char ch = FIRST_CHAR; ch < last_char; ch++)   //  iterate over alphabet (A-Z)
    {
        if (get_hash_sign(by, ch) == sign && get_hash_sign(rest, ch) != COLON)  //  if found character which is not in the same conservative group with the previous one
            return ch;
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
    data->proc_count = 0;

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
void pretty_print_seq_mut(char* seq1, char* seq2, char* mut, double* weights, int offset, int char_offset)
{
    if (!seq1 || !seq2 || !mut || !weights)  return;

    char signs[SEQ2_MAX_LEN] = { '\0' };
    double score = get_score_and_signs(seq1, seq2, weights, offset, signs);    //  evaluate the score of the sequences by the wanted offset, and create the signs sequence

    printf("Original Score: %g\n", score);

    print_with_offset(signs, offset, char_offset);
    printf("\n");

    print_with_offset(seq2, offset, char_offset);
    printf("\n");
    
    printf("%s\n", seq1);       //  print 1st sequence

    print_with_offset(mut, offset, char_offset);
    printf("\n");

    score = get_score_and_signs(seq1, mut, weights, offset, signs);    //  evaluate the score of the sequences by the wanted offset, and create the signs sequence

    print_with_offset(signs, offset, char_offset);
    printf("\n");

    printf("Mutation Score: %g\n", score);
    printf("Seq offset=%3d, Char offset=%3d\n", offset, char_offset);
}

double get_score_and_signs(char* seq1, char* seq2, double* weights, int offset, char* signs)
{
    int idx1 = offset;
    int idx2 = 0;
    int iterations = strlen(seq2);
    double score = 0;
    for (int i = 0; i < iterations; i++, idx1++, idx2++)
    {   
        signs[idx2] = get_hash_sign(seq1[idx1], seq2[idx2]);
        score += get_weight(signs[idx2], weights);
    }
    return score;
}

void print_with_offset(char* chrs, int offset, int char_offset)
{
    if (char_offset < 0)
        char_offset = 0;
    for (int i = 0; i < offset; i++)
        printf(" ");    //  print spaces to align the offset of sequences

    for (int i = 0; i < char_offset; i++)
        printf("%c", chrs[i]);       //  print signs sequence

    printf("\033[0;31m");
    printf("%c", chrs[char_offset]);
    printf("\033[0m");

    for (int i = char_offset + 1; i < strlen(chrs); i++)
        printf("%c", chrs[i]);       //  print signs sequence
}