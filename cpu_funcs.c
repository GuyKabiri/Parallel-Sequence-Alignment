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
#include "cuda_funcs.h"
#include "mpi_funcs.h"

// #define PRINT_SIGN_MAT


// #if (!(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0)))

char hashtable_cpu[NUM_CHARS][NUM_CHARS];
char conservatives_cpu[CONSERVATIVE_COUNT][CONSERVATIVE_MAX_LEN] = { "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
char semi_conservatives_cpu[SEMI_CONSERVATIVE_COUNT][SEMI_CONSERVATIVE_MAX_LEN] = { "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };

// #endif


extern int cuda_percentage;
extern MPI_Datatype program_data_type;
extern MPI_Datatype mutant_type;

void initiate_program(int pid, int num_processes)
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
    }

    if (num_processes > 1)      //  broadcast with MPI only if there are more processes
    {
        MPI_Bcast(&data, 1, program_data_type, ROOT, MPI_COMM_WORLD);
    }

    int total_tasks = strlen(data.seq1) - strlen(data.seq2) + 1;
    int per_proc_tasks = total_tasks / num_processes;

    int first_offset = per_proc_tasks * pid;  //  each process will handle the same amount of tasks, therefore, offset will be multiply by the process index
    int last_offset = per_proc_tasks + first_offset;
    if (pid == num_processes - 1)    //  if the tasks do not divide by the number of processes, the last process will handle any additional tasks
        last_offset += total_tasks % num_processes;



    int gpu_tasks = (last_offset - first_offset) * cuda_percentage / 100;
    int gpu_first_offset = first_offset;
    int gpu_last_offset = gpu_first_offset + gpu_tasks;

    int cpu_tasks = (last_offset - first_offset) - gpu_tasks;
    int cpu_first_offset = gpu_last_offset;
    int cpu_last_offset = cpu_first_offset + cpu_tasks;

    printf("pid %2d, total=%4d, per_proc=%4d, cuda start=%4d, cuda end=%4d, cpu start=%4d, cpu end=%4d\n",
            pid,
            total_tasks,
            last_offset - first_offset,
            gpu_first_offset,
            gpu_last_offset,
            cpu_first_offset,
            cpu_last_offset);

    double gpu_best_score = data.is_max ? INT_MIN : INT_MAX;
    double cpu_best_score = data.is_max ? INT_MIN : INT_MAX;
    Mutant gpu_mutant = { -1, -1, NOT_FOUND_CHAR };
    Mutant cpu_mutant = { -1, -1, NOT_FOUND_CHAR };

// #pragma omp parallel
{
    // printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %d\n", omp_get_num_threads());
    // #pragma omp single
    {
        if (gpu_tasks > 0)
        {
            gpu_best_score = gpu_run_program(&data, &gpu_mutant, gpu_first_offset, gpu_last_offset);
        }
    }

    if (cpu_tasks > 0)
    {   
        cpu_best_score = find_best_mutant_cpu(pid, &data, &cpu_mutant, cpu_first_offset, cpu_last_offset);
    }
}

    printf("cpu tasks=%3d, cpu best score=%lf\ngpu tasks=%3d, gpu best score=%lf\n", cpu_tasks, cpu_best_score, gpu_tasks, gpu_best_score);

    Mutant final_best_mutant = gpu_mutant;
    double final_best_score = gpu_best_score;
    if ((data.is_max && cpu_best_score > gpu_best_score) || 
        (!data.is_max && cpu_best_score < gpu_best_score))
    {
        final_best_mutant = cpu_mutant;
        final_best_score = cpu_best_score;
    }

    double my_best[2] = { 0 };
    my_best[0] = final_best_score;
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
        MPI_Send(&final_best_mutant, 1, mutant_type, ROOT, 0, MPI_COMM_WORLD);
    }
    
    if (pid == ROOT)
    {   
        MPI_Status status;
        if (sender != ROOT)     //  if ROOT process does not have the best score -> retrieve it from the process that does
        {
            final_best_score = gloabl_best[0];        //  best score already sent to all processes by MPI_Allreduce
    	    MPI_Recv(&final_best_mutant, 1, mutant_type, sender, 0, MPI_COMM_WORLD, &status);
        }
        
        char mut[SEQ2_MAX_LEN];
        strcpy(mut, data.seq2);
        mut[final_best_mutant.char_offset] = final_best_mutant.ch;

    	FILE* out_file = fopen(OUTPUT_FILE, "w");
    	if (!out_file)
    	{
    		printf("Error open or write to the output file %s\n", OUTPUT_FILE);
    		MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	if (!write_results_to_file(out_file, mut, final_best_mutant.offset, final_best_score))
    	{
    		printf("Error write to the output file %s\n", OUTPUT_FILE);
			MPI_Abort(MPI_COMM_WORLD, 2);
			exit(1);
    	}
    	fclose(out_file);
        
        if (cpu_tasks == 0) //  if the root CPU did not had any tasks, it's hashtable is empty, therefore, it will fill it now for printing
            fill_hash(data.weights, pid);

        pretty_print(&data, mut, final_best_mutant.offset, final_best_mutant.char_offset);
    }
}

double find_best_mutant_cpu(int pid, ProgramData* data, Mutant* return_mutant, int first_offset, int last_offset)
{
    fill_hash(data->weights, pid);
#ifdef PRINT_SIGN_MAT
    if (pid == ROOT)
        print_hash();
#endif
    
    //  global variable for the best score among all threads
    double gloabl_score = data->is_max ? INT_MIN : INT_MAX;
#pragma omp parallel
{
    double best_score = data->is_max ? INT_MIN : INT_MAX;      //  private variable for thread's best score
    double curr_score;          //  private variable for thread's specific offset score
    Mutant best_mutant;         //  private variable for thread's best mutant
    Mutant temp_mutant;         //  private variable for thread's specific offset mutant
    int to_save;                //  private variable whether to save the current score or not

#pragma omp for nowait      //  each thread will calculate some of the process tasks and save it's best mutant
    for (int curr_offset = first_offset; curr_offset < last_offset; curr_offset++)    //  iterate for amount of tasks
    {     
        //  clculate this offset score, and find the best mutant in that offset
        curr_score = find_best_mutant_offset(data, curr_offset, &temp_mutant);

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

void fill_hash(double* weights, int pid)
{
    char c1, c2;
#pragma omp parallel for
    for (int i = 0; i < NUM_CHARS; i++)
    {
        c1 = FIRST_CHAR + i;               //  FIRST_CHAR = A -> therefore (FIRST_CHAR + i) will represent all characters from A to Z
        for (int j = 0; j <= i; j++)            //  it would be time-consuming to fill the top triangle of a hash table, because it is cyclic (hash[x][y] = hash[y][x])
        {
            c2 = FIRST_CHAR + j;
            hashtable_cpu[i][j] = evaluate_chars(c1, c2);
        }
        hashtable_cpu[i][NUM_CHARS] = SPACE;    //  each char with '-' (hash[ch][-])
    }
}

void print_hash()
{
    char last_char = FIRST_CHAR + NUM_CHARS;
    printf("   ");
    for (int i = FIRST_CHAR; i < last_char; i++)
        printf("%c ", i);
    printf("%c\n", DASH);
    printf("   ");
    for (int i = FIRST_CHAR; i < last_char + 1; i++)
        printf("__");
    printf("\n");
    for (int i = FIRST_CHAR; i < last_char; i++)
    {
        printf("%c |", i);
        for (int j = FIRST_CHAR; j < last_char; j++)
        {
            printf("%c ", get_hash_sign(i, j));
        }
        printf("%c \n", get_hash_sign(i, DASH));
    }
    printf("%c |", DASH);
    for (int i = FIRST_CHAR; i < last_char; i++)
    {
        printf("%c ", get_hash_sign(DASH, i));
    }
    printf("%c ", get_hash_sign(DASH, DASH));
    printf("\n");
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
    data->is_max = strcmp(func_type, MAXIMUM_STR) == 0 ? MAXIMUM_FUNC : MINIMUM_FUNC;    //  saves '1' if it is a maximum, otherwise, saves '0'

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
void pretty_print(ProgramData* data, char* mut, int offset, int char_offset)
{
    if (!data || !mut)  return;

    printf("\033[0;31m%s\033[0m problem\n", data->is_max ? "Maximum" : "Minimum");
    printf("Weights: ");
    for (int i = 0; i < WEIGHTS_COUNT; i++)
        printf("%g ", data->weights[i]);

    if (offset == -1)
    {
        printf("\n\033[0;31mThere are no mutations found\033[0m\n");
        printf("%s\n%s\n", data->seq1, data->seq2);
        return;
    }

    char signs[SEQ2_MAX_LEN] = { '\0' };
    double score = get_score_and_signs(data->seq1, data->seq2, data->weights, offset, signs);    //  evaluate the score of the sequences by the wanted offset, and create the signs sequence

    printf("\nOriginal Score: %g\n", score);

    print_with_offset(signs, offset, char_offset);
    printf("\n");

    print_with_offset(data->seq2, offset, char_offset);
    printf("\n");
    
    printf("%s\n", data->seq1);       //  print 1st sequence

    print_with_offset(mut, offset, char_offset);
    printf("\n");

    score = get_score_and_signs(data->seq1, mut, data->weights, offset, signs);    //  evaluate the score of the sequences by the wanted offset, and create the signs sequence

    print_with_offset(signs, offset, char_offset);
    printf("\n");

    printf("Mutation Score: %g\n", score);
    printf("Seq offset=%3d, Char offset=%3d\n", offset, char_offset);
}

double get_score_and_signs(char* seq1, char* seq2, double* weights, int offset, char* signs)
{
    int idx1 = offset;
    int idx2 = 0;
    int num_chars = strlen(seq2);
    double score = 0;
    for (   ; idx2 < num_chars; idx1++, idx2++)
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

    for (uint i = char_offset + 1; i < strlen(chrs); i++)
        printf("%c", chrs[i]);       //  print signs sequence
}