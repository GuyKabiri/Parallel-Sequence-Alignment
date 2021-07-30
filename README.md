# Parallel Sequence Alignment

This project is part of `Parallel Computation` course.

## Intro
>In bioinformatics, a sequence alignment is a way of arranging the sequences of DNA, RNA, or protein to identify regions of similarity that may be a consequence of functional, structural, or evolutionary relationships between the sequences. [[1]](https://en.wikipedia.org/wiki/Sequence_alignment)  

## Sequence Alignment Evaluation
Each pair of characters generates a special character that indicates the degree of similarity between them.
The special characters are `*` (asterisk), `:` (colon), `.` (dot), and `_` (space).  
The following definitions apply:  
*   Equal characters will produce a `*`.
*   If two characters are not equal, but present in the same conservative group, they will produce a `*` sign.
*   If characters of a pair are not in the same conservative group but are in a semi-conservative group, then they will produce a `.`.
*   If none of the above is true, the characters will result in a `_` sign.

### Equation
Since each sign is weighted, the following equation will result from comparing two sequences:  
$$ S = N_{1} \times W_{1} - N_{2} \times W_{2} - N_{3} \times W_{3} - N_{4} \times W_{4} $$

>   $ N_{i} $ represents the amount, and $ W_{i} $ represents the weight, respectively, of `*`, `:`, `.`, and `_`.

### Groups
<table>
<tr>
<td>Conservative Groups</td><td>Semi-Conservative Groups</td>
</tr>
<tr>
<td>

|	|	|	|
|--|--|--|
|NDEQ  | NEQK|	STA|
|MILV | QHRK | NHQK|
|FYW | HY |MILF |

</td>
<td>

|	|	|	|	|
|--|--|--|--|
|SAG| ATV|	CSA|
|SGND| STPA| STNK|
|NEQHRK| NDEQHK|SNDEQK|
| HFY| FVLIM|    |

</td>
</tr>
</table>


An example of a pair-wise evaluation  
```
PSEKHLQCLLQRHKGK
HSKSHLQHLLQRHKSQ
_*:.***_******.:
```

The following can be seen above:
*   The 2nd pair consists of the characters `S` and `S`, they are equal, and hence result in the `*` sign.
*   The 3rd pair, `E` and `K`, are not equal, but present in the conservative group `NEQK`, so the result is a `:`.
* The 4th pair, `K` and `S`, don't belong to the same conservative group, but rather the same semi-conservative group `STNK`. Therefore, they result in a `.` sign.
*   The 1st pair consists of `P` and `H` without applying any of the rules defined above, so they result in the `_` sign.


The similarity of two sequences `Seq1` and `Seq2` defined as followed:
*	`Seq2` is places under the Sequence Seq1 with offset `n` from the start of `Seq1`. Where `Seq2` do not allowed to pass behind the end of `Seq1`.
*	The letters from `Seq1` that do not have a corresponding letter from `Seq2` are ignored.
*	The Alignment Score is calculated according the pair-wise procedure described above.


Examples:
<table>
<tr>
<td>Sequences</td><td>Results</td>
</tr>
<tr>
<td>

```
LQRHKRTHTGEKPYEPSHLQYHERTHTGEKPYECHQCHQAFKKCSLLQRHKRTH
                     HERTHTGEKPYECHQCRTAFKKCSLLQRHK
                     ****************: ************
```
</td>
<td>

>   Weights: 1.5 2.6 0.3 0.2 <br />
>   Offset: 21<br />
>   Score: 39.2
</td>
</tr>
<tr>
<td>

```
ELMVRTNMYTONEWVFNVJERVMKLWEMVKL
   MSKDVMSDLKWEV
   : .:: :  :* .
```
</td>
<td>

>   Weights: 5 4 3 2<br />
>   Offset: 3<br />
>   Score: -31
</td>
</tr>
</table>


## Mutation
For a given Sequence `S` we define a Mutant Sequence $ MS(n) $ which is received by substitution of one or more characters by other character defined by Substitution Rules:
*	The original character is allowed to be substituted by another character if there is no conservative group that contains both characters.  
    For example:
    *	`N` is not allowed to be substituted by `H` because both characterss present in conservative group `NHQK`.
    *   `N` may be substituted by `W` because there is now conservative group that contains both `N` and `W`.
*   It is not mandatory to substitute all instances of some characters by same substitution character, for example the sequence `PSHLSPSQ` has Mutant Sequence `PFHLSPLQ`.  

## Project Definition
In the given assignment, two sequences `Seq1`, `Seq2`, and a set of weights is provided. A mutation of the sequences `Seq2` and it's offset is need to be found, which produce the `MAX` or `MIN` score (will be given as an input as well).  


## Solution
Initially, a basic iterative solution was implemented. By iterating over the offsets and then for each pair of letters in the offset, the problem can be solved sequentially. Comparing each pair of letters to determine whether they are equal or fall into one of the conservative or semi-conservative groups, then finding their best substitutions (if possible). Hence, save any better substitution found for a pair than the previous one.

Having written the sequential solution, I realized it would be time-wasting to check whether each pair of letters belongs to a conservative or semi-conservative group several times during the run. Despite the fact that iterations over the groups are non-linear ($O(1) $) (since the number of groups and letters in each group is constant), the groups are given ahead of time, so each evaluation of two letters can be done before the program is run, saving significant time.  

Consequently, I created a hashtable of 26 letters and one `-` character (27 X 27). Each pair is still evaluated in $ O(1) $, but this method is much faster than the previous one.  
Additionally, `OpenMP` can be used for filling this hashtable in such a way that every cell is independent of every other cell.  
The hashtable (spaces were used instead of `_`) is as follows:

```
   A B C D E F G H I J K L M N O P Q R S T U V W X Y Z -
   _____________________________________________________
A |*   .       .                 .     : :   .          
B |  *                                                  
C |.   *                               .                
D |      * :   . .     .     :     :   .                
E |      : *     .     :     :     : . .                
F |          *   . :     : :                 . :   :    
G |.     .     *             .         .                
H |      . . .   *     :     :     : :             :    
I |          :     *     : :                 :          
J |                  *                                  
K |      . :     :     *     :     : : . .              
L |          :     :     * :                 :          
M |          :     :     : *                 :          
N |      : :   . :     :     *     : . . .              
O |                            *                        
P |.                             *     . .              
Q |      : :     :     :     :     * : .                
R |        .     :     :     .     : *                  
S |:   . . .   .       .     .   . .   * :              
T |:                   .     .   .     : *   .          
U |                                        *            
V |.         .     :     : :             .   *          
W |          :                                 *   :    
X |                                              *      
Y |          :   :                             :   *    
Z |                                                  *  
- |                                                    * 
```

It is now necessary to implement a parallel solution. As the project will run simultaneously on two machines, each should handle half of the tasks. A single machine should be able to download the input data and write the output data to the file, as specified in the project. The data should be sent between machines using MPI before beginning the search algorithm. Using MPI, one can easily determine the number of processes, so after passing data between processes, one can figure out how many tasks in total will be accomplished. As each process has its own ID, it can determine which specific tasks it will handle (taking into account when dividing the number of tasks unevenly among the number of machines).  

## GPU Implementation
In the beginning, an implementation similar to that on the CPU was performed. The number of threads created was equal to how many offsets the GPU has to handle. On second thought, that could lead to a failure to utilize all of the GPU's resources, when, for example, there are 3 offsets with each 1,000 characters. The GPU will only allocate three threads, although a higher number could have been allocated. CUDA provides a maximum of 1,024 threads per block, and 65,535 blocks (in each dimension of the grid), which results in a maximum of 67,107,840 threads per block (in one dimension block case). The project limitation is 10,000 letters for `Seq1` and 5,000 letters for `Seq2`, which adds up to 25,000,000 pairs of letters. The idea of allocating a thread for each letter and offset would be much better. Now, each thread will handle a specific pair of letters at a specific offset. Once the threads have completed evaluating the letters, the program has an array of mutations for each pair of letters and the original score of the original letters.  

In order to sum up the array and determine which mutation is optimal, a reduction is required. A reduction of pairs in each offset is necessary, in order to sum the offset's score and the optimal offset's mutation. After that, a second reduction is needed to determine which offset has the best mutation. Instead of linear iteration over the array, the reduction could be implemented in parallel.  

While investigating the parallel reduce algorithm, I realized that the mutations for a given offset will often end up in different thread blocks when the given input has a letter sequence that exceeds 1,024 letters. Because CUDA does not support over-grid thread synchronization, but only per block, it will be very difficult to implement the reduced algorithm. Several ways of handling this situation are suggested over the internet, such as using `counter lock`, which acts like a barrier, or CUDA's `cooperative-groups`, which allows threads to synchronize over the whole group.  
Time constraints forced me to take another route. Finally, it was decided to generate the number of blocks as the number of offsets, so that if there are more than 1,024 pairs of letters in each offset, some threads will have to calculate a mutation up to 5 times (in case there are 5,000 pairs of letters). A second problem is that the reduction algorithm only works with a power of 2 amount of threads. If the amount of letters or offsets is not a power of 2, squeezing is necessary before performing a reduction. Finally, for the pair evaluation, the highest power of 2 number equals to or less than the number of letters will be the block dimension. In this way, one can reduce the pairs without having to squeeze them. For the offsets, a squeezing is necessary before performing the reduction.

## Parallel Reduction

Parallel reduction refers to algorithms which combine an array of elements producing a single value as a result. Problems eligible for this algorithm include those which involve operators that are associative and commutative in nature. Some of them include:
*  Sum of an array.
*  Minimum/Maximum of an array.

Assuming one have an array of $ n $ values with $ n $ threads, the reduction algorithm provides a solution with $ log(n) $.
In order to perform reduction over array with $ n $ elemnts, $ n $ must be a power of 2 ($ 2^i $), if not a squeez have to be perform on the array in order to reduce the array size to be power of 2 (will discribed later). At the beginning of the algorithm, a `stride` constant with the size of $ n /_ 2 $ is defined. In each iteration of the algorithm, every cell performing the reduce operation between itself ($ i $) and $ i + stide $. At the end of each iteration, one need to devide `stride` by 2.

![](https://miro.medium.com/max/875/1*l1uoTZpQUW8YaSjFpcMNlw.png)
As can see above, array size is $ 16 $, therefore `stride` will be $ 8 $, and the amount of iterations is $ log(16) = 4 $.


## How To Run
The project was developed using `MPI`, `OpenMP`, and `CUDA`. Therefore, all of those library had to be installed for the project to run.  
An input file with a name of `input.txt` or `input.dat`, and with the following structure has to be present in the root directory:
*   The first line will contain 4 weights (decimal or non decimal) in the exact order of `W1`, `W2`, `W3`, and `W4`.
*   The seconde line will contains the first sequence `Seq1` (up to 10,000 characters).
*   The third line will contains the second sequence `Seq2` (up to 5,000characters).
*   The last line will contain the string `maximum` or `minimum` to define the algorithm which defines the goal of the search.  

The output file will results with the mutant of `Seq2` in the first line, and it's offset and score in the second line.

This project is produce to run on two machines on the same time, therefore a machinefile (`mf`) with machines subnetwork ip's is needed.
After the file exists on the main machine, and both machines contain the executable program, run the following line:  
```bash
mpiexec -np 2 -machinefile mf -map-by node ./{EXECUTABLE} {GPU} 
```
where `{EXECUTABLE}` is the name of the executable file, and `{GPU}` is a number represents the percentage of tasks to run on the machines GPU.



# Difficulties:
*  MPI Datatypes
*  Cuda blocks
*  Parallel reduction - syncing all threads in grid