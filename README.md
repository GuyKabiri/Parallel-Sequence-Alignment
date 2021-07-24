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

## Project Definition:
In the given assignment, two sequences `Seq1`, `Seq2`, and a set of weights is provided. A mutation of the sequences `Seq2` and it's offset is need to be found, which produce the `MAX` or `MIN` score (will be given as an input as well).  


## Solution
I initially implemented a basic iterative solution. It is possible to solve the problem sequentially by iterating over the offsets and the letters for each offset. Compare each pair of letters to see if they are equal or if they fall into one of the conservative or semi-conservative groups. If a substitution is possible for a given pair, and the result is higher than that for the original pair, then save the results. Then, for each offset, the best mutation is provided, and the optimal is returned.  

Having written the sequential solution, I realized it would be time-wasting to check whether each pair of letters belongs to a conservative or semi-conservative group several times during the run. Although iterations over the groups are non-linear ($O(1) $) (since the number of groups and letters in each group is constant), the groups are given ahead of time, so each evaluation of two letters can be done before the code is run, saving a significant amount of time.  
Consequently, I created a hashtable of 26 letters and one `-` character (27 X 27). This still takes $ O(1) $ to evaluate each pair, but is significantly faster than the prior approach.  
In addition, `OpenMP` can be useful for filling this hashtable in such a way that every cell is independent of every other cell.  
Here is the hashtable (spaces were used instead of `_`):

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