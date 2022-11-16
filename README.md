# Sparse-Matrix-Multiplication

## Introduction

 A sparse matrix is defined as a matrix that contains less than 50% non-zero values: meaning that the majority of the values contained in the matrix are zeros. This is important when thinking about large graphs or networks where there are many nodes and little connectivity between all the nodes such as a social network; however, storing these in traditional matrices can be memory intensive as the zero values take up an excessive amount of memory which is why structures such as CSR’s are used to represent sparse matrices.

A CSR is usually defined with 5 fields: num_rows, num_cols, index array,  value array, and pointer array. Obviously, num_rows and num_cols represent the rows and columns of the matrix. The pointer array has a length of num_rows + 1 and is defined as the pair of indices of index_array and value array that represent a specific row. Moreover, the index array and value array both represent the column index of a non-zero value in the matrix. In this paper, A and B will be the input matrices while C is the output matrix.

When parallelizing the matrix multiplication, I decided to parallelize using the sparse dense matrix multiplication method where I dedicated each thread to a block of A, therefore each thread computes the CSR for a subset of C.  When calculating each block, the dense row of matrix A is first created, then each column for a row of matrix B is multiplied by its corresponding column in A which allows for O(1) lookup time. After the sum is calculated, it is appended to the subset CSR. After each subset CSR is calculated, they are all aggregated into one matrix giving the final output C. A full pseudo-code breakdown can be found in Figure 1.

**Figure 1: Parallel Sparse Matrix Multiplication  Pseudo-code.**

```
Matrix A, B, and C

For r_start = 0; r_start< A.num_rows; r_start += block_size
r_end = r_start + block if  r_start + block < A.num_rows else A.num_rows

#pragma omp parallel for
for (r = r_start; r < r_end; r++)
C_nnz, idx_value_index, ptr_idx = 0, 0, 0
	A_dense_vector = {0,0,.......}[A.num_cols]
	for idx = A.pointer_array[r]; idx <  A.pointer_array[r+1]; r++
		A_dense_vector[ A.index_array[idx] ] = A.value_array[idx]

 
for c = 0; c < B_transposed.num_rows; c++
	val = 0
	C_nnz += 1
	for n = B.pointer_array[c]; n < B.pointer_array[c+1]; n++
		val += A_dense_vector[ B.index_array[n] ] * B.value_array[n]
	If val == 0
		Continue
	Else
		C.index_array[idx_val_index] = c
C.value_array[idx_val_index] = val
idx_val_index ++
		subset_C[block_idx] = C

	C = New CSR
	For each c’ in subset_C;
		Add c’ to C in order
Return C
```

## Experiments

A combination of parameters were tested leading to 90 experiments being conducted  where each experiment calculated the time it took the sparse matrix multiplication to complete. The experiment included testing different fill factors(which refer to the percentage of non-zeros in the matrices), thread counts, and matrix sizes.

**Table 1: Experiment Combinations**

| Fill Factor                       | **0.05, 0.10, 0.15, 0.20**                                                              | 4            |
| --------------------------------- | --------------------------------------------------------------------------------------------- | ------------ |
| **Thread Count**            | **1, 2, 4, 8, 12, 14, 16, 20, 24, 28**                                                  | **10** |
| **Matrixes Sizes**          | **10000x10000*10000x10000, <br />20000x5300*5300x50000, <br />9000x35000*35000x5750** | 3            |
| **Total Combination Count** | 4x*10*x3=                                                                                   | 120          |

After getting each experiment’s time, the speedup was calculated using the formula speedup =serial time(1 thread)  n thread time  and the results can be seen in Tables 2 through 4.  The raw execution results can also be found in the appeidices in Tables 5 through 7

**Table 2: 10000x10000*10000x10000 Matrix Speedup Time for Thread Count Vs Fill Factor compared to Sequential (1 Thread)**

| Thread Count/  Fill Factor | 0.05  | 0.10  | 0.15  | 0.20  | Avg Fill Factor |
| -------------------------- | ----- | ----- | ----- | ----- | --------------- |
| 1                          | 1.00  | 1.00  | 1.00  | 1.00  | 1.00            |
| 2                          | 1.89  | 1.87  | 2.04  | 2.00  | 1.95            |
| 4                          | 3.64  | 3.68  | 3.97  | 3.88  | 3.79            |
| 8                          | 7.19  | 7.29  | 7.80  | 7.72  | 7.50            |
| 12                         | 10.54 | 10.81 | 11.66 | 11.51 | 11.13           |
| 14                         | 12.17 | 12.63 | 13.52 | 13.37 | 12.92           |
| 26                         | 13.83 | 14.41 | 15.42 | 15.28 | 14.73           |
| 20                         | 11.81 | 11.92 | 13.18 | 12.75 | 12.41           |
| 24                         | 13.88 | 13.82 | 15.23 | 15.04 | 14.49           |
| 28                         | 13.85 | 14.38 | 15.62 | 15.21 | 14.76           |
| Avg Thread                 | 8.98  | 9.18  | 9.94  | 9.78  | 9.47            |

**Table 3: 20000x5300*5300x50000 Matrix Speedup Time for Thread Count Vs Fill Factor compared to Sequential (1 Thread)**

| Thread Count/  Fill Factor | 0.05  | 0.10  | 0.15  | 0.20  | Avg Fill Factor |
| -------------------------- | ----- | ----- | ----- | ----- | --------------- |
| 1                          | 1.00  | 1.00  | 1.00  | 1.00  | 1.00            |
| 2                          | 1.89  | 1.87  | 2.04  | 2.00  | 1.95            |
| 4                          | 3.64  | 3.68  | 3.97  | 3.88  | 3.79            |
| 8                          | 7.19  | 7.29  | 7.80  | 7.72  | 7.50            |
| 12                         | 10.54 | 10.81 | 11.66 | 11.51 | 11.13           |
| 14                         | 12.17 | 12.63 | 13.52 | 13.37 | 12.92           |
| 26                         | 13.83 | 14.41 | 15.42 | 15.28 | 14.73           |
| 20                         | 11.81 | 11.92 | 13.18 | 12.75 | 12.41           |
| 24                         | 13.88 | 13.82 | 15.23 | 15.04 | 14.49           |
| 28                         | 13.85 | 14.38 | 15.62 | 15.21 | 14.76           |
| Avg Thread                 | 8.98  | 9.18  | 9.94  | 9.78  | 9.47            |

**Table 4: 9000x35000*35000x5750 Matrix Speedup Time for Thread Count Vs Fill Factor compared to Sequential (1 Thread)**

| Thread Count/  Fill Factor | 0.05  | 0.10  | 0.15  | 0.20  | Avg Fill Factor |
| -------------------------- | ----- | ----- | ----- | ----- | --------------- |
| 1                          | 1.00  | 1.00  | 1.00  | 1.00  | 1.00            |
| 2                          | 1.89  | 1.87  | 2.04  | 2.00  | 1.95            |
| 4                          | 3.64  | 3.68  | 3.97  | 3.88  | 3.79            |
| 8                          | 7.19  | 7.29  | 7.80  | 7.72  | 7.50            |
| 12                         | 10.54 | 10.81 | 11.66 | 11.51 | 11.13           |
| 14                         | 12.17 | 12.63 | 13.52 | 13.37 | 12.92           |
| 26                         | 13.83 | 14.41 | 15.42 | 15.28 | 14.73           |
| 20                         | 11.81 | 11.92 | 13.18 | 12.75 | 12.41           |
| 24                         | 13.88 | 13.82 | 15.23 | 15.04 | 14.49           |
| 28                         | 13.85 | 14.38 | 15.62 | 15.21 | 14.76           |
| Avg Thread                 | 8.98  | 9.18  | 9.94  | 9.78  | 9.47            |

## Analysis

Overall, a greater number of threads provided a performance boost compared to the serial code as each thread was able to compute a subset of C  parallelly. The 9000x35000*35000x5750 matrix multiplication performed the best on average with a speedup of 9.70 while 10000x10000*10000x10000 multiplication followed at a close second with an average speedup of 9.41. Lastly, 20000x5300*5300x50000 matrix multiplication had the worse average speedup of 6.61.

Since the program, uses a vector of size A.num_columns as a performance boost to looking up values in matrix A, it follows that matrix multiplications that have larger column and row sizes for matrix A and B respectively will have a greater performance boost from parallelization. Matrix multiplication that has a large row size of A while having a small column size will continually allocate and decollate memory for the dense vector while not being able to utilize it to its full advantage. This is why both  9000x35000*35000x5750 and 10000x10000*10000x10000 matrix multiplications outperformed the 20000x5300*5300x50000 matrix multiplication as they have larger column and row sizes for matrices A and B.

Moreover as seen in figure 2, more threads do not necessarily mean better performance. In the case of 9000x35000*35000x5750 and 10000x10000*10000x10000 multiplication, although a major performance increase was found ranging from going from 1 thread to 16 threads for the 2 matrices mentioned above, both times there was a major degradation in performance at 20 threads. This was very strange because the performance rose afterward for both 24 and 28 threads: this seems to be unexplainable with the current knowledge and more investigation should be done. However, performance always increased for 20000x5300*5300x50000, although, at a slower pace.

**Figure 2: Strong Scaling Chart of Average, in terms of Fill Factor, Speedup for all 3 Matrices**
`<img src="https://github.com/jayteaftw/Sparse-Matrix-Multiplication/blob/master/imgs/figure2.png" height="700" />`

Furthermore as seen in figure 3, the program is not scalable in terms of threads.  The rate of change of speedup decreases stagnates, and or fluctuates when increasing the number of threads.  These diminishing returns show that the greatest performance gains are made around 8 threads.

**Figure 3: Rate of Change Strong Scaling Chart of Average, in terms of Fill Factor, Speedup for all 3 Matrices**
`<img src="https://github.com/jayteaftw/Sparse-Matrix-Multiplication/blob/master/imgs/figure3.png" height="700" />`

When examining, the strong scaling chart for the average speedup in terms of thread count, we can see that there are hardly any gains being made with both the speedup and the speedup’s rate of change being overall constant as seen in figures 4 and 5. Furthermore, both 9000x35000*35000x5750 and 10000x10000*10000x10000 matrix multiplications outperformed the 20000x5300*5300x50000 matrix multiplication which follows since the latter matrix multiplication is much larger than the former matrices, and therefore there is an increases the number of computations and overall execution time of the program. Also as discussed before, the program favors matrices with larger column sizes for matrix A which could be another factor of why the 20000x5300*5300x50000 matrix multiplication performed noticeably worse.

**Figure 4: Strong Scaling Chart of Average, in terms of Thread Count, Speedup for all 3 Matrices**
`<img src="https://github.com/jayteaftw/Sparse-Matrix-Multiplication/blob/master/imgs/figure4.png" height="500" />`

**Figure 5: Rate of Change Strong Scaling Chart of Average, in terms of Thread Count, Speedup for all 3 Matrices**
`<img src="https://github.com/jayteaftw/Sparse-Matrix-Multiplication/blob/master/imgs/figure5.png" height="500" />`

## Appendicies

**Table 5: Raw 10000x10000*10000x10000 Matrix Execution Time**

| Thread Count/  Fill Factor | 0.05           | 0.1            | 0.15           | 0.2            | Avg Fill Factor |
| -------------------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| 1                          | 158,900,000.00 | 317,200,000.00 | 501,500,000.00 | 633,900,000.00 | 1.00            |
| 2                          | 84,080,000.00  | 169,200,000.00 | 246,300,000.00 | 316,700,000.00 | 1.96            |
| 4                          | 43,640,000.00  | 86,270,000.00  | 126,300,000.00 | 163,200,000.00 | 3.82            |
| 8                          | 22,100,000.00  | 43,530,000.00  | 64,290,000.00  | 82,070,000.00  | 7.62            |
| 12                         | 15,070,000.00  | 29,350,000.00  | 43,020,000.00  | 55,060,000.00  | 11.40           |
| 14                         | 13,060,000.00  | 25,110,000.00  | 37,090,000.00  | 47,400,000.00  | 13.26           |
| 16                         | 11,490,000.00  | 22,020,000.00  | 32,520,000.00  | 41,480,000.00  | 15.14           |
| 20                         | 13,460,000.00  | 26,620,000.00  | 38,050,000.00  | 49,730,000.00  | 12.71           |
| 24                         | 11,450,000.00  | 22,950,000.00  | 32,930,000.00  | 42,140,000.00  | 14.87           |
| 28                         | 11,470,000.00  | 22,060,000.00  | 32,110,000.00  | 41,690,000.00  | 15.24           |
| Avg Thread                 | 9.95           | 9.62           | 9.55           | 9.68           | 9.70            |


**Table 6: Raw 20000x5300*5300x50000 Matrix Execution Time**

| Thread Count/  Fill Factor | 0.05           | 0.1            | 0.15           | 0.2            | Avg Fill Factor |
| -------------------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| 1                          | 158,900,000.00 | 317,200,000.00 | 501,500,000.00 | 633,900,000.00 | 1.00            |
| 2                          | 84,080,000.00  | 169,200,000.00 | 246,300,000.00 | 316,700,000.00 | 1.96            |
| 4                          | 43,640,000.00  | 86,270,000.00  | 126,300,000.00 | 163,200,000.00 | 3.82            |
| 8                          | 22,100,000.00  | 43,530,000.00  | 64,290,000.00  | 82,070,000.00  | 7.62            |
| 12                         | 15,070,000.00  | 29,350,000.00  | 43,020,000.00  | 55,060,000.00  | 11.40           |
| 14                         | 13,060,000.00  | 25,110,000.00  | 37,090,000.00  | 47,400,000.00  | 13.26           |
| 16                         | 11,490,000.00  | 22,020,000.00  | 32,520,000.00  | 41,480,000.00  | 15.14           |
| 20                         | 13,460,000.00  | 26,620,000.00  | 38,050,000.00  | 49,730,000.00  | 12.71           |
| 24                         | 11,450,000.00  | 22,950,000.00  | 32,930,000.00  | 42,140,000.00  | 14.87           |
| 28                         | 11,470,000.00  | 22,060,000.00  | 32,110,000.00  | 41,690,000.00  | 15.24           |
| Avg Thread                 | 9.95           | 9.62           | 9.55           | 9.68           | 9.70            |


**Table 7: Raw 9000x35000*35000x5750 Matrix Execution Time**

| Thread Count/  Fill Factor | 0.05           | 0.1            | 0.15           | 0.2            | Avg Fill Factor |
| -------------------------- | -------------- | -------------- | -------------- | -------------- | --------------- |
| 1                          | 158,900,000.00 | 317,200,000.00 | 501,500,000.00 | 633,900,000.00 | 1.00            |
| 2                          | 84,080,000.00  | 169,200,000.00 | 246,300,000.00 | 316,700,000.00 | 1.96            |
| 4                          | 43,640,000.00  | 86,270,000.00  | 126,300,000.00 | 163,200,000.00 | 3.82            |
| 8                          | 22,100,000.00  | 43,530,000.00  | 64,290,000.00  | 82,070,000.00  | 7.62            |
| 12                         | 15,070,000.00  | 29,350,000.00  | 43,020,000.00  | 55,060,000.00  | 11.40           |
| 14                         | 13,060,000.00  | 25,110,000.00  | 37,090,000.00  | 47,400,000.00  | 13.26           |
| 16                         | 11,490,000.00  | 22,020,000.00  | 32,520,000.00  | 41,480,000.00  | 15.14           |
| 20                         | 13,460,000.00  | 26,620,000.00  | 38,050,000.00  | 49,730,000.00  | 12.71           |
| 24                         | 11,450,000.00  | 22,950,000.00  | 32,930,000.00  | 42,140,000.00  | 14.87           |
| 28                         | 11,470,000.00  | 22,060,000.00  | 32,110,000.00  | 41,690,000.00  | 15.24           |
| Avg Thread                 | 9.95           | 9.62           | 9.55           | 9.68           | 9.70            |
