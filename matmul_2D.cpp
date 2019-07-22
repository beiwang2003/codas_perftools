/******************************************************************************************************
  June 13, 2019

  Bei Wang
  Princeton University
  beiwang@princeton.edu

  Sample Matrix-Matrix multiplication code for the CoDas-HEP Summer School 2019

  (This sample code is based on the CoDaS-HEP Summer School 2018 example by Ian A. Cosden.
   Please see https://github.com/cosden/CoDaS-HEP-Perf-Tuning for details.)

  Purpose:
     To use as a example to profile with performance tuning tools.
     The code does not do anything useful and is for illustrative/educational
     use only.  It is not meant to be exhaustive or demonstrating optimal
     matrix-matrix multiplication techniques.

  Description:
     Code generates two matrices with random numbers.  They are stored in 2D
     arrays (named A and B). The result is stored in a 2D array (named C).
     
     NAIVE: the naive approach of perforing matrix-matrix multiplication with i,j,k order
     INTERCHANGE: performing matrix-matrix multiplication with i,k,j order 
     TRIANGULAR: only compute the result for the upper triangular 

  Compile:
     Intel compiler:
     module load intel 
     icpc -g -O3 -xhost -DNAIVE matmul_2D.cpp -o mm_naive.out

     GCC compiler:
     module load rh/devtoolset/7
     g++ -g -O3 -march=native -DNAIVE matmul_2D.cpp -o mm_naive.out

     In the case of multithreading:
     Intel compiler: -qopenmp
     GCC compiler: -fopenmp

/*******************************************************************************************************/

#include "mm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{
  int matrix_size; //N*N matrix
  int max_iters=10; //number of times to call a matrix-matrix function    

  //read command line input
  //set various paramaters
  if(argc<2) {
    cout<<"ERROR: expecting integer matrix size, i.e., N for NxN matrix"<<endl;
    exit(1);
  }
  else {
    matrix_size=atoi(argv[1]);
  }

  cout<<"using matrix size:"<<matrix_size<<endl;
 
  double **A, **B, **C; //2D arrays

  // create memory space for 2D matrixes
  create_matrix_2D(A, B, C, matrix_size);

  // initialize matrixes
  init_matrix_2D(A, B, C, matrix_size);

  for (int r=0; r < max_iters; r++) {
    zero_result(C,matrix_size);
#ifdef NAIVE
    compute_naive(A,B,C,matrix_size);
#elif INTERCHANGE
    compute_interchange(A,B,C,matrix_size);
#elif TRIANGULAR
    compute_triangular(A,B,C,matrix_size);
#endif
  }

  // free memeory space 
  free_matrix_2D(A, B, C, matrix_size);

  return 0;
}


