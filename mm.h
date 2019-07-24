/******************************************************************************************************
  June 13, 2019
  Bei Wang
  Princeton University
  beiwang@princeton.edu

  Sample Matrix-Matrix multiplication code for the CoDas-HEP Summer School 2019
  (This sample code is based on the CoDaS-HEP Summer School 2018 example by Ian A. Cosden. 
   Please see https://github.com/cosden/CoDaS-HEP-Perf-Tuning for details.)    
/******************************************************************************************************/

#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
using namespace std;

#ifdef NAIVE
//NAIVE: 2D matrix-matrix multiplication
__attribute__((noinline)) void compute_naive(double **A, double **B, double **C, int matrix_size) {
  for (int i = 0 ; i < matrix_size; i++) {
    for (int j = 0;  j < matrix_size; j++) {
      for (int k = 0; k < matrix_size; k++) {
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

#elif INTERCHANGE
//INTERCHANGE: 2D matrix-matrix multiplication
__attribute__((noinline)) void compute_interchange(double **A, double **B, double **C, int matrix_size) {
  for (int i = 0 ; i < matrix_size; i++) {
    for (int k = 0; k < matrix_size; k++) {
      for (int j = 0;  j < matrix_size; j++) {
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

#elif TRIANGULAR
//TRIANGULAR: only compute the result for the upper triangular
__attribute__((noinline)) void compute_triangular(double **A, double **B, double **C, int matrix_size) {
#pragma omp parallel for
  for (int i = 0 ; i < matrix_size; i++) {
    for (int j = 0;  j < matrix_size-i; j++) {
      for (int k = 0; k < matrix_size; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
#endif

//memory allocation for 2D matrixes
__attribute__((noinline)) void create_matrix_2D(double **&A, double **&B, double **&C, int matrix_size) {

  A = new double*[matrix_size];
  B = new double*[matrix_size];
  C = new double*[matrix_size];

  for (int i = 0 ; i < matrix_size; i++) {
    A[i] = new double[matrix_size];
    B[i] = new double[matrix_size];
    C[i] = new double[matrix_size];
  }
}

//initialize 2D matrixes
__attribute__((noinline)) void init_matrix_2D(double **A, double **B, double **C, int matrix_size){
  for (int i=0; i<matrix_size; i++) {
    for (int j = 0 ; j < matrix_size; j++) {
      A[i][j]=((double) rand() / (RAND_MAX));
      B[i][j]=((double) rand() / (RAND_MAX));
      C[i][j]=0.0;
    }
  }
}

//free 2D matrixes
__attribute__((noinline)) void free_matrix_2D(double **A, double **B, double **C, int matrix_size) {
  for (int i = 0 ; i < matrix_size; i++)  {
    delete A[i];
    delete B[i];
    delete C[i];
  }
  delete A;
  delete B;
  delete C;
}


//function to print a few elements to std out to use as a check (2D version)
__attribute__((noinline)) void print_check(double **Z, int matrix_size) {

  cout<<Z[0][0]<<" "<<Z[1][1]<<" "<<Z[2][2]<<endl;
}

//zeros all elements of input array (2D version)
__attribute__((noinline)) void zero_result(double **C, int matrix_size) {
  for (int i = 0 ; i < matrix_size; i++) {
    for (int j = 0;  j < matrix_size; j++) {
        C[i][j] = 0.0;
    }
  }
}
