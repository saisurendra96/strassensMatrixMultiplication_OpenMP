#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <omp.h>
#include <vector>

#ifndef VERBOSE
#define VERBOSE 0			// VERBOSE to print output 
#endif

int nlimit;
int num_threads;

using namespace std;

//Initialization to allocate memory
int** initializeMatrix(int n) {
    int** temp = new int* [n];
    for (int i = 0; i < n; i++)
        temp[i] = new int[n];
    return temp;
}

// Function to add the matrices
int** add(int** matrix1, int** matrix2, int n) {
    int** temp = initializeMatrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = matrix1[i][j] + matrix2[i][j];
    return temp;
}

// Function to subtract matrices
int** subtract(int** matrix1, int** matrix2, int n) {
    int** temp = initializeMatrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = matrix1[i][j] - matrix2[i][j];
    return temp;
}

// Printing the matrix for debug
void printMatrix(int** M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << M[i][j] << " ";
        std::cout << std::endl;
    }
    cout << endl;
}

// logic implememtation for multiplication of matrices without parallelization
int** strassen_serial(int** A, int** B, int n, int kprime) {
    if (n == nlimit) { //If threshold value is equal to n we do bruteforce
        int** C = initializeMatrix(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = 0;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++) 
                    C[i][j] += A[i][k] * B[k][j];

        return C;
    }
  

    int k = n / 2; // next level size

    int** A11 = initializeMatrix(k);
    int** A12 = initializeMatrix(k);
    int** A21 = initializeMatrix(k);
    int** A22 = initializeMatrix(k);
    int** B11 = initializeMatrix(k);
    int** B12 = initializeMatrix(k);
    int** B21 = initializeMatrix(k);
    int** B22 = initializeMatrix(k);
    int** C = initializeMatrix(n);
    
    // Assigning their respective values
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][k + j];
            A21[i][j] = A[k + i][j];
            A22[i][j] = A[k + i][k + j];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][k + j];
            B21[i][j] = B[k + i][j];
            B22[i][j] = B[k + i][k + j]; 
        }
        
    int** M1 = strassen_serial(add(A11, A22, k), add(B11, B22, k), k, kprime);
    int** M2 = strassen_serial(add(A21, A22, k), B11, k, kprime);
    int** M3 = strassen_serial(A11, subtract(B12, B22, k), k, kprime);
    int** M4 = strassen_serial(A22, subtract(B21, B11, k), k, kprime);
    int** M5 = strassen_serial(add(A11, A12, k), B22, k, kprime);
    int** M6 = strassen_serial(subtract(A21, A11, k), add(B11, B12, k), k, kprime);
    int** M7 = strassen_serial(subtract(A12, A22, k), add(B21, B22, k), k, kprime);

    int** C11 = subtract(add(add(M1, M4, k), M7, k), M5, k);
    int** C12 = add(M3, M5, k);
    int** C21 = add(M2, M4, k);
    int** C22 = subtract(add(add(M1, M3, k), M6, k), M2, k);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            C[i][j] = C11[i][j];
            C[i][j + k] = C12[i][j];
            C[k + i][j] = C21[i][j];
            C[k + i][k + j] = C22[i][j];
        }

    for (int i = 0; i < k; i++) {        //deallocating memory
        delete[] A11[i];
        delete[] A12[i];
        delete[] A21[i];
        delete[] A22[i];
        delete[] B11[i];
        delete[] B12[i];
        delete[] B21[i];
        delete[] B22[i];
        delete[] M3[i];
        delete[] M5[i];
        delete[] M2[i];
        delete[] M4[i];
        delete[] M1[i];
        delete[] M7[i];
        delete[] M6[i];
        delete[] C11[i];
        delete[] C12[i];
        delete[] C21[i];
        delete[] C22[i];
    }

    delete[] A11;
    delete[] A12;
    delete[] A21;
    delete[] A22;
    delete[] B11;
    delete[] B12;
    delete[] B21;
    delete[] B22;
    delete[] C11;
    delete[] C12;
    delete[] C21;
    delete[] C22;
    delete[] M1;
    delete[] M2;
    delete[] M3;
    delete[] M4;
    delete[] M5;
    delete[] M6;
    delete[] M7;

    return C;
}

//logic implementation for multiplication of matrices using parallelization
int** strassen_parallel(int** A, int** B, int n, int kprime) 
{
    if (n == nlimit) {
        int** C = initializeMatrix(n);

        for (int i = 0; i < n; i++)
           for (int j = 0; j < n; j++)
               C[i][j] = 0;

        #pragma omp for
        for (int i = 0; i < n; i++)
           for (int j = 0; j < n; j++)
              for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
              }
            return C;
        }

    int** C = initializeMatrix(n);
    int k = n / 2;

    int** A11 = initializeMatrix(k);
    int** A12 = initializeMatrix(k);
    int** A21 = initializeMatrix(k);
    int** A22 = initializeMatrix(k);
    int** B11 = initializeMatrix(k);
    int** B12 = initializeMatrix(k);
    int** B21 = initializeMatrix(k);
    int** B22 = initializeMatrix(k);

    for (int i = 0; i < k; i++)
       for (int j = 0; j < k; j++) {
           A11[i][j] = A[i][j];
           A12[i][j] = A[i][k + j];
           A21[i][j] = A[k + i][j];
           A22[i][j] = A[k + i][k + j];
           B11[i][j] = B[i][j];
           B12[i][j] = B[i][k + j];
           B21[i][j] = B[k + i][j];
           B22[i][j] = B[k + i][k + j];
       }

    int** M1;
    int** M2;
    int** M3;
    int** M4;
    int** M5;
    int** M6;
    int** M7;
    int** C11;
    int** C12;
    int** C21;
    int** C22;

 // parallelization code 

#pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
                {
                    M1 = strassen_parallel(add(A11, A22, k), add(B11, B22, k), k, kprime);
                }
            #pragma omp task
                {
                    M2 = strassen_parallel(add(A21, A22, k), B11, k, kprime);
                }
            #pragma omp task
                {
                    M3 = strassen_parallel(A11, subtract(B12, B22, k), k, kprime);
                }
            #pragma omp task
                {
                    M4 = strassen_parallel(A22, subtract(B21, B11, k), k, kprime);
                }
            #pragma omp task
                {
                    M5 = strassen_parallel(add(A11, A12, k), B22, k, kprime);
                }
            #pragma omp task
                {
                    M6 = strassen_parallel(subtract(A21, A11, k), add(B11, B12, k), k, kprime);
                }
            #pragma omp task
                {
                    M7 = strassen_parallel(subtract(A12, A22, k), add(B21, B22, k), k, kprime);
                } 
            #pragma omp taskwait
        }

        #pragma omp single
                {
                #pragma omp task
                    {
                        C11 = subtract(add(add(M1, M4, k), M7, k), M5, k);
                    }
                #pragma omp task
                    {
                        C12 = add(M3, M5, k);
                    }
                #pragma omp task
                    {
                        C21 = add(M2, M4, k);
                    }
                #pragma omp task
                    {
                        C22 = subtract(add(add(M1, M3, k), M6, k), M2, k);
                    }
                }
    }
                
     for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            C[i][j] = C11[i][j];
            C[i][j + k] = C12[i][j];
            C[k + i][j] = C21[i][j];
            C[k + i][k + j] = C22[i][j];
        }

     if (VERBOSE) {
         cout << "---------------------------------------------------------------------------------------------------- " << endl;
         printMatrix(C, k);
     }
     //Deallocating memory
     for (int i = 0; i < k; i++) {
            delete[] A11[i];
            delete[] A12[i];
            delete[] A21[i];
            delete[] A22[i];
            delete[] B11[i];
            delete[] B12[i];
            delete[] B21[i];
            delete[] B22[i];
            delete[] C11[i];
            delete[] C12[i];
            delete[] C21[i];
            delete[] C22[i];
            delete[] M1[i];
            delete[] M2[i];
            delete[] M3[i];
            delete[] M4[i];
            delete[] M5[i];
            delete[] M6[i]; 
            delete[] M7[i];
     }

     delete[] A11;
     delete[] A12;
     delete[] A21;
     delete[] A22;
     delete[] B11;
     delete[] B12;
     delete[] B21;
     delete[] B22;
     delete[] C11;
     delete[] C12;
     delete[] C21;
     delete[] C22;
     delete[] M1;
     delete[] M2;
     delete[] M3;
     delete[] M4;
     delete[] M5;
     delete[] M6;
     delete[] M7;
     
     return C;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        printf("Usage: ./<executable_name> <k> <k'> <number of threads>\n");
        exit(0);
    }

    int k,n;
    int kprime;
    struct timespec start, stop_parallel_strassen, stop_strassen;
    double strassen_parallel_time, strassen_serial_time;

    k = atoi(argv[argc - 3]);
    n = pow(2, k);
    kprime =  atoi(argv[argc - 2]);
    num_threads = atoi(argv[argc - 1]);
    nlimit = n / pow(2, kprime);
    
    if (n < pow(2, kprime)) { cout << "k should be greater than k'\n"; exit(0); }

    int** A = initializeMatrix(n);
    int** B = initializeMatrix(n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 100;
                B[i][j] = rand() % 100;
        }

    if (VERBOSE) {
        cout << "Matrix A:" << endl;
        printMatrix(A, n);

        cout << "Matrix B:" << endl;
        printMatrix(B, n);
    }

    int** C = initializeMatrix(n);
    int** serial_strassan = initializeMatrix(n);

    omp_set_num_threads(num_threads);

    // Compute time taken
    clock_gettime(CLOCK_REALTIME, &start);
    C = strassen_parallel(A, B, n, kprime);
    clock_gettime(CLOCK_REALTIME, &stop_parallel_strassen);
    strassen_parallel_time = (stop_parallel_strassen.tv_sec - start.tv_sec) + 0.000000001 * (stop_parallel_strassen.tv_nsec - start.tv_nsec);

    serial_strassan = strassen_serial(A, B, n, kprime);
    clock_gettime(CLOCK_REALTIME, &stop_strassen);
    strassen_serial_time = (stop_strassen.tv_sec - stop_parallel_strassen.tv_sec) + 0.000000001 * (stop_strassen.tv_nsec - stop_parallel_strassen.tv_nsec);

    if (VERBOSE) {
        cout << "Multipliction result:" << std::endl;
        printMatrix(C, n);
    }
    
    int error = 0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (C[j][i] != serial_strassan[j][i]) error = error +  1;
        }
    }

    if (error != 0) {
        printf("Houston, we have a problem!\n");
    }

    //printf("Matrix Size = %d x %d, k=%d, k'=%d, error = %d, threads = %d, strassen_parallel_time (sec) = %8.4f, strassen_serial_time (sec) = %8.4f, bruteforce_multiplication_time = %8.4f\n", n, n, k, kprime, error,num_threads, strassen_parallel_time, strassen_serial_time, brute_force_time);
    printf("Matrix Size = %d x %d, k=%d, k'=%d, error = %d, threads = %d, strassen_parallel_time (sec) = %8.4f, strassen_serial_time (sec) = %8.4f\n", n, n, k, kprime, error, num_threads, strassen_parallel_time, strassen_serial_time);


    for (int i = 0; i < n; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
        delete[] serial_strassan[i];
        //delete[] brute_force_result[i];
    }
    delete[] A;        
    delete[] B;
    delete[] C;    
    delete[] serial_strassan;
    //delete[] brute_force_result;

    return 0;
}