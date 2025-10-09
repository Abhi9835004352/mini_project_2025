#include <stdio.h>
#include <stdlib.h>

// --- Configuration ---
// Increase this value for a longer-running, more intensive benchmark.
// A size of 512 is a good starting point.
#define MATRIX_SIZE 512

// Function to allocate memory for a matrix
double** create_matrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    return matrix;
}

// Function to free the memory of a matrix
void free_matrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    int n = MATRIX_SIZE;

    // 1. Allocate memory for three matrices
    double** A = create_matrix(n);
    double** B = create_matrix(n);
    double** C = create_matrix(n);

    // 2. Initialize matrices A and B with some values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(i + j);
            B[i][j] = (double)(i - j);
            C[i][j] = 0.0;
        }
    }

    // 3. Perform matrix multiplication (C = A * B)
    // This is the computationally intensive part that FOGA will optimize.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // 4. Print a single result to prevent dead code elimination
    // This ensures the compiler must perform the calculation.
    printf("Result checksum: %f\n", C[0][0]);

    // 5. Free allocated memory
    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}