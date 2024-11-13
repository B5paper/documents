#include <stdlib.h>
#include "matmul.h"
#include "timeit.h"

int main()
{
    const int mat_N = 2048;

    int *A, *B, *C;
    A = (int*) malloc(mat_N * mat_N * sizeof(int));
    B = (int*) malloc(mat_N * mat_N * sizeof(int));
    C = (int*) malloc(mat_N * mat_N * sizeof(int));

    for (int i = 0; i < mat_N; ++i)
    {
        for (int j = 0; j < mat_N; ++j)
        {
            A[i * mat_N + j] = rand() % 5;
            B[i * mat_N + j] = rand() % 5;
        }
    }

    float fsecs;
    timeit(TIMEIT_START, NULL);
    matmul_i32(A, B, C, mat_N, mat_N, mat_N);
    timeit(TIMEIT_END, NULL);
    timeit(TIMEIT_GET_SEC, &fsecs);
    printf("cpu 1 process, time consumption: %.2f secs\n", fsecs);

    return 0;
}
