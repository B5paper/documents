#include <openmpi/shmem.h>
#include <stdlib.h>
#include <stdio.h>
#include "matmul.h"
#include "timeit.h"

void print_mat(int *M, int n_row, int n_col)
{
    for (int i = 0; i < n_row; ++i)
    {
        for (int j = 0; j < n_col; ++j)
        {
            // if (j != n_col - 1)
                printf("%d, ", M[i * n_col + j]);
            // else
            //     printf("%d", M[i * n_col + j]);
        }
        putchar('\n');
    }
}

void print_vec(int *V, int len)
{
    for (int i = 0; i < len; ++i)
    {
        if (i != len - 1)
            printf("%d, ", V[i]);
        else
            printf("%d", V[i]);
    }
    putchar('\n');
}

void print_vec_with_headline(char *headline, int *V, int len)
{
    puts(headline);
    print_vec(V, len);
    putchar('\n');
}

int compare_arr_i32(int *A, int *B, int len)
{
    // all equal: return 0
    // else: return -1

    for (int i = 0; i < len; ++i)
    {
        if (A[i] != B[i])
        {
            printf("not equal, A[%d] = %d, B[%d] = %d\n", i, i, A[i], B[i]);
            return -1;
        }
    }
    return 0;
}

long pSync[SHMEM_BCAST_SYNC_SIZE];

int main()
{
    shmem_init();
    int my_pe = shmem_my_pe();
    int n_pes = shmem_n_pes();

    const int mat_N = 2048;

    if (mat_N % n_pes != 0)
    {
        printf("mat_N %% n_pes != 0, mat_N = %d, n_pes = %d\n", mat_N, n_pes);
        return -1;
    }

    int *A, *B, *C, *C_ref;
    if (my_pe == 0)
    {
        A = (int*) malloc(mat_N * mat_N * sizeof(int));
        B = (int*) malloc(mat_N * mat_N * sizeof(int));
        C = (int*) malloc(mat_N * mat_N * sizeof(int));
        C_ref = (int*) malloc(mat_N * mat_N * sizeof(int));

        for (int i = 0; i < mat_N; ++i)
        {
            for (int j = 0; j < mat_N; ++j)
            {
                A[i * mat_N + j] = rand() % 5;
                B[i * mat_N + j] = rand() % 5;
            }
        }
        // print_mat_with_headline("A:", A, mat_N, mat_N);
        // print_mat_with_headline("B:", B, mat_N, mat_N);
    }

    for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; ++i)
        pSync[i] = SHMEM_SYNC_VALUE;

    int A_pe_nrow = mat_N / n_pes;

    int *A_pe = shmem_malloc(A_pe_nrow * mat_N * sizeof(int));
    int *B_pe = shmem_malloc(mat_N * mat_N * sizeof(int));
    int *C_pe = shmem_malloc(A_pe_nrow * mat_N * sizeof(int));
    
    timeit(TIMEIT_START, NULL);
    if (my_pe == 0)
    {
        for (int i = 0; i < n_pes; ++i)
        {
            shmem_put32(A_pe, &A[i * A_pe_nrow * mat_N + 0], A_pe_nrow * mat_N, i);
            shmem_put32(B_pe, B, mat_N * mat_N, i);
        }
    }
    // shmem_get32(A_pe, &A[my_pe * A_pe_nrow * mat_N + 0], A_pe_nrow * mat_N, 0);

    shmem_barrier_all();  // this line is necessary
    // return -1;

    for (int i = 0; i < A_pe_nrow; ++i)
    {
        for (int j = 0; j < mat_N; ++j)
        {
            C_pe[i * mat_N + j] = 0;
            for (int k = 0; k < mat_N; ++k)
            {
                C_pe[i * mat_N + j] += A_pe[i * mat_N + k] * B_pe[k * mat_N + j];
            }
        }
    }
    shmem_barrier_all();

    if (my_pe == 0)
    {
        for (int i = 0; i < n_pes; ++i)
        {
            shmem_get32(&C[i * A_pe_nrow * mat_N + 0], C_pe, A_pe_nrow * mat_N, i);
        }
    }
    shmem_barrier_all();

    timeit(TIMEIT_END, NULL);
    float fsecs;
    timeit(TIMEIT_GET_SEC, &fsecs);

    if (my_pe == 0)
    {
        printf("shmem %d process(es), time consumption: %.2f secs\n", n_pes, fsecs);
    }

    // if (my_pe == 0)
    // {
    //     puts("pe 0, C:");
    //     print_mat(C, mat_N, mat_N);
    //     putchar('\n');
    // }

    // static int ser_val = 0;
    // shmem_int_wait_until(&ser_val, SHMEM_CMP_EQ, my_pe);
    // printf("pe %d, A_pe:\n", my_pe);
    // print_mat(A_pe, A_pe_nrow, mat_N);
    // printf("pe %d, B_pe:\n", my_pe);
    // print_mat(B_pe, mat_N, mat_N);
    // printf("pe %d, C_pe:\n", my_pe);
    // print_mat(C_pe, A_pe_nrow, mat_N);
    // putchar('\n');
    // if (my_pe < n_pes - 1)
    //     shmem_int_add(&ser_val, my_pe + 1, my_pe + 1);
    // shmem_barrier_all();

    // cpu mat mul
    if (my_pe == 0)
    {
        timeit(TIMEIT_START, NULL);
        matmul_i32(A, B, C_ref, mat_N, mat_N, mat_N);
        timeit(TIMEIT_END, NULL);
        timeit(TIMEIT_GET_SEC, &fsecs);
        printf("cpu 1 process, time consumption: %.2f secs\n", fsecs);

        timeit(TIMEIT_START, NULL);
        int ret = compare_arr_i32(C, C_ref, mat_N * mat_N);
        timeit(TIMEIT_END, NULL);
        timeit(TIMEIT_GET_SEC, &fsecs);
        printf("result check, time consumption: %.2f secs\n", fsecs);
        
        if (ret != 0)
            goto ERR_C_ELM;
        printf("all results are correct.\n");
    }
    
ERR_C_ELM:
    shmem_free(A_pe);
    shmem_free(B_pe);
    shmem_free(C_pe);
    
    if (my_pe == 0)
    {
        free(A);
        free(B);
        free(C);
    }

    shmem_finalize();
    return 0;
}
