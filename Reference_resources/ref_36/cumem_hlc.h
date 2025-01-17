#ifndef CUMEM_HLC_H
#define CUMEM_HLC_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <typeinfo>

template<typename T>
void sibling_alloc_buf_assign_rand_int(T **buf, T **cubuf, size_t num_elm, int MAX_RAND_VAL = 5)
{
    *buf = (T*) malloc(sizeof(T) * num_elm);
    cudaMalloc(cubuf, sizeof(T) * num_elm);
    for (size_t i = 0; i < num_elm; ++i)
    {
        (*buf)[i] = rand() % (MAX_RAND_VAL + 1);
    }
    cudaMemcpy(*cubuf, *buf, sizeof(T) * num_elm, cudaMemcpyHostToDevice);
}

template<typename T>
void sibling_free_buf(T *buf, T *cubuf)
{
    free(buf);
    cudaFree(cubuf);
}

void assign_vec_rand_int(float *vec, size_t len)
{
    for (int i = 0; i < len; ++i)
    {
        vec[i] = rand() % 5;
    }
}

template<typename T>
void assign_cubuf_rand_int(T *cubuf, size_t num_elm)
{
    T *buf = (T*) malloc(num_elm * sizeof(T));
    for (size_t i = 0; i < num_elm; ++i)
    {
        buf[i] = rand() % 5;
    }
    cudaMemcpy(cubuf, buf, num_elm * sizeof(T), cudaMemcpyHostToDevice);
    free(buf);
}

void assign_cubuf_rand_int(float *cubuf, size_t num_elm)
{
    float *buf = (float*) malloc(num_elm * sizeof(float));
    for (size_t i = 0; i < num_elm; ++i)
    {
        buf[i] = rand() % 5;
    }
    cudaMemcpy(cubuf, buf, num_elm * sizeof(float), cudaMemcpyHostToDevice);
    free(buf);
}

void print_vec(float *vec, size_t len)
{
    for (int i = 0; i < len; ++i)
    {
        printf("%.1f, ", vec[i]);
    }
    putchar('\n');
}

template<typename T>
void print_cubuf(T *cubuf, size_t num_elm)
{
    T *buf = (T*) malloc(num_elm * sizeof(T));
    cudaMemcpy(buf, cubuf, num_elm * sizeof(T), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_elm; ++i)
    {
        if (typeid(T) == typeid(float))
            printf("%.1f, ", buf[i]);
        else if (typeid(T) == typeid(int))
            printf("%d, ", buf[i]);
        else
        {
            printf("unknown type\n");
            exit(-1);
        }
    }
    putchar('\n');
    free(buf);
}

template<>
void print_cubuf<float>(float *cubuf, size_t num_elm)
{
    using elm_type = float;
    elm_type *buf = (elm_type*) malloc(num_elm * sizeof(elm_type));
    cudaMemcpy(buf, cubuf, num_elm * sizeof(elm_type), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_elm; ++i)
    {
        printf("%.1f, ", buf[i]);
    }
    printf("specialized as float");
    putchar('\n');
    free(buf);
}

template<>
void print_cubuf<int>(int *cubuf, size_t num_elm)
{
    using elm_type = int;
    elm_type *buf = (elm_type*) malloc(num_elm * sizeof(elm_type));
    cudaMemcpy(buf, cubuf, num_elm * sizeof(elm_type), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_elm; ++i)
    {
        printf("%d, ", buf[i]);
    }
    printf("specialized as int");
    putchar('\n');
    free(buf);
}


// void print_cubuf(float *cubuf, size_t num_elm)
// {
//     float *buf = (float*) malloc(num_elm * sizeof(float));
//     cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
//     for (size_t i = 0; i < num_elm; ++i)
//     {
//         printf("%.1f, ", buf[i]);
//     }
//     putchar('\n');
//     free(buf);
// }

void print_mat(float *M, size_t n_rows, size_t n_cols)
{
    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_cols; ++j)
        {
            printf("%.1f, ", M[i * n_cols + j]);
        }
        putchar('\n');
    }
}

void print_mat_cu(float *M, size_t n_rows, size_t n_cols)
{
    float *buf = (float*) malloc(n_rows * n_cols * sizeof(float));
    cudaMemcpy(buf, M, n_rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    print_mat(buf, n_rows, n_cols);
    free(buf);
}

void print_cumat(float *M, size_t n_rows, size_t n_cols)
{
    print_mat_cu(M, n_rows, n_cols);
}

template<typename T>
void matmul_cpu(T *buf_A, T *buf_B, T *buf_C,
    size_t num_rows_A, size_t num_cols_A, size_t num_rows_B, size_t num_cols_B)
{
    if (num_cols_A != num_rows_B)
    {
        printf("shape not match, fail to do matmul");
        return;
    }

    int num_rows_C = num_rows_A;
    int num_cols_C = num_cols_B;

    for (size_t i = 0; i < num_rows_C; ++i)
    {
        for (size_t j = 0; j < num_cols_C; ++j)
        {
            float val = 0;
            for (size_t k = 0; k < num_cols_A; ++k)
            {
                val += buf_A[i * num_cols_A + k] * buf_B[k * num_cols_B + j];
            }
            buf_C[i * num_cols_C + j] = val;
        }
    }
}

template<typename T>
void compare_buf_cubuf(T *buf, T *cubuf, size_t num_elm, bool *all_equal)
{
    T *buf_2 = (T*) malloc(num_elm * sizeof(T));
    cudaMemcpy(buf_2, cubuf, num_elm * sizeof(T), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_elm; ++i)
    {
        if (buf[i] != buf_2[i])
        {
            if (typeid(T) == typeid(float))
            {
                printf("the %d elm not equal, buf_1[%d] = %.1ff, buf_2[%d] = %.1f\n",
                    i, i, buf[i], i, buf_2[i]);
            }
            else if (typeid(T) == typeid(int))
            {
                printf("the %d elm not equal, buf_1[%d] = %d, buf_2[%d] = %d\n",
                    i, i, buf[i], i, buf_2[i]);
            }
            else
            {
                printf("unknown type\n");
                exit(-1);
            }

            *all_equal = false;
            free(buf_2);
            return;
        }
    }
    *all_equal = true;
    free(buf_2);
}

template<>
void compare_buf_cubuf<int>(int *buf, int *cubuf, size_t num_elm, bool *all_equal)
{
    printf("in compare_buf_cubuf(), specialized as int\n");
    using elm_type = int;
    elm_type *buf_2 = (elm_type*) malloc(num_elm * sizeof(elm_type));
    cudaMemcpy(buf_2, cubuf, num_elm * sizeof(elm_type), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_elm; ++i)
    {
        if (buf[i] != buf_2[i])
        {
            printf("the %d elm not equal, buf_1[%d] = %d, buf_2[%d] = %d, specialized as int\n",
                i, i, buf[i], i, buf_2[i]);
            *all_equal = false;
            free(buf_2);
            return;
        }
    }
    *all_equal = true;
    free(buf_2);
}

template<>
void compare_buf_cubuf<float>(float *buf, float *cubuf, size_t num_elm, bool *all_equal)
{
    printf("in compare_buf_cubuf(), specialized as float\n");
    using elm_type = float;
    elm_type *buf_2 = (elm_type*) malloc(num_elm * sizeof(elm_type));
    cudaMemcpy(buf_2, cubuf, num_elm * sizeof(elm_type), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_elm; ++i)
    {
        if (buf[i] != buf_2[i])
        {
            printf("the %d elm not equal, buf_1[%d] = %.2f, buf_2[%d] = %.2f, specialized as float\n",
                i, i, buf[i], i, buf_2[i]);
            *all_equal = false;
            free(buf_2);
            return;
        }
    }
    *all_equal = true;
    free(buf_2);
}

template<typename T>
void compare_buf_buf(T *buf_1, T *buf_2, size_t num_elm, bool *all_equal)
{    
    for (int i = 0; i < num_elm; ++i)
    {
        if (buf_1[i] != buf_2[i])
        {
            printf("the %d-th elm not equal, buf_1[%d] = %.1f, buf_2[%d] = %.1f\n", 
                i, i, buf_1[i], i, buf_2[i]);
            *all_equal = false;
            return;
        }
    }
    *all_equal = true;
}

template<typename T>
void compare_cubuf(T *cubuf_1, T *cubuf_2, size_t num_elm, bool *all_equal)
{
    T *buf_1 = (T*) malloc(num_elm * sizeof(T));
    cudaMemcpy(buf_1, cubuf_1, num_elm * sizeof(T), cudaMemcpyDeviceToHost);
    T *buf_2 = (T*) malloc(num_elm * sizeof(T));
    cudaMemcpy(buf_2, cubuf_2, num_elm * sizeof(T), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < num_elm; ++i)
    {
        if (buf_1[i] != buf_2[i])
        {
            printf("the %d-th elm not equal, buf_1[%d] = %.1ff, buf_2[%d] = %.1f\n",
                i, i, buf_1[i], i, buf_2[i]);
            *all_equal = false;
            free(buf_1);
            free(buf_2);
            return;
        }
    }
    *all_equal = true;

    free(buf_1);
    free(buf_2);
}

#endif