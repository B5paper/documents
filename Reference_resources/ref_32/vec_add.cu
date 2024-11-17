#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "mem_disp.h"

__global__ void vec_add(float *A, float *B, float *C, int num_elm)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elm)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int num_elm = 8;
    float *buf_A = (float*) malloc(num_elm * sizeof(float));
    float *buf_B = (float*) malloc(num_elm * sizeof(float));
    float *buf_C = (float*) malloc(num_elm * sizeof(float));
    float *buf_C_ref = (float*) malloc(num_elm * sizeof(float));
    for (int i = 0; i < num_elm; ++i)
    {
        buf_A[i] = rand() % 5;
        buf_B[i] = rand() % 5;
    }
    
    printf("buf_A:\n");
    print_vec_f32(buf_A, num_elm);
    printf("buf_B:\n");
    print_vec_f32(buf_B, num_elm);
    for (int i = 0; i < num_elm; ++i)
    {
        buf_C_ref[i] = buf_A[i] + buf_B[i];
    }
    printf("buf_C_ref:\n");
    print_vec_f32(buf_C_ref, num_elm);
    putchar('\n');

    // cudaError_t cu_ret;
    float *cubuf_A, *cubuf_B, *cubuf_C;
    cudaMalloc(&cubuf_A, num_elm * sizeof(float));
    cudaMalloc(&cubuf_B, num_elm * sizeof(float));
    cudaMalloc(&cubuf_C, num_elm * sizeof(float));
    cudaMemcpy(cubuf_A, buf_A, num_elm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cubuf_B, buf_B, num_elm * sizeof(float), cudaMemcpyHostToDevice);
    vec_add<<<num_elm, 1>>>(cubuf_A, cubuf_B, cubuf_C, num_elm);
    cudaMemcpy(buf_C, cubuf_C, num_elm * sizeof(float), cudaMemcpyDeviceToHost);

    printf("cubuf_A:\n");
    print_vec_cuf32(cubuf_A, num_elm);
    printf("cubuf_B:\n");
    print_vec_cuf32(cubuf_B, num_elm);
    printf("cubuf_C:\n");
    print_vec_cuf32(cubuf_C, num_elm);

    cudaFree(cubuf_A);
    cudaFree(cubuf_B);
    cudaFree(cubuf_C);
    free(buf_A);
    free(buf_B);
    free(buf_C);
    free(buf_C_ref);
    return 0;
}