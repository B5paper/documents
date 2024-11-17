#ifndef MEM_DISP_H
#define MEM_DISP_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void print_vec_cuf32(void *cubuf, int num_elm)
{
    float *buf = (float*) malloc(num_elm * sizeof(float));
    cudaError_t cu_ret;
    cu_ret = cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
    if (cu_ret != cudaSuccess)
    {
        printf("fail to cuda memcpy\n");
        exit(-1);
    }
    for (int i = 0; i < num_elm; ++i)
    {
        printf("%.1f, ", buf[i]);
    }
    putchar('\n');
    free(buf);
}

void print_vec_f32(float *buf, int num_elm)
{
    for (int i = 0; i < num_elm; ++i)
    {
        printf("%.1f, ", buf[i]);
    }
    putchar('\n');
}

#endif