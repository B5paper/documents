#include <stdlib.h>
#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include "mem_disp.h"
#include "timeit.h"

// resources on a cuda device
typedef struct
{
    float *cubuf_A;
    float *cubuf_B;
    cudaStream_t cu_stream;
} CudevRes;

struct CalcRes
{
    int num_elm;
    float *buf_A;
    float *buf_B;
    float *buf_C;
    float *buf_C_ref;
    float *cubuf_A;
    float *cubuf_B;
    float *cubuf_C;
};

__global__ void vec_add(float *A, float *B, float *C, int num_elm)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float a, b, c;
    if (i < num_elm)
    {
        a = A[i];
        b = B[i];
        
        // for (int j = 0; j < 1024 * 1024 * 830; ++j)
        for (int j = 0; j < 1024 * 1024 * 47; ++j)
            c = a + b;

        C[i] = c;
    }
}

// run all reduce and vec add by sequence
int test_1()
{
    // prepare resource for nccl all reduce
    int num_cuda_devs = 2;
    ncclComm_t *nccl_comms = (ncclComm_t*) malloc(num_cuda_devs * sizeof(ncclComm_t));
    int *cuda_dev_ids = (int*) malloc(num_cuda_devs * sizeof(int));
    for (int i = 0; i < num_cuda_devs; ++i)
        cuda_dev_ids[i] = i;
    ncclResult_t ret = ncclCommInitAll(nccl_comms, num_cuda_devs, cuda_dev_ids);
    if (ret != ncclSuccess)
    {
        printf("fail to comm init all\n");
        return -1;
    }
    printf("successfully comm init all\n");

    int num_elms = 1024 * 1024 * 1024;
    float *buf_A_dev_0 = (float*) malloc(num_elms * sizeof(float));
    float *buf_A_dev_1 = (float*) malloc(num_elms * sizeof(float));
    float *buf_B = (float*) malloc(num_elms * sizeof(float));
    // for (int i = 0; i < num_elms; ++i)
    // {
    //     buf_A_dev_0[i] = rand() % 5;
    //     buf_A_dev_1[i] = rand() % 5;
    //     buf_B[i] = rand() % 5;
    // }
    // for (int i = 0; i < num_elms; ++i)
    // {
    //     buf_B[i] = buf_A_dev_0[i] + buf_A_dev_1[i];
    // }
    // printf("buf_A_dev_0:\n");
    // print_vec_f32(buf_A_dev_0, num_elms);
    // printf("buf_A_dev_1:\n");
    // print_vec_f32(buf_A_dev_1, num_elms);
    // printf("buf_B:\n");
    // print_vec_f32(buf_B, num_elms);
    // putchar('\n');

    CudevRes *cudev_reses = (CudevRes*) malloc(num_cuda_devs * sizeof(CudevRes));
    for (int i = 0; i < num_cuda_devs; ++i)
    {
        CudevRes *cudev_res = &cudev_reses[i];

        cudaSetDevice(i);
        cudaMalloc((void**) &cudev_res->cubuf_A, num_elms * sizeof(float));
        cudaMalloc((void**) &cudev_res->cubuf_B, num_elms * sizeof(float));
        cudaStreamCreate(&cudev_res->cu_stream);
        // printf("allocate resources from cuda device %d\n", i);

        if (i == 0)
            cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_0, num_elms * sizeof(float), cudaMemcpyHostToDevice);
        else if (i == 1)
            cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_1, num_elms * sizeof(float), cudaMemcpyHostToDevice);
        else
        {
            printf("error\n");
            return -1;
        }

        cudaMemset(cudev_res->cubuf_B, 0, num_elms * sizeof(float));
        // printf("assign cuda mem data for dev %d\n", i);
    }
    // for (int i = 0; i < num_cuda_devs; ++i)
    // {
    //     cudaSetDevice(i);
    //     printf("cu dev %d:\n", i);
    //     printf("\tcubuf_A: ");
    //     print_vec_cuf32(cudev_reses[i].cubuf_A, num_elms);
    //     printf("\tcubuf_B: ");
    //     print_vec_cuf32(cudev_reses[i].cubuf_B, num_elms);
    // }

    // prepare resources for vec add
    CalcRes calc_res;
    // calc_res.num_elm = 1024 * 1024 * 1024;
    calc_res.num_elm = 8;
    calc_res.buf_A = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_B = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_C = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_C_ref = (float*) malloc(calc_res.num_elm * sizeof(float));
    for (int i = 0; i < calc_res.num_elm; ++i)
    {
        calc_res.buf_A[i] = rand() % 5;
        calc_res.buf_B[i] = rand() % 5;
    }

    cudaMalloc(&calc_res.cubuf_A, calc_res.num_elm * sizeof(float));
    cudaMalloc(&calc_res.cubuf_B, calc_res.num_elm * sizeof(float));
    cudaMalloc(&calc_res.cubuf_C, calc_res.num_elm * sizeof(float));
    cudaMemcpy(calc_res.cubuf_A, calc_res.buf_A, calc_res.num_elm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc_res.cubuf_B, buf_B, calc_res.num_elm * sizeof(float), cudaMemcpyHostToDevice);

    // run two tasks by sequence
    timeit(TIMEIT_START, NULL);
    int num_round = 2;
    puts("all reduce sum start...");
    for (int round = 0; round < num_round; ++round)
    {
        ncclGroupStart();
        for (int i = 0; i < num_cuda_devs; ++i)
        {
            CudevRes cudev_res = cudev_reses[i];
            cudaSetDevice(i);
            ncclAllReduce(cudev_res.cubuf_A, cudev_res.cubuf_B, num_elms, ncclFloat, ncclSum, nccl_comms[i], cudev_res.cu_stream);
        }
        ncclGroupEnd();
    }

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        CudevRes cudev_res = cudev_reses[i];
        cudaSetDevice(i);
        cudaStreamSynchronize(cudev_res.cu_stream);
    }
    printf("nccl all reduce ended.\n");
    timeit(TIMEIT_END, NULL);
    float fsecs;
    timeit(TIMEIT_GET_SEC, &fsecs);
    printf("all reduce time: %.2f secs\n", fsecs);

    // for (int i = 0; i < num_cuda_devs; ++i)
    // {
    //     cudaSetDevice(i);
    //     printf("cu dev %d:\n", i);
    //     printf("\tcubuf_A: ");
    //     print_vec_cuf32(cudev_reses[i].cubuf_A, num_elms);
    //     printf("\tcubuf_B: ");
    //     print_vec_cuf32(cudev_reses[i].cubuf_B, num_elms);
    // }

    puts("start calc...");
    timeit(TIMEIT_START, NULL);
    vec_add<<<calc_res.num_elm, 1>>>(calc_res.cubuf_A, calc_res.cubuf_B, calc_res.cubuf_C, calc_res.num_elm);
    cudaDeviceSynchronize();

    timeit(TIMEIT_END, NULL);
    puts("end calc.");
    // float fsecs;
    timeit(TIMEIT_GET_SEC, &fsecs);
    printf("vec add time: %.2f secs\n", fsecs);
    cudaMemcpy(calc_res.buf_C, calc_res.cubuf_C, calc_res.num_elm * sizeof(float), cudaMemcpyDeviceToHost);

    // release res
    cudaFree(calc_res.cubuf_A);
    cudaFree(calc_res.cubuf_B);
    cudaFree(calc_res.cubuf_C);
    free(calc_res.buf_A);
    free(calc_res.buf_B);
    free(calc_res.buf_C);
    free(calc_res.buf_C_ref);

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        cudaSetDevice(i);
        CudevRes cudev_res = cudev_reses[i];
        cudaFree(cudev_res.cubuf_A);
        cudaFree(cudev_res.cubuf_B);
        cudaStreamDestroy(cudev_res.cu_stream);
    }
    printf("cuda dev resource free\n");

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        ncclCommDestroy(nccl_comms[i]);
    }
    free(nccl_comms);
    free(cuda_dev_ids);

    return 0;
}

// run all reduce and vec add calc together
int test_2()
{
    // prepare resource for nccl all reduce
    int num_cuda_devs = 2;
    ncclComm_t *nccl_comms = (ncclComm_t*) malloc(num_cuda_devs * sizeof(ncclComm_t));
    int *cuda_dev_ids = (int*) malloc(num_cuda_devs * sizeof(int));
    for (int i = 0; i < num_cuda_devs; ++i)
        cuda_dev_ids[i] = i;
    ncclResult_t ret = ncclCommInitAll(nccl_comms, num_cuda_devs, cuda_dev_ids);
    if (ret != ncclSuccess)
    {
        printf("fail to comm init all\n");
        return -1;
    }
    printf("successfully comm init all\n");

    int num_elms = 1024 * 1024 * 1024;
    float *buf_A_dev_0 = (float*) malloc(num_elms * sizeof(float));
    float *buf_A_dev_1 = (float*) malloc(num_elms * sizeof(float));
    float *buf_B = (float*) malloc(num_elms * sizeof(float));

    CudevRes *cudev_reses = (CudevRes*) malloc(num_cuda_devs * sizeof(CudevRes));
    for (int i = 0; i < num_cuda_devs; ++i)
    {
        CudevRes *cudev_res = &cudev_reses[i];

        cudaSetDevice(i);
        cudaMalloc((void**) &cudev_res->cubuf_A, num_elms * sizeof(float));
        cudaMalloc((void**) &cudev_res->cubuf_B, num_elms * sizeof(float));
        cudaStreamCreate(&cudev_res->cu_stream);

        if (i == 0)
            cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_0, num_elms * sizeof(float), cudaMemcpyHostToDevice);
        else if (i == 1)
            cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_1, num_elms * sizeof(float), cudaMemcpyHostToDevice);
        else
        {
            printf("error\n");
            return -1;
        }

        cudaMemset(cudev_res->cubuf_B, 0, num_elms * sizeof(float));
    }

    // prepare resources for vec add
    CalcRes calc_res;
    // calc_res.num_elm = 1024 * 1024 * 1024;
    calc_res.num_elm = 8;
    calc_res.buf_A = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_B = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_C = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_C_ref = (float*) malloc(calc_res.num_elm * sizeof(float));
    for (int i = 0; i < calc_res.num_elm; ++i)
    {
        calc_res.buf_A[i] = rand() % 5;
        calc_res.buf_B[i] = rand() % 5;
    }

    cudaMalloc(&calc_res.cubuf_A, calc_res.num_elm * sizeof(float));
    cudaMalloc(&calc_res.cubuf_B, calc_res.num_elm * sizeof(float));
    cudaMalloc(&calc_res.cubuf_C, calc_res.num_elm * sizeof(float));
    cudaMemcpy(calc_res.cubuf_A, calc_res.buf_A, calc_res.num_elm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc_res.cubuf_B, buf_B, calc_res.num_elm * sizeof(float), cudaMemcpyHostToDevice);

    // run two tasks together
    timeit(TIMEIT_START, NULL);
    int num_round = 2;
    puts("all reduce sum start...");
    for (int round = 0; round < num_round; ++round)
    {
        ncclGroupStart();
        for (int i = 0; i < num_cuda_devs; ++i)
        {
            CudevRes cudev_res = cudev_reses[i];
            cudaSetDevice(i);
            ncclAllReduce(cudev_res.cubuf_A, cudev_res.cubuf_B, num_elms, ncclFloat, ncclSum, nccl_comms[i], cudev_res.cu_stream);
        }
        ncclGroupEnd();
    }

    puts("start calc...");
    timeit(TIMEIT_START, NULL);
    vec_add<<<calc_res.num_elm, 1>>>(calc_res.cubuf_A, calc_res.cubuf_B, calc_res.cubuf_C, calc_res.num_elm);

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        CudevRes cudev_res = cudev_reses[i];
        cudaSetDevice(i);
        cudaStreamSynchronize(cudev_res.cu_stream);
    }
    printf("nccl all reduce ended.\n");

    cudaDeviceSynchronize();
    puts("end calc.");

    timeit(TIMEIT_END, NULL);
    float fsecs;
    timeit(TIMEIT_GET_SEC, &fsecs);
    printf("all reduce and vec add time: %.2f secs\n", fsecs);
    cudaMemcpy(calc_res.buf_C, calc_res.cubuf_C, calc_res.num_elm * sizeof(float), cudaMemcpyDeviceToHost);

    // release res
    cudaFree(calc_res.cubuf_A);
    cudaFree(calc_res.cubuf_B);
    cudaFree(calc_res.cubuf_C);
    free(calc_res.buf_A);
    free(calc_res.buf_B);
    free(calc_res.buf_C);
    free(calc_res.buf_C_ref);

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        cudaSetDevice(i);
        CudevRes cudev_res = cudev_reses[i];
        cudaFree(cudev_res.cubuf_A);
        cudaFree(cudev_res.cubuf_B);
        cudaStreamDestroy(cudev_res.cu_stream);
    }
    printf("cuda dev resource free\n");

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        ncclCommDestroy(nccl_comms[i]);
    }
    free(nccl_comms);
    free(cuda_dev_ids);

    return 0;
}

// run vec add in another cuda stream
int test_3()
{
    cudaStream_t calc_stream;
    cudaStreamCreate(&calc_stream);

    // prepare resource for nccl all reduce
    int num_cuda_devs = 2;
    ncclComm_t *nccl_comms = (ncclComm_t*) malloc(num_cuda_devs * sizeof(ncclComm_t));
    int *cuda_dev_ids = (int*) malloc(num_cuda_devs * sizeof(int));
    for (int i = 0; i < num_cuda_devs; ++i)
        cuda_dev_ids[i] = i;
    ncclResult_t ret = ncclCommInitAll(nccl_comms, num_cuda_devs, cuda_dev_ids);
    if (ret != ncclSuccess)
    {
        printf("fail to comm init all\n");
        return -1;
    }
    printf("successfully comm init all\n");

    int num_elms = 1024 * 1024 * 1024;
    float *buf_A_dev_0 = (float*) malloc(num_elms * sizeof(float));
    float *buf_A_dev_1 = (float*) malloc(num_elms * sizeof(float));
    float *buf_B = (float*) malloc(num_elms * sizeof(float));

    CudevRes *cudev_reses = (CudevRes*) malloc(num_cuda_devs * sizeof(CudevRes));
    for (int i = 0; i < num_cuda_devs; ++i)
    {
        CudevRes *cudev_res = &cudev_reses[i];

        cudaSetDevice(i);
        cudaMalloc((void**) &cudev_res->cubuf_A, num_elms * sizeof(float));
        cudaMalloc((void**) &cudev_res->cubuf_B, num_elms * sizeof(float));
        cudaStreamCreate(&cudev_res->cu_stream);

        if (i == 0)
            cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_0, num_elms * sizeof(float), cudaMemcpyHostToDevice);
        else if (i == 1)
            cudaMemcpy(cudev_res->cubuf_A, buf_A_dev_1, num_elms * sizeof(float), cudaMemcpyHostToDevice);
        else
        {
            printf("error\n");
            return -1;
        }

        cudaMemset(cudev_res->cubuf_B, 0, num_elms * sizeof(float));
    }

    // prepare resources for vec add
    CalcRes calc_res;
    // calc_res.num_elm = 1024 * 1024 * 1024;
    calc_res.num_elm = 8;
    calc_res.buf_A = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_B = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_C = (float*) malloc(calc_res.num_elm * sizeof(float));
    calc_res.buf_C_ref = (float*) malloc(calc_res.num_elm * sizeof(float));
    for (int i = 0; i < calc_res.num_elm; ++i)
    {
        calc_res.buf_A[i] = rand() % 5;
        calc_res.buf_B[i] = rand() % 5;
    }

    cudaMalloc(&calc_res.cubuf_A, calc_res.num_elm * sizeof(float));
    cudaMalloc(&calc_res.cubuf_B, calc_res.num_elm * sizeof(float));
    cudaMalloc(&calc_res.cubuf_C, calc_res.num_elm * sizeof(float));
    cudaMemcpy(calc_res.cubuf_A, calc_res.buf_A, calc_res.num_elm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc_res.cubuf_B, buf_B, calc_res.num_elm * sizeof(float), cudaMemcpyHostToDevice);

    // run two tasks together
    timeit(TIMEIT_START, NULL);
    int num_round = 2;
    puts("all reduce sum start...");
    for (int round = 0; round < num_round; ++round)
    {
        ncclGroupStart();
        for (int i = 0; i < num_cuda_devs; ++i)
        {
            CudevRes cudev_res = cudev_reses[i];
            cudaSetDevice(i);
            ncclAllReduce(cudev_res.cubuf_A, cudev_res.cubuf_B, num_elms, ncclFloat, ncclSum, nccl_comms[i], cudev_res.cu_stream);
        }
        ncclGroupEnd();
    }

    puts("start calc...");
    timeit(TIMEIT_START, NULL);
    vec_add<<<calc_res.num_elm, 1, 0, calc_stream>>>(calc_res.cubuf_A, calc_res.cubuf_B, calc_res.cubuf_C, calc_res.num_elm);

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        CudevRes cudev_res = cudev_reses[i];
        cudaSetDevice(i);
        cudaStreamSynchronize(cudev_res.cu_stream);
    }
    printf("nccl all reduce ended.\n");

    cudaDeviceSynchronize();
    puts("end calc.");

    timeit(TIMEIT_END, NULL);
    float fsecs;
    timeit(TIMEIT_GET_SEC, &fsecs);
    printf("all reduce and vec add time: %.2f secs\n", fsecs);
    cudaMemcpy(calc_res.buf_C, calc_res.cubuf_C, calc_res.num_elm * sizeof(float), cudaMemcpyDeviceToHost);

    // release res
    cudaFree(calc_res.cubuf_A);
    cudaFree(calc_res.cubuf_B);
    cudaFree(calc_res.cubuf_C);
    free(calc_res.buf_A);
    free(calc_res.buf_B);
    free(calc_res.buf_C);
    free(calc_res.buf_C_ref);

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        cudaSetDevice(i);
        CudevRes cudev_res = cudev_reses[i];
        cudaFree(cudev_res.cubuf_A);
        cudaFree(cudev_res.cubuf_B);
        cudaStreamDestroy(cudev_res.cu_stream);
    }
    printf("cuda dev resource free\n");

    for (int i = 0; i < num_cuda_devs; ++i)
    {
        ncclCommDestroy(nccl_comms[i]);
    }
    free(nccl_comms);
    free(cuda_dev_ids);

    return 0;
}

int main()
{
    int cu_dev_cnt;
    cudaGetDeviceCount(&cu_dev_cnt);
    printf("there are totally %d cuda devices\n\n", cu_dev_cnt);

    // int cur_cu_dev_id;
    // cudaGetDevice(&cur_cu_dev_id);
    // printf("current cuda device: %d\n", cur_cu_dev_id);

    puts("test 1:");
    test_1();
    putchar('\n');

    puts("test 2:");
    test_2();
    putchar('\n');

    puts("test 3:");
    test_3();

    return 0;
}
