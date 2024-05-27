
#include "../ocl_simple/simple_ocl.hpp"
#include "../timeit/timeit.hpp"
#include <unistd.h>

void sum_cpu(const float *arr, float *out, const int len, float *time_ms)
{
    timeit_ms("start");
    float sum = 0.0f;
    for (int i = 0; i < len; ++i)
    {
        sum += arr[i];
    }
    *out = sum;
    *time_ms = timeit_ms("end");
}

void test_1(const float *arr, float *out, const int len, float *time_ms, const size_t time_N)
{
    init_ocl_env("./kernels.cl", {"sum_1"});
    add_buf("arr", sizeof(float), len);
    add_buf("out", sizeof(float), 1);
    timeit_ms("start");
    for (size_t i = 0; i < time_N; ++i)
    {
        write_buf("arr", (void*) arr);
        run_kern("sum_1", {(size_t) 256}, "arr", "out", len);
        read_buf(out, "out");
    }
    *time_ms = timeit_ms("end");
    *time_ms /= time_N;
    exit_ocl_env();
}

void test_2(const float *arr, float *out, const int len, float *time_ms, const size_t time_N)
{
    init_ocl_env("./kernels.cl", {"sum_2"});
    add_buf("arr", sizeof(float), len);
    add_buf("out", sizeof(float), 1);
    timeit_ms("start");
    for (size_t i = 0; i < time_N; ++i)
    {
        write_buf("arr", (void*) arr);
        run_kern("sum_2", {(size_t) 256}, "arr", "out", len);
        read_buf(out, "out");
    }
    *time_ms = timeit_ms("end");
    *time_ms /= time_N;
    exit_ocl_env();
}

void gen_rand_arr(float *arr, const int len)
{
    for (int i = 0; i < len; ++i)
    {
        arr[i] = rand() % 10;
    }
}

int main()
{
    int N = 1024;
    float *arr = (float*) malloc(sizeof(float) * N);
    gen_rand_arr(arr, N);

    // cpu
    float sum_out_cpu;
    float time_ms_cpu;
    sum_cpu(arr, &sum_out_cpu, N, &time_ms_cpu);
    printf("cpu sum: %.2f\n", sum_out_cpu);
    printf("cpu time consumption: %.3f ms\n", time_ms_cpu);
    putchar('\n');

    // gpu sum_1
    float sum_out_gpu_sum_1;
    float time_ms_gpu_sum_1;
    test_1(arr, &sum_out_gpu_sum_1, N, &time_ms_gpu_sum_1, 102400);
    if (sum_out_gpu_sum_1 != sum_out_cpu)
        printf("not correct!\n");
    printf("gpu sum 1: %.2f\n", sum_out_gpu_sum_1);
    printf("gpu time consumption: %.3f ms\n", time_ms_gpu_sum_1);
    putchar('\n');

    // gpu sum 2
    float sum_out_gpu_sum_2;
    float time_ms_gpu_sum_2;
    test_2(arr, &sum_out_gpu_sum_2, N, &time_ms_gpu_sum_2, 102400);
    if (sum_out_gpu_sum_2 != sum_out_cpu)
        printf("not correct!\n");
    printf("gpu sum 2: %.2f\n", sum_out_gpu_sum_2);
    printf("gpu time consumption: %.3f ms\n", time_ms_gpu_sum_2);
    putchar('\n');

    free(arr);    
    return 0;
}