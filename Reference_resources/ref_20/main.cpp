#include "../ocl_simple/simple_ocl.hpp"
#include <vector>
#include <time.h>
using namespace std;

float timeit_ms(const char *cmd)
{
    static timespec tmspec_start, tmspec_end;
    if (strcmp(cmd, "start") == 0)
    {
        timespec_get(&tmspec_start, TIME_UTC);
        return 0.0f;
    }
    else if (strcmp(cmd, "end") == 0)
    {
        timespec_get(&tmspec_end, TIME_UTC);
        float dur_ms = (float) (tmspec_end.tv_sec - tmspec_start.tv_sec) * 1000.0f 
            + (float) (tmspec_end.tv_nsec - tmspec_start.tv_nsec) / 1000.0f / 1000.0f;
        return dur_ms;
    }
    else
    {
        printf("unknown timeit_ms() command\n");
        return -1.0;
    }
}

void csum(float *arr, float *output, const int len)
{
    float s = 0.0f;
    for (int i = 0; i < len; ++i)
    {
        s += arr[i];
    }
    *output = s;
}

int main()
{
    init_ocl_env("./kernels.cl", {"gsum"});
    int len = 256;
    vector<float> arr(len);
    for (int i = 0; i < len; ++i)
    {
        arr[i] = rand() % 10;
    }

    // cpu sum reference
    float s_ref;
    timeit_ms("start");
    csum(arr.data(), &s_ref, len);
    float dur_ms = timeit_ms("end");
    printf("cpu sum ref: %.2f\n", s_ref);
    printf("time consumption: %.3f\n\n", dur_ms);

    // gpu opencl reduction
    float s;
    add_buf("arr", sizeof(float), len);
    add_buf("output", sizeof(float), 1);
    add_local_buf("loc_arr", sizeof(float), len);
    timeit_ms("start");
    write_buf("arr", arr.data());
    run_kern("gsum", {(size_t) len}, {(size_t) len},
        "arr", "output", len, "loc_arr");
    read_buf(&s, "output");
    dur_ms = timeit_ms("end");
    printf("gpu sum: %.2f\n", s);
    printf("time consumption: %.3f\n\n", dur_ms);

    if (s != s_ref)
    {
        printf("uncorrect result:\n");
        printf("cpu: %f\n", s_ref);
        printf("gpu: %f\n", s);
        return -1;
    }
    printf("[checked] correct.\n");
    
    return 0;
}