# OpenCL Note QA

[unit]
[u_0]
请写一个向量相加的算子`add`。
[u_1]
`kernels.cl`:

```c
kernel void add(global const int *A, global const int *B, global int *C)
{
    size_t i = get_global_id(0);
    C[i] = A[i] + B[i];
}
```
[title]
向量相加 kernel

[unit]
[title]
向量相加
[deps]
向量相加 kernel
[u_0]
请写一个向量相加的主程序。
[u_1]
(2024.03.03 version)

```c
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>

int main()
{
    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, NULL);
    cl_device_id dev;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    FILE *f = fopen("./kernels.cl", "r");
    fseek(f, 0, SEEK_END);
    size_t file_len = ftell(f);
    char *prog_content = (char*) malloc(file_len);
    fseek(f, 0, SEEK_SET);
    fread(prog_content, file_len, 1, f);
    fclose(f);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**) &prog_content, &file_len, NULL);
    free(prog_content);
    clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    cl_kernel k_add = clCreateKernel(prog, "add", NULL);
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0};
    cl_mem buf_A = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * sizeof(float), A, NULL);
    cl_mem buf_B = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * sizeof(float), B, NULL);
    cl_mem buf_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(float), NULL, NULL);
    clSetKernelArg(k_add, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(k_add, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(k_add, 2, sizeof(cl_mem), &buf_C);
    cl_command_queue cmd_que = clCreateCommandQueueWithProperties(ctx, dev, NULL, NULL);
    size_t glb_work_size = 4;
    size_t glb_work_offset = 0;
    clEnqueueNDRangeKernel(cmd_que, k_add, 1, &glb_work_offset, &glb_work_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cmd_que, buf_C, CL_TRUE, 0, sizeof(float) * 4, C, 0, NULL, NULL);
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", C[i]);
    printf("\n");
    return 0;
}
```

```c
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    cl_uint num_plats;
    clGetPlatformIDs(0, nullptr, &num_plats);
    cl_platform_id *plats = (cl_platform_id*) malloc(num_plats * sizeof(cl_platform_id));
    clGetPlatformIDs(num_plats, plats, &num_plats);
    printf("opencl platform number: %d\n", num_plats);

    cl_uint num_devs;
    clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devs);
    cl_device_id *devs = (cl_device_id*) malloc(num_devs * sizeof(cl_device_id));
    clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, num_devs, devs, &num_devs);
    printf("platform %d has %d available device.\n", 0, num_devs);

    cl_int ret_code;
    cl_context ctx = clCreateContext(nullptr, 1, &devs[0], nullptr, nullptr, &ret_code);
    cl_command_queue cmd_queue = clCreateCommandQueueWithProperties(ctx, devs[0], nullptr, &ret_code);
    cl_mem mem_1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(float), nullptr, &ret_code);
    cl_mem mem_2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(float), nullptr, &ret_code);
    cl_mem mem_3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sizeof(float), nullptr, &ret_code);

    FILE *f = fopen("kernels.cl", "r");
    fseek(f, 0, SEEK_END);
    size_t p_program_length = ftell(f);
    char *program_content = (char*) malloc(p_program_length);
    fseek(f, 0, SEEK_SET);
    fread(program_content, p_program_length, 1, f);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **) &program_content, &p_program_length, &ret_code);
    free(program_content);
    ret_code = clBuildProgram(prog, 1, &devs[0], nullptr, nullptr, nullptr);
    cl_kernel kern_add = clCreateKernel(prog, "add", &ret_code);

    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0};
    clEnqueueWriteBuffer(cmd_queue, mem_1, CL_TRUE, 0, sizeof(float) * 4, A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cmd_queue, mem_2, CL_TRUE, 0, sizeof(float) * 4, B, 0, nullptr, nullptr);
    size_t global_work_offset = 0;
    size_t global_work_size = 4;
    clSetKernelArg(kern_add, 0, sizeof(cl_mem), &mem_1);
    clSetKernelArg(kern_add, 1, sizeof(cl_mem), &mem_2);
    clSetKernelArg(kern_add, 2, sizeof(cl_mem), &mem_3);
    clEnqueueNDRangeKernel(cmd_queue, kern_add, 1, &global_work_offset, &global_work_size,
        nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(cmd_queue, mem_3, CL_TRUE, 0, sizeof(float) * 4, C, 0, nullptr, nullptr);

    // display
    printf("A: [");
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", A[i]);
    printf("]\n");

    printf("B: [");
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", B[i]);
    printf("]\n");

    printf("C: [");
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", C[i]);
    printf("]\n");
    
    return 0;
}
```
