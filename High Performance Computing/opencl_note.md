# OpenCL Note

Home page: <https://www.khronos.org/api/opencl>

Ref:

* <https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/>

## Introduction

OpenCL 可以创建 kernel programs (or kernels)，支持在不同的硬件上并行计算。

The OpenCL framework contains two APIs. The Platform Layer API is run on the host CPU and is used first to enable a program to discover what parallel processors or compute devices are available in a system. By querying for what compute devices are available an application can run portably across diverse systems - adapting to different combinations of accelerator hardware. Once the compute devices are discovered, the Platform API enables the application to select and initialize the devices it wants to use.

The second API is the Runtime API which enables the application's kernel programs to be compiled for the compute devices on which they are going to run, loaded in parallel onto those processors and executed. Once the kernel programs finish execution the Runtime API is used to gather the results.

我们编译时使用的语言叫 OpenCL C。OpenCL C is based on C99 and is defined as part of the OpenCL specification。其他语言也可以写 OpenCL 的 kernel，生成的是中间形式 SPIR-V。

An OpenCL program is a collection of kernels and functions (similar to dynamic library with run-time linking).

An OpenCL `command queue` is used by the host application to send kernels and data transfer functions to a device for execution. By `enqueueing` commands into a command queue, kernels and data transfer functions may execute asynchronously and in parallel with application host code.

A complete sequence for executing an OpenCL program is:

1. Query for available OpenCL platforms and devices

1. Create a context for one or more OpenCL devices in a platform

1. Create and build programs for OpenCL devices in the context

1. Select kernels to execute from the programs

1. Create memory objects for kernels to operate on

1. Create command queues to execute commands on an OpenCL device

1. Enqueue data transfer commands into the memory objects, if needed

1. Enqueue kernels into the command queue for execution

1. Enqueue commands to transfer data back to the host, if needed

Aspects of OpenCL programming:

* Check for OpenCL-capable device(s);
* Memory allocation on the device;
* Data transfer to the device;
* Retrieve data from the device;
* Compile C/C++ programs that launch OpenCL kernels.

## Installation

### Windows + Intel

首先下载 Intel 的 OpenCL SDK：<https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/choose-download.html>

下载并安装完成后，找到这个目录：`..\IntelSWTools\system_studio_2020\OpenCL\sdk`

其中的`include`和`lib`文件夹，就是我们后面编程所需要的。

接下来下载 Visual Studio，或许也可以使用 MS build tools，反正只有用微软的编译环境才能编译成功。尝试过使用 mingw64 的 gcc 编译，失败了。

首先我们写个算子：

`vector_add_kernel.cl`:

```c
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}
```

然后写主程序：

`main.c`

```c
#define _CRT_SECURE_NO_DEPRECATE
#pragma warning(disable: 4996)
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int* A = (int*)malloc(sizeof(int) * LIST_SIZE);
    int* B = (int*)malloc(sizeof(int) * LIST_SIZE);
    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");  // 打开我们刚才写的算子文件
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
        &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

    // Display the result to the screen
    for (i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}
```

编译的时候，需要在 visual studio 里面加上头文件目录的搜索路径`../include`以及库文件的搜索路径`../lib/x64`，还需要加上库文件：`OpenCL.lib`。

运行后，会在 console 输出两数和为 1024 的所有情况。

### Ubuntu + Intel

(empty)

## Quickstart

