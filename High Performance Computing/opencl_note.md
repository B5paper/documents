# OpenCL Note

Home page: <https://www.khronos.org/api/opencl>

Ref:

* <https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/>

* <https://registry.khronos.org/OpenCL/>，这里面资料挺多的

* <https://www.khronos.org/opencl/resources>

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

**OpenCL Platform Mode**

One *Host* control one or more *OpenCL Devices*.

(question: does the host participate in computing?)

Each OpenCL Device is composed of one or more *Compute Units*.

Each Compute Unit is divided into one or more *Processing Elements*.

(question: what's the difference between compute units and processing elements?)

Memory divided into *host memory* and *device memory*

kernel:

```c
__kernel void
mul(__global const float *a,
    __global const float *b,
    __global float *c)
{
    int id = get_global_id(0);
    c[id] = a[id] * b[id];
}
```

An N-dimensional domain of work-items 

* Global Dimensions:

    1024x1024 (whole problem space)

    Synchronization between work-items possible only within work-groups: barriers and memory fences

* Local Dimensions:

    128x128 (work-group, executes together) 

    Cannot synchronize between work-groups within a kernel

（我的理解是，同一个 *work-group* 中的 *work_item* 之间可以交换数据做同步，而不同 *work_group* 之间则不行）

Choose the dimensions (1, 2, or 3) that are “best” for your algorithm

（这里的`1`似乎指的是`Buffer`，`2`和`3`指的是`Image`图像。但我目前还不知道他们有什么区别）

**OpenCL Memory model**

* Private Memory

    Per work-item

* Local Memory

    Shared within a work-group

* Global Memory Constant Memory

    Visible to all work-groups

* Host memory

    On the CPU 

Memory management is explicit:

You are responsible for moving data from

`host -> global -> local and back`

**Context and Command-Queues**

Context: The environment within which kernels execute and in which synchronization and memory management is defined. 

The context includes:

* One or more devices
* Device memory
* One or more command-queues

All commands for a device (kernel execution, synchronization, and memory operations) are submitted through a command-queue. 

* Each command-queue points to a single device within a context. 

kernel:

```c
__kernel void times_two(
 __global float* input,
 __global float* output)
{
 int i = get_global_id(0);
 output[i] = 2.0f * input[i];
}
```

**Building Program Objects**

The program object encapsulates:

* A context
* The program source or binary, and
* List of target devices and build options

OpenCL uses runtime compilation

because in general you don’t know the details of the target device when you ship the program

Example:

`cl::Program program(context, KernelSource, true);`

kernel:

```c
__kernel void
horizontal_reflect(read_only image2d_t src,
 write_only image2d_t dst)
{
 int x = get_global_id(0); // x-coord
 int y = get_global_id(1); // y-coord
 int width = get_image_width(src);
 float4 src_val = read_imagef(src, sampler,
 (int2)(width-1-x, y));
 write_imagef(dst, (int2)(x, y), src_val);
}
```

## Common used API reference

* `clGetPlatformIDs`

    ```c
    cl_int clGetPlatformIDs(
        cl_uint num_entries,
        cl_platform_id* platforms,
        cl_uint* num_platforms);
    ```

    Get the list of platforms available.

    * `num_entries` is the number of `cl_platform_id` entries that can be added to platforms. If platforms is not NULL, the `num_entries` must be greater than zero.

        其实就是缓冲区的大小。通常这个填 1 就行了。

        不清楚如果有多个 platform 的话该怎么填。

    * `platforms` returns a list of OpenCL platforms found. The cl_platform_id values returned in platforms can be used to identify a specific OpenCL platform. If `platforms` is NULL, this argument is ignored. The number of OpenCL platforms returned is the minimum of the value specified by `num_entries` or the number of OpenCL platforms available.

        其实就是缓冲区的地址。需要注意的是`cl_platform_id`是一个不可见的结构体，所以我们无法看到它的字段。

        （如果我自己想创建一个这样的结构体，代码该怎么写呢？）

    * `num_platforms` returns the number of OpenCL platforms available. If `num_platforms` is NULL, this argument is ignored.

        其实就是实际使用了多少缓冲区。

    Return value:

    `clGetPlatformIDs` returns `CL_SUCCESS` if the function is executed successfully. Otherwise, it returns one of the following errors:

    （目前只要知道这个就可以了，具体的错误类型以后再说）

    * `CL_INVALID_VALUE` if `num_entries` is equal to zero and platforms is not NULL or if both `num_platforms` and platforms are NULL.

    * `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

* `clGetPlatformInfo`

    拿到`platform_id`后，可以用这个函数将 id 转换成一些字符串信息。

    Syntax:

    ```c
    cl_int clGetPlatformInfo(
        cl_platform_id platform,
        cl_platform_info param_name,
        size_t param_value_size,
        void* param_value,
        size_t* param_value_size_ret);
    ```

    `param_value_size`是字符串缓冲区大小，`param_value`是缓冲区地址，`param_value_size_ret`是实际使用的缓冲区大小。如果`param_value_size`不够大，`clGetPlatformInfo()`会失败。

    常用的`param_name`如下：

    | Name | Description |
    | - | - |
    | `CL_PLATFORM_PROFILE` | `FULL_PROFILE` - functionality defined as part of the core specification and does not require any extensions to be supported. <br> `EMBEDDED_PROFILE` - The embedded profile is defined to be a subset for each version of OpenCL. The embedded profile for OpenCL is described in OpenCL Embedded Profile. |
    |  |  |

* `clGetDeviceIDs`

    ```c
    cl_int clGetDeviceIDs(
        cl_platform_id platform,
        cl_device_type device_type,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices);
    ```

    `platform` refers to the platform ID returned by clGetPlatformIDs or can be NULL. If platform is NULL, the behavior is implementation-defined.

    `device_type` is a bitfield that identifies the type of OpenCL device. The device_type can be used to query specific OpenCL devices or all OpenCL devices available. The valid values for device_type are specified in the Device Types table.

    `num_entries` is the number of cl_device_id entries that can be added to devices. If devices is not NULL, the num_entries must be greater than zero.

    `devices` returns a list of OpenCL devices found. The cl_device_id values returned in devices can be used to identify a specific OpenCL device. If devices is NULL, this argument is ignored. The number of OpenCL devices returned is the minimum of the value specified by num_entries or the number of OpenCL devices whose type matches device_type.

    `num_devices` returns the number of OpenCL devices available that match device_type. If num_devices is NULL, this argument is ignored.

* `clCreateContext`

    ```c
    cl_context clCreateContext(
        const cl_context_properties* properties,
        cl_uint num_devices,
        const cl_device_id* devices,
        void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
        void* user_data,
        cl_int* errcode_ret);
    ```

    * `properties` specifies a list of context property names and their corresponding values. Each property name is immediately followed by the corresponding desired value. The list is terminated with 0. The list of supported properties is described in the Context Properties table. properties can be NULL in which case the platform that is selected is implementation-defined.

        通常这个填 NULL 就行。

    * `num_devices` is the number of devices specified in the devices argument.

        有几个设备填几，如果是 GPU 的话填 1。

    * `devices` is a pointer to a list of unique devices returned by `clGetDeviceIDs` or sub-devices created by `clCreateSubDevices` for a platform. [11]

        `device_id`是由`clGetDeviceIDs()`得到的。

    * `pfn_notify` is a callback function that can be registered by the application. This callback function will be used by the OpenCL implementation to report information on errors during context creation as well as errors that occur at runtime in this context. This callback function may be called asynchronously by the OpenCL implementation. It is the applications responsibility to ensure that the callback function is thread-safe. If pfn_notify is NULL, no callback function is registered.

    * `user_data` will be passed as the user_data argument when pfn_notify is called. user_data can be NULL.

    * `errcode_ret` will return an appropriate error code. If errcode_ret is NULL, no error code is returned.

* `clCreateCommandQueue`

    ```c
    cl_command_queue clCreateCommandQueue(
        cl_context context,
        cl_device_id device,
        cl_command_queue_properties properties,
        cl_int* errcode_ret);
    ```

    * `context`
        
        must be a valid OpenCL context.

    * `device`
    
        must be a device or sub-device associated with context. It can either be in the list of devices and sub-devices specified when context is created using clCreateContext or be a root device with the same device type as specified when context is created using clCreateContextFromType.

    * `properties`
    
        specifies a list of properties for the command-queue. This is a bit-field and the supported properties are described in the table below. Only command-queue properties specified in this table can be used, otherwise the value specified in properties is considered to be not valid. properties can be 0 in which case the default values for supported command-queue properties will be used.

        |Command-Queue Properties|Description|
        | - | - |
        |`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`|Determines whether the commands queued in the command-queue are executed in-order or out-of-order. If set, the commands in the command-queue are executed out-of-order. Otherwise, commands are executed in-order.|
        |`CL_QUEUE_PROFILING_ENABLE`|Enable or disable profiling of commands in the command-queue. If set, the profiling of commands is enabled. Otherwise profiling of commands is disabled.|

    * `errcode_ret`
    
        will return an appropriate error code. If errcode_ret is NULL, no error code is returned.

    `clCreateCommandQueue` returns a valid non-zero command-queue and `errcode_ret` is set to `CL_SUCCESS` if the command-queue is created successfully. Otherwise, it returns a NULL value with one of the following error values returned in `errcode_ret`:

    `CL_INVALID_CONTEXT` if context is not a valid context.

    `CL_INVALID_DEVICE` if device is not a valid device or is not associated with context.

    `CL_INVALID_VALUE` if values specified in properties are not valid.

    `CL_INVALID_QUEUE_PROPERTIES` if values specified in properties are valid but are not supported by the device.

    `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

* `clCreateBuffer`

    Syntax:

    ```c
    cl_mem clCreateBuffer(
        cl_context context,
        cl_mem_flags flags,
        size_t size,
        void* host_ptr,
        cl_int* errcode_ret);
    ```

    这个函数作用类似于 C 语言中的`malloc()`，申请一块内存（或者显存）。

    * `context` is a valid OpenCL context used to create the buffer object.

        这里的`context`通常在我们调用这个函数前就已经创建好。

    * `flags` is a bit-field that is used to specify allocation and usage information about the image memory object being created and is described in the supported memory flag values table.

        类似打开文件时的读写权限，还有一些其他功能。常用的枚举有：

        * `CL_MEM_READ_WRITE`

        * `CL_MEM_WRITE_ONLY`

        * `CL_MEM_READ_ONLY`

            前三个看名字就能猜出来意思，没什么好说的。

        * `CL_MEM_USE_HOST_PTR`

            这个不知道干啥用的。文档的解释如下：

            This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation to use memory referenced by host_ptr as the storage bits for the memory object.

            The contents of the memory pointed to by host_ptr at the time of the clCreateBuffer, clCreateBufferWithProperties, clCreateImage, clCreateImageWithProperties, clCreateImage2D, or clCreateImage3D call define the initial contents of the memory object.

            OpenCL implementations are allowed to cache the contents pointed to by host_ptr in device memory. This cached copy can be used when kernels are executed on a device.

            The result of OpenCL commands that operate on multiple buffer objects created with the same host_ptr or from overlapping host or SVM regions is considered to be undefined.

    * `size` is the size in bytes of the buffer memory object to be allocated.

    * `host_ptr` is a pointer to the buffer data that may already be allocated by the application. The size of the buffer that host_ptr points to must be greater than or equal to size bytes.

    * `errcode_ret` may return an appropriate error code. If errcode_ret is NULL, no error code is returned.

* `clEnqueueWriteBuffer`

    ```c
    cl_int clEnqueueWriteBuffer(
        cl_command_queue command_queue,
        cl_mem buffer,
        cl_bool blocking_write,
        size_t offset,
        size_t size,
        const void* ptr,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);
    ```

    Parameters:

    * `command_queue` is a valid host command-queue in which the read / write command will be queued. command_queue and buffer must be created with the same OpenCL context.

    * `buffer` refers to a valid buffer object.

    * `blocking_read` and `blocking_write` indicate if the read and write operations are blocking or non-blocking (see below).

    * `offset` is the offset in bytes in the buffer object to read from or write to.

    * `size` is the size in bytes of data being read or written.

    * `ptr` is the pointer to buffer in host memory where data is to be read into or to be written from.

    * `event_wait_list` and `num_events_in_wait_list` specify events that need to complete before this particular command can be executed. If event_wait_list is NULL, then this particular command does not wait on any event to complete. If event_wait_list is NULL, num_events_in_wait_list must be 0. If event_wait_list is not NULL, the list of events pointed to by event_wait_list must be valid and num_events_in_wait_list must be greater than 0. The events specified in event_wait_list act as synchronization points. The context associated with events in event_wait_list and command_queue must be the same. The memory associated with event_wait_list can be reused or freed after the function returns.

    * `event` returns an event object that identifies this read / write command and can be used to query or queue a wait for this command to complete. If event is NULL or the enqueue is unsuccessful, no event will be created and therefore it will not be possible to query the status of this command or to wait for this command to complete. If event_wait_list and event are not NULL, event must not refer to an element of the event_wait_list array.

    Return value:

    `clEnqueueReadBuffer` and `clEnqueueWriteBuffer` return CL_SUCCESS if the function is executed successfully.

* `clRetainContext`

    增加 context 的引用计数，防止 opencl 库突然释放资源。