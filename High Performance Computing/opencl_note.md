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

### Ubuntu + AMD GPU

Ref: <https://askubuntu.com/questions/1406137/how-enable-opencl-on-amd-gpu-ubuntu-22-04-lts>

首先需要在官网上下载一个 driver：<https:/evice_id;/www.amd.com/en/support/graphics/amd-radeon-5600-series/amd-radeon-rx-5600-series/amd-radeon-rx-5600-xt>

找到自己系统对应的版本，使用`dpkg -i xxx.deb`安装。

然后参考安装文档：<https://amdgpu-install.readthedocs.io/en/latest/>，指定要安装什么库。目前我安装的是

```bash
amdgpu-install --usecase=workstation,rocm,opencl --opencl=rocr,legacy --vulkan=pro --accept-eula
```

接下来可以在`/opt/rocm/opencl/`中找到 OpenCL 相关的头文件与库。（这个好像不行，看看`/usr/lib/x86_64-linux-gnu/`这个文件夹中有没有 OpenCL 相关的库，目前这个是正常的）

另外还需要安装：`apt install mesa-opencl-icd` （这个好像也不好使，也有可能需要安装的是`libclc-amdgcn`）

此时运行`clinfo`，如果正常，说明安装成功。如果不正常，可以重启后再试试。

amd 的 opencl 有三个版本，mesa, amd, ROCm

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

    * `platform` refers to the platform ID returned by clGetPlatformIDs or can be NULL. If platform is NULL, the behavior is implementation-defined.

    * `device_type` is a bitfield that identifies the type of OpenCL device. The device_type can be used to query specific OpenCL devices or all OpenCL devices available. The valid values for device_type are specified in the Device Types table.

        目前填的是`CL_DEVICE_TYPE_DEFAULT`。

    * `num_entries` is the number of `cl_device_id` entries that can be added to devices. If devices is not NULL, the `num_entries` must be greater than zero.

        目前填的是 1。

    * `devices` returns a list of OpenCL devices found. The cl_device_id values returned in devices can be used to identify a specific OpenCL device. If devices is NULL, this argument is ignored. The number of OpenCL devices returned is the minimum of the value specified by num_entries or the number of OpenCL devices whose type matches device_type.

    * `num_devices` returns the number of OpenCL devices available that match device_type. If num_devices is NULL, this argument is ignored.

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

        填 NULL

    * `user_data` will be passed as the user_data argument when pfn_notify is called. user_data can be NULL.

        填 NULL

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

        置 NULL

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

    * `host_ptr` is a pointer to the buffer data that may already be allocated by the application. The size of the buffer that `host_ptr` points to must be greater than or equal to `size` bytes.

        填 NULL。

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

        目前填的是`CL_TRUE`。

    * `offset` is the offset in bytes in the buffer object to read from or write to.

        目前填 0。

    * `size` is the size in bytes of data being read or written.

        通常填缓冲区的大小。

    * `ptr` is the pointer to buffer in host memory where data is to be read into or to be written from.

        填缓冲区的起始地址。

    * `event_wait_list` and `num_events_in_wait_list` specify events that need to complete before this particular command can be executed. If event_wait_list is NULL, then this particular command does not wait on any event to complete. If event_wait_list is NULL, num_events_in_wait_list must be 0. If event_wait_list is not NULL, the list of events pointed to by event_wait_list must be valid and num_events_in_wait_list must be greater than 0. The events specified in event_wait_list act as synchronization points. The context associated with events in event_wait_list and command_queue must be the same. The memory associated with event_wait_list can be reused or freed after the function returns.

        都置 NULL。

    * `event` returns an event object that identifies this read / write command and can be used to query or queue a wait for this command to complete. If event is NULL or the enqueue is unsuccessful, no event will be created and therefore it will not be possible to query the status of this command or to wait for this command to complete. If event_wait_list and event are not NULL, event must not refer to an element of the event_wait_list array.

        置 NULL。

    Return value:

    `clEnqueueReadBuffer` and `clEnqueueWriteBuffer` return CL_SUCCESS if the function is executed successfully.

* `clRetainContext`

    增加 context 的引用计数，防止 opencl 库突然释放资源。

* `clCreateProgramWithSource`

    Syntax:

    ```c
    cl_program clCreateProgramWithSource(
        cl_context context,
        cl_uint count,
        const char** strings,
        const size_t* lengths,
        cl_int* errcode_ret);
    ```

    * `context` must be a valid OpenCL context.

    * `count`

        目前填 1，不知道有什么用。

    * `strings` is an array of count pointers to optionally null-terminated character strings that make up the source code.

        这里给的字符串，必须是使用`malloc`申请的内存才行。使用数组会报错。

    * `lengths` argument is an array with the number of chars in each string (the string length). If an element in lengths is zero, its accompanying string is null-terminated. If lengths is `NULL`, all strings in the strings argument are considered null-terminated. Any length value passed in that is greater than zero excludes the null terminator in its count.

        源代码字符串的实际长度。

    `errcode_ret` will return an appropriate error code. If errcode_ret is NULL, no error code is returned.

    The source code specified by strings will be loaded into the program object.

    The devices associated with the program object are the devices associated with context. The source code specified by strings is either an OpenCL C program source, header or implementation-defined source for custom devices that support an online compiler. OpenCL C++ is not supported as an online-compiled kernel language through this interface.

    `clCreateProgramWithSource` returns a valid non-zero program object and `errcode_ret` is set to `CL_SUCCESS` if the program object is created successfully. Otherwise, it returns a `NULL` value with one of the following error values returned in `errcode_ret`:

    * `CL_INVALID_CONTEXT` if context is not a valid context.

    * `CL_INVALID_VALUE` if count is zero or if strings or any entry in strings is NULL.

    * `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    * `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

* `clBuildProgram`

    Syntax:

    ```c
    cl_int clBuildProgram(
        cl_program program,
        cl_uint num_devices,
        const cl_device_id* device_list,
        const char* options,
        void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
        void* user_data);
    ```

    * `program` is the program object.

    * `num_devices` is the number of devices listed in `device_list`.

    * `device_list` is a pointer to a list of devices associated with program. If `device_list` is a `NULL` value, the program executable is built for all devices associated with program for which a source or binary has been loaded. If `device_list` is a non-NULL value, the program executable is built for devices specified in this list for which a source or binary has been loaded.

        这个参数是使用`clGetDeviceIDs()`得到的。

    * `options` is a pointer to a null-terminated string of characters that describes the build options to be used for building the program executable. The list of supported options is described in Compiler Options. If the program was created using clCreateProgramWithBinary and options is a `NULL` pointer, the program will be built as if options were the same as when the program binary was originally built. If the program was created using clCreateProgramWithBinary and options string contains anything other than the same options in the same order (whitespace ignored) as when the program binary was originally built, then the behavior is implementation defined. Otherwise, if options is a `NULL` pointer then it will have the same result as the empty string.

        目前这一项直接置`NULL`.

    * `pfn_notify` is a function pointer to a notification routine. The notification routine is a callback function that an application can register and which will be called when the program executable has been built (successfully or unsuccessfully). If pfn_notify is not NULL, clBuildProgram does not need to wait for the build to complete and can return immediately once the build operation can begin. Any state changes of the program object that result from calling clBuildProgram (e.g. build status or log) will be observable from this callback function. The build operation can begin if the context, program whose sources are being compiled and linked, list of devices and build options specified are all valid and appropriate host and device resources needed to perform the build are available. If pfn_notify is NULL, clBuildProgram does not return until the build has completed. This callback function may be called asynchronously by the OpenCL implementation. It is the applications responsibility to ensure that the callback function is thread-safe.

        目前这一项也直接置 NULL

    * `user_data` will be passed as an argument when pfn_notify is called. user_data can be NULL.

        这一项也是置 NULL

    The program executable is built from the program source or binary for all the devices, or a specific device(s) in the OpenCL context associated with program. OpenCL allows program executables to be built using the source or the binary. clBuildProgram must be called for program created using clCreateProgramWithSource, clCreateProgramWithIL or clCreateProgramWithBinary to build the program executable for one or more devices associated with program. If program is created with clCreateProgramWithBinary, then the program binary must be an executable binary (not a compiled binary or library).

    The executable binary can be queried using clGetProgramInfo(program, CL_PROGRAM_BINARIES, …​) and can be specified to clCreateProgramWithBinary to create a new program object.

    clBuildProgram returns CL_SUCCESS if the function is executed successfully. Otherwise, it returns one of the following errors:

    * `CL_INVALID_PROGRAM` if program is not a valid program object.

    * `CL_INVALID_VALUE` if device_list is NULL and num_devices is greater than zero, or if device_list is not NULL and num_devices is zero.

    * `CL_INVALID_VALUE` if pfn_notify is NULL but user_data is not NULL.

    * `CL_INVALID_DEVICE` if any device in device_list is not in the list of devices associated with program.

    * `CL_INVALID_BINARY` if program is created with clCreateProgramWithBinary and devices listed in device_list do not have a valid program binary loaded.

    * `CL_INVALID_BUILD_OPTIONS` if the build options specified by options are invalid.

    * `CL_COMPILER_NOT_AVAILABLE` if program is created with clCreateProgramWithSource or clCreateProgramWithIL and a compiler is not available, i.e. CL_DEVICE_COMPILER_AVAILABLE specified in the Device Queries table is set to CL_FALSE.

    * `CL_BUILD_PROGRAM_FAILURE` if there is a failure to build the program executable. This error will be returned if clBuildProgram does not return until the build has completed.

    * `CL_INVALID_OPERATION` if the build of a program executable for any of the devices listed in device_list by a previous call to clBuildProgram for program has not completed.

    * `CL_INVALID_OPERATION` if there are kernel objects attached to program.

    * `CL_INVALID_OPERATION` if program was not created with clCreateProgramWithSource, clCreateProgramWithIL or clCreateProgramWithBinary.

    * `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    * `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

* `clCreateKernel`

    Syntax:

    ```c
    cl_kernel clCreateKernel(
        cl_program program,
        const char* kernel_name,
        cl_int* errcode_ret);
    ```

    * `program` is a program object with a successfully built executable.

    * `kernel_name` is a function name in the program declared with the `__kernel` qualifier.

    * `errcode_ret` will return an appropriate error code. If errcode_ret is NULL, no error code is returned.

    `clCreateKernel` returns a valid non-zero kernel object and `errcode_ret` is set to `CL_SUCCESS` if the kernel object is created successfully. Otherwise, it returns a `NULL` value with one of the following error values returned in `errcode_ret`:

    * `CL_INVALID_PROGRAM` if program is not a valid program object.

    * `CL_INVALID_PROGRAM_EXECUTABLE` if there is no successfully built executable for program.

    * `CL_INVALID_KERNEL_NAME` if kernel_name is not found in program.

    * `CL_INVALID_KERNEL_DEFINITION` if the function definition for __kernel function given by kernel_name such as the number of arguments, the argument types are not the same for all devices for which the program executable has been built.

    * `CL_INVALID_VALUE` if kernel_name is NULL.

    * `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    * `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

* `clSetKernelArg`

    ```c
    cl_int clSetKernelArg(
        cl_kernel kernel,
        cl_uint arg_index,
        size_t arg_size,
        const void* arg_value);
    ```

    * `kernel` is a valid kernel object.

    * `arg_index` is the argument index. Arguments to the kernel are referred by indices that go from 0 for the leftmost argument to n - 1, where n is the total number of arguments declared by a kernel (see below).

    * `arg_size` specifies the size of the argument value. If the argument is a memory object, the `arg_size` value must be equal to `sizeof(cl_mem)`. For arguments declared with the local qualifier, the size specified will be the size in bytes of the buffer that must be allocated for the local argument. If the argument is of type `sampler_t`, the `arg_size` value must be equal to `sizeof(cl_sampler)`. If the argument is of type `queue_t`, the arg_size value must be equal to `sizeof(cl_command_queue)`. For all other arguments, the size will be the size of argument type.

    * `arg_value` is a pointer to data that should be used as the argument value for argument specified by arg_index. The argument data pointed to by arg_value is copied and the arg_value pointer can therefore be reused by the application after clSetKernelArg returns. The argument value specified is the value used by all API calls that enqueue kernel (clEnqueueNDRangeKernel and clEnqueueTask) until the argument value is changed by a call to clSetKernelArg for kernel.

    example:

    ```c
    kernel void image_filter (int n,
                            int m,
                            constant float *filter_weights,
                            read_only image2d_t src_image,
                            write_only image2d_t dst_image)
    {
    ...
    }
    ```

    Argument index values for `image_filter` will be `0` for `n`, `1` for `m`, `2` for `filter_weights`, `3` for src`_image and `4` for `dst_image`.

    If the argument is a memory object (buffer, pipe, image or image array), the arg_value entry will be a pointer to the appropriate buffer, pipe, image or image array object. The memory object must be created with the context associated with the kernel object. If the argument is a buffer object, the `arg_value` pointer can be `NULL` or point to a `NULL` value in which case a `NULL` value will be used as the value for the argument declared as a pointer to global or constant memory in the kernel. If the argument is declared with the local qualifier, the `arg_value` entry must be `NULL`. If the argument is of type `sampler_t`, the `arg_value` entry must be a pointer to the sampler object. If the argument is of type `queue_t`, the `arg_value` entry must be a pointer to the device queue object.

    If the argument is declared to be a pointer of a built-in scalar or vector type, or a user defined structure type in the global or constant address space, the memory object specified as argument value must be a buffer object (or `NULL`). If the argument is declared with the constant qualifier, the size in bytes of the memory object cannot exceed `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` and the number of arguments declared as pointers to constant memory cannot exceed `CL_DEVICE_MAX_CONSTANT_ARGS`.

    The memory object specified as argument value must be a pipe object if the argument is declared with the pipe qualifier.

    The memory object specified as argument value must be a 2D image object if the argument is declared to be of type image2d_t. The memory object specified as argument value must be a 2D image object with image channel order = CL_DEPTH if the argument is declared to be of type image2d_depth_t. The memory object specified as argument value must be a 3D image object if argument is declared to be of type image3d_t. The memory object specified as argument value must be a 1D image object if the argument is declared to be of type image1d_t. The memory object specified as argument value must be a 1D image buffer object if the argument is declared to be of type image1d_buffer_t. The memory object specified as argument value must be a 1D image array object if argument is declared to be of type image1d_array_t. The memory object specified as argument value must be a 2D image array object if argument is declared to be of type image2d_array_t. The memory object specified as argument value must be a 2D image array object with image channel order = CL_DEPTH if argument is declared to be of type image2d_array_depth_t.

    For all other kernel arguments, the arg_value entry must be a pointer to the actual data to be used as argument value.

    `clSetKernelArg` returns `CL_SUCCESS` if the function was executed successfully. Otherwise, it returns one of the following errors:

    `CL_INVALID_KERNEL` if kernel is not a valid kernel object.

    `CL_INVALID_ARG_INDEX` if arg_index is not a valid argument index.

    `CL_INVALID_ARG_VALUE` if arg_value specified is not a valid value.

    `CL_INVALID_MEM_OBJECT` for an argument declared to be a memory object when the specified arg_value is not a valid memory object.

    `CL_INVALID_SAMPLER` for an argument declared to be of type sampler_t when the specified arg_value is not a valid sampler object.

    `CL_INVALID_DEVICE_QUEUE` for an argument declared to be of type queue_t when the specified arg_value is not a valid device queue object. This error code is missing before version 2.0.

    `CL_INVALID_ARG_SIZE` if arg_size does not match the size of the data type for an argument that is not a memory object or if the argument is a memory object and arg_size != sizeof(cl_mem) or if arg_size is zero and the argument is declared with the local qualifier or if the argument is a sampler and arg_size != sizeof(cl_sampler).

    `CL_MAX_SIZE_RESTRICTION_EXCEEDED` if the size in bytes of the memory object (if the argument is a memory object) or arg_size (if the argument is declared with local qualifier) exceeds a language- specified maximum size restriction for this argument, such as the MaxByteOffset SPIR-V decoration. This error code is missing before version 2.2.

    `CL_INVALID_ARG_VALUE` if the argument is an image declared with the read_only qualifier and arg_value refers to an image object created with cl_mem_flags of CL_MEM_WRITE_ONLY or if the image argument is declared with the write_only qualifier and arg_value refers to an image object created with cl_mem_flags of CL_MEM_READ_ONLY.

    `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

    When `clSetKernelArg` returns an error code different from CL_SUCCESS, the internal state of kernel may only be modified when that error code is CL_OUT_OF_RESOURCES or CL_OUT_OF_HOST_MEMORY. When the internal state of kernel is modified, it is implementation-defined whether:

    The argument value that was previously set is kept so that it can be used in further kernel enqueues.

    The argument value is unset such that a subsequent kernel enqueue fails with `CL_INVALID_KERNEL_ARGS`. 

    Example:

    ```c
    cl_int ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    ```

* `clFlush`

    `command_queue` is the command queue to flush.

    All previously queued OpenCL commands in command_queue are issued to the device associated with `command_queue`. `clFlush` only guarantees that all queued commands to command_queue will eventually be submitted to the appropriate device. There is no guarantee that they will be complete after `clFlush` returns.

    Any blocking commands queued in a command-queue and `clReleaseCommandQueue` perform an implicit flush of the command-queue. These blocking commands are `clEnqueueReadBuffer`, `clEnqueueReadBufferRect`, `clEnqueueReadImage`, with `blocking_read` set to `CL_TRUE`; `clEnqueueWriteBuffer`, `clEnqueueWriteBufferRect`, `clEnqueueWriteImage` with `blocking_write` set to `CL_TRUE`; `clEnqueueMapBuffer`, `clEnqueueMapImage` with `blocking_map` set to `CL_TRUE`; `clEnqueueSVMMemcpy` with `blocking_copy` set to `CL_TRUE`; `clEnqueueSVMMap` with `blocking_map` set to `CL_TRUE` or clWaitForEvents.

    To use event objects that refer to commands enqueued in a command-queue as event objects to wait on by commands enqueued in a different command-queue, the application must call a clFlush or any blocking commands that perform an implicit flush of the command-queue where the commands that refer to these event objects are enqueued.

    `clFlush` returns `CL_SUCCESS` if the function call was executed successfully. Otherwise, it returns one of the following errors:

    * `CL_INVALID_COMMAND_QUEUE` if command_queue is not a valid host command-queue.

    * `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    * `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

* `clEnqueueNDRangeKernel`

    Syntax:

    ```c
    cl_int clEnqueueNDRangeKernel(
        cl_command_queue command_queue,
        cl_kernel kernel,
        cl_uint work_dim,
        const size_t* global_work_offset,
        const size_t* global_work_size,
        const size_t* local_work_size,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);
    ```

    To enqueue a command to execute a kernel on a device.

    * `command_queue` is a valid host command-queue. The kernel will be queued for execution on the device associated with `command_queue`.

    * `kernel` is a valid kernel object. The OpenCL context associated with kernel and command-queue must be the same.

    * `work_dim` is the number of dimensions used to specify the global work-items and work-items in the work-group. `work_dim` must be greater than zero and less than or equal to `CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS`. If `global_work_size` is `NULL`, or the value in any passed dimension is `0` then the kernel command will trivially succeed after its event dependencies are satisfied and subsequently update its completion event. The behavior in this situation is similar to that of an enqueued marker, except that unlike a marker, an enqueued kernel with no events passed to `event_wait_list` may run at any time.

    * `global_work_offset` can be used to specify an array of `work_dim` unsigned values that describe the offset used to calculate the global ID of a work-item. If `global_work_offset` is `NULL`, the global IDs start at offset `(0, 0, 0)`. `global_work_offset` must be NULL before version 1.1.

    * `global_work_size` points to an array of work_dim unsigned values that describe the number of global work-items in `work_dim` dimensions that will execute the kernel function. The total number of global work-items is computed as `global_work_size[0] × …​ × global_work_size[work_dim - 1]`.

    * `local_work_size` points to an array of work_dim unsigned values that describe the number of work-items that make up a work-group (also referred to as the size of the work-group) that will execute the kernel specified by kernel. The total number of work-items in a work-group is computed as `local_work_size[0] × …​ × local_work_size[work_dim - 1]`. The total number of work-items in the work-group must be less than or equal to the CL_KERNEL_WORK_GROUP_SIZE value specified in the Kernel Object Device Queries table, and the number of work-items specified in local_work_size[0], …​, local_work_size[work_dim - 1] must be less than or equal to the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], …​, CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]. The explicitly specified local_work_size will be used to determine how to break the global work-items specified by global_work_size into appropriate work-group instances.

    * `event_wait_list` and num_events_in_wait_list specify events that need to complete before this particular command can be executed. If `event_wait_list` is NULL, then this particular command does not wait on any event to complete. If `event_wait_list` is NULL, `num_events_in_wait_list` must be 0. If `event_wait_list` is not NULL, the list of events pointed to by `event_wait_list` must be valid and `num_events_in_wait_list` must be greater than 0. The events specified in event_wait_list act as synchronization points. The context associated with events in `event_wait_list` and command_queue must be the same. The memory associated with `event_wait_list` can be reused or freed after the function returns.

    * `event` returns an event object that identifies this command and can be used to query or wait for this command to complete. If event is NULL or the enqueue is unsuccessful, no event will be created and therefore it will not be possible to query the status of this command or to wait for this command to complete. If event_wait_list and event are not NULL, event must not refer to an element of the event_wait_list array.

    An ND-range kernel command may require uniform work-groups or may support non-uniform work-groups. To support non-uniform work-groups:

    1. The device associated with command_queue must support non-uniform work-groups.

    2. The program object associated with kernel must support non-uniform work-groups. Specifically, this means:

        1. If the program was created with clCreateProgramWithSource, the program must be compiled or built using the -cl-std=CL2.0 or -cl-std=CL3.0 build option and without the -cl-uniform-work-group-size build option.

        1. If the program was created with clCreateProgramWithIL or clCreateProgramWithBinary, the program must be compiled or built without the -cl-uniform-work-group-size build options.

        1. If the program was created using clLinkProgram, all input programs must support non-uniform work-groups.

    If non-uniform work-groups are supported, any single dimension for which the global size is not divisible by the local size will be partitioned into two regions. One region will have work-groups that have the same number of work-items as was specified by the local size parameter in that dimension. The other region will have work-groups with less than the number of work items specified by the local size parameter in that dimension. The global IDs and group IDs of the work-items in the first region will be numerically lower than those in the second, and the second region will be at most one work-group wide in that dimension. Work-group sizes could be non-uniform in multiple dimensions, potentially producing work-groups of up to 4 different sizes in a 2D range and 8 different sizes in a 3D range.

    If non-uniform work-groups are supported and local_work_size is NULL, the OpenCL runtime may choose a uniform or non-uniform work-group size.

    Otherwise, when non-uniform work-groups are not supported, the size of each work-group must be uniform. If local_work_size is specified, the values specified in global_work_size[0], …​, global_work_size[work_dim - 1] must be evenly divisible by the corresponding values specified in local_work_size[0], …​, local_work_size[work_dim - 1]. If local_work_size is NULL, the OpenCL runtime must choose a uniform work-group size.

    The work-group size to be used for kernel can also be specified in the program source or intermediate language. In this case the size of work-group specified by local_work_size must match the value specified in the program source.

    These work-group instances are executed in parallel across multiple compute units or concurrently on the same compute unit.

    Each work-item is uniquely identified by a global identifier. The global ID, which can be read inside the kernel, is computed using the value given by global_work_size and global_work_offset. In addition, a work-item is also identified within a work-group by a unique local ID. The local ID, which can also be read by the kernel, is computed using the value given by local_work_size. The starting local ID is always (0, 0, …​, 0).

    clEnqueueNDRangeKernel returns `CL_SUCCESS` if the kernel-instance was successfully queued. Otherwise, it returns one of the following errors:

    * `CL_INVALID_PROGRAM_EXECUTABLE` if there is no successfully built program executable available for device associated with command_queue.

    * `CL_INVALID_COMMAND_QUEUE` if command_queue is not a valid host command-queue.

    * `CL_INVALID_KERNEL` if kernel is not a valid kernel object.

    * `CL_INVALID_CONTEXT` if context associated with command_queue and kernel are not the same or if the context associated with command_queue and events in event_wait_list are not the same.

    * `CL_INVALID_KERNEL_ARGS` if the kernel argument values have not been specified.

    * `CL_INVALID_WORK_DIMENSION` if work_dim is not a valid value (i.e. a value between 1 and CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS).

    * `CL_INVALID_GLOBAL_WORK_SIZE` if global_work_size is NULL or if any of the values specified in global_work_size[0], …​global_work_size[work_dim - 1] are 0. Returning this error code under these circumstances is deprecated by version 2.1.

    * `CL_INVALID_GLOBAL_WORK_SIZE` if any of the values specified in global_work_size[0], …​ global_work_size[work_dim - 1] exceed the maximum value representable by size_t on the device on which the kernel-instance will be enqueued.

    * `CL_INVALID_GLOBAL_OFFSET` if the value specified in global_work_size + the corresponding values in global_work_offset for any dimensions is greater than the maximum value representable by size t on the device on which the kernel-instance will be enqueued, or if global_work_offset is non-NULL before version 1.1.

    * `CL_INVALID_WORK_GROUP_SIZE` if local_work_size is specified and does not match the required work-group size for kernel in the program source.

    * `CL_INVALID_WORK_GROUP_SIZE` if local_work_size is specified and is not consistent with the required number of sub-groups for kernel in the program source.

    * `CL_INVALID_WORK_GROUP_SIZE` if local_work_size is specified and the total number of work-items in the work-group computed as local_work_size[0] × …​ local_work_size[work_dim - 1] is greater than the value specified by CL_KERNEL_WORK_GROUP_SIZE in the Kernel Object Device Queries table.

    * `CL_INVALID_WORK_GROUP_SIZE` if the work-group size must be uniform and the local_work_size is not NULL, is not equal to the required work-group size specified in the kernel source, or the global_work_size is not evenly divisible by the local_work_size.

    * `CL_INVALID_WORK_ITEM_SIZE` if the number of work-items specified in any of local_work_size[0], …​ local_work_size[work_dim - 1] is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], …​, CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].

    * `CL_MISALIGNED_SUB_BUFFER_OFFSET` if a sub-buffer object is specified as the value for an argument that is a buffer object and the offset specified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue. This error code is missing before version 1.1.

    * `CL_INVALID_IMAGE_SIZE` if an image object is specified as an argument value and the image dimensions (image width, height, specified or compute row and/or slice pitch) are not supported by device associated with queue.

    * `CL_IMAGE_FORMAT_NOT_SUPPORTED` if an image object is specified as an argument value and the image format (image channel order and data type) is not supported by device associated with queue.

    * `CL_OUT_OF_RESOURCES` if there is a failure to queue the execution instance of kernel on the command-queue because of insufficient resources needed to execute the kernel. For example, the explicitly specified local_work_size causes a failure to execute the kernel because of insufficient resources such as registers or local memory. Another example would be the number of read-only image args used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of write-only and read-write image args used in kernel exceed the CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS value for device or the number of samplers used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device.

    * `CL_MEM_OBJECT_ALLOCATION_FAILURE` if there is a failure to allocate memory for data store associated with image or buffer objects specified as arguments to kernel.

    * `CL_INVALID_EVENT_WAIT_LIST` if event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events.

    * `CL_INVALID_OPERATION` if SVM pointers are passed as arguments to a kernel and the device does not support SVM or if system pointers are passed as arguments to a kernel and/or stored inside SVM allocations passed as kernel arguments and the device does not support fine grain system SVM allocations.

    * `CL_OUT_OF_RESOURCES` if there is a failure to allocate resources required by the OpenCL implementation on the device.

    * `CL_OUT_OF_HOST_MEMORY` if there is a failure to allocate resources required by the OpenCL implementation on the host.

## 函数调用流程简记

```c
clGetPlatformIDs
clGetDeviceIDs
clCreateContext
clCreateCommandQueue
clCreateBuffer
clEnqueueWriteBuffer
clCreateProgramWithSource
clBuildProgram
clCreateKernel
clSetKernelArg
clEnqueueNDRangeKernel
clEnqueueReadBuffer
clFlush
clFinish
clReleaseKernel
clReleaseProgram
clReleaseMemObject
clReleaseCommandQueue
clReleaseContext
```

## 另一个 example

这个程序主要用于给一个长为 1024 的随机数组排序。

`my_cl_op.cl`:

```c
void swap(__global int *a, __global int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

__kernel void cl_sort(__global int *A, __global int *n_elm)
{
    int n = *n_elm;
    int temp;
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - 1 - i; ++j)
        {
            if (A[j] > A[j+1])
            {
                swap(&A[j], &A[j+1]);
            }
        }
    }
}
```

`main.c`:

```c
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>

int main()
{
    cl_platform_id platform;
    cl_uint num_platforms;
    cl_int ret_val;
    ret_val = clGetPlatformIDs(1, &platform, &num_platforms);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to get platform id.\n");
        printf("return value: %d", ret_val);
        getchar();
        exit(-1);
    }

    cl_device_id device_id;
    cl_uint num_devices;
    ret_val = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to get device ids\n");
        printf("return value: %d", ret_val);
        getchar();
        exit(-1);
    }

    cl_context context;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create context\n");
        printf("return value: %d\n", ret_val);
        getchar();
        exit(-1);
    }

    cl_command_queue cq;
    cq = clCreateCommandQueue(context, device_id, NULL, &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create command queue\n");
        printf("return value: %d\n", ret_val);
        getchar();
        exit(-1);
    }

    const int ARR_SIZE = 1024;
    int *A = malloc(ARR_SIZE * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < ARR_SIZE; ++i)
    {
        A[i] = rand() % 1000;
    }
    cl_mem mem_A;
    mem_A = clCreateBuffer(context, CL_MEM_READ_WRITE, ARR_SIZE * sizeof(int), NULL, &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create buffer A\n");
        printf("return value: %d\n", ret_val);
        getchar();
        exit(-1);
    }

    int *N = malloc(1 * sizeof(int));
    *N = 1024;
    cl_mem mem_N;
    mem_N = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(int), NULL, &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create buffer N\n");
        printf("return value: %d\n", ret_val);
        getchar();
        exit(-1);
    }
    
    ret_val = clEnqueueWriteBuffer(cq, mem_A, CL_TRUE, 0, ARR_SIZE * sizeof(int), A, NULL, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to enqueue write buffer A\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    ret_val = clEnqueueWriteBuffer(cq, mem_N, CL_TRUE, 0, 1 * sizeof(int), N, NULL, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to enqueue write buffer N\n");
        printf("return value %d\n", ret_val);
        getchar();
    }

    FILE *f = fopen("my_cl_op.cl", "r");
    // char buf[1024] = {0};
    char *buf = malloc(1024 * sizeof(char));  // 必须用 malloc 申请的内存才行，如果用数组会报错
    int n_read = 0;
    n_read = fread(buf, sizeof(char), 1024, f);
    fclose(f);
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &buf, &n_read, &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create program with source\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    ret_val = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to build program\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    cl_kernel kernel;
    kernel = clCreateKernel(program, "cl_sort", &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create kernel\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    ret_val = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_A);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to set kernel arg mem_A\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    ret_val = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_N);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to set kernel arg mem_N\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    size_t global_item_size = ARR_SIZE;
    size_t local_item_size = ARR_SIZE;
    ret_val = clEnqueueNDRangeKernel(cq, kernel, 1, 0, &global_item_size, &local_item_size, 0, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to enqueue nd range kernel\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }
    
    ret_val = clEnqueueReadBuffer(cq, mem_A, CL_TRUE, 0, ARR_SIZE * sizeof(int), A, 0, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to enqueue read buffer\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    for (int i = 0; i < ARR_SIZE; ++i)
    {
        printf("%d, ", A[i]);
    }
    printf("\n");

    clFlush(cq);
    clFinish(cq);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(mem_A);
    clReleaseMemObject(mem_N);
    clReleaseCommandQueue(cq);
    clReleaseContext(context);

    free(A);
    free(N);
    free(buf);
    return 0;
}
```

## time based note

1. `CL_MEM_COPY_HOST_PTR`

    如果需要在创建 buffer 时就把数据从 host 上复制到 buffer 里，那么可以这样写：

    ```cpp
    cl_mem buf_rand_seed_vbuf = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ray_count * sizeof(cl_long), random_seeds, &ret);
    ```

    此时会把`random_seeds`中的数据写入到 buffer 中。

1. set arg

    猜测 set arg 的行为是在内存中开辟一个栈，存储一些数据，这些数据为作为 kernel 的参数。这些数据通常有两大类，一类是`cl_mem`，另一类是`cl_float3`，`cl_int`之类的数值。

    猜测 set arg 是按值复制，所以不用担心数值类的局部变量消失后参数失效的问题。由于`cl_mem`是个指针，如果不显式释放，其对应的 buffer （即显存）也不会被释放，所以也不用担心`cl_mem`失效。

    综上，一个`kernel`在整个程序的生命周期中，只需要设置一次 arg 就可以了。后面只需要 writer buffer, nd range kernel 就会触发计算。

## Problems shooting

* `Memory access fault by GPU node-1 (Agent handle: 0x55555670bd80) on address 0x7fffe0e00000. Reason: Page not present or supervisor privilege.`

    原因是 buffer 的 read 次数和 write 次数不匹配，多次循环后，导致 buffer 溢出。