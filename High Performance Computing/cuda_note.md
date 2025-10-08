# CUDA Note

## cache

* cuda 的跨进程 vram 访问

    见`ref_39`

    compile: `make`

    run:

    `./server`

    `./client`

    output:

    `server`:

    ```
    1, 4, 3, 1, 5, 1, 4, 0, specialized as int
    sizeof handle: 64
    cubuf: 0x7f53f1a00000
    ```

    `client`:

    ```
    cubuf 0x7f7fb9a00000
    1, 4, 3, 1, 5, 1, 4, 0, specialized as int
    cubuf 2: 0x7f7fb9c00000
    1, 4, 3, 1, 5, 1, 4, 0, specialized as int
    ```

    可以看到 cuda 的 mem handle 是一个 64 字节的数据结构。在一个进程里申请的 memory 到另一个进程里使用时，需要以 handle 导入的方式拿到 va。不同进程里，同一块 memory 的 va 不相同。使用 handle 拿到的 va，仍然具有 device p2p y访存的能力，但是需要提前把 p2p 开关打开。

    其中比较关键的两个函数为：`cudaIpcGetMemHandle()`, `cudaIpcOpenMemHandle()`。

* cuda 中代码对函数进行偏特化时，`nvcc`的报错

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include "../utils/cumem_hlc.h"
    #include "../utils/timeit.h"

    enum Op
    {
        op_sum,
        op_minus
    };

    template<typename T, Op op>
    __global__ void do_calc(T *a, T *b, T *out);

    template<typename T>
    __global__ void do_calc<T, op_sum>(T *a, T *b, T *out)
    {
        *out = *a + *b; 
    }

    template<typename T>
    __global__ void do_calc<T, op_minus>(T *a, T *b, T *out)
    {
        *out = *a - *b;
    }

    int main()
    {
        float *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, sizeof(float));
        cudaMalloc(&cubuf_2, sizeof(float));
        assign_cubuf_rand_int(cubuf_1, 1);
        assign_cubuf_rand_int(cubuf_2, 1);
        print_cubuf(cubuf_1, 1);
        print_cubuf(cubuf_2, 1);
        do_calc<float, op_sum><<<1, 1>>>(cubuf_1, cubuf_2, cubuf_1);
        cudaDeviceSynchronize();
        printf("after:\n");
        print_cubuf(cubuf_1, 1);
        print_cubuf(cubuf_2, 1);
        return 0;
    }
    ```

    compile: `nvcc -g -G main.cu -o main`

    compiling output:

    ```
    main_9.cu(21): error: function template "do_calc(T *, T *, T *)" has already been defined
      __attribute__((global)) void do_calc<T, op_minus>(T *a, T *b, T *out)
                                   ^

    1 error detected in the compilation of "main_9.cu".
    ```

    nvcc 编译器认为函数不能有偏特化，因此去找函数`do_calc()`的第一个实现，结果在代码中找到的是`do_calc<T, op_sum>()`。后来又找到了`do_calc<T, op_minus>()`的实现，因此就认为是重复定义了`do_calc()`，所以报错。

* `__any_sync()`

    syntax:

    ```cpp
    __any_sync(unsigned mask, predicate);
    ```

    在一个 warp 32 个线程中，每个线程提供一个元素`predicate`，只要有一个是非 0，那么`__any_sync()`返回非 0 值（实测为 1）。

    example:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void reduce_sum(elm_type *cubuf_1, elm_type *cubuf_2)
    {
        int tid = threadIdx.x;
        int predicate = tid;
        elm_type ret = __any_sync(0xffffffff, predicate);
        cubuf_1[tid] = ret;
        predicate = 0;
        ret = __any_sync(0xffffffff, predicate);
        cubuf_2[tid] = ret;
    }

    int main()
    {
        using elm_type = int;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));

        reduce_sum<elm_type><<<1, 32>>>(cubuf_1, cubuf_2);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);

        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    output:

    ```
    cubuf 1:
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, specialized as int
    cubuf 2:
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, specialized as int
    ```

* `T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);`

    手动指定`srcLane`，从`srcLane`拿数据。

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void my_kern(elm_type *cubuf_1, elm_type *cubuf_2)
    {
        int tid = threadIdx.x;
        elm_type val = __shfl_sync(0xffffffff, cubuf_1[tid], tid+2);
        cubuf_2[tid] = val;
    }

    int main()
    {
        using elm_type = float;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));
        assign_cubuf_order_int<elm_type>(cubuf_1, num_elm);

        my_kern<elm_type><<<1, 32>>>(cubuf_1, cubuf_2);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);
        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    output:

    ```
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 0.0, 1.0, specialized as float
    ```

    可以看到，这里是可以循环取值的。

* `__shfl_up_sync()`

    `__shfl_up_sync()`的作用与`__shfl_down_sync()`相似，只不过是向左 shift。

    syntax:

    ```cpp
    T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
    ```

* `__shfl_xor_sync()`

    syntax:

    ```cpp
    T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
    ```

    如果`laneMask`是偶数，那么按`laneMask`个元素左右交换数据。

    example:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void my_kern(elm_type *cubuf_1, elm_type *cubuf_2, int laneMask)
    {
        int tid = threadIdx.x;
        elm_type val = __shfl_xor_sync(0xffffffff, cubuf_1[tid], laneMask);
        cubuf_2[tid] = val;
    }

    int main()
    {
        using elm_type = float;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));
        assign_cubuf_order_int<elm_type>(cubuf_1, num_elm);

        int laneMask = 2;
        my_kern<elm_type><<<1, 32>>>(cubuf_1, cubuf_2, laneMask);
        cudaDeviceSynchronize();

        printf("laneMask = %d:\n", laneMask);
        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);
        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        laneMask = 4;
        my_kern<elm_type><<<1, 32>>>(cubuf_1, cubuf_2, laneMask);
        cudaDeviceSynchronize();

        putchar('\n');
        printf("laneMask = %d:\n", laneMask);
        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);
        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    output:

    ```
    laneMask = 2:
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0, 10.0, 11.0, 8.0, 9.0, 14.0, 15.0, 12.0, 13.0, 18.0, 19.0, 16.0, 17.0, 22.0, 23.0, 20.0, 21.0, 26.0, 27.0, 24.0, 25.0, 30.0, 31.0, 28.0, 29.0, specialized as float

    laneMask = 4:
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0, 8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 16.0, 17.0, 18.0, 19.0, 28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0, specialized as float
    ```

    如果将`laneMask`设置为 5，则会出现比较混乱的结果：

    ```
    laneMask = 5:
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    5.0, 4.0, 7.0, 6.0, 1.0, 0.0, 3.0, 2.0, 13.0, 12.0, 15.0, 14.0, 9.0, 8.0, 11.0, 10.0, 21.0, 20.0, 23.0, 22.0, 17.0, 16.0, 19.0, 18.0, 29.0, 28.0, 31.0, 30.0, 25.0, 24.0, 27.0, 26.0, specialized as float
    ```

    目前不知道原因。

* cuda 中`__shfl_down_sync()`的含义

    syntax:

    ```cpp
    T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
    ```

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void my_kern(elm_type *cubuf_1, elm_type *cubuf_2)
    {
        int tid = threadIdx.x;
        elm_type val = __shfl_down_sync(0xffffffff, cubuf_1[tid], 4);
        cubuf_2[tid] = val;
    }

    int main()
    {
        using elm_type = float;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));
        my_kern<elm_type>(cubuf_1, num_elm);

        sum_reduce<elm_type><<<1, 32>>>(cubuf_1, cubuf_2);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);
        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.cu
    	nvcc -g -G main.cu -o main

    clean:
    	rm -f main
    ```

    output:

    ```
    cubuf 1:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, specialized as float
    cubuf 2:
    3.0, 0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 2.0, 0.0, 4.0, 2.0, specialized as float
    ```

    可以看到，`cubuf 2`中的第 1 个元素`3.0`，是`cubuf 1`中的第 5 个元素；`cubuf 2`中的第 2 个元素`0.0`，是`cubuf 1`中的第 6 个元素。以此类推。

    `__shfl_down_sync()`的作用是每个线程出一个数据`val`，然后 thread 在 warp 内进行交换数据，函数的返回值就是交换完后得到的数据。

* 当 cuda kernel 中 printf 过多时，会出现乱码

    下面是截取的一段逐渐乱码的过程：

    ```
    after Unroll, Unroll: 4, BytePerPack: 16
    after Unroll, Unroll: 4, BytePerPack: 16
    after Unroll, Unroll: 4, BytePerPack: 16
    after Unroll, Unroll: 4, BytePerPack: 16
    after UnMoll, Unroll: 4, BytePerPack: 16
    after UnMill, Unroll: 4, BytePerPack: 16
    after UnMinS, Unroll: 4, BytePerPack: 16
    after UnMinSrcUnroll: 4, BytePerPack: 16
    after UnMinSrcs:roll: 4, BytePerPack: 16
    after UnMinSrcs: %: 4, BytePerPack: 16
    after UnMinSrcs: 4, MinDstytePerPack: 16
    after UnMinSrcs: 4, MinDsts: PerPack: 16
    after UnMinSrcs: 4, MinDsts: 16rPack: 1702127201
    after UnMinSrcs: 4, MinDsts: 16, Byk: 1702127201
    after UnMinSrcs: 4, MinDsts: 16, ByteP1702127201
    after UnMinSrcs: 4, MinDsts: 16, BytePer
    after UnMinSrcs: 4, MinDsts: 16, BytePerPacafter UnMinSrcs: 4, MinDsts: 16, BytePerPack:after UnMinSrcs: 4, MinDsts: 16, BytePerPack: after UnMinSrcs: 4, MinDsts: 16, BytePerPack: %after UnMinSrcs: 4, MinDsts: 16, BytePerPack: 1702127201
    after UnMinSrcs: 1, MinDsts: 16, BytePerPack: 1702127201
    ^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^P^PMinSrcs: 1, MinDsts: 1, BytePerPack: 16
    MinSrcs: 1, MinDsts: 1, BytePerPack: 16
    MinSrcs: 1, MinDsts: 1, BytePerPack: 16
    MinSrcs: 1, MinDsts: 1, BytePerPack: 16
    ```

* cuda 线程不同步的一个现象

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define N 64

    __global__ void test_kern(float *vec)
    {
        int x = threadIdx.x;

        float val = 0;
        for (int i = 0; i <= x; ++i)
        {
            val += vec[i];
        }
        vec[x] = val;
        // __syncthreads();

        printf("thread %d, val %.1f\n", x, vec[N - 1]);
    }

    int main()
    {
        // int N = 32;
        float *buf = (float*) malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i)
            buf[i] = i;
        float *cubuf;
        cudaMalloc(&cubuf, sizeof(float) * N);
        cudaMemcpy(cubuf, buf, sizeof(float) * N, cudaMemcpyHostToDevice);

        test_kern<<<1, N>>>(cubuf);
        cudaDeviceSynchronize();

        free(buf);
        return 0;
    }
    ```

    compile: `nvcc -g -G main.cu -o main`

    run: `./main`

    output:

    ```
    thread 0, val 63.0
    thread 1, val 63.0
    thread 2, val 63.0
    thread 3, val 63.0
    thread 4, val 63.0
    thread 5, val 63.0
    thread 6, val 63.0
    thread 7, val 63.0
    thread 8, val 63.0
    thread 9, val 63.0
    thread 10, val 63.0
    thread 11, val 63.0
    thread 12, val 63.0
    thread 13, val 63.0
    thread 14, val 63.0
    thread 15, val 63.0
    thread 16, val 63.0
    thread 17, val 63.0
    thread 18, val 63.0
    thread 19, val 63.0
    thread 20, val 63.0
    thread 21, val 63.0
    thread 22, val 63.0
    thread 23, val 63.0
    thread 24, val 63.0
    thread 25, val 63.0
    thread 26, val 63.0
    thread 27, val 63.0
    thread 28, val 63.0
    thread 29, val 63.0
    thread 30, val 63.0
    thread 31, val 63.0
    thread 32, val 2016.0
    thread 33, val 2016.0
    thread 34, val 2016.0
    thread 35, val 2016.0
    thread 36, val 2016.0
    thread 37, val 2016.0
    thread 38, val 2016.0
    thread 39, val 2016.0
    thread 40, val 2016.0
    thread 41, val 2016.0
    thread 42, val 2016.0
    thread 43, val 2016.0
    thread 44, val 2016.0
    thread 45, val 2016.0
    thread 46, val 2016.0
    thread 47, val 2016.0
    thread 48, val 2016.0
    thread 49, val 2016.0
    thread 50, val 2016.0
    thread 51, val 2016.0
    thread 52, val 2016.0
    thread 53, val 2016.0
    thread 54, val 2016.0
    thread 55, val 2016.0
    thread 56, val 2016.0
    thread 57, val 2016.0
    thread 58, val 2016.0
    thread 59, val 2016.0
    thread 60, val 2016.0
    thread 61, val 2016.0
    thread 62, val 2016.0
    thread 63, val 2016.0
    ```

    将` // __syncthreads();`取消注释后，重新编译运行，输出为

    ```
    thread 0, val 2016.0
    thread 1, val 2016.0
    thread 2, val 2016.0
    thread 3, val 2016.0
    thread 4, val 2016.0
    thread 5, val 2016.0
    thread 6, val 2016.0
    thread 7, val 2016.0
    thread 8, val 2016.0
    thread 9, val 2016.0
    thread 10, val 2016.0
    thread 11, val 2016.0
    thread 12, val 2016.0
    thread 13, val 2016.0
    thread 14, val 2016.0
    thread 15, val 2016.0
    thread 16, val 2016.0
    thread 17, val 2016.0
    thread 18, val 2016.0
    thread 19, val 2016.0
    thread 20, val 2016.0
    thread 21, val 2016.0
    thread 22, val 2016.0
    thread 23, val 2016.0
    thread 24, val 2016.0
    thread 25, val 2016.0
    thread 26, val 2016.0
    thread 27, val 2016.0
    thread 28, val 2016.0
    thread 29, val 2016.0
    thread 30, val 2016.0
    thread 31, val 2016.0
    thread 32, val 2016.0
    thread 33, val 2016.0
    thread 34, val 2016.0
    thread 35, val 2016.0
    thread 36, val 2016.0
    thread 37, val 2016.0
    thread 38, val 2016.0
    thread 39, val 2016.0
    thread 40, val 2016.0
    thread 41, val 2016.0
    thread 42, val 2016.0
    thread 43, val 2016.0
    thread 44, val 2016.0
    thread 45, val 2016.0
    thread 46, val 2016.0
    thread 47, val 2016.0
    thread 48, val 2016.0
    thread 49, val 2016.0
    thread 50, val 2016.0
    thread 51, val 2016.0
    thread 52, val 2016.0
    thread 53, val 2016.0
    thread 54, val 2016.0
    thread 55, val 2016.0
    thread 56, val 2016.0
    thread 57, val 2016.0
    thread 58, val 2016.0
    thread 59, val 2016.0
    thread 60, val 2016.0
    thread 61, val 2016.0
    thread 62, val 2016.0
    thread 63, val 2016.0
    ```

    我们根据不同 thread 编号的大小，让 thread 执行长度不同的任务，然后让所有 thread 取一个只有当所有 thread 都算完才能得到的结果。

    可以看到，0 到 31 号 thread 取到了相同的值，但是是错的；32 到 63 号 thread 取得的值相同，并且是正确的。加上`__syncthreads();`后，所有的值都是正确的。

* `cudaLaunchKernel()` example 2

    ```cpp
    #include <cuda_runtime.h>
    #include <stdlib.h>
    #include <stdio.h>

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

    void print_cubuf(float *cubuf, size_t num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < num_elm; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');
        free(buf);
    }

    __global__ void vec_add(float *A, float *B, float *C)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        C[x] = A[x] + B[x];
    }

    int main()
    {
        float *cubuf_A, *cubuf_B, *cubuf_C;
        cudaMalloc(&cubuf_A, 8 * sizeof(float));
        cudaMalloc(&cubuf_B, 8 * sizeof(float));
        cudaMalloc(&cubuf_C, 8 * sizeof(float));

        assign_cubuf_rand_int(cubuf_A, 8);
        assign_cubuf_rand_int(cubuf_B, 8);

        puts("cubuf_A:");
        print_cubuf(cubuf_A, 8);
        puts("cubuf_B:");
        print_cubuf(cubuf_B, 8);

        // void *args[3] = {&cubuf_A, &cubuf_B, &cubuf_C};
        void **args = (void**) malloc(3 * sizeof(void*));
        args[0] = &cubuf_A;
        args[1] = &cubuf_B;
        args[2] = &cubuf_C;
        cudaLaunchKernel((const void *) vec_add, dim3(2, 1, 1), dim3(4, 1, 1), args, 0, NULL);
        // cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

        puts("cubuf_C:");
        print_cubuf(cubuf_C, 8);

        cudaFree(cubuf_A);
        cudaFree(cubuf_B);
        cudaFree(cubuf_C);

        return 0;
    }
    ```

    output:

    ```
    cubuf_A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    cubuf_B:
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    cubuf_C:
    7.0, 2.0, 4.0, 2.0, 3.0, 4.0, 4.0, 3.0,
    ```

* `cudaLaunchKernel()`的 example

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdlib.h>
    #include <stdio.h>

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

    void print_cubuf(float *cubuf, size_t num_elm)
    {
        float *buf = (float*) malloc(num_elm * sizeof(float));
        cudaMemcpy(buf, cubuf, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < num_elm; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');
        free(buf);
    }

    __global__ void vec_add(float *A, float *B, float *C)
    {
        int x = threadIdx.x;
        C[x] = A[x] + B[x];
    }

    int main()
    {
        float *cubuf_A, *cubuf_B, *cubuf_C;
        cudaMalloc(&cubuf_A, 8 * sizeof(float));
        cudaMalloc(&cubuf_B, 8 * sizeof(float));
        cudaMalloc(&cubuf_C, 8 * sizeof(float));

        assign_cubuf_rand_int(cubuf_A, 8);
        assign_cubuf_rand_int(cubuf_B, 8);

        puts("cubuf_A:");
        print_cubuf(cubuf_A, 8);
        puts("cubuf_B:");
        print_cubuf(cubuf_B, 8);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        void *args[3] = {&cubuf_A, &cubuf_B, &cubuf_C};
        // void **args = (void**) malloc(3 * sizeof(void*));
        // args[0] = &cubuf_A;
        // args[1] = &cubuf_B;
        // args[2] = &cubuf_C;
        cudaLaunchKernel((const void *) vec_add, 1, 8, args, 0, stream);
        cudaStreamSynchronize(stream);

        puts("cubuf_C:");
        print_cubuf(cubuf_C, 8);

        cudaFree(cubuf_A);
        cudaFree(cubuf_B);
        cudaFree(cubuf_C);
        cudaStreamDestroy(stream);

        return 0;
    }
    ```

    compile:

    `nvcc -g -G main.cu -o main`

    run:

    `./main`

    output:

    ```
    cubuf_A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    cubuf_B:
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    cubuf_C:
    7.0, 2.0, 4.0, 2.0, 3.0, 4.0, 4.0, 3.0, 
    ```

    `cudaLaunchKernel()`，第一个参数是 kernel 函数地址，需要用`(const void *)`或`(void *)`类型转换一下，第二个参数是 grid dim，可以指定`dim3`，如果是一维的，直接指定 scalar 就可以了，第三个参数是 block dim，同参数二。

    第四个参数是 kernel 函数的参数列表，虽然指定的类型是`void**`，即`void*`的数组，但是实际传递的并不是 buffer 的 va，而是 buffer va 的 va。可以看到`void *args[3] = {&cubuf_A, &cubuf_B, &cubuf_C};`里，对`cubuf_A`等`float*`类型的变量又多加了一层取地址。不加这个会报 segment fault。

    第 5 个参数直接填 0 就可以，目前用不到。

    第 6 个参数可以填 stream，也可以填`NULL`，只不过这时要用`cudaDeviceSynchronize();`来阻塞等待 kernel launch。

* cuda shared memory 优化矩阵乘法的一个例子

    `main_4.cu`:

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>
    #include <cuda_runtime.h>

    #define BLOCK_SIZE 4

    typedef struct {
        int width;
        int height;
        int stride;
        float* elements;
    } Matrix;

    __global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

    void assign_mat_rand_int(float *m, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                m[i * n_cols + j] = rand() % 5;
            }
        }
    }

    void display_mat(float *mat, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                printf("%.1f, ", mat[i * n_cols + j]);
            }
            putchar('\n');
        }
    }

    int main()
    {
        Matrix A, B, C;
        A.width = 8;
        A.height = 8;
        A.elements = (float*) malloc(A.width * A.height * sizeof(float));
        B.width = 8;
        B.height = 8;
        B.elements = (float*) malloc(B.width * B.height * sizeof(float));
        C.width = 8;
        C.height = 8;
        C.elements = (float*) malloc(C.width * C.height * sizeof(float));

        assign_mat_rand_int(A.elements, A.height, A.width);
        assign_mat_rand_int(B.elements, B.height, B.width);

        puts("A:");
        display_mat(A.elements, A.height, A.width);
        puts("B:");
        display_mat(B.elements, B.height, B.width);

        Matrix d_A;
        d_A.width = d_A.stride = A.width;
        d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

        Matrix d_B;
        d_B.width = d_B.stride = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

        Matrix d_C;
        d_C.width = d_C.stride = C.width;
        d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

        puts("C:");
        display_mat(C.elements, C.height, C.width);

        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        free(A.elements);
        free(B.elements);
        free(C.elements);

        return 0;
    }

    __device__ float GetElement(const Matrix A, int row, int col)
    {
        return A.elements[row * A.stride + col];
    }

    __device__ void SetElement(Matrix A, int row, int col, float value)
    {
        A.elements[row * A.stride + col] = value;
    }

     __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
    {
        Matrix Asub;
        Asub.width    = BLOCK_SIZE;
        Asub.height   = BLOCK_SIZE;
        Asub.stride   = A.stride;
        Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
        return Asub;
    }

     __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
    {
        int blockRow = blockIdx.y;
        int blockCol = blockIdx.x;
        Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
        float Cvalue = 0;
        int row = threadIdx.y;
        int col = threadIdx.x;
        for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
        {
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            Matrix Bsub = GetSubMatrix(B, m, blockCol);
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
            As[row][col] = GetElement(Asub, row, col);
            Bs[row][col] = GetElement(Bsub, row, col);
            __syncthreads();
            for (int e = 0; e < BLOCK_SIZE; ++e)
                Cvalue += As[row][e] * Bs[e][col];
            __syncthreads();
        }
        SetElement(Csub, row, col, Cvalue);
    }
    ```

    compile: `nvcc -g -G main_4.cu -o main_4`

    run: `./main_4`

    output:

    ```
    A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 4.0, 
    2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 
    2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 1.0, 2.0, 
    4.0, 3.0, 1.0, 4.0, 4.0, 2.0, 3.0, 4.0, 
    0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 
    2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0, 2.0, 
    B:
    1.0, 0.0, 1.0, 4.0, 3.0, 2.0, 4.0, 0.0, 
    2.0, 0.0, 4.0, 2.0, 4.0, 4.0, 3.0, 0.0, 
    2.0, 3.0, 1.0, 3.0, 3.0, 4.0, 3.0, 1.0, 
    4.0, 4.0, 2.0, 0.0, 1.0, 3.0, 4.0, 2.0, 
    1.0, 1.0, 4.0, 4.0, 0.0, 0.0, 4.0, 3.0, 
    2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 
    1.0, 0.0, 1.0, 1.0, 0.0, 4.0, 4.0, 4.0, 
    0.0, 4.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 
    C:
    13.0, 17.0, 24.0, 37.0, 23.0, 24.0, 39.0, 15.0, 
    29.0, 22.0, 26.0, 37.0, 34.0, 47.0, 54.0, 22.0, 
    19.0, 30.0, 24.0, 28.0, 25.0, 33.0, 35.0, 18.0, 
    24.0, 28.0, 24.0, 30.0, 19.0, 39.0, 52.0, 30.0, 
    30.0, 32.0, 39.0, 45.0, 38.0, 46.0, 57.0, 22.0, 
    39.0, 41.0, 52.0, 56.0, 43.0, 56.0, 80.0, 35.0, 
    12.0, 26.0, 13.0, 20.0, 16.0, 22.0, 24.0, 12.0, 
    12.0, 15.0, 11.0, 19.0, 14.0, 29.0, 33.0, 19.0,
    ```

    这个例子先按行切第一个矩阵，按列切第二个矩阵，切完后，再将第一个子矩阵竖着切成一小块一小块，每一个小块都是边长为`BLOCK_SIZE`的小正方形。用`m`来标记当前进行到了第几个小正方形。

    `__shared__`关键字来申请 shared memory，用来存放小正方形。`GetElement()`用于往小正方形中填充数据，每个线程填充一个元素。

    `e`用于标记小正方形单行/单列的各个元素序号。因为直接使用的二维坐标来标记线程，所以相当于脱掉了矩阵乘法三层循环的外面两层。

    `__syncthreads();`用于做线程同步，每次等矩阵数据加载完成，或矩阵乘法计算完成后，翥需要同步一下。

* cuda 实现矩阵乘法的一个例子

    由 cuda programming guide 里的一个 example 改编而来。

    `main_3.cu`:

    ```cpp
    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime.h>

    // Matrices are stored in row-major order:
    // M(row, col) = *(M.elements + row * M.width + col)
    typedef struct {
        int width;
        int height;
        float* elements;
    } Matrix;

    #define BLOCK_SIZE 4  // threads per block

    __global__ void MatMulKernel(const Matrix mat_1, const Matrix mat_2, Matrix mat_out);

    void assign_mat_rand_int(float *m, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                m[i * n_cols + j] = rand() % 5;
            }
        }
    }

    void display_mat(float *mat, int n_rows, int n_cols)
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                printf("%.1f, ", mat[i * n_cols + j]);
            }
            putchar('\n');
        }
    }

    int main()
    {
        Matrix A, B, C;
        A.width = 8;
        A.height = 8;
        A.elements = (float*) malloc(A.width * A.height * sizeof(float));
        B.width = 8;
        B.height = 8;
        B.elements = (float*) malloc(B.width * B.height * sizeof(float));
        C.width = 8;
        C.height = 8;
        C.elements = (float*) malloc(C.width * C.height * sizeof(float));

        assign_mat_rand_int(A.elements, A.height, A.width);
        assign_mat_rand_int(B.elements, B.height, B.width);

        puts("A:");
        display_mat(A.elements, A.height, A.width);
        puts("B:");
        display_mat(B.elements, B.height, B.width);

        Matrix d_A;
        d_A.width = A.width; d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

        Matrix d_B;
        d_B.width = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

        Matrix d_C;
        d_C.width = C.width; d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);

        puts("C:");
        display_mat(C.elements, C.height, C.width);

        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        free(A.elements);
        free(B.elements);
        free(C.elements);

        return 0;
    }

    __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
    {
        float Cvalue = 0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.width; ++e)
            Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
        C.elements[row * C.width + col] = Cvalue;
    }
    ```

    compile: `nvcc -g -G main_3.cu -o main_3`

    run: `./main_3`

    output:

    ```
    A:
    3.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 
    4.0, 1.0, 2.0, 2.0, 0.0, 4.0, 3.0, 1.0, 
    0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 4.0, 
    2.0, 0.0, 2.0, 3.0, 2.0, 0.0, 4.0, 2.0, 
    2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 1.0, 2.0, 
    4.0, 3.0, 1.0, 4.0, 4.0, 2.0, 3.0, 4.0, 
    0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 
    2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0, 2.0, 
    B:
    1.0, 0.0, 1.0, 4.0, 3.0, 2.0, 4.0, 0.0, 
    2.0, 0.0, 4.0, 2.0, 4.0, 4.0, 3.0, 0.0, 
    2.0, 3.0, 1.0, 3.0, 3.0, 4.0, 3.0, 1.0, 
    4.0, 4.0, 2.0, 0.0, 1.0, 3.0, 4.0, 2.0, 
    1.0, 1.0, 4.0, 4.0, 0.0, 0.0, 4.0, 3.0, 
    2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 
    1.0, 0.0, 1.0, 1.0, 0.0, 4.0, 4.0, 4.0, 
    0.0, 4.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 
    C:
    13.0, 17.0, 24.0, 37.0, 23.0, 24.0, 39.0, 15.0, 
    29.0, 22.0, 26.0, 37.0, 34.0, 47.0, 54.0, 22.0, 
    19.0, 30.0, 24.0, 28.0, 25.0, 33.0, 35.0, 18.0, 
    24.0, 28.0, 24.0, 30.0, 19.0, 39.0, 52.0, 30.0, 
    30.0, 32.0, 39.0, 45.0, 38.0, 46.0, 57.0, 22.0, 
    39.0, 41.0, 52.0, 56.0, 43.0, 56.0, 80.0, 35.0, 
    12.0, 26.0, 13.0, 20.0, 16.0, 22.0, 24.0, 12.0, 
    12.0, 15.0, 11.0, 19.0, 14.0, 29.0, 33.0, 19.0,
    ```

    这个 example 是对 A 矩阵横着切，B 矩阵竖着切，每个 kernel 只计算一行/一列。通过 grid 保证所有的行/列都会被覆盖到。

* cuda 中同一 block 中的 thread 可以有共享显存（shared memory），这个共享显存通常是 L1 cache

* 3d grid 的使用

    `main_3.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define N_0 16
    #define N_1 4
    #define N_2 8

    __global__ void volume_add(float A[N_0][N_1][N_2], float B[N_0][N_1][N_2])
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        A[x][y][z] = A[x][y][z] + B[x][y][z];
    }

    int main()
    {
        float *cubuf_A, *cubuf_B;
        cudaMalloc(&cubuf_A, N_0 * N_1 * N_2 * sizeof(float));
        cudaMalloc(&cubuf_B, N_0 * N_1 * N_2 * sizeof(float));

        float buf_A[N_0][N_1][N_2];
        float buf_B[N_0][N_1][N_2];
        for (int i = 0; i < N_0; ++i)
        {
            for (int j = 0; j < N_1; ++j)
            {
                for (int k = 0; k < N_2; ++k)
                {
                    buf_A[i][j][k] = rand() % 5;
                    buf_B[i][j][k] = rand() % 5;
                }
            }
        }

        size_t num_elm = N_0 * N_1 * N_2;

        printf("buf A:\n");
        for (int i = 0; i < num_elm; ++i)
            printf("%.1f, ", buf_A[i / (N_1 * N_2)][i / N_2 % N_1][i % N_2]);
        putchar('\n');

        printf("buf B:\n");
        for (int i = 0; i < num_elm; ++i)
            printf("%.1f, ", buf_B[i / (N_1 * N_2)][i / N_2 % N_1][i % N_2]);
        putchar('\n');

        cudaMemcpy(cubuf_A, buf_A, num_elm * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cubuf_B, buf_B, num_elm * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 thread_per_block(2, 2, 2);
        dim3 block_per_grid(8, 2, 4);
        volume_add<<<block_per_grid, thread_per_block>>>((float(*)[N_1][N_2]) cubuf_A, (float(*)[N_1][N_2]) cubuf_B);
        cudaMemcpy(buf_A, cubuf_A, num_elm * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("buf A:\n");
        for (int i = 0; i < num_elm; ++i)
            printf("%.1f, ", buf_A[i / (N_1 * N_2)][i / N_2 % N_1][i % N_2]);
        putchar('\n');

        return 0;
    }
    ```

    compile: `nvcc -g -G main_3.cu -o main_3`

    run: `./main_3`

    output:

    ```
    buf A:
    3.0, 2.0, 3.0, 1.0, 4.0, ...
    buf B:
    1.0, 0.0, 0.0, 2.0, 1.0, ...
    buf A:
    4.0, 2.0, 3.0, 3.0, 5.0, ...
    ```

    可以看到，我们的线程组总是`(2, 2, 2)`，但是每个 grid 中有`(8, 2, 4)`个 block，一共 1 个 grid。

    同一个 block 中的多个 threads 是保证并行执行的，但是 block 与 block 之间并不保证并行。

    其实 grid 中的 block 有点像 batch 的感觉。

* cuda 实现 vec add

    见`ref_32`

* block 也可以被组织为一／二／三维的 grid。这么做主要为了适配需要计算的数据。通常数据的 dim length 是会超过 gpu 中流处理器的数量的

    example:

    `main_3.cu`:

    ```cpp
    __global__ void vec_add_blk(float *A, float *B)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        A[x] = A[x] + B[x];
        printf("%d, %d, %d\n", blockDim.x, blockDim.y, blockDim.z);
    }

    int main()
    {
        float *cubuf_A, *cubuf_B;
        cudaMalloc(&cubuf_A, 8 * sizeof(float));
        cudaMalloc(&cubuf_B, 8 * sizeof(float));
        float *buf_A, *buf_B;
        buf_A = (float*) malloc(8 * sizeof(float));
        buf_B = (float*) malloc(8 * sizeof(float));
        for (int i = 0; i < 8; ++i)
        {
            buf_A[i] = rand() % 5;
            buf_B[i] = rand() % 5;
        }

        printf("buf A:\n");
        for (int i = 0; i < 8; ++i)
            printf("%.2f, ", buf_A[i]);
        putchar('\n');

        printf("buf B:\n");
        for (int i = 0; i < 8; ++i)
            printf("%.2f, ", buf_B[i]);
        putchar('\n');

        cudaMemcpy(cubuf_A, buf_A, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cubuf_B, buf_B, 8 * sizeof(float), cudaMemcpyHostToDevice);
        // vec_add<<<1, 8>>>(cubuf_A, cubuf_B);
        vec_add_blk<<<4, 2>>>(cubuf_A, cubuf_B);
        cudaMemcpy(buf_A, cubuf_A, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("buf A:\n");
        for (int i = 0; i < 8; ++i)
            printf("%.2f, ", buf_A[i]);
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    buf A:
    3.00, 2.00, 3.00, 1.00, 4.00, 2.00, 0.00, 3.00, 
    buf B:
    1.00, 0.00, 0.00, 2.00, 1.00, 2.00, 4.00, 1.00, 
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    2, 1, 1
    buf A:
    4.00, 2.00, 3.00, 3.00, 5.00, 4.00, 4.00, 4.00,
    ```

    可以看到`blockDim`其实就是每个维度的 length。

    grid 有点像 batch 的概念，比如一个 shape 为`(20, 8, 4)`的数据，我既可以一次性处理，也可以拆分成 4 个`(5, 2, 1)`进行处理。前者的 grid 即为 1，后者的 grid 即为`(4, 4, 4)`。

    如果拆成 grid 进行处理，那么在确定数组索引时，就需要用`int x = blockIdx.x * blockDim.x + threadIdx.x;`这种方式。

    grid 与 grid 之间并不保证是并行执行的，可能是并行，也可能是串行。

    > Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series.

* 一个 block (thread block) 中的 threads 数量是有限的，因为每个 block 会被绑定到一个 sm (streaming multiprocessor) 上，这个 block 中的所有 thread 都会在这个 sm 上执行。

    > On current GPUs, a thread block may contain up to 1024 threads.

    当前的 gpu，每个 block 最多有 1024 个 thread。

* `cudaPointerGetAttributes()`用法

    `main_6.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        float *cubuf;
        cudaMalloc(&cubuf, 8 * sizeof(float));
        
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, cubuf);
        printf("cubuf: %p\n", cubuf);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        
        cudaFree(cubuf);
        return 0;
    }
    ```

    compile: `nvcc -g main_6.cu -o main_6`

    run: `./main_6`

    output:

    ```
    cubuf: 0x7fd3ffa00000
    attr.device: 0
    attr.devicePointer: 0x7fd3ffa00000
    attr.hostPointer: (nil)
    attr.type: 2
    ```

    可以看到，`cudaPointerGetAttributes()`可以拿到 ptr 的 device, addr，以及 type 这三个信息。

    type 一共就 4 种：

    ```cpp
    /**
     * CUDA memory types
     */
    enum __device_builtin__ cudaMemoryType
    {
        cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
        cudaMemoryTypeHost         = 1, /**< Host memory */
        cudaMemoryTypeDevice       = 2, /**< Device memory */
        cudaMemoryTypeManaged      = 3  /**< Managed memory */
    };
    ```

    如果传递给它的是`malloc()`申请的内存，则会返回 nil：

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        float *buf = (float*) malloc(8 * sizeof(float));

        cudaPointerAttributes attr;
        cudaError_t ret = cudaPointerGetAttributes(&attr, buf);
        if (ret != cudaSuccess)
        {
            printf("fail to get pointer attr, ret: %d\n", ret);
            return -1;
        }
        printf("cubuf: %p\n", buf);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        
        free(buf);
        return 0;
    }
    ```

    output:

    ```
    cubuf: 0x561a01324520
    attr.device: -2
    attr.devicePointer: (nil)
    attr.hostPointer: (nil)
    attr.type: 0
    ```

    `cudaPointerGetAttributes()`还能对 range 进行判断：

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        float *cubuf;
        cudaMalloc(&cubuf, 8 * sizeof(float));
        
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, cubuf + 7);
        printf("cubuf: %p\n", cubuf + 7);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        putchar('\n');

        cudaPointerGetAttributes(&attr, cubuf + 8);
        printf("cubuf: %p\n", cubuf + 8);
        printf("attr.device: %d\n", attr.device);
        printf("attr.devicePointer: %p\n", attr.devicePointer);
        printf("attr.hostPointer: %p\n", attr.hostPointer);
        printf("attr.type: %d\n", attr.type);
        
        cudaFree(cubuf);
        return 0;
    }
    ```

    output:

    ```
    cubuf: 0x7f764ba0001c
    attr.device: 0
    attr.devicePointer: 0x7f764ba0001c
    attr.hostPointer: (nil)
    attr.type: 2

    cubuf: 0x7f764ba00020
    attr.device: -2
    attr.devicePointer: (nil)
    attr.hostPointer: (nil)
    attr.type: 0
    ```

* 不需要 enable peer access，也可以使用`cudaMemcpy()`将数据从一个 device 复制到另一个 device

    `main_5.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        cudaSetDevice(0);
        float *cubuf_0;
        cudaMalloc(&cubuf_0, 8 * sizeof(float));
        cudaSetDevice(1);
        float *cubuf_1;
        cudaMalloc(&cubuf_1, 8 * sizeof(float));

        // cudaSetDevice(0);
        float *buf = (float*) malloc(8 * sizeof(float));
        for (int i = 0; i < 8; ++i)
            buf[i] = 123;
        cudaMemcpy(cubuf_0, buf, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaError_t ret;
        ret = cudaMemcpy(cubuf_1, cubuf_0, 8 * sizeof(float), cudaMemcpyDeviceToDevice);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy cubuf_0 to cubuf_1\n");
            return -1;
        }

        // cudaSetDevice(1);
        cudaMemcpy(buf, cubuf_1, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 8; ++i)
            printf("%.1f, ", buf[i]);
        putchar('\n');

        cudaFree(cubuf_0);
        cudaFree(cubuf_1);
        free(buf);
        return 0;
    }
    ```

    compile: `nvcc -g main_5.cu -o main_5`

    run: `./main_5`

    output:

    ```
    123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0, 123.0,
    ```

    如果 enable peer access，则可以直接在 launch kernel 时，使用 peer device 的指针，省去了 cuda memcpy 将数据从当前 deivce 复制到 peer device 的步骤。

    说明：

    1. `cudaMemcpy()`不需要显式用`cudaSetDevice()`指定 device。看起来只要有能辨别 device 的信息（比如指针，device id），就不需要显式指定 device。

* cuda 不同进程间的 va 分配情况

    运行下面的程序 10 次，

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        cudaSetDevice(0);
        int *cubuf_0;
        cudaMalloc(&cubuf_0, 8 * sizeof(float));
        printf("cubuf 0: %p\n", cubuf_0);

        cudaSetDevice(1);
        int *cubuf_1;
        cudaMalloc(&cubuf_1, 8 * sizeof(float));
        printf("cubuf 1: %p\n", cubuf_1);

        cudaFree(cubuf_0);
        cudaFree(cubuf_1);
        return 0;
    }
    ```

    输出如下：

    ```
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f0907a00000
    cubuf 1: 0x7f08f3a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8a47a00000
    cubuf 1: 0x7f8a2fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f68ada00000
    cubuf 1: 0x7f6893a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f441da00000
    cubuf 1: 0x7f4403a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f4d47a00000
    cubuf 1: 0x7f4d2fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8063a00000
    cubuf 1: 0x7f8049a00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8277a00000
    cubuf 1: 0x7f825fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f1857a00000
    cubuf 1: 0x7f183fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7f8197a00000
    cubuf 1: 0x7f817fa00000
    (base) huliucheng@zjxj:~/Documents/Projects/cumem_test$ ./main_4
    cubuf 0: 0x7fefe5a00000
    cubuf 1: 0x7fefcba00000
    ```

    可以看到，所有的 va 都以`0x7f`开头，后续的两位 16 进制数是根据进程随机分配的，再往后两位不固定，再往后一位总是`a`。

* cuda 在同一个进程里分配的 va range 相同

    `main_4.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        int num_round = 5;
        for (int i = 0; i < num_round; ++i)
        {
            printf("round: %d\n", i);

            cudaSetDevice(0);
            int *cubuf_0;
            cudaMalloc(&cubuf_0, 8 * sizeof(float));
            printf("cubuf 0: %p\n", cubuf_0);

            cudaSetDevice(1);
            int *cubuf_1;
            cudaMalloc(&cubuf_1, 8 * sizeof(float));
            printf("cubuf 1: %p\n", cubuf_1);

            cudaFree(cubuf_0);
            cudaFree(cubuf_1);

            putchar('\n');
        }

        return 0;
    }
    ```

    compile: `nvcc -g main_4.cu -o main_4`

    run: `./main_4`

    output:

    ```
    round: 0
    cubuf 0: 0x7ff44da00000
    cubuf 1: 0x7ff433a00000

    round: 1
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000

    round: 2
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000

    round: 3
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000

    round: 4
    cubuf 0: 0x7ff433a00000
    cubuf 1: 0x7ff433c00000
    ```

    注释掉`cudaFree()`，代码变为

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        int num_round = 5;
        for (int i = 0; i < num_round; ++i)
        {
            printf("round: %d\n", i);

            cudaSetDevice(0);
            int *cubuf_0;
            cudaMalloc(&cubuf_0, 8 * sizeof(float));
            printf("cubuf 0: %p\n", cubuf_0);

            cudaSetDevice(1);
            int *cubuf_1;
            cudaMalloc(&cubuf_1, 8 * sizeof(float));
            printf("cubuf 1: %p\n", cubuf_1);

            // cudaFree(cubuf_0);
            // cudaFree(cubuf_1);

            putchar('\n');
        }

        return 0;
    }
    ```

    output:

    ```
    round: 0
    cubuf 0: 0x7fa897a00000
    cubuf 1: 0x7fa883a00000

    round: 1
    cubuf 0: 0x7fa897a00200
    cubuf 1: 0x7fa883a00200

    round: 2
    cubuf 0: 0x7fa897a00400
    cubuf 1: 0x7fa883a00400

    round: 3
    cubuf 0: 0x7fa897a00600
    cubuf 1: 0x7fa883a00600

    round: 4
    cubuf 0: 0x7fa897a00800
    cubuf 1: 0x7fa883a00800
    ```

    可以看到，地址每次都增加`0x200`，猜测 page size 为 512 Byte。

    cu mem 的最小显存分配粒度为 2M，为什么这里可以做到 512 Byte？

* cuda peer access

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    __global__ void arr_inc_1(float *arr)
    {
        int i = threadIdx.x;
        arr[i] += 1;
    }

    int main()
    {    
        float* p0;
        cudaSetDevice(0);
        cudaMalloc(&p0, 4 * sizeof(float));
        cudaMemset(&p0, 0, 4 * sizeof(float));
        // vec_add<<<1, 4>>>(p0);

        cudaSetDevice(1);
        int canAccessPeer;
        cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
        if (!canAccessPeer)
        {
            printf("fail to access peer 0\n");
            return -1;
        }
        cudaDeviceEnablePeerAccess(0, 0);
        arr_inc_1<<<1, 4>>>(p0);

        cudaSetDevice(0);
        float buf[4] = {0};
        cudaMemcpy(buf, p0, 4 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; ++i)
        {
            printf("%.1f, ", buf[i]);
        }
        putchar('\n');

        cudaFree(p0);
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -o main`

    run: `./main`

    output:

    ```
    1.0, 1.0, 1.0, 1.0,
    ```

    我们在 dev 0 上申请显存，然后在 dev 1 上 enable dev 0 的 peer access，再在 dev 1 上 launch kernel，用的数据是 dev 0 上的数据，最后把 dev 0 里的数据拿出来，可以看到是正确的结果。

    说明：

    1. `cudaDeviceEnablePeerAccess(0, 0);`表示从当前 device （由`cudaSetDevice(1);`设定）可以获取 remote device (dev 0) 上的数据，是单向链路。而不是 dev 0 的数据可以由任何其他 dev 获取。

    2. 根据官网资料，peer access 可能走的是 pcie 或 nvlink

        > Depending on the system properties, specifically the PCIe and/or NVLINK topology, devices are able to address each other’s memory

        是否可以走网络或者 host 中转？目前不清楚。

        这里的 peer access 似乎更关注虚拟地址的处理，而不是底层通路。

    3. 根据官网资料，一个 dev 似乎最多能 peer access 8 个其他 dev

        > On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections.

* 在 50 机器上写如下程序

    `main.cu`:

    ```cpp
    #include <cuda.h>
    #include <stdlib.h>
    #include <stdio.h>

    __global__ void vec_add(float *A, float *B, float *C)
    {
        int id = blockIdx.x;
        C[id] = A[id] + B[id];
    }

    int main()
    {
        float *h_A, *h_B, *h_C;
        h_A = (float*) malloc(8 * sizeof(float));
        h_B = (float*) malloc(8 * sizeof(float));
        h_C = (float*) malloc(8 * sizeof(float));
        for (int i = 0; i < 8; ++i)
        {
            h_A[i] = rand() % 10;
            h_B[i] = rand() % 10;
        }
        float *A, *B, *C;
        cudaMalloc(&A, 8 * sizeof(float));
        cudaMalloc(&B, 8 * sizeof(float));
        cudaMalloc(&C, 8 * sizeof(float));
        cudaMemcpy(A, h_A, 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B, h_B, 8 * sizeof(float), cudaMemcpyHostToDevice);
        vec_add<<<8, 1>>>(A, B, C);
        cudaMemcpy(h_C, C, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 8; ++i)
        {
            printf("%.1f + %.1f = %.1f\n", h_A[i], h_B[i], h_C[i]);
        }
        return 0;
    }
    ```

    `Makefile`:

    ```makefile
    main: main.cu
    	nvcc -g -I../cuda-samples-12.1/Common -o main main.cu

    clean:
    	rm -f main
    ```

    在 vscode 中，使用如下`launch.json`:

    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [{
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/main"
        }]
    }
    ```

    在`vec_add()`中设置断点后，F5 运行无法 hit 断点。目前不清楚原因。

    （2024.12.20）：目前看来，应该是编译时没加`-G`。

* 224 机器上的 nccl cuda-gdb 依然很慢，起码需要半个小时以上才能 hit 断点

* 使用 b address `(cuda-gdb) b *0x7ffe85823040`会导致直接 cuda-gdb 直接报错退出。

    ```
    (cuda-gdb) b *0x7ffe85823040
    Breakpoint 3 at 0x7ffe85823040
    (cuda-gdb) c
    Continuing.
    warning: Cuda API error detected: cuLaunchKernelEx returned (0x190)


    zjxj:55959:55959 [1] enqueue.cc:1451 NCCL WARN Cuda failure 400 'invalid resource handle'
    zjxj:55959:55959 [1] NCCL INFO group.cc:231 -> 1
    zjxj:55959:55959 [1] NCCL INFO group.cc:453 -> 1
    zjxj:55959:55959 [1] NCCL INFO group.cc:546 -> 1
    zjxj:55959:55959 [1] NCCL INFO group.cc:101 -> 1
    fail to group end
    [Thread 0x7fff590fd000 (LWP 55998) exited]
    [Thread 0x7fff598fe000 (LWP 55997) exited]
    [Thread 0x7fff615ac000 (LWP 55994) exited]
    [Thread 0x7fff61dad000 (LWP 55993) exited]
    [Thread 0x7fffc0dff000 (LWP 55992) exited]
    [Thread 0x7fffc924a000 (LWP 55991) exited]
    [Thread 0x7fffc2a4f000 (LWP 55990) exited]
    [Thread 0x7fffc8a49000 (LWP 55985) exited]
    [Thread 0x7fffc9b3d000 (LWP 55983) exited]
    [Thread 0x7fffd4909000 (LWP 55963) exited]
    [Inferior 1 (process 55959) exited with code 0377]
    ```

* cuda 12.4 在编译的时候必须使用 compute 80，使用 compute 70 无法正常运行

    好像 compute 70, 80, 90 分别对应三种不同的 nv gpu 架构。

* `cudaSuccess`是 cuda runtime 里的枚举常量，`CUDA_SUCCESS`是 cuda umd 里的枚举常量

* thread id 与 thread idx 并不是同一个东西

    thread id 是 thread idx 的 flatten 版本。

    对于一维的情况，`thread_id = thread idx`

    对于二维 size 为`(dim_len_x, dim_len_y)`的情况，`thread_id = x + y * dim_len_x`

    对于三维 size 为`(dim_len_x, dim_len_y, dim_len_z)`的情况，`thread_id = x + y * dim_len_x + z * dim_len_x * dim_len_z`

    example:

    `main.cu`:

    ```cpp
    #define dim_x 3
    #define dim_y 4
    #define dim_z 5

    __global__ void vol_add(
        int A[dim_x][dim_y][dim_z], 
        int B[dim_x][dim_y][dim_z],
        int C[dim_x][dim_y][dim_z])
    {
        int i = threadIdx.x;
        int j = threadIdx.y;
        int k = threadIdx.z;

        C[i][j][k] = A[i][j][k] + B[i][j][k];
    }

    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        // int dim_x = 3, dim_y = 4, dim_z = 5;
        int num_elm = dim_x * dim_y * dim_z;
        int *host_buf_A = (int*) malloc(num_elm * sizeof(int));
        int *host_buf_B = (int*) malloc(num_elm * sizeof(int));
        int *host_buf_C = (int*) malloc(num_elm * sizeof(int));
        for (int i = 0; i < num_elm; ++i)
        {
            host_buf_A[i] = rand() % 5;
            host_buf_B[i] = rand() % 5;
        }

        int *dev_buf_A, *dev_buf_B, *dev_buf_C;
        cudaError_t ret;
        ret = cudaMalloc(&dev_buf_A, num_elm * sizeof(int));
        if (ret != cudaSuccess)
        {
            printf("fail to cuda malloc dev buf A\n");
            return -1;
        }
        printf("successfully cuda malloc dev buf A\n");

        ret = cudaMalloc(&dev_buf_B, num_elm * sizeof(int));
        if (ret != cudaSuccess)
        {
            printf("fail to cuda malloc dev buf B\n");
            return -1;
        }
        printf("successfully cuda malloc dev buf B\n");

        ret = cudaMalloc(&dev_buf_C, num_elm * sizeof(int));
        if (ret != cudaSuccess)
        {
            printf("fail to cuda malloc dev buf C\n");
            return -1;
        }
        printf("successfully cuda malloc dev buf C\n");

        ret = cudaMemcpy(dev_buf_A, host_buf_A, num_elm * sizeof(int), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy host buf A\n");
            return -1;
        }
        printf("successfully cuda memcpy host buf A\n");

        ret = cudaMemcpy(dev_buf_B, host_buf_B, num_elm * sizeof(int), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy host buf B\n");
            return -1;
        }
        printf("successfully cuda memcpy host buf B\n");

        dim3 threads_per_block(dim_x, dim_y, dim_z);
        vol_add<<<1, threads_per_block>>>(
            (int (*)[4][5]) dev_buf_A, 
            (int (*)[4][5]) dev_buf_B,
            (int (*)[4][5]) dev_buf_C);

        ret = cudaMemcpy(host_buf_C, dev_buf_C, num_elm * sizeof(int), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess)
        {
            printf("fail to cuda memcpy host buf C\n");
            return -1;
        }
        printf("successfully cuda memcpy host buf C\n");

        printf("host buf A:\n");
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%d, ", host_buf_A[i]);
        }
        putchar('\n');

        printf("host buf B:\n");
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%d, ", host_buf_B[i]);
        }
        putchar('\n');

        printf("host buf C:\n");
        for (int i = 0; i < num_elm; ++i)
        {
            printf("%d, ", host_buf_C[i]);
        }
        putchar('\n');

        cudaFree(dev_buf_A);
        cudaFree(dev_buf_B);
        cudaFree(dev_buf_C);
        free(host_buf_A);
        free(host_buf_B);
        free(host_buf_C);
        return 0;
    }
    ```

    compile: `nvcc -g -G -o main main.cu`

    run: `./main`

    output:

    ```
    successfully cuda malloc dev buf A
    successfully cuda malloc dev buf B
    successfully cuda malloc dev buf C
    successfully cuda memcpy host buf A
    successfully cuda memcpy host buf B
    successfully cuda memcpy host buf C
    host buf A:
    3, 2, 3, 1, 4, 2, 0, 3, 0, 2, 1, 2, 2, 2, 2, 4, 2, 4, 3, 1, 4, 1, 4, 3, 0, 3, 1, 1, 2, 1, 0, 4, 1, 1, 3, 4, 2, 4, 4, 3, 2, 1, 3, 3, 4, 2, 1, 4, 1, 4, 0, 4, 2, 2, 2, 2, 1, 1, 0, 4, 
    host buf B:
    1, 0, 0, 2, 1, 2, 4, 1, 1, 1, 3, 4, 0, 3, 0, 2, 3, 2, 1, 2, 3, 4, 2, 4, 0, 1, 0, 3, 0, 1, 0, 2, 0, 4, 2, 0, 0, 2, 4, 0, 3, 3, 4, 1, 4, 0, 3, 2, 1, 4, 0, 3, 1, 2, 2, 1, 0, 1, 4, 4, 
    host buf C:
    4, 2, 3, 3, 5, 4, 4, 4, 1, 3, 4, 6, 2, 5, 2, 6, 5, 6, 4, 3, 7, 5, 6, 7, 0, 4, 1, 4, 2, 2, 0, 6, 1, 5, 5, 4, 2, 6, 8, 3, 5, 4, 7, 4, 8, 2, 4, 6, 2, 8, 0, 7, 3, 4, 4, 3, 1, 2, 4, 8,
    ```

    看起来一个 block 是一个组织 thread 的方式。

    每个 block 最大有 1024 个 thread。

    如果我们只使用一维 thread，手动将`threadIdx.x`, `threadIdx.y`, `threadIdx.z`映射到一维的 idx 上，和直接使用 x, y, z 有什么不同？

    说明：

    1. `vol_add<<<numBlocks, threadsPerBlock>>>()`

        `threadsPerBlock`可以是一个数字，也可以是`dim3`类型的变量。

        如果数组中有 2 个元素或 3 个元素，那么必须使用`dim3`类型定义。

* thread block: cuda 中指一维的多线程，二维的多线程或三维的多线程

    vector, matrix, or volume 分别代表一维的数据，二维矩阵，三维体素

* cuda low level memory operation example

    `main.cu`:

    ```cpp
    #include <cuda.h>
    #include <stdio.h>

    int main()
    {
        cuInit(0);

        int device_id = 0;

        CUcontext ctx;
        CUresult ret;
        ret = cuCtxCreate(&ctx, 0, device_id);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to create cuda ctx\n");
            return -1;
        }
        printf("successfully create cuda ctx\n");

        CUmemGenericAllocationHandle handle;
        size_t size = 4096;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        // prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
        unsigned long long flags = 0;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        size_t padded_size = max(size, granularity);
        printf("granularity: %lu\n", granularity);
        printf("padded size: %lu\n", padded_size);

        ret = cuMemCreate(&handle, padded_size, &prop, flags);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to create cu mem\n");
            return -1;
        }
        printf("successfully create cu mem\n");

        CUdeviceptr ptr;
        ret = cuMemAddressReserve(&ptr, padded_size, 0, 0, 0);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to reserve mem addr\n");
            return -1;
        }
        printf("successfully reserve mem addr\n");
        printf("\tptr: %p\n", (void*) ptr);
        printf("\tsizeof(unsigned long long): %lu\n", sizeof(unsigned long long));

        ret = cuMemMap(ptr, padded_size, 0, handle, 0);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to map mem\n");
            return -1;
        }
        printf("successfully map mem\n");

        ret = cuMemRelease(handle);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to release cu mem\n");
            return -1;
        }
        printf("successfully release cu mem\n");

        CUmemAccessDesc access_desc;
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_id;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ret = cuMemSetAccess(ptr, padded_size, &access_desc, 1);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to set mem access\n");
            return -1;
        }
        printf("successfully set mem access\n");

        int *host_mem = (int*) malloc(128 * sizeof(int));
        if (!host_mem)
        {
            printf("fail to malloc host_mem\n");
            return -1;
        }
        printf("successfully malloc host_mem\n");

        ret = cuMemcpyHtoD(ptr, host_mem, 128 * sizeof(int));
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to memcpy h to d\n");
            return -1;
        }
        printf("successfully memcpy h to d\n");

        // for (int i = 0; i < 128; ++i)
        // {
        //     ((int*) ptr)[i] = i;
        // }

        ret = cuMemUnmap(ptr, padded_size);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to unmap mem\n");
            return -1;
        }
        printf("successfully unmap mem\n");

        ret = cuMemAddressFree(ptr, padded_size);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to free mem addr\n");
            return -1;
        }
        printf("successfully free mem addr\n");

        ret = cuCtxDestroy(ctx);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to destroy cuda ctx\n");
            return -1;
        }
        printf("successfully destroy cuda ctx\n");

        free(host_mem);
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -lcuda -o main`

    run: `./main`

    output:

    ```
    successfully create cuda ctx
    granularity: 2097152
    padded size: 2097152
    successfully create cu mem
    successfully reserve mem addr
    	ptr: 0x7f72fda00000
    	sizeof(unsigned long long): 8
    successfully map mem
    successfully release cu mem
    successfully set mem access
    successfully malloc host_mem
    successfully memcpy h to d
    successfully unmap mem
    successfully free mem addr
    successfully destroy cuda ctx
    ```

    说明：

    1. 不执行`cuInit(0);`，后面所有与 cuda 相关的函数都会执行失败

    2. `cuCtxCreate()`是必须的，不然`cuMemcpyHtoD()`会失败

    3. `cuMemMap()`之后马上就可以`cuMemRelease()`了，为什么？ release 不是释放显存的意思吗？

    4. `cuMemAddressReserve()`应该是只分配了 va 地址范围，device 这时可能还不知道这个 va 范围是多少，也没有把 va 写入到 mmu 里。

        device 在什么时候能看到其他 device 在 user space 的 va？这时候大概率就要写入 device 的 mmu 了。

    5. `cuMemSetAccess()`必须执行，不然`cuMemcpyHtoD()`会失败

    6. 只能使用`cuMemcpyHtoD()`将数据复制到 device 内，不能像 vulkan 那样 mmap 之后可以直接解引用赋值：

        ```cpp
        // for (int i = 0; i < 128; ++i)
        // {
        //     ((int*) ptr)[i] = i;  // segment fault
        // }
        ```

* 使用`cuMemCreate()`申请 device memory

    `main.cu`:

    ```cpp
    #include <cuda.h>
    #include <stdio.h>

    int main()
    {
        cuInit(0);

        int device_id = 0;

        CUmemGenericAllocationHandle handle;
        size_t size = 4096;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        // prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
        unsigned long long flags = 0;

        size_t granularity = 0;
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        size_t padded_size = max(size, granularity);
        printf("granularity: %lu\n", granularity);
        printf("padded size: %lu\n", padded_size);

        CUresult ret;
        ret = cuMemCreate(&handle, padded_size, &prop, flags);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to create cu mem\n");
            return -1;
        }
        printf("successfully create cu mem\n");

        ret = cuMemRelease(handle);
        if (ret != CUDA_SUCCESS)
        {
            printf("fail to release cu mem\n");
            return -1;
        }
        printf("successfully release cu mem\n");

        return 0;
    }
    ```

    compile:
    
    `nvcc -g main.cu -lcuda -o main`

    run:

    `./main`

    output:

    ```
    granularity: 2097152
    padded size: 2097152
    successfully create cu mem
    successfully release cu mem
    ```

    可以看到，`prop.type`, `prop.location.type`和`prop.location.id`是必填项，其他的项可以置 0。

    当前 device 支持的最小显存申请大小为 2 MB。

    说明：

    1. 必须先执行`cuInit(0);`，`cuMemCreate()`才能成功返回。

    1. `prop.type`只有一种选择。`cuda.h`中定义的 enum 如下：

        ```cpp
        /**
        * Defines the allocation types available
        */
        typedef enum CUmemAllocationType_enum {
            CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,

            /** This allocation type is 'pinned', i.e. cannot migrate from its current
              * location while the application is actively using it
              */
            CU_MEM_ALLOCATION_TYPE_PINNED  = 0x1,
            CU_MEM_ALLOCATION_TYPE_MAX     = 0x7FFFFFFF
        } CUmemAllocationType;
        ```

    1. `prop.location.type`也只有一种选择，`cuda.h`中定义的 enum 如下：

        ```cpp
        /**
         * Specifies the type of location
         */
        typedef enum CUmemLocationType_enum {
            CU_MEM_LOCATION_TYPE_INVALID = 0x0,
            CU_MEM_LOCATION_TYPE_DEVICE  = 0x1,  /**< Location is a device location, thus id is a device ordinal */
            CU_MEM_LOCATION_TYPE_MAX     = 0x7FFFFFFF
        } CUmemLocationType;
        ```

* 当 src 中有`cuda.h`的函数时，无论 src 的扩展名是`.cu`还是`.cpp`，`nvcc`都不会自动链接库`-lcuda`，需要手动添加:

    `nvcc -g main.cu -lcuda -o main`

    如果使用`gcc`编译，需要将扩展名改成`.c`，并且手动指定 include 目录：

    `gcc -g main.c -I/usr/local/cuda/include -lcuda -o main`

* `cuMemCreate()`之类的函数在`cuda.h`中，不在`cuda_runtime.h`中。

    `cudaMalloc()`之类的函数在`cuda_runtime.h`中，不在`cuda.h`中。

    看起来`cuda.h`比`cuda_runtime.h`更底层一点。

* asm 基本语法

    `asm("template-string" : "constraint"(output) : "constraint"(input)"));`

    一个使用`vabsdiff4`命令的 example:

    `asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r" (result):"r" (A), "r" (B), "r" (C));`

    从例子中可以看出，前两个字符串应该是拼接在一起的完整命令：`"vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"`。根据后面的命令可以猜出，`=`表示需要写入的变量，其余的是需要读取的变量。`r`不清楚是什么意思。

* nvcc 编译不同的架构支持`-gencode arch=compute_70,code=sm_70`

    如果硬件是 80 的兼容性，那么这个硬件支持使用 70 兼容性编译出来的 code。

    相反，如果硬件是 70 的兼容性，那么它跑不起来使用 80 兼容性编译出来的 code.

* cuda-gdb hit 断点时，可以使用`info cuda kernels`查看当前的 kernel 函数，sm，block, grid, device 使用情况等信息。

* nvcc 加上`-g`后支持 host code 调试，加上`-G`后支持 cuda kernel code 调试

* cuda kernel example

    ```cpp
    // Kernel definition
    __global__ void VecAdd(float* A, float* B, float* C)
    {
        int i = threadIdx.x;
        C[i] = A[i] + B[i];
    }

    int main()
    {
        ...
        // Kernel invocation with N threads
        VecAdd<<<1, N>>>(A, B, C);
        ...
    }
    ```

    cuda 编程模型是 c++ 的一个扩展。

    cuda kernel 前面要加上`__global__`。（为什么不是加`__device__`？）

* 每个 cuda kernel 在运行时对应一个 cuda thread，不是 block，也不是 sm

* SMs - Streaming Multiprocessors

* 在 cuda 中，软件使用的 block 会被 cuda 根据实际的物理 block 数重新排布

    比如软件定义 8 个 block 一起计算，如果 card 0 只有 2 个 sm，那么程序会变成 2 个 block 2 个 block 执行，一共执行 4 轮；如果 card 1 有 4 个 sm，那么程序会变成 4 个 block 4 个 block 执行，一共执行 2 轮。

    这个能力被官网称作 Automatic Scalability。

    注：这里的 block 可能指的并不是 thread，一个 block 可能会包含多个 thread。

* 官网介绍的 cuda 的核心三个抽象

    At its core are three key abstractions — a hierarchy of thread groups, shared memories, and barrier synchronization — that are simply exposed to the programmer as a minimal set of language extensions.

* 假如把 pcie p2p 和 nvlink p2p 都看作 peer access 能力，那么可以使用`cudaDeviceCanAccessPeer()`判断两个 dev 是否可以通过 pcie/nvlink 进行 p2p 互联

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    int main()
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            for (int j = 0; j < deviceCount; ++j)
            {
                if (i != j)
                {
                    int canAccessPeer = 0;
                    cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    if (canAccessPeer)
                    {
                        printf("dev peer access %d - %d, OK\n", i, j);
                    }
                    else 
                    {
                        printf("dev peer access %d - %d, Error\n", i, j);
                    }
                }
            }
        }

        return 0;
    }
    ```

    compile: `nvcc -g main.cu -L/usr/local/cuda-12.1/lib64 -lcudart -o main`

    run: `./main`

    output:

    ```
    dev peer access 0 - 1, OK
    dev peer access 1 - 0, OK
    ```

* a100 x 8 机器的 p2p 测速输出

    ```
    hlc@a147:~/Data/Projects/cuda-samples-12.4.1/bin/x86_64/linux/release$ ./p2pBandwidthLatencyTest 
    [P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
    Device: 0, NVIDIA A100-SXM4-80GB, pciBusID: 26, pciDeviceID: 0, pciDomainID:0
    Device: 1, NVIDIA A100-SXM4-80GB, pciBusID: 2c, pciDeviceID: 0, pciDomainID:0
    Device: 2, NVIDIA A100-SXM4-80GB, pciBusID: 66, pciDeviceID: 0, pciDomainID:0
    Device: 3, NVIDIA A100-SXM4-80GB, pciBusID: 6b, pciDeviceID: 0, pciDomainID:0
    Device: 4, NVIDIA A100-SXM4-80GB, pciBusID: a4, pciDeviceID: 0, pciDomainID:0
    Device: 5, NVIDIA A100-SXM4-80GB, pciBusID: a9, pciDeviceID: 0, pciDomainID:0
    Device: 6, NVIDIA A100-SXM4-80GB, pciBusID: e1, pciDeviceID: 0, pciDomainID:0
    Device: 7, NVIDIA A100-SXM4-80GB, pciBusID: e7, pciDeviceID: 0, pciDomainID:0
    Device=0 CAN Access Peer Device=1
    Device=0 CAN Access Peer Device=2
    Device=0 CAN Access Peer Device=3
    Device=0 CAN Access Peer Device=4
    Device=0 CAN Access Peer Device=5
    Device=0 CAN Access Peer Device=6
    Device=0 CAN Access Peer Device=7
    Device=1 CAN Access Peer Device=0
    Device=1 CAN Access Peer Device=2
    Device=1 CAN Access Peer Device=3
    Device=1 CAN Access Peer Device=4
    Device=1 CAN Access Peer Device=5
    Device=1 CAN Access Peer Device=6
    Device=1 CAN Access Peer Device=7
    Device=2 CAN Access Peer Device=0
    Device=2 CAN Access Peer Device=1
    Device=2 CAN Access Peer Device=3
    Device=2 CAN Access Peer Device=4
    Device=2 CAN Access Peer Device=5
    Device=2 CAN Access Peer Device=6
    Device=2 CAN Access Peer Device=7
    Device=3 CAN Access Peer Device=0
    Device=3 CAN Access Peer Device=1
    Device=3 CAN Access Peer Device=2
    Device=3 CAN Access Peer Device=4
    Device=3 CAN Access Peer Device=5
    Device=3 CAN Access Peer Device=6
    Device=3 CAN Access Peer Device=7
    Device=4 CAN Access Peer Device=0
    Device=4 CAN Access Peer Device=1
    Device=4 CAN Access Peer Device=2
    Device=4 CAN Access Peer Device=3
    Device=4 CAN Access Peer Device=5
    Device=4 CAN Access Peer Device=6
    Device=4 CAN Access Peer Device=7
    Device=5 CAN Access Peer Device=0
    Device=5 CAN Access Peer Device=1
    Device=5 CAN Access Peer Device=2
    Device=5 CAN Access Peer Device=3
    Device=5 CAN Access Peer Device=4
    Device=5 CAN Access Peer Device=6
    Device=5 CAN Access Peer Device=7
    Device=6 CAN Access Peer Device=0
    Device=6 CAN Access Peer Device=1
    Device=6 CAN Access Peer Device=2
    Device=6 CAN Access Peer Device=3
    Device=6 CAN Access Peer Device=4
    Device=6 CAN Access Peer Device=5
    Device=6 CAN Access Peer Device=7
    Device=7 CAN Access Peer Device=0
    Device=7 CAN Access Peer Device=1
    Device=7 CAN Access Peer Device=2
    Device=7 CAN Access Peer Device=3
    Device=7 CAN Access Peer Device=4
    Device=7 CAN Access Peer Device=5
    Device=7 CAN Access Peer Device=6

    ***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
    So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

    P2P Connectivity Matrix
         D\D     0     1     2     3     4     5     6     7
         0	     1     1     1     1     1     1     1     1
         1	     1     1     1     1     1     1     1     1
         2	     1     1     1     1     1     1     1     1
         3	     1     1     1     1     1     1     1     1
         4	     1     1     1     1     1     1     1     1
         5	     1     1     1     1     1     1     1     1
         6	     1     1     1     1     1     1     1     1
         7	     1     1     1     1     1     1     1     1
    Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1524.39  16.69  20.20  20.58  18.53  20.63  19.52  20.58 
         1  16.98 1537.89  20.05  20.08  17.80  20.15  19.42  20.61 
         2  20.19  20.25 1534.87  16.91  17.70  20.17  19.25  20.61 
         3  20.12  20.23  17.01 1539.41  17.79  20.47  19.62  20.61 
         4  18.36  19.99  18.57  19.74 1573.51  11.79  19.22  18.40 
         5  19.46  19.00  19.69  18.76  12.10 1570.35  17.80  19.22 
         6  18.39  20.07  18.55  19.62  18.64  19.95 1571.93  11.72 
         7  19.43  18.18  19.65  18.59  19.60  18.97  11.83 1575.10 
    Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1545.50 270.83 274.85 273.60 274.38 272.17 273.49 274.86 
         1 270.19 1556.27 274.34 273.93 274.58 274.03 274.48 274.67 
         2 270.85 271.61 1540.93 274.54 274.32 274.63 274.53 275.05 
         3 271.98 273.56 272.55 1591.14 274.67 275.17 274.81 275.42 
         4 272.06 273.89 272.11 275.67 1587.91 274.89 276.05 275.05 
         5 272.75 273.16 273.78 275.51 275.13 1584.69 275.54 274.60 
         6 273.19 273.46 273.59 275.04 275.15 275.83 1584.69 273.68 
         7 273.62 273.78 273.35 275.28 275.49 275.44 274.88 1586.29 
    Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1559.38  12.03  27.82  27.85  20.26  21.38  21.41  21.52 
         1  12.23 1596.02  28.21  28.04  20.75  21.53  21.47  21.82 
         2  28.26  28.20 1597.65  12.29  20.25  21.27  21.39  21.14 
         3  28.20  28.04  12.28 1600.10  20.36  21.32  21.39  21.17 
         4  22.80  22.50  22.37  22.59 1596.02  10.55  20.90  20.69 
         5  22.13  22.51  22.39  22.52  11.88 1598.47  20.94  20.87 
         6  22.05  22.35  22.35  22.19  21.61  20.96 1599.28  10.55 
         7  22.18  22.37  22.08  22.15  21.08  21.05  12.34 1602.56 
    Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
       D\D     0      1      2      3      4      5      6      7 
         0 1556.27 423.17 423.40 423.51 424.20 424.43 424.31 425.35 
         1 424.43 1556.27 424.20 424.66 422.25 424.89 424.89 425.68 
         2 421.79 422.59 1560.16 424.08 425.81 425.35 425.35 424.08 
         3 425.52 424.08 424.31 1563.28 425.35 423.02 425.12 425.00 
         4 426.45 426.44 426.40 426.60 1607.51 518.15 518.66 521.47 
         5 425.39 426.69 426.76 427.09 517.28 1597.65 520.94 517.97 
         6 426.85 426.74 426.48 427.34 520.39 517.80 1596.02 521.47 
         7 426.40 427.50 426.43 426.02 426.93 518.39 519.35 1594.39 
    P2P=Disabled Latency Matrix (us)
       GPU     0      1      2      3      4      5      6      7 
         0   3.04  23.59  23.62  23.88  24.51  24.48  24.31  24.48 
         1  23.66   2.85  23.65  23.58  24.49  24.48  24.30  24.48 
         2  23.61  23.60   3.03  23.58  24.54  24.49  24.40  24.48 
         3  23.59  23.57  23.59   3.15  24.50  24.49  24.46  24.49 
         4  24.61  24.62  24.62  24.60   2.99  24.51  24.60  24.53 
         5  24.57  24.61  24.62  24.60  24.59   2.91  24.60  24.53 
         6  24.60  24.41  24.62  24.60  24.59  24.59   2.29  24.30 
         7  24.64  24.65  24.64  24.64  24.60  24.60  24.43   2.48 

       CPU     0      1      2      3      4      5      6      7 
         0   2.40   7.08   6.87   6.92   8.45   8.39   8.45   8.38 
         1   6.99   2.32   6.91   6.85   8.48   8.44   8.49   8.47 
         2   6.97   6.88   2.34   6.88   8.47   8.41   8.49   8.37 
         3   6.80   6.90   6.75   2.33   8.53   8.54   8.45   8.39 
         4   7.95   7.88   7.89   7.87   2.86   9.56   9.56   9.54 
         5   7.88   7.85   7.85   7.82   9.42   2.85   9.55   9.47 
         6   7.91   7.89   7.90   7.82   9.49   9.45   2.90   9.57 
         7   7.91   7.85   7.86   7.80   9.46   9.46   9.55   2.89 
    P2P=Enabled Latency (P2P Writes) Matrix (us)
       GPU     0      1      2      3      4      5      6      7 
         0   3.02   3.42   3.36   3.36   3.40   3.35   3.42   3.43 
         1   3.36   2.87   3.41   3.47   3.54   3.55   3.48   3.54 
         2   3.36   3.41   3.02   3.44   3.43   3.37   3.36   3.40 
         3   3.36   3.37   3.37   3.18   3.37   3.40   3.43   3.37 
         4   3.23   3.19   3.18   3.18   2.98   3.19   3.18   3.19 
         5   3.19   3.22   3.19   3.23   3.15   2.91   3.24   3.25 
         6   2.82   2.77   2.81   2.79   2.78   2.81   2.28   2.84 
         7   2.82   2.78   2.78   2.82   2.82   2.82   2.82   2.47 

       CPU     0      1      2      3      4      5      6      7 
         0   2.41   2.07   2.07   2.10   2.10   2.09   2.08   2.09 
         1   2.10   2.42   2.06   2.33   2.31   2.32   2.29   2.28 
         2   2.18   2.02   2.44   2.03   2.00   2.05   2.01   2.09 
         3   2.07   2.03   2.02   2.43   2.02   2.05   1.99   2.10 
         4   2.69   2.63   2.61   2.61   2.96   2.64   2.59   2.60 
         5   2.69   2.62   2.63   2.65   2.66   2.98   2.70   2.63 
         6   2.69   2.61   2.63   2.64   2.63   2.63   2.95   2.59 
         7   2.71   2.64   2.65   2.64   2.65   2.74   2.64   3.00 

    NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
    ```

* 使用 cuda-sample 里提供的程序也可以测速，测出来数据和自己写代码测的差不多

    run: `(base) huliucheng@zjxj:~/Documents/Projects/cuda-samples-12.1/bin/x86_64/linux/release$ ./p2pBandwidthLatencyTest`

    output:

    ```
    [P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
    Device: 0, Tesla V100-PCIE-32GB, pciBusID: b1, pciDeviceID: 0, pciDomainID:0
    Device: 1, Tesla V100-PCIE-32GB, pciBusID: e3, pciDeviceID: 0, pciDomainID:0
    Device=0 CAN Access Peer Device=1
    Device=1 CAN Access Peer Device=0

    ***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
    So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

    P2P Connectivity Matrix
         D\D     0     1
         0	     1     1
         1	     1     1
    Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1 
         0 771.60  12.22 
         1  12.19 774.28 
    Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
       D\D     0      1 
         0 708.94  11.35 
         1  11.36 773.90 
    Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
       D\D     0      1 
         0 712.33  16.60 
         1  16.42 775.05 
    Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
       D\D     0      1 
         0 711.85  22.07 
         1  22.07 776.40 
    P2P=Disabled Latency Matrix (us)
       GPU     0      1 
         0   1.96  78.17 
         1  14.45   1.97 

       CPU     0      1 
         0   2.63   6.44 
         1   6.42   2.59 
    P2P=Enabled Latency (P2P Writes) Matrix (us)
       GPU     0      1 
         0   1.91   1.74 
         1   1.74   1.97 

       CPU     0      1 
         0   2.58   2.00 
         1   1.92   2.65 

    NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
    ```

* deepseek 帮忙写的检测环境中是否有 nvlink 的代码

    `main.cu`:

    ```cpp
    #include <nvml.h>
    #include <iostream>

    int main() {
        nvmlInit();

        unsigned int deviceCount;
        nvmlDeviceGetCount(&deviceCount);

        for (unsigned int i = 0; i < deviceCount; ++i) {
            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(i, &device);

            for (unsigned int j = 0; j < deviceCount; ++j) {
                if (i != j) {
                    nvmlPciInfo_t pciInfo1, pciInfo2;
                    nvmlDeviceGetPciInfo(device, &pciInfo1);

                    nvmlDevice_t peerDevice;
                    nvmlDeviceGetHandleByIndex(j, &peerDevice);
                    nvmlDeviceGetPciInfo(peerDevice, &pciInfo2);

                    nvmlEnableState_t isEnabled;
                    nvmlDeviceGetNvLinkState(device, j, &isEnabled);

                    if (isEnabled == NVML_FEATURE_ENABLED) {
                        std::cout << "Device " << i << " is connected to Device " << j << " via NVLink." << std::endl;
                    } else {
                        std::cout << "Device " << i << " is not connected to Device " << j << " via NVLink." << std::endl;
                    }
                }
            }
        }

        nvmlShutdown();
        return 0;
    }
    ```

    compile: `nvcc -g main.cu -lnvidia-ml -o main`

    run: `./main`

    output:

    ```
    Device 0 is not connected to Device 1 via NVLink.
    Device 1 is not connected to Device 0 via NVLink.
    ```

    这个是使用 nvml 库来实现的。

* cuda pcie p2p 的一段代码

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>

    #include <iostream>
    #include <string>
    #include <vector>

    // Enhanced error checking macro
    #define checkCudaErrors(val) \
      checkCudaErrorsImpl((val), #val, __FILE__, __LINE__)

    void checkCudaErrorsImpl(cudaError_t err, const char* func, const char* file,
                             int line) {
      if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " in " << func
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    int main(int argc, char** argv) {
      if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <from_device1> <to_device1> [<from_device2> <to_device2> "
                     "...] [--enable-peer-access]"
                  << std::endl;
        return EXIT_FAILURE;
      }

      bool enablePeerAccess = false;
      std::vector<std::pair<int, int>> devicePairs;

      // Parse command-line arguments
      for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--enable-peer-access") {
          enablePeerAccess = true;
        } else {
          if (i + 1 < argc) {
            int fromDevice = std::stoi(argv[i]);
            int toDevice = std::stoi(argv[i + 1]);
            devicePairs.emplace_back(fromDevice, toDevice);
            ++i;  // Skip the next argument as it's part of the device pair
          }
        }
      }

      // Enable peer access between devices if possible and requested
      if (enablePeerAccess) {
        for (const auto& pair : devicePairs) {
          int fromDevice = pair.first;
          int toDevice = pair.second;

          int canAccessPeer = 0;
          checkCudaErrors(
              cudaDeviceCanAccessPeer(&canAccessPeer, fromDevice, toDevice));
          if (canAccessPeer) {
            cudaSetDevice(fromDevice);
            cudaError_t err = cudaDeviceEnablePeerAccess(toDevice, 0);
            if (err == cudaSuccess) {
              std::cout << "Peer access enabled from device " << fromDevice
                        << " to device " << toDevice << std::endl;
            } else {
              std::cout << "Failed to enable peer access from device " << fromDevice
                        << " to device " << toDevice << ": "
                        << cudaGetErrorString(err) << std::endl;
            }
          } else {
            std::cout << "Peer access not supported from device " << fromDevice
                      << " to device " << toDevice << std::endl;
          }

          checkCudaErrors(
              cudaDeviceCanAccessPeer(&canAccessPeer, toDevice, fromDevice));
          if (canAccessPeer) {
            cudaSetDevice(toDevice);
            cudaError_t err = cudaDeviceEnablePeerAccess(fromDevice, 0);
            if (err == cudaSuccess) {
              std::cout << "Peer access enabled from device " << toDevice
                        << " to device " << fromDevice << std::endl;
            } else {
              std::cout << "Failed to enable peer access from device " << toDevice
                        << " to device " << fromDevice << ": "
                        << cudaGetErrorString(err) << std::endl;
            }
          } else {
            std::cout << "Peer access not supported from device " << toDevice
                      << " to device " << fromDevice << std::endl;
          }
        }
      }

      size_t sizes[] = {
          // 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
          // 1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
          // 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
        //   4294967296};           // 2KB - 4GB
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL,
        24 * 1024ULL * 1024ULL * 1024ULL};
      const int numRepeats = 1;  // Number of repetitions for averaging

      for (size_t size : sizes) {
        std::cout << "Testing size: " << size << " bytes" << std::endl;

        std::vector<cudaStream_t> streams(devicePairs.size());
        std::vector<void*> d_srcs(devicePairs.size());
        std::vector<void*> d_dsts(devicePairs.size());
        std::vector<float> totalMilliseconds(devicePairs.size(), 0.0f);

        for (size_t i = 0; i < devicePairs.size(); ++i) {
          int fromDevice = devicePairs[i].first;
          int toDevice = devicePairs[i].second;

          // Allocate source memory on fromDevice
          cudaSetDevice(fromDevice);
          checkCudaErrors(cudaMalloc(&d_srcs[i], size));

          // Allocate destination memory on toDevice
          cudaSetDevice(toDevice);
          checkCudaErrors(cudaMalloc(&d_dsts[i], size));

          // Create stream on fromDevice
          cudaSetDevice(fromDevice);  // Set to fromDevice before creating stream
          checkCudaErrors(cudaStreamCreate(&streams[i]));
        }

        for (int repeat = 0; repeat < numRepeats; ++repeat) {
          std::vector<cudaEvent_t> startEvents(devicePairs.size());
          std::vector<cudaEvent_t> stopEvents(devicePairs.size());

          for (size_t i = 0; i < devicePairs.size(); ++i) {
            int fromDevice = devicePairs[i].first;
            // Ensure device is set to fromDevice
            cudaSetDevice(
                fromDevice);  // Set device to fromDevice where stream[i] resides

            checkCudaErrors(cudaEventCreate(&startEvents[i]));
            checkCudaErrors(cudaEventCreate(&stopEvents[i]));

            checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));
            checkCudaErrors(cudaMemcpyAsync(d_dsts[i], d_srcs[i], size,
                                            cudaMemcpyDeviceToDevice, streams[i]));
            checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
          }

          // Synchronize all streams to ensure all operations for the current size
          // are completed
          for (size_t i = 0; i < devicePairs.size(); ++i) {
            int fromDevice = devicePairs[i].first;
            cudaSetDevice(fromDevice);  // Ensure device is set before synchronizing

            checkCudaErrors(cudaStreamSynchronize(streams[i]));

            float milliseconds = 0;
            checkCudaErrors(
                cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]));
            totalMilliseconds[i] += milliseconds;

            checkCudaErrors(cudaEventDestroy(startEvents[i]));
            checkCudaErrors(cudaEventDestroy(stopEvents[i]));
          }
        }

        for (size_t i = 0; i < devicePairs.size(); ++i) {
          int fromDevice = devicePairs[i].first;
          int toDevice = devicePairs[i].second;

          float averageMilliseconds = totalMilliseconds[i] / numRepeats;
          double bandwidth = (size * 1e-9) / (averageMilliseconds * 1e-3);
          std::cout << "  Device Pair " << fromDevice << " -> " << toDevice
                    << ": Average Bandwidth = " << bandwidth
                    << " GB/s, milliseconds " << averageMilliseconds << std::endl;

          // Free source memory on fromDevice
          cudaSetDevice(fromDevice);
          checkCudaErrors(cudaFree(d_srcs[i]));

          // Free destination memory on toDevice
          cudaSetDevice(toDevice);
          checkCudaErrors(cudaFree(d_dsts[i]));

          // Destroy stream on fromDevice
          cudaSetDevice(
              fromDevice);  // Ensure device is set before destroying stream
          checkCudaErrors(cudaStreamDestroy(streams[i]));
        }
      }

      return EXIT_SUCCESS;
    }
    ```

    compile: `nvcc -g main.cu -L/usr/local/cuda/lib64 -lcudart -o main`

    run & output:

    * `./main 0 1`

        ```
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4125 GB/s, milliseconds 2076.12
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4143 GB/s, milliseconds 2075.82
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4187 GB/s, milliseconds 2075.08
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.2919 GB/s, milliseconds 2096.49
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4177 GB/s, milliseconds 2075.25
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4205 GB/s, milliseconds 2074.77
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4222 GB/s, milliseconds 2074.49
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.3017 GB/s, milliseconds 2094.81
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4195 GB/s, milliseconds 2074.95
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 12.4187 GB/s, milliseconds 2075.07
        ```

    * `./main 0 1 --enable-peer-access`

        ```
        Peer access enabled from device 0 to device 1
        Peer access enabled from device 1 to device 0
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.24
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3263 GB/s, milliseconds 2275.22
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3339 GB/s, milliseconds 2273.68
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.24
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3263 GB/s, milliseconds 2275.22
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.24
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        Testing size: 25769803776 bytes
          Device Pair 0 -> 1: Average Bandwidth = 11.3262 GB/s, milliseconds 2275.23
        ```

    可以观察到下面几点：

    * 在传输数据时，无论是否开始 peer access，cpu 都会单核跑满

        看来 cpu 单核跑满与 nccl 无关，是 async copy 的结果。目前不清楚机制。

    * 假如机器止是 pcie 3.0 理论峰值速度是 16 GB/s，如果是 pcie 4.0 理论峰值速度是 32 GB/s，但是无论是否开始 peer access 实测的速度最大只有 12 GB/s 左右

        猜想：

        1. 不开启 peer access 时，host memory 只分配几个小 size buffer 作为中转，这个小 size buffer 可能是 pageable memory，此时影响速度的因素可能有：
        
            1. pageable memory 导致一直在查表；
            
            2. device 往 host memory 上写速度会拉低一些

        2. 开启 peer access 时，有可能 device memory 也只分配了一点点 buffer 映射到 bar 空间上，并不是把所有的 device memory 映射出来。

* cuda stream 在创建时，靠`cudaSetDevice()`来指定具体的 device

    这个操作看起来比较像 opengl，提前指定好上下文环境。

* 可以直接用 apt 安装 nvidia-cuda-tookit，这样可以安装上`nvcc`等开发环境

    cuda 的版本会落后一些，但是提供了提示和编译环境，可以用来跳转和写代码。

* nvcc 需要安装 g++ 编译器

## topics

### ptx

* `.reg .pred p;`是一个声明变量的语句，`.reg`表示是寄存器空间，`.pred`表示 predicate (谓词)，可能是用于条件分支的，相当于 if 语句。

    example:

    ```c
    if (i < n)
        j = j + 1;
    ```

    上面的 c 代码等价于下面这两种 ptx 代码：

    ```asm
          setp.lt.s32  p, i, n;    // p = (i < n)
    @p    add.s32      j, j, 1;    // if i < n, add 1 to j
    ```

    ```asm
          setp.lt.s32  p, i, n;    // compare i to n
    @!p   bra  L1;                 // if False, branch over
          add.s32      j, j, 1;
    L1:     ...
    ```

* `.step`用于比较大小，并将结果存放到指定寄存器中

    ```asm
    setp.lt.s32  p|q, a, b;  // p = (a < b); q = !(a < b);
    ```

* True 和 False 被称作 guard predicate

* `@{!}p    instruction;`被称作 Predicated execution.

    当`{!}p`为 true 时，才执行 instruction，否则不执行。

    example:

    ```asm
        setp.eq.f32  p,y,0;     // is y zero?
    @!p div.f32      ratio,x,y  // avoid division by zero

    @q  bra L23;                // conditional branch
    ```

* `bra`跳转

    ```asm
    @p   bra{.uni}  tgt;           // tgt is a label
         bra{.uni}  tgt;           // unconditional branch
    ```

    `bra.uni` is guaranteed to be non-divergent, i.e. all active threads in a warp that are currently executing this instruction have identical values for the guard predicate and branch target.

    这个可能的含义是当所有线程的`p`或`tgt`都一致时，才使用`.uni`。（什么时候`tgt`会不一致？如果`p`或`tgt`不一致，但是使用了`.uni`，会报什么错？）

    `bra`看起来像是 branch 的缩写。

* `bar{.cta}, barrier{.cta}`

    barrier

    同步 cta 中的线程。猜想：cta 为一种用于同步的资源，当线程运行到 barrier 后停下，当所有线程都运行到 barrier 后，由 cta 唤醒线程继续运行。每个 cta 有 16 个用于同步的资源，编号为 0 到 15.

    > The optional .cta qualifier simply indicates CTA-level applicability of the barrier and it doesn’t change the semantics of the instruction.

    看上去 cta 只是一个提示词，无论使用还是不使用都不影响功能。

* `barrier{.cta}.sync{.aligned}      a{, b};`

    `.sync`表示当参与 barrier 的线程到达这条指令后，等待其他线程。

    `a`表示使用第几个 cta，可取值为 0 到 15.

    `b`表示有多少线程参与 barrier，这个数必须是 warp size 的整数倍。如果不指定`b`，则所有参与 barrier 的 thread 所在的 warp，都会进入 barrier。

    这里的`.aligned`与`.cta`同理，都只是一个提示词，不具备实际功能。

* `barrier{.cta}.arrive{.aligned}    a, b;`

    与`.sync`相对，`.arrive`不会阻塞 thread。它似乎仅用于标记这里有个 barrier。

    官网举的例子是 producer-consumer 模型，整个过程分为两步：
    
    1. 一部分 thread 作为 producer 执行`barrier.arrive`，另一部分 thread 作为 consumer 执行`barrier.sync`等待 producer 生产出资源

    2. 刚才作为 producer 的 threads 执行`barrier.sync`等待 consumer 消耗资源，而刚才作为 consumer 的 threads 执行`barrier.arrive`消耗资源

    目前没有看到实际的 example。

    注意，在`.arrive`中，`b`必须指定。（为什么？）

* `barrier{.cta}.red.popc{.aligned}.u32  d, a{, b}, {!}c;`

    `.red`表示 reduce，`.popc`表示 population-count，`d`表示目标寄存器，`c`表示谓词（predicate）。

    `.popc`表示统计`c`中有多少个 true，并把结果存储到寄存器`d`中。

* `barrier{.cta}.red.op{.aligned}.pred   p, a{, b}, {!}c;`

    其中的`.op`可以为`.and`，也可以为`.or`，分别用于判断`c`是否全为 true，或只有部分为 true。将判断的结果存储到寄存器`p`中。

* Cooperative thread arrays (CTAs) implement CUDA thread blocks

* The Parallel Thread Execution (PTX) programming model is explicitly parallel:

* The thread identifier is a three-element vector tid, (with elements tid.x, tid.y, and tid.z) 

* Each CTA has a 1D, 2D, or 3D shape specified by a three-element vector ntid (with elements ntid.x, ntid.y, and ntid.z).

* Threads within a CTA execute in SIMT (single-instruction, multiple-thread) fashion in groups called warps.

* A warp is a maximal subset of threads from a single CTA

    不懂这句什么意思

* PTX includes a run-time immediate constant, WARP_SZ

    warp size

* Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate with each other via shared memory. 

    一个 cluster 由多个 warp 组成，多个 warp 之间可以共享内存。

* Cluster-wide barriers can be used to synchronize all the threads within the cluster. 

    cluster-wide 并不是拿来同步 cluster 的，而是拿来同步 cluster 中的 thread 的

    不同 cluster 之间的 thread 无法同步，也无法通信。

* Each CTA in a cluster has a unique CTA identifier within its cluster (cluster_ctaid).

    cta 在 cluster 中有独立的 id 来标识。

* Each cluster of CTAs has 1D, 2D or 3D shape specified by the parameter cluster_nctaid

    cluster 也可以按 1d, 2d, 3d 组合。这里只提供了一个`cluster_nctaid`，不同维度的 id 该如何获取？

* Each CTA in the cluster also has a unique CTA identifier (cluster_ctarank) across all dimensions. total number of CTAs across all the dimensions in the cluster is specified by cluster_nctarank.

    这个应该是把 dimension flatten 后得到的 id

* Threads may read and use these values through predefined, read-only special registers %cluster_ctaid, %cluster_nctaid, %cluster_ctarank, %cluster_nctarank.

* Each cluster has a unique cluster identifier (clusterid) within a grid of clusters. Each grid of clusters has a 1D, 2D , or 3D shape specified by the parameter nclusterid. Each grid also has a unique temporal grid identifier (gridid). Threads may read and use these values through predefined, read-only special registers %tid, %ntid, %clusterid, %nclusterid, and %gridid.

    cluster id 的标识，没啥可说的，和 thread id 差不多。

* Each CTA has a unique identifier (ctaid) within a grid. Each grid of CTAs has 1D, 2D, or 3D shape specified by the parameter nctaid. Thread may use and read these values through predefined, read-only special registers %ctaid and %nctaid.

     cta 在 grid 中的 id 标识，注意这个是区别于 cluster 的。

* Grids may be launched with dependencies between one another

    不同 grid 之间可以指定依赖关系后串行执行。这个依赖关系和 cuda graph 相关。

* ach thread has a private local memory. Each thread block (CTA) has a shared memory visible to all threads of the block and to all active blocks in the cluster and with the same lifetime as the block. Finally, all threads have access to the same global memory.

    其他的都比较清楚，唯一不清楚的是，shared memory 是每个 cta 有一份，还是一个 cluster 中只有一份？

* Constant and texture memory are read-only; surface memory is readable and writable.

* The global, constant, param, texture, and surface state spaces are optimized for different memory usages.

* texture and surface memory is cached, and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes,

    texture 和 surface 总是读缓存里的数据，因此如果在一个 kernel 调用内先存入新的值，再立即去读取值，很有可能读出的值并不是存入的值。

* The global, constant, and texture state spaces are persistent across kernel launches by the same application.

    不懂这个是啥意思。

    前面一直没提到过 param memory 是干什么用的，有什么特性。

* The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs).

* A multiprocessor consists of multiple Scalar Processor (SP) cores, a multithreaded instruction unit, and on-chip shared memory. It implements a single-instruction barrier synchronization.

    The multiprocessor maps each thread to one scalar processor core, and each scalar thread executes independently with its own instruction address and register state. 

    每个 sp 对应一个线程吗？

* The multiprocessor SIMT unit creates, manages, schedules, and executes threads in groups of parallel threads called warps. (This term originates from weaving, the first parallel thread technology.)

    simt unit 指的是管理多个 sp 的组件吗？

* When a multiprocessor is given one or more thread blocks to execute, it splits them into warps that get scheduled by the SIMT unit.

    SM 拿到 blocks，SM 将 blocks 分成多个 warps。每个 warp 对应一个 simt unit.

* At every instruction issue time, the SIMT unit selects a warp that is ready to execute and issues the next instruction to the active threads of the warp.

    warp 和 simt unit 并不是绑定的关系，simt unit 会动态选择准备就绪的 warp 进行处理。

    一个 warp 中的 threads 也不是都为 active 状态，可能只有部分是 active 状态。（猜测这里的 active 和 if 条件语句有关）

* A warp executes one common instruction at a time, so full efficiency is realized when all threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp serially executes each branch path taken, disabling threads that are not on that path, and when all paths complete, the threads converge back to the same execution path. Branch divergence occurs only within a warp; 

    如果一个 warp 中的多个 thread 根据 if 语句发生了 diverge，一部分 threads 走 branch 1，另一部分 threads 走 branch 2，那么 warp 会先执行 branch 1 的代码，再执行 branch 2 的代码，最后再把所有 threads 合并到一起，执行 common 代码。

    显然，最坏的情况是 branch 1 上有 N - 1 个 threads，而 branch 2 上只有 1 个 thread，并且 branch 2 的代码很长。为了这一个 thread，其他 thread 要等很长时间，彻底破坏了 simt 的优势。

* How many blocks a multiprocessor can process at once depends on how many registers per thread and how much shared memory per block are required for a given kernel since the multiprocessor’s registers and shared memory are split among all the threads of the batch of blocks. 

    看来 sm 中的寄存器和 shared memory 的数量是有限的，根据这些资源的占用情况，动态决定一个 sm 能同时处理多少个 block。

    推论：或者是编译，或者是固件，一定有个地方会计算 sm 中寄存器资源的占用情况，并在 dispatch 任务时会根据这些占用情况进行派发。

* 从 vulta 架构开始，引入了 Independent Thread Scheduling，此时不再以 warp + mask 的形式进行线程调度，提高了灵活性。

    目前并不清楚这点是怎么 work 的，有时间了再看。

* on-chip memory

    片上内存，速度快。一共有 4 种。

    * register，每个 thread 独占。

    * shared memory, 位置在 sm 上，逻辑上 block 共享

    * read-only constant cache，位置在 sm 上，逻辑情况未知

    * read-only texture cache，位置在 sm 上，逻辑情况未知

* PTX is case sensitive and uses lowercase for keywords.

* ptx 语句有两种，一种是 directive，另一种是 instruction

    常见的 directive 如下：

    ```
            .reg     .b32 r1, r2;
            .global  .f32  array[N];

    start:  mov.b32   r1, %tid.x;
            shl.b32   r1, r1, 2;          // shift thread id by 2 bits
            ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
            add.f32   r2, r2, 0.5;        // add 1/2
    ```

    常见的 insrruction 如下：

    `abs`, `cvta`, `membar`

* identifier

    ptx 中的 identifier 与 c/c++ 中的相似。

    regular expression:

    ```
    followsym:   [a-zA-Z0-9_$]
    identifier:  [a-zA-Z]{followsym}* | {[_$%]{followsym}+
    ```

* Predefined Identifiers

    ptx 预定义了一些常量。常用的常量：

    `%clock`, `%laneid`, `%lanemask_gt`

* integer 有 signed 和 unsigned 之分，以 64 位为例，分别表示为`.s64`, `.u64`

* 整型的字面量  Integer literals

    常见的格式如下：

    ```
    hexadecimal literal:  0[xX]{hexdigit}+U?
    octal literal:        0{octal digit}+U?
    binary literal:       0[bB]{bit}+U?
    decimal literal       {nonzero-digit}{digit}*U?
    ```

* floating point

    ptx 支持使用十六进制数按照 IEEE 754 格式精确表示浮点数。

    ```
    0[fF]{hexdigit}{8}      // single-precision floating point
    0[dD]{hexdigit}{16}     // double-precision floating point
    ```

    example:

    ```asm
    mov.f32  $f3, 0F3f800000;       //  1.0
    ```

* A state space is a storage area with particular characteristics.

    常用的 state space:

    * `.reg`: Registers, fast.

    * `.sreg`: Special registers. Read-only; pre-defined; platform-specific.

        The special register (.sreg) state space holds predefined, platform-specific registers, such as grid, cluster, CTA, and thread parameters, clock counters, and performance monitoring registers. All special registers are predefined.

    * `.const`: Shared, read-only memory.

    * `.global`: Global memory, shared by all threads.

    * `.local`: Local memory, private to each thread.

    * `.param`: Kernel parameters, defined per-grid; or Function or local parameters, defined per-thread.

    * `.shared`: Addressable memory, defined per CTA, accessible to all threads in the cluster throughout the lifetime of the CTA that defines it.

    * `.tex`: Global texture memory (deprecated).

* The constant (.const) state space is a read-only memory initialized by the host. Constant memory is accessed with a `ld.const` instruction.

* Constant memory is restricted in size, currently limited to 64 KB which can be used to hold statically-sized constant variables.

    常量的空间竟然有 64 KB，感觉还是比较大的。

* There is an additional 640 KB of constant memory, organized as ten independent 64 KB regions. The driver may allocate and initialize constant buffers in these regions and pass pointers to the buffers as kernel function parameters.

    没有明白这个是啥意思，猜测：默认的字面量和 const 变量都是存在 64 KB 的 constant memory 中，如果有额外的大量常量数据的需要，比如传入一个世界地图的数据之类的，可以在驱动层面额外申请，并且显式地把数据写入进去。推论：640 KB 的 constant memory 是所有 sm 共享的。

* ptx asm, `ld`相关指令的注意事项

    `ld`一个字节的代码：

    ```cpp
    template<>
    __device__ __forceinline__ BytePack<1> ld_volatile_global<1>(uintptr_t *ptr)
    {
        uint32_t ans;
        asm("ld.volatile.global.b8 %0, [%1];" : "=r"(ans) : "l"(ptr));
        return *(BytePack<1>*) &ans;
    }
    ```

    由于这里实现的是`ld_volatile_global()`模板函数的一个特化，要求返回值类型和参数类型的**形式**要和原函数声明一样，因此返回值选用了`BytePack<ElmSize>`，这样可以返回不定长的值，而输入参数使用了`uniptr_t *`，由于指针的长度总是固定的（64 位），所以不需要变长类型，直接统一使用`uniptr_t*`了。

    而且`uniptr_t *`也是下面 asm 命令的要求，`"l"(ptr)`，ptr 的约束必须为`l`，不然编译通不过。

    函数中声明了一个`uint32_t ans;`的变量，这个也是 asm 的要求，`.b8`指令对应的`ans`必须为 32 位。不然编译不通过。因为是 32 位，所以 constraint 使用的是`r`。后面会详细讨论 constraint 相关。

    接下来看 asm 指令，`.b8`的`ld`指令，读 1 个字节，但是`ans`是 32 位，`ptr`是 64 位的指针。说明可能硬件上可能取 32 位数据的前 8 位当作 1 个字节来用。

    return 时使用了强制类型转换。直接`return (BytePack<1>) ans;`显然是编译不过去的。如果写成

    ```cpp
        BytePack<1> ret;
        ret.native = ans;
        return ret;
    ```

    这样又显得繁琐，并且创建了一个中间变量。所以最终选择直接做类型转换。

* 在 ptx 中使用多个 const bank

    ```asm
    .extern .const[2] .b32 const_buffer[];
    ```

    看起来是将`.const[2]`的地址赋给`const_buffer`。

    ```asm
    .extern .const[2] .b32 const_buffer[];
    ld.const[2].b32  %r1, [const_buffer+4]; // load second word
    ```

    这个看上去是从`const_buffer+4`处，取一个`.b32`值。看来即使有了地址，`.const[2]`这些 bank 信息还是不能省。

* Use `ld.global`, `st.global`, and `atom.global` to access global variables.

* The local state space (.local) is typically standard memory with cache. Use ld.local and st.local to access local variables.

    loadl memory 是缓存吗？为什么其他地方说和全局变量一样，是片外的？

* 关于 .local memory，如果架构支持 stack，那么 .local memory 被存在 stack 中，如果没有 stack，那么 .local variable 被存在 fixed address 中，因此函数也不支持递归调用。

* Additional sub-qualifiers ::entry or ::func can be specified on instructions with .param state space to indicate whether the address refers to kernel function parameter or device function parameter.

    看来`__global__` kernel 和`__device__` kernel 的作用还不一样。

    `::entry`和`::func`并没有默认情况，只有根据上下文指令而自动指定。比如`st.param`会对应到`st.param::func`，`isspacep.param`会对应到`isspacep.param::entry`。

* `.param`可能会被映射到 register, stack 或 global memory 上，这一点我们无法确定，具体位置在哪需要参考 ABI 的实现。

### algo

* cuda 实现 reverse

    ```cpp
    #include <iostream>
    #include <cuda_runtime.h>

    // 简单的 reverse kernel
    __global__ void reverse_kernel(int* input, int* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int rev_idx = n - 1 - idx;
            output[rev_idx] = input[idx];
        }
    }

    // 检查 CUDA 错误
    void checkCudaError(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int main() {
        const int n = 10;
        int size = n * sizeof(int);
        
        // 主机数据
        int h_input[n], h_output[n];
        for (int i = 0; i < n; i++) {
            h_input[i] = i + 1;  // 1, 2, 3, ..., 10
        }
        
        // 设备内存分配
        int *d_input, *d_output;
        checkCudaError(cudaMalloc(&d_input, size), "cudaMalloc d_input failed");
        checkCudaError(cudaMalloc(&d_output, size), "cudaMalloc d_output failed");
        
        // 数据拷贝到设备
        checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice),
                      "cudaMemcpy HostToDevice failed");
        
        // 启动 kernel
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        
        reverse_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
        checkCudaError(cudaGetLastError(), "Kernel execution failed");
        
        // 拷贝结果回主机
        checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost),
                      "cudaMemcpy DeviceToHost failed");
        
        // 输出结果
        std::cout << "Original: ";
        for (int i = 0; i < n; i++) std::cout << h_input[i] << " ";
        std::cout << std::endl;
        
        std::cout << "Reversed: ";
        for (int i = 0; i < n; i++) std::cout << h_output[i] << " ";
        std::cout << std::endl;
        
        // 清理
        cudaFree(d_input);
        cudaFree(d_output);
        
        return 0;
    }
    ```

    output:

    ```
    Original: 1 2 3 4 5 6 7 8 9 10 
    Reversed: 10 9 8 7 6 5 4 3 2 1
    ```

    in-place 交换：

    ```cpp
    __global__ void reverse_inplace_kernel(int* data, int n) {
        int tid = threadIdx.x;
        int block_start = blockIdx.x * blockDim.x * 2;  // 每个块处理两倍数据
        
        int left_idx = block_start + tid;
        int right_idx = block_start + blockDim.x * 2 - 1 - tid;
        
        if (left_idx < right_idx && right_idx < n) {
            // 交换对称位置的元素
            int temp = data[left_idx];
            data[left_idx] = data[right_idx];
            data[right_idx] = temp;
        }
    }
    ```

* cuda 矩阵乘的 example

    假设我们要计算：

    C = A * B

    其中 A 是 M x K 的矩阵，B 是 K x N 的矩阵，那么结果 C 就是 M x N 的矩阵。

    example:

    ```cpp
    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime.h>
    #include <assert.h>

    // CUDA 矩阵乘法内核函数
    __global__ void matrixMultiplyKernel(float* A, float* B, float* C, 
                                       int M, int N, int K) {
        // 计算当前线程负责的结果矩阵位置
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        // 检查是否在有效范围内
        if (row < M && col < N) {
            float sum = 0.0f;
            // 计算A的第row行和B的第col列的点积
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    // CPU版本的矩阵乘法，用于验证结果
    void matrixMultiplyCPU(float* A, float* B, float* C, int M, int N, int K) {
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;
                for (int i = 0; i < K; i++) {
                    sum += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }

    // 比较两个矩阵是否相等（允许一定的误差）
    bool compareMatrices(float* A, float* B, int size, float tolerance = 1e-5f) {
        for (int i = 0; i < size; i++) {
            if (fabs(A[i] - B[i]) > tolerance) {
                printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, A[i], B[i]);
                return false;
            }
        }
        return true;
    }

    // 初始化矩阵
    void initializeMatrix(float* matrix, int rows, int cols) {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = static_cast<float>(rand() % 10); // 0-9的随机数
        }
    }

    // 打印矩阵（小矩阵用于调试）
    void printMatrix(float* matrix, int rows, int cols, const char* name) {
        printf("%s:\n", name);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%6.2f ", matrix[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    int main() {
        // 矩阵维度
        const int M = 512;  // A的行数，C的行数
        const int K = 256;  // A的列数，B的行数  
        const int N = 384;  // B的列数，C的列数
        
        printf("Matrix Multiplication: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", 
               M, K, K, N, M, N);
        
        // 分配主机内存
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);
        
        float* h_A = (float*)malloc(size_A);
        float* h_B = (float*)malloc(size_B);
        float* h_C = (float*)malloc(size_C);      // GPU结果
        float* h_C_CPU = (float*)malloc(size_C);  // CPU结果（用于验证）
        
        // 初始化矩阵
        srand(2024); // 固定随机种子以便重现结果
        initializeMatrix(h_A, M, K);
        initializeMatrix(h_B, K, N);
        
        // 分配设备内存
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);
        
        // 拷贝数据到设备
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
        
        // 配置线程块和网格维度
        dim3 blockSize(16, 16); // 256个线程 per block
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                      (M + blockSize.y - 1) / blockSize.y);
        
        printf("Grid: (%d, %d), Block: (%d, %d)\n", 
               gridSize.x, gridSize.y, blockSize.x, blockSize.y);
        
        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // 执行GPU计算
        cudaEventRecord(start);
        matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        
        // 等待内核执行完成
        cudaEventSynchronize(stop);
        
        // 计算GPU运行时间
        float gpuTime = 0.0f;
        cudaEventElapsedTime(&gpuTime, start, stop);
        
        // 拷贝结果回主机
        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
        
        // 执行CPU计算并计时
        clock_t cpuStart = clock();
        matrixMultiplyCPU(h_A, h_B, h_C_CPU, M, N, K);
        clock_t cpuEnd = clock();
        double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;
        
        // 验证结果
        bool resultsMatch = compareMatrices(h_C, h_C_CPU, M * N);
        
        // 输出结果
        printf("\n=== 性能结果 ===\n");
        printf("GPU Time: %.2f ms\n", gpuTime);
        printf("CPU Time: %.2f ms\n", cpuTime);
        printf("Speedup: %.2fx\n", cpuTime / gpuTime);
        printf("Results Match: %s\n", resultsMatch ? "Yes" : "No");
        
        // 打印前3x3部分结果（对于大矩阵）
        if (M >= 3 && N >= 3) {
            printf("\n=== 前3x3结果验证 ===\n");
            printf("GPU Result (top-left 3x3):\n");
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    printf("%8.2f ", h_C[i * N + j]);
                }
                printf("\n");
            }
            
            printf("\nCPU Result (top-left 3x3):\n");
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    printf("%8.2f ", h_C_CPU[i * N + j]);
                }
                printf("\n");
            }
        }
        
        // 清理资源
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_CPU);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        printf("\nDemo completed successfully!\n");
        return 0;
    }
    ```

    output:

    ```
    Matrix Multiplication: A(512 x 256) * B(256 x 384) = C(512 x 384)
    Grid: (24, 32), Block: (16, 16)

    === 性能结果 ===
    GPU Time: 0.07 ms
    CPU Time: 167.03 ms
    Speedup: 2298.38x
    Results Match: Yes

    === 前3x3结果验证 ===
    GPU Result (top-left 3x3):
     5172.00  5910.00  5610.00 
     4534.00  4625.00  4653.00 
     4819.00  5186.00  5153.00 

    CPU Result (top-left 3x3):
     5172.00  5910.00  5610.00 
     4534.00  4625.00  4653.00 
     4819.00  5186.00  5153.00 

    Demo completed successfully!
    ```

*  CUDA 向量加（Vector Addition） example

    ```cpp
    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime.h>

    // CUDA 内核函数：向量加法
    __global__ void vectorAdd(const float* A, const float* B, float* C, int numElements)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < numElements) {
            C[i] = A[i] + B[i];
        }
    }

    // 初始化主机数据的辅助函数
    void initData(float* ptr, int size, float value)
    {
        for (int i = 0; i < size; ++i) {
            ptr[i] = value + i;
        }
    }

    // 验证结果的辅助函数
    bool verifyResult(const float* A, const float* B, const float* C, int numElements)
    {
        const float epsilon = 1.0e-6f;
        for (int i = 0; i < numElements; ++i) {
            if (fabs(C[i] - (A[i] + B[i])) > epsilon) {
                printf("验证失败! 在索引 %d: 期望 %.2f, 得到 %.2f\n", 
                       i, A[i] + B[i], C[i]);
                return false;
            }
        }
        printf("验证成功! 所有元素计算正确。\n");
        return true;
    }

    int main(void)
    {
        // 设置向量长度
        const int numElements = 50000;
        const size_t size = numElements * sizeof(float);
        printf("[信息] 向量加法，每个向量长度为 %d 元素\n", numElements);

        // 分配主机内存
        float *h_A, *h_B, *h_C;
        h_A = (float*)malloc(size);
        h_B = (float*)malloc(size);
        h_C = (float*)malloc(size);

        if (!h_A || !h_B || !h_C) {
            printf("主机内存分配失败!\n");
            exit(EXIT_FAILURE);
        }

        // 初始化主机数组
        initData(h_A, numElements, 1.0f);  // A = [1, 2, 3, ...]
        initData(h_B, numElements, 2.0f);  // B = [2, 3, 4, ...]
        // 期望结果: C = [3, 5, 7, ...]

        // 分配设备内存
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);

        // 拷贝输入数据从主机到设备
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // 启动内核配置
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        printf("[信息] 启动内核配置: %d 个线程块, 每个块 %d 个线程\n", 
               blocksPerGrid, threadsPerBlock);

        // 执行CUDA内核
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        // 等待所有GPU操作完成
        cudaDeviceSynchronize();

        // 检查内核执行是否有错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA 内核执行错误: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // 拷贝结果从设备到主机
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // 验证结果
        verifyResult(h_A, h_B, h_C, numElements);

        // 打印前10个结果作为示例
        printf("\n前10个结果示例:\n");
        for (int i = 0; i < 10 && i < numElements; ++i) {
            printf("C[%d] = %.2f (A[%d] + B[%d] = %.2f + %.2f)\n", 
                   i, h_C[i], i, i, h_A[i], h_B[i]);
        }

        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // 释放主机内存
        free(h_A);
        free(h_B);
        free(h_C);

        printf("\n[信息] 程序执行完成，内存已释放。\n");

        // 重置设备以便性能分析工具获取更准确的数据
        cudaDeviceReset();

        return 0;
    }
    ```

    threadIdx.x：线程在其线程块（Block）内的索引。

    blockIdx.x：线程块在网格（Grid）中的索引。

    blockDim.x：每个线程块中的线程数量。

    compile:

    `nvcc -o vector_add vector_add.cu`

    run:

    `./vector_add`

    output:

    ```
    [信息] 向量加法，每个向量长度为 50000 元素
    [信息] 启动内核配置: 196 个线程块, 每个块 256 个线程
    验证成功! 所有元素计算正确。

    前10个结果示例:
    C[0] = 3.00 (A[0] + B[0] = 1.00 + 2.00)
    C[1] = 5.00 (A[1] + B[1] = 2.00 + 3.00)
    C[2] = 7.00 (A[2] + B[2] = 3.00 + 4.00)
    C[3] = 9.00 (A[3] + B[3] = 4.00 + 5.00)
    C[4] = 11.00 (A[4] + B[4] = 5.00 + 6.00)
    C[5] = 13.00 (A[5] + B[5] = 6.00 + 7.00)
    C[6] = 15.00 (A[6] + B[6] = 7.00 + 8.00)
    C[7] = 17.00 (A[7] + B[7] = 8.00 + 9.00)
    C[8] = 19.00 (A[8] + B[8] = 9.00 + 10.00)
    C[9] = 21.00 (A[9] + B[9] = 10.00 + 11.00)

    [信息] 程序执行完成，内存已释放。
    ```

### accel

* shift 与 reduce

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void reduce_sum(elm_type *cubuf_1, elm_type *cubuf_2)
    {
        int tid = threadIdx.x;
        elm_type val = cubuf_1[tid];
        for (int laneMask = 16; laneMask >= 1; laneMask /= 2)
            val += __shfl_xor_sync(0xffffffff, val, laneMask);
        cubuf_2[tid] = val;
    }

    int main()
    {
        using elm_type = float;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));
        assign_cubuf_order_int<elm_type>(cubuf_1, num_elm);

        reduce_sum<elm_type><<<1, 32>>>(cubuf_1, cubuf_2);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);
        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    output:

    ```
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, specialized as float
    ```

    将`__shfl_xor_sync()`改成`__shfl_down_sync()`，也可以实现类似的效果，输出如下：

    ```
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    496.0, 512.0, 528.0, 544.0, 560.0, 576.0, 592.0, 608.0, 624.0, 640.0, 656.0, 672.0, 688.0, 704.0, 720.0, 736.0, 752.0, 768.0, 784.0, 800.0, 816.0, 832.0, 848.0, 864.0, 880.0, 896.0, 912.0, 928.0, 944.0, 960.0, 976.0, 992.0, specialized as float
    ```

    其中`cubuf_2`只有第一个数字是有效的，其他的都是无效数字。

    如果使用`__shfl_up_sync()`，则最终累加的结果在最后一个：

    ```
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 272.0, 288.0, 304.0, 320.0, 336.0, 352.0, 368.0, 384.0, 400.0, 416.0, 432.0, 448.0, 464.0, 480.0, 496.0, specialized as float
    ```

* 一个能跑通的`__shared__` example:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    __shared__ int val;

    __global__ void test_kern()
    {
        val = 123;
        printf("%d\n", val);
    }

    int main()
    {
        test_kern<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    ```

    output:

    ```
    123
    ```

* 当`__shared__`在 kernel 外声明时，依然有基本功能：被`__shared__`修饰的数据，只在同一个 block 中相冋，并且只能被同一个 block 中的 thread 访问

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>

    __shared__ int a;

    __global__ void my_kern()
    {
        if (threadIdx.x == 0)
        {
            a = blockIdx.x;
        }
        __syncthreads();
        if (threadIdx.x == 1)
        {
            printf("in block %d, a = %d\n", blockIdx.x, a);
        }
    }

    int main()
    {
        my_kern<<<2, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    ```

    compile: `nvcc main.cu -o main`

    run: `./main`

    output:

    ```
    in block 0, a = 0
    in block 1, a = 1
    ```

    可以看到，不同 block 访问到的 shared 数据不同。

### cuda api

* `cvta_to_global()`会将`T*`转换成`uintptr_t`, `uintptr_t`有点像`void*`，其他类型的地址都可以转成这个类型，方便处理。

* `__syncwarp()`

    syntax:

    `void __syncwarp(unsigned mask=0xffffffff);`

    只是一个普通的基于 warp 的同步函数，没有什么特别的。`mask`用于指定进入同步的线程。

* cuda 中的 kernel 无论是 template 形式, 用 cudaLaunchKernel 启动，还是`__global__`修饰，`__device__`修饰，都是可以打断点的。nccl 中用的都是 cuda kernel，因此也是可以使用 cuda gdb 打断点的。通常 hit 一次断点需要 30 分钟以上，目前不清楚原因。

* `__all_sync()`

    原理与`__any_sync()`相似，当一个 warp 中每个线程提供的值都为 1 时，则返回 1，否则返回 0.

    syntax:

    ```cpp
    __any_sync(unsigned mask, predicate);
    ```

    example:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void reduce_sum(elm_type *cubuf_1, elm_type *cubuf_2)
    {
        int tid = threadIdx.x;
        int predicate = tid;
        elm_type ret = __all_sync(0xffffffff, predicate);
        cubuf_1[tid] = ret;
        predicate = 1;
        ret = __all_sync(0xffffffff, predicate);
        cubuf_2[tid] = ret;
    }

    int main()
    {
        using elm_type = int;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));

        reduce_sum<elm_type><<<1, 32>>>(cubuf_1, cubuf_2);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);

        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    output:

    ```
    cubuf 1:
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, specialized as int
    cubuf 2:
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, specialized as int
    ```

* `__ballot_sync()`

    每个线程给定一个值 val，`__ballot_sync()`返回一个 32 位数 ret，如果 val 为非 0，假如 lane id 为`tid`，则将 ret 值的第`tid`位置 1，否则置 0。

    example:

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void reduce_sum(elm_type *cubuf_1, elm_type *cubuf_2, elm_type *cubuf_3)
    {
        int tid = threadIdx.x;
        int predicate = tid % 2;
        unsigned ret = __ballot_sync(0xffffffff, predicate);
        cubuf_1[tid] = ret;
        predicate = 1;
        ret = __ballot_sync(0xffffffff, predicate);
        cubuf_2[tid] = ret;
        predicate = 0;
        ret = __ballot_sync(0xffffffff, predicate);
        cubuf_3[tid] = ret;
    }

    int main()
    {
        using elm_type = unsigned int;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2, *cubuf_3;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_3, num_elm * sizeof(elm_type));

        reduce_sum<elm_type><<<1, 32>>>(cubuf_1, cubuf_2, cubuf_3);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);

        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        printf("cubuf 3:\n");
        print_cubuf<elm_type>(cubuf_3, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        cudaFree(cubuf_3);
        return 0;
    }
    ``` 

    output:

    ```
    cubuf 1:
    2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, 2863311530, specialized as unsigned int
    cubuf 2:
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, specialized as unsigned int
    cubuf 3:
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, specialized as unsigned int
    ```

    `2863311530`的二进制为`1010 1010 1010 1010 1010 1010 1010 1010`, `4294967295`的二进制为`1111 1111 1111 1111 1111 1111 1111 1111`。

* `__shfl_xor_sync()`中，当`laneMask ＝ 3`时，会 4 个元素一组倒序取值

    `main.cu`:

    ```cpp
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include "../utils/cumem_hlc.h"

    template<typename elm_type>
    __global__ void my_kern(elm_type *cubuf_1, elm_type *cubuf_2)
    {
        int tid = threadIdx.x;
        elm_type val = __shfl_xor_sync(0xffffffff, cubuf_1[tid], 3);
        cubuf_2[tid] = val;
    }

    int main()
    {
        using elm_type = float;
        int num_elm = 32;
        elm_type *cubuf_1, *cubuf_2;
        cudaMalloc(&cubuf_1, num_elm * sizeof(elm_type));
        cudaMalloc(&cubuf_2, num_elm * sizeof(elm_type));
        assign_cubuf_order_int<elm_type>(cubuf_1, num_elm);

        my_kern<elm_type><<<1, 32>>>(cubuf_1, cubuf_2);
        cudaDeviceSynchronize();

        printf("cubuf 1:\n");
        print_cubuf<elm_type>(cubuf_1, num_elm);
        printf("cubuf 2:\n");
        print_cubuf<elm_type>(cubuf_2, num_elm);

        cudaFree(cubuf_1);
        cudaFree(cubuf_2);
        return 0;
    }
    ```

    output:

    ```
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    3.0, 2.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0, 11.0, 10.0, 9.0, 8.0, 15.0, 14.0, 13.0, 12.0, 19.0, 18.0, 17.0, 16.0, 23.0, 22.0, 21.0, 20.0, 27.0, 26.0, 25.0, 24.0, 31.0, 30.0, 29.0, 28.0, specialized as float
    ```

    目前不明白是啥原理。

    当`laneMask = 7`时，会 8 个一组反转：

    ```
    cubuf 1:
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, specialized as float
    cubuf 2:
    7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, specialized as float
    ```

* cuda 中没有方法能直接拿到 thread id，只能通过 idx 计算得来

* cuda stream

    `__host__ ​cudaError_t cudaStreamCreate ( cudaStream_t* pStream ) `

    在当前 host thread 最新的 context 上创建一个 stream。

    如果当前 host thread 没有 context，那么在 device 的 primary context 上创建一个 stream，并将这个 context 作为当前 host thread 的 context.

    销毁 stream:

    `__host__ ​ __device__ ​cudaError_t cudaStreamDestroy ( cudaStream_t stream ) `

## ptx

### cache

* `.param` state space 是可寻址（addressable）的，只读（read-only）的。

* The kernel parameter variables are shared across all CTAs from all clusters within a grid.

    这里的 grid 只是编程上的概念，并不是硬件上的概念。在 grid 上共享，那么它在硬件的什么地方？在 global memory 上吗？

* `ld.param`可以将 param 的数据加载到寄存器里，`mov`可以将 param 的地址加载到寄存器里

    examples:

    ```asm
    .entry foo ( .param .b32 N, .param .align 8 .b8 buffer[64] )
    {
        .reg .u32 %n;
        .reg .f64 %d;

        ld.param.u32 %n, [N];
        ld.param.f64 %d, [buffer];
        ...
    ```

    ```asm
    .entry bar ( .param .b32 len )
    {
        .reg .u32 %ptr, %n;

        mov.u32      %ptr, len;
        ld.param.u32 %n, [%ptr];
        ...
    ```

* Kernel function parameters may represent normal data values, or they may hold addresses to objects in constant, global, local, or shared state spaces.

    `.param`变量可能会来自不同的地址空间，在 ptx 汇编中，需要提供它来自哪个 state space。

    但是上面的 example 并没有提供 state space，为什么？

* The current implementation does not allow creation of generic pointers to constant variables (cvta.const) in programs that have pointers to constant buffers passed as kernel parameters.

    generic pointers 不允许指向常数变量，有专门的`cvta.const`指令创建指向 const 变量的指针。

* `.ptr`可以为`.param`添加额外的属性（attribute），这里的属性主要有两种，一个是`.space`，另一个是`.align`。

    语法如下：

    ```asm
    .param .type .ptr .space .align N  varname
    .param .type .ptr        .align N  varname

    .space = { .const, .global, .local, .shared };
    ```

    `.align`表示 4 字节对齐（aligned to a 4 byte boundary.）。

    example:

    ```asm
    .entry foo ( .param .u32 param1,
                 .param .u32 .ptr.global.align 16 param2,
                 .param .u32 .ptr.const.align 8 param3,
                 .param .u32 .ptr.align 16 param4  // generic address
                                                   // pointer
    ) { .. }
    ```

    这个形式看起来比较像一个函数，`.param`标记了一个参数的开始。是否存在不是`.param`的参数？或者说，当一个参数不被`.param`标记时，是否会报错？