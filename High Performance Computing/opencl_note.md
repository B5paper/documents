# OpenCL Note

Home page: <https://www.khronos.org/api/opencl>

Ref:

* <https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/>

* <https://registry.khronos.org/OpenCL/>，这里面资料挺多的

* <https://www.khronos.org/opencl/resources>

## cache

* 在 cpu 上运行 opencl

    安装库：`sudo apt install libpocl2`

    如果需要编译代码，还需要安装头文件：`sudo apt install opencl-headers`

    实测 windows 10 host + i7-1360P + virtual box 虚拟机 + Ubuntu 22.04 可以正常运行。

* `CL_MEM_COPY_HOST_PTR`

    如果需要在创建 buffer 时就把数据从 host 上复制到 buffer 里，那么可以这样写：

    ```cpp
    cl_mem buf_rand_seed_vbuf = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ray_count * sizeof(cl_long), random_seeds, &ret);
    ```

    此时会把`random_seeds`中的数据写入到 buffer 中。

* set arg

    猜测 set arg 的行为是在内存中开辟一个栈，存储一些数据，这些数据为作为 kernel 的参数。这些数据通常有两大类，一类是`cl_mem`，另一类是`cl_float3`，`cl_int`之类的数值。

    猜测 set arg 是按值复制，所以不用担心数值类的局部变量消失后参数失效的问题。由于`cl_mem`是个指针，如果不显式释放，其对应的 buffer （即显存）也不会被释放，所以也不用担心`cl_mem`失效。

    综上，一个`kernel`在整个程序的生命周期中，只需要设置一次 arg 就可以了。后面只需要 writer buffer, nd range kernel 就会触发计算。

* warm up 对 gpu 计算的影响

    可能是因为在创建`OclEnv`的时候对 opencl 环境进行了初始化，但是在析构`OclEnv`对象时，并没有 destroy opencl 环境，而是只 release buffer。导致了 warm up 占用的时间比较长。

    见`ref_23`。 output:

    ```
    cpu sum: 4643.00
    cpu time consumption: 0.008 ms

    opencl device name: gfx1034
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: arr
    gpu sum 1: 4643.00
    gpu time consumption: 0.506 ms

    opencl device name: gfx1034
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: arr
    gpu sum 2: 4643.00
    gpu time consumption: 0.170 ms

    opencl device name: gfx1034
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: arr
    gpu sum 1: 4643.00
    gpu time consumption: 0.153 ms
    ```

    同样是`gpu sum 1`，第一次运行的时候，时间消耗为`0.506 ms`，第二次运行时，时间消耗为`0.153 ms`，相差很大。

    以后再测试性能的时候，可以全程只 init 一次 opencl 环境，保证环境的公平性。

* 对于 256 线程的算子需要归约 8 次，假如现在有 1024 个数据，有两种处理方案：

    1. 让 256 线程先每个线程处理 4 个数据，再按 256 线程归约 256 个数据

        理论分析这样的时间复杂度是`4 + 8 = 12 `

    2. 让 256 线程处理前 256 个数据，然后按这样处理 4 次，最终得到 4 个数据，再从 256 线程中挑出来 4 个线程对这 4 个数据进行归约，或者挑出来 1 个线程对这 4 个数据进行处理。

        理论上，时间复杂度为`8 * 4 + 2 = 34`, or `8 * 4 + 4 = 36`

    3. 同样是先让 256 线程循环批量处理 256 个数据，得到 4 个数据后，再起 kernel，用 4 个线程处理这 4 个数据；或者用 cpu 处理这 4 个数据。

        时间复杂度：同上。

    在实际的测试中，方法 1 和方法 2 的时间几乎相同。

    test code: 见`ref_22`

    output:

    ```
    cpu sum: 4643.00
    cpu time consumption: 0.008 ms

    opencl device name: gfx1034
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: arr
    gpu sum 1: 4643.00
    gpu time consumption: 0.052 ms

    opencl device name: gfx1034
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: arr
    gpu sum 2: 4643.00
    gpu time consumption: 0.048 ms
    ```

    根据结果来看，甚至方法 1 的速度还更快一些。

* N = 1024, max group size = 256 时的一个加法归约算子实现

    ```opencl
    kernel void sum_2(global float *arr, global float *out, const int len)
    {
        size_t glb_id = get_global_id(0);
        size_t work_span = len / 256;
        float s = arr[glb_id * work_span];
        for (int i = 1; i < work_span; ++i)
        {
            s += arr[glb_id * work_span + i];
        }
        arr[glb_id * work_span] = s;
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int span_len = work_span * 2; span_len <= len; span_len *= 2)
        {
            if ((glb_id * work_span) % span_len == 0)
            {
                arr[glb_id * work_span] += arr[glb_id * work_span + span_len / 2];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        
        if (glb_id == 0)
            *out = arr[0];
    }
    ```

    这个思路是先以 4 个数据为长度分组，每组分一个线程。每个线程遍历 4 个数据求和，这样就把 1024 个数字缩减成了 256 个数据。接下来再按照 256 线程去求解就可以了。

    这个算法可以保证计算结果总是正确，不会有数据同步造成的问题。

    output:

    ```
    opencl device name: gfx1034
    gpu time duration: 0.510
    cpu time duration: 0.002
    correct.
    [Warning] destroy ocl env
    release mem: out
    release mem: arr
    release ocl buffer: out
    release ocl buffer: arr
    ```

* gpu 最大同步线程数限制对归约算法的影响

    对于一个数组`float arr[N];`，当`N = 1024`时，使用下面的归约算子并不能得到正确的结果：

    ```opencl
    kernel void my_sum(global float *arr, const int len)
    {
        size_t glb_id = get_global_id(0);
        size_t glb_size = get_global_size(0);  // number of all threads
        size_t max_id = glb_size;
        for (int span_len = 2; span_len <= len; span_len *= 2)
        {
            if (glb_id > max_id)
                continue;
            max_id /= 2;
            size_t pos = glb_id * span_len;
            arr[pos] += arr[pos + span_len / 2];
        }
    }
    ```

    因为 gpu 通常有个最大同步线程数限制，它表示了最大能有多少个线程同步执行。目前我的机器上这个数字是`256`。这个数字即 max group size。

    下面分析一下，上面的代码会出什么错。
    
    当`N = 1024`，`max_group_size = 256`时，我们 run kernel 实际起了`N / 2 = 512`个线程。由于最大只能保证 256 个线程同时执行，所以其实这 512 个线程被分成了两组。这两组线程是乱序执行的。

    为了便于分析，我们将这个问题转换成，当`N = 8`，`max_group_size = 4`时，会是什么情况：

    ```cpp
    for (int i = 0; i < 10240; ++i)
    {
        sync_cpu_to_gpu({"arr", "out"});
        run_kern("my_sum", {(size_t) N}, {(size_t) 4}, "arr", "out", N);
        sync_gpu_to_cpu({"out"});
        
        if (s != *out)
        {
            printf("i: %d\n", i);
            printf("fault result.\n");
            printf("cpu sum: %f\n", s);
            printf("gpu sum: %f\n", *out);
            for (int j = 0; j < N; ++j)
                printf("%.1f, ", arr[j]);
            printf("\n");
            sync_gpu_to_cpu({"arr"});
            for (int j = 0; j < N; ++j)
                printf("%.1f, ", arr[j]);
            printf("\n");
            return 0;
        }
    }
    ```

    每次的输出都是不一样的，多次运行程序后，一个可能的输出如下：

    ```
    i: 9912
    fault result.
    cpu sum: 41.000000
    gpu sum: 33.000000
    7.0, 1.0, 6.0, 8.0, 5.0, 6.0, 7.0, 1.0, 
    33.0, 1.0, 14.0, 8.0, 19.0, 6.0, 8.0, 1.0, 
    [Warning] destroy ocl env
    release mem: out
    release mem: arr
    release ocl buffer: out
    release ocl buffer: arr
    ```

    初始数组是`[7, 1, 6, 8, 5, 6, 7, 1]`。

    第一次归约，我们得到：`[8, 1, 14, 8, 11, 6, 8, 1]`

    第二次归约，我们得到：`[22, 1, 14, 8, 19, 6, 8, 1]`

    可以看到，cpu 计算出的结果，执行的是`22 + 19 = 41`，而 gpu 计算出的结果，执行的是`22 + 11 = 33`。gpu 同时使用了当前轮的结果和上一轮的结果。

    对于归约算法来说，当前轮的结果永远依赖于上一轮的结果。但是 gpu 对不同 group 的乱序执行违反了这一规则，因此计算结果是错误的。

    如何解决这一问题？opencl 对不同 group 之间的同步性并不能做保证，即使`barrier();`也只能保证 group 内的线程同步。唯一的解决办法是从算法入手。

* sum reduce 算子实现的线性思考

    假如现在要对一个一维的`float`数组`arr`求和，为了简化问题，我们假设数组的长度`N`满足 2 的 n 次幂。

    首先能想到的是，我们起`N`个线程，线程的 id 从`0`到`N - 1`。

    在做第一次归约的时候，我们需要计算`arr[0] += arr[1];`, `arr[2] += arr[3]`，`arr[4] += arr[5]`, ...

    如下图所示：

    <div style='text-align:center'>
    <img src='../../Reference_resources/ref_21/pics/pic_1.png'>
    </div>

    针对这一过程，写出的代码如下：

    ```opencl
    kernel void my_sum(global float *arr, const int len)
    {
        size_t glb_id = get_global_id(0);
        if (glb_id % 2 == 0)  // selecte the work items located at 0, 2, 4, 6, ...
        {
            arr[glb_id] += arr[glb_id + 1];
        }
    }
    ```

    可以看到，第一次归约我们只用到了 0, 2, 4, 6, ... 这些位置上的线程，因此其实我们不需要起`N`个线程，只需要起`N / 2`个线程就可以了。这个问题我们暂时先放着，继续往下走。

    第二次归约，我们需要做`arr[0] += arr[2]`，`arr[4] += arr[6]`, ... 这样的计算，如下图所示：

    <div style='text-align:center'>
    <img src='../../Reference_resources/ref_21/pics/pic_2.png'>
    </div>

    写成代码如下：

    ```opencl
    kernel void my_sum(global float *arr, const int len)
    {
        size_t glb_id = get_global_id(0);
        if (glb_id % 2 == 0)  // select the work items located at 0, 2, 4, 6, ...
        {
            arr[glb_id] += arr[glb_id + 1];
        }

        if (glb_id % 4 == 0)  // select the work items located at 0, 4, 8, ...
        {
            arr[glb_id] += arr[glb_id + 2];
        }
    }
    ```

    到这里我们差不多可以对代码抽象一下了，我们的取模从 2 开始，然后不断乘 2 翻倍。那么这个变量到什么时候停止？可以观察到这个变量恰好是每个小区间的长度，我们给这个变量取个名字`span_len`。显然对于最后一次归约，`span_len`应该等于数组长度`len`才对。

    同时我们观察到加法的第二个加数的索引，好正是`glb_id + span_len / 2`。这一点我们结合实际意义也能想通，即每次都相加区间一半处的值。

    由此我们可以将代码归纳为：

    ```opencl
    kernel void my_sum(global float *arr, const int len)
    {
        size_t glb_id = get_global_id(0);
        for (int span_len = 2; span_len <= len; span_len *= 2)
        {
            if (glb_id % span_len == 0)
            {
                arr[glb_id] += arr[glb_id + span_len / 2];
            }
        }
    }
    ```

    重新回到我们前面放着的问题，由于第一次归约只需要起`N / 2`的线程，这意味着索引编号变成了`[00, N / 2 - 1]`，因此使用取模`%`来筛选线程不适用了。我们重构代码如下：

    ```opencl
    kernel void my_sum(global float *arr, const int len)
    {
        size_t glb_id = get_global_id(0);
        for (int span_len = 2; span_len <= len; span_len *= 2)
        {
            size_t pos = glb_id * span_len;
            if (pos < len)
            {
                arr[pos] += arr[pos + span_len / 2];
            }
        }
    }
    ```

    在调用 kernel 时，我们也需要将 global size 改为`N / 2`。

    我们通过`glb_id * span_len`的方式让索引“散开”，然后使用`pos < len`限制索引范围，防止数组越界。

    由于`span_len`是指数级增加，所以`glb_id * span_len`会增长得非常快，发生溢出。

    我们知道归约需要的线程数是每次折半的，因此可以写出这样的代码：

    ```opencl
    kernel void my_sum(global float *arr, const int len)
    {
        size_t glb_id = get_global_id(0);
        size_t glb_size = get_global_size(0);  // number of all threads
        size_t max_id = glb_size;
        for (int span_len = 2; span_len <= len; span_len *= 2)
        {
            if (glb_id > max_id)
                continue;
            max_id /= 2;
            size_t pos = glb_id * span_len;
            arr[pos] += arr[pos + span_len / 2];
        }
    }
    ```

* 求 sum 的 reduce 算法

    example code 见`ref_20`。

    这个代码的想法是只申请一个 work group，把 global memory 上的数据全部拷贝到 local memory 上。然后利用 local memory 的快速访问进行归约。

    经过测试，在当前的 gpu 上，work group size 最大为 256。再大会发生 opencl kernel 启动失败的情况。

    是一个 work group 中默认有 256 个 work item 线程，还是软件定义了一个虚拟的最大值？

    显然正常情况下一个`arr`的 length 会远远大于 256，那么该怎么改代码，使得其对无论多大的 length 都有效？

* 或许可以使用`clGetKernelArgInfo()`得到 kernel function 的参数类型信息，从而动态判断是否将某个字符串类型的参数转换成 opencl memory buffer。

    example code:

    ```cpp
    int ret;
    cl_kernel_arg_address_qualifier addr_qlf;
    ret = clGetKernelArgInfo(kernel, cur_arg_idx, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(addr_qlf), &addr_qlf, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("fail to get kernel arg info\n");
        exit(-1);
    }
    if (addr_qlf == CL_KERNEL_ARG_ADDRESS_LOCAL)
    {
        OclLocalBuf &local_buf = ocl_env->local_bufs[arg];
        ret = clSetKernelArg(kernel, cur_arg_idx, local_buf.size, NULL);
    }
    ```

* opencl 中 local memory 的使用可以提高显存访问效率

    ```opencl
    kernel void mat_mul_ocl_mem_opt(global float *M_A, global float *M_B, global float *output,
        const uint n_row_A, const uint n_col_A, const uint n_col_B,
        local float *vec_B)
    {
        size_t glb_id = get_global_id(0);
        size_t grp_id = get_group_id(0);
        size_t loc_id = get_local_id(0);
        size_t grp_size = get_local_size(0);
        size_t num_grps = n_row_A / grp_size;
        size_t micro_len = n_col_A / grp_size;

        // copy from global to private
        private float vec_A[1024];
        for (int i = 0; i < n_col_A; ++i)
            vec_A[i] = M_A[glb_id * n_col_A + i];  // M_A 的第 glb_id 行，第 i 列
        
        for (int i = 0; i < n_col_B; ++i)
        {
            // copy from global to local shared
            for (int j = 0; j < micro_len; ++j)
            {
                vec_B[loc_id * micro_len + j] = M_B[(loc_id * micro_len + j) * n_col_B + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // calc vector dot product, and store the result into global memory
            float prod_sum = 0.0f;
            for (int k = 0; k < n_col_A; ++k)
                prod_sum += vec_A[k] * vec_B[k];
            output[glb_id * n_col_B + i] = prod_sum;   
        }
    }
    ```

    ```cpp
    void test_mat_mul_ocl_mem_opt(float A[], float B[], float C[],
        int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
    {
        init_ocl_env("./kernels.cl", {"mat_mul_ocl_mem_opt"});
        int elm_num_A = n_row_A * n_col_A;
        int elm_num_B = n_col_A * n_col_B;
        int elm_num_C = n_row_A * n_col_B;
        add_buf("A", sizeof(float), elm_num_A);
        add_buf("B", sizeof(float), elm_num_B);
        add_buf("C", sizeof(float), elm_num_C);
        global_ocl_env->add_local_buf("vec_B", sizeof(float), 1024);

        timeit_ms("start");
        write_buf("A", A);
        write_buf("B", B);
        run_kern("mat_mul_ocl_mem_opt",
            {(size_t) n_row_A}, {(size_t) n_row_A / 8},
            "A", "B", "C", n_row_A, n_col_A, n_col_B,
            "vec_B");
        read_buf(C, "C");
        float dur_ms = timeit_ms("end");

        if (disp_mat)
        {
            printf("A:\n");
            print_mat(A, n_row_A, n_col_A);
            putchar('\n');
            printf("B:\n");
            print_mat(B, n_col_A, n_col_B);
            putchar('\n');
            printf("C:\n");
            print_mat(C, n_row_A, n_col_B);
            putchar('\n');
        }

        printf("time duration: %.3f ms\n", dur_ms);
        exit_ocl_env();
    }
    ```

    对于`A[1024][1024]`与`B[1024][1024]`的矩阵相乘，使用 2d 版本的 opencl 算子，速度可以达到`364.218 ms`，但是使用上面的代码，速度可以达到`27.263 ms`。

    上面的 opencl 算子将矩阵 A 按行分成几个 block（上面的代码是分成了 8 个 block），对于矩阵 B，每次都拷贝一列到 local mem，然后让每个 block 去和矩阵 B 的一列做矩阵乘法，并保存计算结果。每个 work item 负责一个 bloack。

    几个关键点：

    1. 要想使用 local memory (在 cuda 里叫 shared memory)，必须在 kernel 函数的参数里加上使用`local`修饰的指针

        比如上面代码里的`local float *vec_B`。

    2. 在 host 程序上，使用下面的方式为 local memory 分配显存

        ```cpp
        clSetKernelArg(kernel, cur_arg_idx, buf_size, NULL);
        ```

        其中`buf_size`指的是 local mem 占用多少字节，第四个参数填`NULL`。

    3. 由于是使用 group 里的所有 work item 将全局显存`B`中的数据搬到 local memory，所以还需要加一个 barrier，保证 group 里所有的 work item 都完成任务

        ```cpp
        barrier(CLK_LOCAL_MEM_FENCE);
        ```

    4. 程序有时候会出现计算错误的情况，目前不清楚是怎么回事

* opencl 中通常二维的 work size 比一维的 work size 要快

    ```opencl
    kernel void mat_mul(global float *M_A, global float *M_B, global float *output,
        const uint n_row_A, const uint n_col_A, const uint n_col_B)
    {
        size_t gid = get_global_id(0);
        for (uint p = 0; p < n_col_B; ++p)
        {
            float prod_sum = 0.0f;
            for (uint k = 0; k < n_col_A; ++k)
            {
                prod_sum += M_A[gid * n_col_A + k] * M_B[k * n_col_B + p];
            }
            output[gid * n_col_B + p] = prod_sum;
        }
    }

    kernel void mat_mul_2d(global float *M_A, global float *M_B, global float *output,
        const uint n_row_A, const uint n_col_A, const uint n_col_B)
    {
        size_t gid_0 = get_global_id(0);
        size_t gid_1 = get_global_id(1);
        float prod_sum = 0.0f;
        for (uint k = 0; k < n_col_A; ++k)
        {
            prod_sum += M_A[gid_0 * n_col_A + k] * M_B[k * n_col_B + gid_1];
        }
        output[gid_0 * n_col_B + gid_1] = prod_sum;
    }
    ```

    上面的两个矩阵乘法算子，`mat_mul()`是将第一个矩阵的行拆成`n_row_A`份，然后让矩阵`A`的一秆和矩阵`B`的每一列相乘，算出输出矩阵`C`的结果。`mat_mul_2d()`是将第一个矩阵`A`拆成`n_row_A`行，将`B`拆成`n_col_B`列，然后再做向量点积，计算出结果。

    对应的 cpp 代码：

    ```cpp
    void test_mat_mul_ocl(float A[], float B[], float C[],
        int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
    {
        init_ocl_env("./kernels.cl", {"mat_mul"});
        int elm_num_A = n_row_A * n_col_A;
        int elm_num_B = n_col_A * n_col_B;
        int elm_num_C = n_row_A * n_col_B;
        add_buf("A", sizeof(float), elm_num_A);
        add_buf("B", sizeof(float), elm_num_B);
        add_buf("C", sizeof(float), elm_num_C);
        timeit_ms("start");
        write_buf("A", A);
        write_buf("B", B);
        run_kern("mat_mul", {(size_t) n_row_A}, "A", "B", "C",
            n_row_A, n_col_A, n_col_B);
        read_buf(C, "C");
        float dur_ms = timeit_ms("end");

        if (disp_mat)
        {
            printf("A:\n");
            print_mat(A, n_row_A, n_col_A);
            putchar('\n');
            printf("B:\n");
            print_mat(B, n_col_A, n_col_B);
            putchar('\n');
            printf("C:\n");
            print_mat(C, n_row_A, n_col_B);
            putchar('\n');
        }
        printf("time duration: %.3f ms\n", dur_ms);
        exit_ocl_env();
    }

    void test_mat_mul_2d(float A[], float B[], float C[],
        int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
    {
        init_ocl_env("./kernels.cl", {"mat_mul_2d"});
        int elm_num_A = n_row_A * n_col_A;
        int elm_num_B = n_col_A * n_col_B;
        int elm_num_C = n_row_A * n_col_B;
        add_buf("A", sizeof(float), elm_num_A);
        add_buf("B", sizeof(float), elm_num_B);
        add_buf("C", sizeof(float), elm_num_C);
        timeit_ms("start");
        write_buf("A", A);
        write_buf("B", B);
        run_kern("mat_mul_2d", {(size_t) 1024, (size_t) 1024},
            "A", "B", "C", n_row_A, n_col_A, n_col_B);
        read_buf(C, "C");
        float dur_ms = timeit_ms("end");

        if (disp_mat)
        {
            printf("A:\n");
            print_mat(A, n_row_A, n_col_A);
            putchar('\n');
            printf("B:\n");
            print_mat(B, n_col_A, n_col_B);
            putchar('\n');
            printf("C:\n");
            print_mat(C, n_row_A, n_col_B);
            putchar('\n');
        }
        printf("time duration: %.3f ms\n", dur_ms);
        exit_ocl_env();
    }
    ```

* opencl 算子中的 global memory 访问会降低速度

    比如，对于这样一段 opencl 代码：

    ```opencl
    kernel void mat_mul(global float *M_A, global float *M_B, global float *output,
        const uint n_row_A, const uint n_col_A, const uint n_col_B)
    {
        size_t gid = get_global_id(0);
        for (uint p = 0; p < n_col_B; ++p)
        {
            float prod_sum = 0.0f;
            for (uint k = 0; k < n_col_A; ++k)
            {
                prod_sum += M_A[gid * n_col_A + k] * M_B[k * n_col_B + p];
            }
            output[gid * n_col_B + p] = prod_sum;
        }
    }
    ```

    ```cpp
    void test_mat_mul_ocl(float A[], float B[], float C[],
        int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
    {
        init_ocl_env("./kernels.cl", {"mat_mul"});
        int elm_num_A = n_row_A * n_col_A;
        int elm_num_B = n_col_A * n_col_B;
        int elm_num_C = n_row_A * n_col_B;
        add_buf("A", sizeof(float), elm_num_A);
        add_buf("B", sizeof(float), elm_num_B);
        add_buf("C", sizeof(float), elm_num_C);
        timeit_ms("start");
        write_buf("A", A);
        write_buf("B", B);
        run_kern("mat_mul", {(size_t) n_row_A}, "A", "B", "C",
            n_row_A, n_col_A, n_col_B);
        read_buf(C, "C");
        float dur_ms = timeit_ms("end");
        printf("time duration: %.3f ms\n", dur_ms);
        exit_ocl_env();
    }
    ```

    当`n_row_A = 1024`, `n_col_A = 1024`, `n_col_B = 1024`时，运行时间为`655.946 ms`。

    但是如果把 opencl 代码改成

    ```opencl
    kernel void mat_mul(global float *M_A, global float *M_B, global float *output,
        const uint n_row_A, const uint n_col_A, const uint n_col_B)
    {
        size_t gid = get_global_id(0);
        for (uint p = 0; p < n_col_B; ++p)
        {
            output[gid * n_col_B + p] = 0.0f;
            for (uint k = 0; k < n_col_A; ++k)
            {
                output[gid * n_col_B + p] += M_A[gid * n_col_A + k] * M_B[k * n_col_B + p];
            }
        }
    }
    ```

    运行时间就变成了`1311.735 ms`。可以看出运行时间是之前的两倍。

    `output`是放在 global 上的显存，每次对它的访问都有时延。但是当我们使用一个`private`值`prod_sum`存储计算的中间结果时，访问速度就会快很多。

    （理论上`private`和`global`的显存都放在 device memorty 里，但是为什么访问速度不一样？）

* opencl programming guide 中，矩阵乘法 matrix multiplication 的 example，其索引二维数组的方式很可能是错的

    对于一个 shape 为`M x N`的数组`float A[M][N]`，设`i`为行索引，`j`为列索引，那么索引方式应该为

    ```cpp
    *(A + i * N + j);  // A[i][j]
    ```

    显然如果`i`的范围为`0 ~ M-1`，和`M`相关，那么说到第`i`行时，就应该让`i`去乘`N`。

    但是书上的`i`的取值和`M`相关，`i`乘的也同样是`M`，这肯定是有问题的。

    猜想：如果从一个维度中取索引，那么就应该让它乘其他所有维度的长度的乘积。

* opencl 中 global work size 的每个维度的长度，都必须是 local work size 的对应维度长度的整数倍

    如果无法被恰好分成整数倍，多余的数据必须重新调用一次 ndrange。

* opencl read image 提供了各种内置函数，可以将图片读取成浮点数形式，还可以读取成整数形式

	可以以整数坐标获取像素值，还可以以浮点数坐标获取像素值。

* opencl miscellaneous functions

	* shuffle 可以给一个 vector 排序

		比如这段代码会把`1, 2, 3, 4`变成`4, 3, 2, 1`：

		```c
		kernel void test_shuffle(global float4 *out)
		{
			uint4 mask = (uint4)(3, 2, 1, 0);
			float4 a = (float4)(1, 2, 3, 4);
			float4 r = shuffle(a, mask);
			*out = r;
		}
		```

		`shuffle()`主要是将`a`按顺序赋给`r`的`mask`索引处的值。

		问题：如果是取`a`在`mask`索引处的值赋给`r`，那么和上面的描述相比有什么不同？

		```c
		kernel void test_shuffle(global float8 *out)
		{
			uint8 mask = (uint8)(3, 2, 1, 0, 7, 6, 5, 4);
			float4 a = (float4)(1, 2, 3, 4), b = (float4)(5, 6, 7, 8);
			float8 r = shuffle2(a, b, mask);
			*out = r;
		}
		```

		上面代码输出为

		```
		4.00, 3.00, 2.00, 1.00, 8.00, 7.00, 6.00, 5.00
		```

		> The elements of the input vectors are numbered from left to right across one or both of the vectors. For this purpose, the number of elements in a vector is given by vec_step(gentypem).

		根据这个描述，看起来如果有 3 元素的向量，那么它被对待成 4 元素。

	* `vec_step()`可以返回指定向量或数据类型包含的元素个数

		example:

		```opencl
		kernel void test_shuffle(global int *out)
		{
			float vec_1;
			float2 vec_2;
			float3 vec_3;
			float4 vec_4;
			int elm_num_1 = vec_step(vec_1);  // 1
			int elm_num_2 = vec_step(vec_2);  // 2
			int elm_num_3 = vec_step(vec_3);  // 4
			int elm_num_4 = vec_step(vec_4);  // 4

			out[0] = elm_num_1;
			out[1] = elm_num_2;
			out[2] = elm_num_3;
			out[3] = elm_num_4;
		}
		```

		注意，`float3`其实有 4 个元素。

* opencl build-in functions

	syntax:

	```c
	int atomic_add(volatile global int *p, int val)
	```

	Read the 32-bit value (referred to as old) stored at the location pointed by p. Compute (old + val) and store the result at the location pointed by p. The function returns old.

	类似的函数还有

	```c
	int atomic_sub(volatile global int *p, int val)

	int atomic_inc(volatile global int *p)
	int atomic_dec(volatile global int *p)
	```

	其余的几个看函数名就能猜出来意思，这里就不多写了。

	其他的一些 atomic 函数：

	* `atomic_xchg`

		syntax:

		```c
		int	atomic_xchg(volatile global int *p, int val)
		```

		> Swap the old stored at location p with new value given by val. The function returns old.

	* `atomic_cmpxchg`

		syntax:

		```c
		int atomic_cmpxchg(volatile global int *p, int cmp, int val)
		```

		Read the 32-bit value (referred to as old) stored at the location pointed by p. Compute (old == cmp) ? val : old and store the result at the location pointed by p. The function returns old.

	* `atomic_min`

		syntax:

		```c
		int atomic_min(volatile global int *p, int val)
		```

		Read the 32-bit value (referred to as old) stored at the location pointed by p. Compute min(old, val) and store the result at the location pointed by p. The function returns old.

		相似的函数：

		```c
		int atomic_max(volatile global int *p, int val)
		```

	* `atomic_and`

		syntax:

		```c
		int atomic_and(volatile global int *p, int val)
		```

		Read the 32-bit value (referred to as old) stored at the location pointed by p. Compute (old & val) and store the result at the location pointed by p. The function returns old.

		其他几个相似的函数：

		```c
		int atomic_or(volatile global int *p, int val)
		int atomic_xor(volatile global int *p, int val)
		```

* opencl： 并没有一个方法可以提前知道 kernel 中参数的 size。

* `clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, info_size, (void*) info.data(), &info_size);`

	返回的直接就是整数类型，不是字符串。

* opencl synchronization functions

    * `barrier`

        ```cpp
        void barrier(cl_mem_fence_flags flags)
        ```

        > All work-items in a work-group executing the kernel on a compute unit must execute this function before any are allowed to continue execution beyond the barrier.

        parameters:

        * `CLK_LOCAL_MEM_FENCE`: The barrier function will either flush any variables stored in local memory or queue a memory fence to ensure correct ordering of memory operations to local memory.

            如果`barrier()`前面的代码不涉及到 global memory 的写入，就用这个

        * `CLK_GLOBAL_MEM_FENCE`： The barrier function will either flush any variables stored in global memory or queue a memory fence to ensure correct ordering of memory operations to global memory. This is needed when work- items in a work-group, for example, write to a buffer object in global memory and then read the updated data.

            如果`barrier()`前面的代码涉及到 global memory 的写入，并且需要同步，那么就用这个。

    * `async_work_group_copy()`

        syntax:

        ```cpp
        event_t async_work_group_copy(
            local gentype *dst,
            const global gentype *src,
            size_t num_gentypes,
            event_t event)

        event_t async_work_group_copy(global gentype *dst,
            const local gentype *src,
            size_t num_gentypes,
            event_t event)
        ```

        异步复制数据。函数的返回值`event`是用来让函数`wait_group_events()`使用的，等待事件列表完成。而参数中的`event`是用来让多个函数 share 的，也就是说多个函数都注册到这个 event 上。（不清楚这些个函数是串行执行还是并行，应该是并行）

    * `async_work_group_strided_copy`

        syntac:

        ```cpp
        event_t async_work_group_strided_copy(
            local gentype *dst,
            const global gentype *src,
            size_t num_gentypes,
            size_t src_stride,
            event_t event)

        event_t async_work_group_strided_copy(
            global gentype *dst,
            const local gentype *src,
            size_t num_gentypes,
            size_t dst_stride,
            event_t event)
        ```

        按`stride`去增加偏移量。其他的和`async_work_group_copy`相同。

    * `wait_group_events`

        syntax:

        ```cpp
        void wait_group_events(int num_events,
            event_t *event_list)
        ```

        Wait for events that identify the copy operations associated with `async_work_group_copy` and `async_work_group_strided_copy` functions to complete. The event objects specified in `event_list` will be released after the wait is performed.

        等待事件完成。

    * `prefetch`

        ```c
        void prefetch(const global gentype *p,
            size_t num_gentypes)
        ```

        Prefetch num_gentypes * sizeof(gentype) bytes into the global cache. The prefetch function is applied to a work-item in a work-group and does not affect the functional behavior of the kernel.

        先把一些数据放到缓存里，以提高命中率。

* `vloadn`

    从指定地址 + 指定偏移读取数据。

    syntax:

    ```c
    gentypen vloadn(size_t offset,
        const global gentype *p)
    ```

    Returns sizeof(gentypen) bytes of data read from address (p + (offset * n)).

    The address computed as `(p + (offset * n))` must be 8-bit aligned if gentype is `char` or `uchar`; 16-bit aligned if gentype is `short` or `ushort`; 32-bit aligned if gentype is `int`, `uint`, or `float`; 64-bit aligned if gentype is `long`, `ulong`, or `double`.

    vloadn is used to do an unaligned vector load.

    example:

    `kernels.cl`:

    ```opencl
    kernel void test(global float *in, global float3 *out)
    {
        *out = vload3(1, in);
    }
    ```

    `main.cpp`:

    ```cpp
    #include "../ocl_simple/global_ocl_env.h"

    int main()
    {
        float arr[] = {
            0, 1, 2,
            3, 4, 5
        };
        cl_float3 out;
        init_global_ocl_env("kernels.cl", {"test"});
        add_buf("arr", sizeof(float), sizeof(arr) / sizeof(float), arr);
        add_buf("out", sizeof(cl_float3), 1);
        run_kern("test", {1}, "arr", "out");
        read_buf(&out, "out");
        printf("%f, %f, %f, %f\n", out.s[0], out.s[1], out.s[2], out.s[3]);
        return 0;
    }
    ```

    输出：

    ```
    opencl device name: gfx1034
    3.000000, 4.000000, 5.000000, 0.000000
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: arr
    ```

    要求`arr`必须是`float`类型，同样，给 kernel 传递参数时，也需要以`float`指针来传递。如果使用`cl_float3`存储数据以及传递，数据会出错。

    因为`cl_float3`实际上占用的是`sizeof(float) * 4`个字节的内存，所以会有一些问题。

* `vstoren`

    syntax:

    ```c
    gentypen vstoren(gentypen data,
        size_t offset,
        global gentype *p)
    ```

    Write sizeof(gentypen) bytes given by data to address (p + (offset * n)).

    这个函数的返回值类型应该是`void`吧，好像书上写错了。

    这个函数的使用和`vloadn()`正好相反。

    example:

    `kernels.cl`:

    ```c
    kernel void test(global float3 *in, global float *out)
    {
        float3 data = *in;
        vstore3(data, 1, out);
    }
    ```

    `main.cpp`:

    ```cpp
    #include "../ocl_simple/global_ocl_env.h"

    int main()
    {
        cl_float3 in = {1, 2, 3};
        float out[6] = {0};
        init_global_ocl_env("kernels.cl", {"test"});
        add_buf("in", sizeof(cl_float3), sizeof(cl_float3) / sizeof(cl_float3), &in);
        add_buf("out", sizeof(float), 6);
        run_kern("test", {1}, "in", "out");
        read_buf(&out, "out");
        printf("%f, %f, %f\n", out[3], out[4], out[5]);
        return 0;
    }
    ```

    输出：

    ```
    opencl device name: gfx1034
    1.000000, 2.000000, 3.000000
    [Warning] destroy ocl env
    release ocl buffer: out
    release ocl buffer: in
    ```

    可以看到，`in`中的数组被存储到了`out`的 3，4，5 这三个索引处。

* `vload_half()`与`vload_halfn()`，`vstore_harf()`与`vstore_harfn()`

    syntax:

    ```c
    float vload_half(size_t offset,
        const global half *p)

    floatn vload_halfn(size_t offset,
        const global half *p)

    void vstore_half(float data,
        size_t offset,
        global half *p)

    void vstore_halfn(floatn data,
        size_t offset,
        global half *p)
    ```

    Returns sizeof(half) bytes of data read from address (p + offset).

* 有关`half`类型和`e`, `z`, `p`, `n`

    ```cpp
    void vstore_half_rte(double data, size_t offset, half *p)
    void vstore_half_rtz(double data, size_t offset, half *p)
    void vstore_half_rtp(double data, size_t offset, half *p)
    void vstore_half_rtn(double data, size_t offset, half *p)
    ```

    这几个都涉及到浮点数到整数的转换，猜测`rte`的意思是向偶数取整，`rtz`是向零取整，`rtp`是向正无穷取整，`rtn`是向负方向取整。

    `rt`表示 round to，`e`表示 even，`z`表示 zero，`p`表示 positive，`n`表示 negative。

* `vloada_halfn`

    `load`或`store`后面带一个`a`的，表示 aligned。指针的偏移必须是 1, 2, 4, 8, 16 的整数倍。

    如果碰到`float3`这样的数据，会自动对齐到`float4`。

* opencl 

    `clGetPlatformIDs()`的第一个参数`num_entries`经测试也可以设置为 0.

    其实可以像 vulkan 那样，用这种方式获得 platform id:

    ```cpp
    #define CL_TARGET_OPENCL_VERSION 300
    #include <CL/cl.h>
    #include <stdio.h>

    int main()
    {
        cl_uint num_plats;
        clGetPlatformIDs(0, nullptr, &num_plats);
        cl_platform_id *plats = (cl_platform_id*) malloc(num_plats * sizeof(cl_platform_id));
        clGetPlatformIDs(num_plats, plats, &num_plats);
        printf("opencl platform number: %d\n", num_plats);
        return 0;
    }
    ```

    `malloc()`可以换成 vector。

    既然`cl_platform_id`已经是个独立的类型了，其实 id 就代表了 platform，那么变量的命名也就没必要再加上 id 了。就好像 vulkan 中的句柄（handle）。

* opencl 中一个 ctx 可以有多个 deivce，每个 device 可以有多个 command queue。

    但是每个 command queue 只能对应到一个 device 上。

* opencl 记录一些感觉可能有用的内置函数

    * `clamp`

        ```c
        gentype clamp(gentype x,
            gentype minval,
            gentype maxval)
        ```

        Returns `fmin(fmax(x, minval), maxval)`.

        钳位函数，返回最小值和最大值定义的区间中的数

    * `gentype degrees(gentype radians)`

        将弧度转化为角度。

        与函数`gentype radians(gentype degrees)`配置使用

    * `mix`

        ```c
        gentype mix(gentype x,
            gentype y, gentype a)
        ```

        Returns the linear blend of x and y implemented as

        x + (y – x) * a

        a must be a value in the range 0.0 … 1.0. If a is not in this range, the return values are undefined.

        线性混合。

    * `gentype step(gentype edge, gentype x)`

        Returns 0.0 if x < edge; otherwise it returns 1.0. The step function can be used to create a discontinuous jump at an arbitrary point.

        阶跃函数。

    * `gentype sign(gentype x)`

        Returns 1.0 if x > 0, -0.0 if x = -0.0, +0.0 if x = +0.0, or -1.0 if x < 0. Returns 0.0 if x is a NaN.

        返回`x`的符号。

    * `float4 cross(float4 p0, float4 p1)`

        Returns the cross-product of p0.xyz and p1.xyz. The w compo- nent of a 4-component vector result returned will be 0.

        向量叉乘。参数可以同时为`float3`，也可以同时为`float4`，但是不能一个`float3`，另一个`float4`。

    * `float dot(gentypef p0, gentypef p1)`

        点乘。

        这里没说`p0`和`p1`是什么类型。可能任意长度的向量都支持。

    * `float length(gentypef p)`

        Returns the length of vector p, i.e., $\sqrt{p.x^2 + p.y^2 + …}$

        The length is calculated without overflow or extraordinary precision loss due to underflow.

        不清楚这里的 *underflow* 说的是什么意思。

    * `gentypef normalize(gentypef p)`

        Returns a vector in the same direction as p but with a length of 1.

    * `intn isequal(floatn x, floatn y)`

        Returns the component-wise compare of x == y.

    * `intn isless(floatn x, floatn y)`

        Returns the component-wise compare of x < y.

    * `intn isfinite(floatn x)`

        Tests for the finite value of x.

    * `intn isinf(floatn x)`

    * `intn isnan(floatn x)`

    * `intn isnormal(floatn x)`

        Tests for a normal value (i.e., x is neither zero, denormal, infinite, nor NaN).

        检测`x`是否为一个正常数字。

    * `int any(sgentype x)`

        Returns 1 if the most significant bit in any component of x is set; otherwise returns 0.

        不明白这里的 significant bit 是什么意思

    * `select`

        ```c
        entype select(gentype a,
            gentype b,
            sgentype c)
        ```

        For each component of a vector type `result[i] = if MSB of c[i] is set ? b[i] : a[i]`

        这个函数有点像 numpy 的`where()`.

        cached task: 有时间可以做个实验试试。

* opencl 中内置函数的运算有一定的误差，官方文档给出了误差上界

    比如

    `1.0f/x`的误差为`<= 2.5 ulp`

    `cos`的误差为`<= 4 ulp`

    `fabs`的误差为`0 ulp`

    其中`ulp`指的是两个相邻最近的离散值的距离。

* opencl 中的浮点数和整数用的是两套函数

    比如整数的绝对值用的是

    `ugentype abs(gentype x)`

    如果使用`abs(1.0);`，则会在编译时报错。

* opencl 中一些常见的宏

    ```c
    #define CHAR_BIT 8
    #define INT_MAX 2147483647
    #define LONG_MIN (-0x7fffffffffffffffL – 1)
    #define SCHAR_MAX 127
    #define SHRT_MIN (-32767 – 1)
    #define UCHAR_MAX 255
    #define UINT_MAX 0xffffffff
    ```

* opencl build in functions

    * `get_work_dim`

        syntax:

        ```c
        uint get_work_dim()
        ```

    * `get_global_size`

        syntax:

        ```c
        size_t get_global_size(uint dimindx)
        ```

    * `get_global_id`

        ```c
        size_t get_global_id(uint dimindx)
        ```

    * `get_num_groups`

        ```c
        size_t get_num_groups(uint dimindx)
        ```

    * `get_group_id`

        ```c
        size_t get_group_id(uint dimindx)
        ```

    * `get_local_id`

        ```c
        size_t get_local_id(uint dimindx)
        ```
    
    * `get_local_size`

        ```c
        size_t get_local_size(uint dimindx)
        ```
    
    opencl 中的三角函数都是以弧度为单位。

    `gentype acospi(gentype x)`计算的是`acos(x) / PI`。

    比如`acospi(0.5)`，先计算出来弧度为`PI/3`，再把这个数除以`PI`，得到`1/3`。

    * `gentype atan(gentype y_over_x)`

        试了一下，这里的`y`指的是直角三角形的对边，`x`指的是另一条直角边。

        输入的参数是`y / x`。

    * `gentype cbrt(gentype x)`

        求立方根。

        C 的数学库里也有这个函数。之前竟然都不知道。

    * `gentype copysign(gentype x, gentype y)`

        将`x`的符号变成和`y`一样。

    * `gentype fmax(gentype x,gentype y)`

        Returns y if x < y; otherwise it returns x. If one argument is a NaN, fmax() returns the other argument.
        
        If both arguments are NaNs, fmax() returns a NaN.

    * `gentype logb(gentype x)`

        Compute the exponent of x, which is the integral part of logr|x|.

        不知道这里的`r`是什么意思。

    * `gentype rint(gentype x)`

        按四舍五入法将小数转换成整数。

        如果是`x.5`，似乎只会舍入到偶数。比如`0.5`变成`0`，`1.5`变成 2.

    * `gentype rootn(gentype x, intn y)`

        Compute x to the power 1/y.

    * `gentype round(gentype x)`

        严格执行四舍，五入

    * `gentypef half_cos(gentypef x)`

        Compute the cosine of x. x must be in the range `[-2^16… +2^16]`.

        这里的`x`仍是浮点数，不清楚`2^16`这个数字是怎么来的。

        `half`可能指的是半精度。

    常用的常量：

    * `M_E_F`, `M_E`

        Value of e

    * `M_PI_F`, `M_PI`

        Value of pi

    * `M_1_PI_F`, `M_1_PI`

        Value of 1/pi

* 如果一个 kernel 函数内部有`local`声明的变量，那么这个函数不能被其他 kernel 函数调用

    测试了下，好像没什么问题。

    example:

    ```c
    kernel void func_a(global int *output)
    {
        local int a;
        a = 100;
        *output = a;
    }

    kernel void func_b(global int *output)
    {
        func_a(output);
    }
    ```

    这段代码是正常的，在`main.cpp`中，`output`的输出为 100。

* 可以给 opencl kernel 加上下面这些修饰符，帮助编译器优化

    ```c
    __attribute__((work_group_size_hint(X, Y, Z)))

    __attribute__((reqd_work_group_size(X, Y, Z)))

    __attribute__((vec_type_hint(<type>)))
    ```

    第一个是 hint，说明只是提示，不是确定的。

    第二个是 reqd，说明是要求，必须和 host 代码保持一致。（如果不一致会怎么样？）

    第三个的`<type>`通常是`int`，不清楚这个有什么用。

* kernel 函数不能有 private address space 的指针作为参数

* local variable 必须在 function 的最外层（kernel function scope）申请，且不能初始化。

    local address space 主要是用于 work-group 共享的。当 work-group 结束时，这些变量就会被释放掉。

* 指针只能在同一个 address space 下被赋值，跨 address space 的赋值是不合法的

    example:

    ```cpp
    void func_a(global float4 *pointer)
    {
        global float4 *g_pointer = pointer;
        // local float4 *l_pointer = pointer;  // compiling error
        // private float4 *p_pointer = pointer;  // compiling error
    }
    ```

* 如果一个变量是`image2d_t`，那么还可以给参数加上`read_only`或`write_only`修饰，因为当前的 GPU 不允许同时对图片读和写

    这是因为 GPU 对图片数据做了缓存，读图时从缓存中读，但是写图片时不会改变缓存中的内容。

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

`kernels.cl`:

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
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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
    cq = clCreateCommandQueueWithProperties(context, device_id, nullptr, &ret_val);
    // cq = clCreateCommandQueue(context, device_id, 0, &ret_val);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to create command queue\n");
        printf("return value: %d\n", ret_val);
        getchar();
        exit(-1);
    }

    const int ARR_SIZE = 1024;
    int *A = (int*) malloc(ARR_SIZE * sizeof(int));
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

    int *N = (int*) malloc(1 * sizeof(int));
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
    
    ret_val = clEnqueueWriteBuffer(cq, mem_A, CL_TRUE, 0, ARR_SIZE * sizeof(int), A, 0, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to enqueue write buffer A\n");
        printf("return value: %d\n", ret_val);
        getchar();
    }

    ret_val = clEnqueueWriteBuffer(cq, mem_N, CL_TRUE, 0, 1 * sizeof(int), N, 0, NULL, NULL);
    if (ret_val != CL_SUCCESS)
    {
        printf("fail to enqueue write buffer N\n");
        printf("return value %d\n", ret_val);
        getchar();
    }

    FILE *f = fopen("kernels.cl", "r");
    // char buf[1024] = {0};
    char *buf = (char*) malloc(1024 * sizeof(char));  // 必须用 malloc 申请的内存才行，如果用数组会报错
    size_t n_read = 0;
    n_read = fread(buf, sizeof(char), 1024, f);
    fclose(f);
    cl_program program;
    program = clCreateProgramWithSource(context, 1, (const char**)&buf, &n_read, &ret_val);
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

## opencl kernel programming (OpenCL Language)

opencl kernel 里可以执行`printf()`，并且这个函数还是 atomic 的

opencl 的 kernel 函数中，`printf()`只能打印常量字符串，不能打印变长字符串，也不能打印`global char *`的字符串。

```opencl
kernel void copy_str(global char *str)
{
    char local_str[] = "aaaaa";
    constant char *const_str = "const strrrrr";
    printf("%s\n", str);  // invalid
    printf("%s\n", local_str);  // invalid
    printf("%s\n", const_str);  // valid
    printf("hello, world\n");  // valid
}
```

更多资料：<https://man.opencl.org/printfFunction.html>

### vloadn 与 vstoren

`vloadn()`用于从某个内存地址读取一个 vector，`vstoren()`用于将一个 vector 存储到某个内存目标地址中。

这两个函数还挺有用的，尤其是`vloadn()`。opencl 里的 float3 其实是 float4，只不过最后一个数字不参与计算。但是在 host 上如果我们定义了一个`glm::vec3`，或者`float arr[3]`，那可是只有 3 个数字。如果按照 opencl 的标准来，必须把之前写的`vec3`都改成`cl_float3`，这样得改动好多之前写好的代码。那么能不能把数据加载进显存的时候，用`glm::vec3`，而让 opencl 计算的时候，用`float3`？可以的，我们只需要用`vload3()`创建局部`float3`对象就可以了。

Refs:

1. <https://man.opencl.org/vloadn.html>

1. <https://man.opencl.org/vstoren.html>

1. <https://man.opencl.org/vectorDataLoadandStoreFunctions.html>

Syntax:

```c
gentypen vloadn(size_t offset,
                  const gentype *p)

gentypen vloadn(size_t offset,
                  const constant gentype *p)

void vstore n(gentype n data,
              size_t offset,
              gentype *p)
```

Example:

`main.cpp`:

```cpp
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include "../ocl_simple/global_ocl_env.h"
using namespace std;

ostream& operator<<(ostream &cout, cl_float3 &vec)
{
    cout << "[";
    for (int i = 0; i < 3; ++i)
        cout << vec.s[i] << ", ";
    cout << vec.s[3] << "]";
    return cout;
}

int main()
{
    init_global_ocl_env("./kernels.cl", {"test_vstore"});
    cl_float3 vec_1 = {1, 2, 3, 4}, vec_2 = {2, 3, 4, 5};
    add_buf("vec_1", sizeof(cl_float3), 1, &vec_1);
    add_buf("vec_2", sizeof(cl_float3), 1, &vec_2);
    run_kern("test_vstore", {1}, "vec_1", "vec_2");
    read_buf(&vec_1, "vec_1");
    read_buf(&vec_2, "vec_2");
    cout << "vec_1: " << vec_1 << endl;
    cout << "vec_2: " << vec_2 << endl;
    return 0;
}
```

`kernels.cl`:

```c
__kernel void test_vstore(global float *a, global float *b)
{
    float3 vec = vload3(0, a);
    vstore3(vec, 0, b);
}
```

编译：

```bash
g++ -g main.cpp -lOpenCL -o main
```

运行：

```bash
./main
```

输出：

```
opencl device name: gfx1034
vec_1: [1, 2, 3, 4]
vec_2: [1, 2, 3, 5]
[Warning] destroy ocl env
release ocl buffer: vec_2
release ocl buffer: vec_1
```

可以看到，`vec_1`中前 3 个数据被复制到了`vec_2`中。我们知道，`cl_float3`和`cl_float4`是等价的，都是 4 个元素，但是`vload3()`和`vstore3()`都只对前 3 个元素作用。

如果我们使用 offset 参数指定下一个 vector，也会出现步长为 3 的情况：

`main.cpp`:

```cpp
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include "../ocl_simple/global_ocl_env.h"
using namespace std;

ostream& operator<<(ostream &cout, cl_float3 &vec)
{
    cout << "[";
    for (int i = 0; i < 3; ++i)
        cout << vec.s[i] << ", ";
    cout << vec.s[3] << "]";
    return cout;
}

int main()
{
    init_global_ocl_env("./kernels.cl", {"test_vstore"});
    cl_float3 vec_1[2] = {1, 2, 3, 4, 5, 6, 7, 8};
    cl_float3 vec_2[2] = {2, 3, 4, 5, 6, 7, 8, 9};
    add_buf("vec_1", sizeof(cl_float3), 2, vec_1);
    add_buf("vec_2", sizeof(cl_float3), 2, vec_2);
    run_kern("test_vstore", {1}, "vec_1", "vec_2");
    read_buf(vec_1, "vec_1");
    read_buf(vec_2, "vec_2");
    cout << "vec_1: " << vec_1[0] << ", " << vec_1[1] << endl;
    cout << "vec_2: " << vec_2[0] << ", " << vec_2[1] << endl;
    return 0;
}
```

`kernels.cl`:

```c
__kernel void test_vstore(global float *a, global float *b)
{
    float3 vec = vload3(1, a);  // offset 设置为 1，下面一行同理
    vstore3(vec, 1, b);
}
```

输出：

```
opencl device name: gfx1034
vec_1: [1, 2, 3, 4], [5, 6, 7, 8]
vec_2: [2, 3, 4, 4], [5, 6, 8, 9]
[Warning] destroy ocl env
release ocl buffer: vec_2
release ocl buffer: vec_1
```

可以看到，`offset`设置为 1 时，`vec_1`中的`[4, 5, 6]`被赋值给了`vec_2`中的`[5, 6, 7]`所在位置，`vload3()`和`vstore3()`的步长都为 3 个元素。

所以只有用`vload4()`和`vstore4()`，才能正确作用于`cl_float3`类型。

### vector data types

可以在`int`, `float`等类型后加上一个数字`n`，表示一个向量。比如`float3`，`int4`。目前支持的`n`有`2`, `3`, `4`, `8`, `16`。

For 3-component vector data types, the size of the data type is `4 × sizeof(component)`. This means that a 3-component vector data type will be aligned to a `4 × sizeof(component)` boundary.

向量字面量，以`float4`为例：

```c
(float4)( float, float, float, float )
(float4)( float2, float, float )
(float4)( float, float2, float )
(float4)( float, float, float2 )
(float4)( float2, float2 )
(float4)( float3, float )
(float4)( float, float3 )
(float4)( float )

float4 f = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
uint4 u = (uint4)(1); // u will be (1, 1, 1, 1)
float4 f = (float4)((float2)(1.0f, 2.0f), (float2)(3.0f, 4.0f));
float4 f = (float4)(1.0f, (float2)(2.0f, 3.0f), 4.0f);
```

可以用`xyzw`表示 4 个元素及以下的向量的各个分量：

```c
float4 c;
c.xyzw = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
c.z = 1.0f;
c.xy = (float2)(3.0f, 4.0f);
c.xyz = (float3)(3.0f, 4.0f, 5.0f);

float4 pos = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
float4 swiz = pos.wzyx; // swiz = (4.0f, 3.0f, 2.0f, 1.0f)
float4 dup = pox.xxyy; // dup = (1.0f, 1.0f, 2.0f, 2.0f)
```

还可以使用`x` + 十六进制数字进行索引分量：

```c
float4 f;
float4 v_A = f.xs123;  // is illegal
float4 v_B = f.s012w;  // is illegal
```

还可以使用`lo`和`hi`拿到低位分量和高位分量，`odd`，`even`拿到索引为奇数和偶数的分量：

```c
float4 vf;
float2 low = vf.lo;  // returns vf.xy
float2 high = vf.hi;  // returns vf.zw
float x = low.lo;  // returns low.x
float y = low.hi;  // returns low.y
float2 odd = vf.odd;  // returns vf.yw
float2 even = vf.even; // returns vf.xz
```

如果是`float3`，那么按照`float4`处理，并保持`w`分量未定义。

Example，验证`lo`和`hi`：

```cpp
#include "../ocl_simple/global_ocl_env.h"

template<typename T, size_t arr_elm_num>
ostream& operator<<(ostream &cout, array<T, arr_elm_num> &arr)
{
    cout << "[";
    for (int i = 0; i < arr.size() - 1; ++i) {
        cout << arr[i] << ", ";
    }
    cout << arr[arr.size() - 1] << "]";
    return cout;
}

int main()
{
    init_global_ocl_env("./kernels.cl", {"low_to_high_4", "low_to_high_8"});

    array<float, 4> vec_1{0, 1, 2, 3};
    add_buf("vec_1", sizeof(cl_float4), 1, &vec_1);
    run_kern("low_to_high_4", {1}, "vec_1");
    read_buf(&vec_1, "vec_1");
    cout << vec_1 << endl;

    array<float, 8> vec_2{0, 1, 2, 3, 4, 5, 6, 7};
    add_buf("vec_2", sizeof(cl_float8), 1, &vec_2);
    run_kern("low_to_high_8", {1}, "vec_2");
    read_buf(&vec_2, "vec_2");
    cout << vec_2 << endl;

    return 0;
}
```

`kernels.cl`:

```c
kernel void low_to_high_4(global float4 *vec)
{
    (*vec).hi = (*vec).lo;
}

kernel void low_to_high_8(global float8 *vec)
{
    (*vec).hi = (*vec).lo;
}
```

输出：

```
opencl device name: gfx1034
[0, 1, 0, 1]
[0, 1, 2, 3, 0, 1, 2, 3]
[Warning] destroy ocl env
release ocl buffer: vec_2
release ocl buffer: vec_1
```

使用 xyza 索引和 s数字 索引：

`main.cpp`:

```cpp
int main()
{
    init_global_ocl_env("./kernels.cl", {"assign"});
    cl_float3 vec = {0, 1, 2};
    cl_float3 vec_2 = {2, 1, 0};
    add_buf("vec", sizeof(cl_float3), 1, &vec);
    add_buf("vec_2", sizeof(cl_float3), 1, &vec_2);
    run_kern("assign", {1}, "vec", "vec_2");
    read_buf(&vec, "vec");
    read_buf(&vec_2, "vec_2");
    cout << vec << endl;
    cout << vec_2 << endl;
    return 0;
}
```

`kernels.cl`:

```cpp
kernel void assign(global float3 *vec_1, global float3 *vec_2)
{
    (*vec_1).s1 = (*vec_1).s0;
    vec_1->s2 = vec_1->s0;

    (*vec_2).y = (*vec_2).x;
    vec_2->z = vec_2->x;
}
```

输出：

```
opencl device name: gfx1034
[0, 0, 0, 0]
[2, 2, 2, 0]
[Warning] destroy ocl env
release ocl buffer: vec_2
release ocl buffer: vec
```

（没想到指针也可以使用便捷索引）

显式类型转换的语法：

```cpp
destType convert_destType<_sat><_roundingMode> (sourceType)
destType convert_destTypen<_sat><_roundingMode> (sourceTypen)
```

These provide a full set of type conversions for the following scalar types: char, uchar, short, ushort, int, uint, long, ulong, float, double, half, and the built-in vector types derived therefrom.

（`half`和`double`类型的转换需要机器支持这个类型才可以）

Example:

`main.cpp`:

```cpp
int main()
{
    init_global_ocl_env("./kernels.cl", {"convert_float_to_int"});
    float val[3] = {1.3, 1.5, 1.7};
    int dst_val[3];
    add_buf("val_0", sizeof(float), 1, &val[0]);
    add_buf("val_1", sizeof(float), 1, &val[1]);
    add_buf("val_2", sizeof(float), 1, &val[2]);
    add_buf("dst_val_0", sizeof(int), 1, &dst_val[0]);
    add_buf("dst_val_1", sizeof(int), 1, &dst_val[1]);
    add_buf("dst_val_2", sizeof(int), 1, &dst_val[2]);
    run_kern("convert_float_to_int", {1}, "dst_val_0", "val_0");
    run_kern("convert_float_to_int", {1}, "dst_val_1", "val_1");
    run_kern("convert_float_to_int", {1}, "dst_val_2", "val_2");
    read_buf(&dst_val[0], "dst_val_0");
    read_buf(&dst_val[1], "dst_val_1");
    read_buf(&dst_val[2], "dst_val_2");
    cout << dst_val[0] << endl;
    cout << dst_val[1] << endl;
    cout << dst_val[2] << endl;
    return 0;
}
```

`kernels.cl`:

```c
kernel void convert_float_to_int(global int *dst_val, global float *src_val)
{
    *dst_val = convert_int(*src_val);
}
```

输出：

```
1
1
1
```

如果将 kernel 改成

```c
kernel void convert_float_to_int(global int *dst_val, global float *src_val)
{
    *dst_val = convert_int_rte(*src_val);
}
```

那么效果如下：

```
1
2
2
```

其他支持的模式：

| Rounding Mode Modifier | Rounding Mode Description |
| - | - |
| `_rte` | Round to nearest even. |
| `_rtz` | Round toward zero. |
| `_rtp` | Round toward positive infinity. |
| `_rtn` | Round toward negative infinity. |
| No modifier specified | Use the default rounding mode for this destination type: _rtz for conversion to integers or _rte for conversion to floating-point types. |

没太看懂`_sat`是干嘛用的，好像不是很重要。

可以使用`as_type`和`as_typen`进行内存类型转换：

```cpp
int main()
{
    init_global_ocl_env("./kernels.cl", {"reinterpret_float_as_int"});
    float src = 1.3;
    int dst = -1;
    add_buf("src", sizeof(float), 1, &src);
    add_buf("dst", sizeof(int), 1, &dst);
    run_kern("reinterpret_float_as_int", {1}, "dst", "src");
    read_buf(&dst, "dst");
    cout << dst << endl;
    return 0;
}
```

`kernels.cl`:

```c
kernel void reinterpret_float_as_int(global int *dst, global float *src)
{
    *dst = as_int(*src);
}
```

输出：

```
opencl device name: gfx1034
1067869798
[Warning] destroy ocl env
release ocl buffer: dst
release ocl buffer: src
```

可以看到，`as_type`是对内存的重新解释，并不是数值类型转换。

### qualifiers

#### function qualifiers

**Function Qualifiers**

`kernel` is used to specify that a function in the program source is a kernel function.

The following rules apply to kernel functions:

* The return type must be void. If the return type is not void, it will result in a compilation error.

* The function can be executed on a device by enqueuing a command to execute the kernel from the host.

* The function behaves as a regular function if it is called from a kernel function. The only restriction is that a kernel function with variables declared inside the function with the local qualifier cannot be called from another kernel function.

**Kernel Attribute Qualifiers**

`__attribute__` is used to declare the following additional information about the kernel:

* `__attribute__((work_group_size_hint(X, Y, Z)))` is a hint to the compiler and is intended to specify the work-group size that will most likely be used, that is, the value specified in the `local_work_size` argument to `clEnqueueNDRangeKernel`.

    如果有没有用到的 dim，那么让没有用到的 dim 填 1。

    global worker size 最终会划分到 local worker size 上。如果 local worder size 设置为空，那么 local worker size 等于 global worker size。

    或许只有进一步弄明白 global, group, local 代表的意义，才能明白这个 attribute 是怎么做优化的。

* `__attribute__((reqd_work_group_size(X, Y, Z)))` is intended to specify the work-group size that will be used, that is, the value specified in the `local_work_size` argument to clEnqueueN- DRangeKernel. This provides an opportunity for the compiler to perform specific optimizations that depend on knowing what the work-group size is.

* `__attribute__((vec_type_hint(<type>)))` is a hint to the compiler on the computational width of the kernel, that is, the size of the data type the kernel is operating on. This serves as a hint to an auto-vectorizing compiler. The default value of `<type>` is `int`, `indi-` cating that the kernel is scalar in nature and the auto-vectorizer can therefore vectorize the code across the SIMD lanes of the vector unit for multiple work-items.

指定这些修饰符的话，具体能快多少？

**Address Space Qualifiers**

The type qualifier can be `global` (or `__global`), `local` (or `__local`), `constant` (or `__constant`), or `private` (or `__private`).

`global`是全局的显存，`local`是 work group 共享的显存，`private`是 work item 独有的显存。work iterm 访问 work group 的显存比访问全局的显存要快一些。所以一种可能的加速做法是，对于频繁访问的地址，将 global 的显存复制到 local 里，然后再让 work iterm 多次访问 local memory。

凡是没有指定地址修饰符的，默认都是`private`类型。

* `global`

    The global address qualifier should not be used for image types.

    Pointers to the global address space are allowed as arguments to functions
    (including kernel functions) and variables declared inside functions. Vari-
    ables declared inside a function cannot be allocated in the global address
    space.

    examples:

    ```cpp
    void my_func(global float4 *vA, global float4 *vB)
    {
        global float4 *p;  // legal
        global float4 a;  // illegal
    }
    ```

    其中`global float4 *p;`指的是`p`指向一个全局的`float4`对象。

### C++ support

## opencl c++ binding

opencl 的 c++ binding，相当于是 opencl c 接口的官方 c++ wrapper.

<https://github.khronos.org/OpenCL-CLHPP/namespaces.html>

## C++ grammar for OpenCL kernels

opencl 的 kernel 语言支持 c++ 了，看起来是以 c++17 为标准，舍弃了部分特性，添加了部分特性.

<https://www.khronos.org/opencl/assets/CXX_for_OpenCL.html>

有时间了看下，感觉我应该用不到。

## Examples

### vector add

## ocl simple

### vector add

`kernels.cl`:

```cl
kernel void vec_add(global float *A, global float *B, global float *C)
{
    size_t gid = get_global_id(0);
    C[gid] = A[gid] + B[gid];
}
```

`main.cpp`:

```cpp
#include "../ocl_simple/simple_ocl.hpp"

int main()
{
    init_ocl_env("./kernels.cl", {"vec_add"});
    int vec_len = 4;
    float *A = (float*) add_buf_mem("A", sizeof(float), vec_len);
    float *B = (float*) add_buf_mem("B", sizeof(float), vec_len);
    float *C = (float*) add_buf_mem("C", sizeof(float), vec_len);
    for (int i = 0; i < vec_len; ++i)
    {
        A[i] = random() % 10;
        B[i] = random() % 10;
    }
    sync_cpu_to_gpu({"A", "B"});
    run_kern("vec_add", {(size_t) vec_len}, "A", "B", "C");
    sync_gpu_to_cpu({"C"});
    for (int i = 0; i < vec_len; ++i)
    {
        printf("%.1f + %.1f = %.1f\n", A[i], B[i], C[i]);
    }
    return 0;
}
```

`Makefile`:

```makefile
main: main.cpp
	g++ -g main.cpp -lOpenCL -o main

clean:
	rm -f main
```

compile: `make`

run: `./main`

output:

```
opencl device name: pthread-13th Gen Intel(R) Core(TM) i7-1360P
9.0 + 9.0 = 18.0
5.0 + 6.0 = 11.0
0.0 + 8.0 = 8.0
2.0 + 9.0 = 11.0
[Warning] destroy ocl env
release mem: B
release mem: C
release mem: A
release ocl buffer: B
release ocl buffer: C
release ocl buffer: A
```

说明：

* 使用 cpu 跑 opencl 时，需要把`simple_ocl.hpp`中的`clGetDeviceIDs()`函数的第二个参数改为`CL_DEVICE_TYPE_ALL`

## Problems shooting

* `Memory access fault by GPU node-1 (Agent handle: 0x55555670bd80) on address 0x7fffe0e00000. Reason: Page not present or supervisor privilege.`

    原因是 buffer 的 read 次数和 write 次数不匹配，多次循环后，导致 buffer 溢出。

* 向量类型转换 vector type casting 

    ref: <https://blog.csdn.net/10km/article/details/51171911>

* copy global data to private space

    用的是`async_work_group_copy()`：

    <https://stackoverflow.com/questions/45575072/opencl-copy-character-from-global-to-local-memory>

    对于 struct 之类的对象，直接在 private space 创建一个新对象就可以了。对于`char*`字符串，确实没想过这个问题。