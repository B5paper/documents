# Python HPC Note

主要记录使用 python 进行高性能计算的方法。

## cache

* 混合编程（Python + Thrust C++）

    可以使用 PyBind11 或 Cython 包装 Thrust 代码：
        
    ```cpp
    // thrust_module.cpp
    #include <pybind11/pybind11.h>
    #include <thrust/device_vector.h>
    #include <thrust/sort.h>

    namespace py = pybind11;

    py::list thrust_sort(py::list input) {
        // 转换 Python 列表到 thrust::device_vector
        thrust::device_vector<float> d_vec(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            d_vec[i] = input[i].cast<float>();
        }
        
        // 执行排序
        thrust::sort(d_vec.begin(), d_vec.end());
        
        // 转换回 Python 列表
        py::list result;
        for (auto val : d_vec) {
            result.append(val);
        }
        return result;
    }

    PYBIND11_MODULE(thrust_module, m) {
        m.def("sort", &thrust_sort, "Sort array using Thrust");
    }
    ```

    编译并 Python 调用：

    ```bash
    # 编译
    g++ -O3 -shared -std=c++14 -fPIC \
        -I/usr/local/cuda/include \
        -I/path/to/pybind11/include \
        thrust_module.cpp -o thrust_module$(python3-config --extension-suffix) \
        -L/usr/local/cuda/lib64 -lcudart
    ```

    ```python
    import thrust_module
    result = thrust_module.sort([5, 3, 1, 4, 2])
    ```

* Numba + CUDA

    使用装饰器编写 GPU 内核：

    ```python
    from numba import cuda
    import numpy as np

    # 设备函数（类似 __device__ 函数）
    @cuda.jit(device=True)
    def device_square(x):
        return x * x

    # 内核函数（类似 __global__ 函数）
    @cuda.jit
    def square_kernel(x, y):
        idx = cuda.grid(1)
        if idx < x.shape[0]:
            y[idx] = device_square(x[idx])

    # 执行
    x = np.arange(100, dtype=np.float32)
    y = np.empty_like(x)

    # 传输到 GPU
    d_x = cuda.to_device(x)
    d_y = cuda.device_array_like(y)

    # 启动内核
    threads_per_block = 128
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
    square_kernel[blocks_per_grid, threads_per_block](d_x, d_y)

    # 复制回主机
    y = d_y.copy_to_host()
    ```

    安装：

    * 使用 conda（推荐）

        ```bash
        # 创建新环境
        conda create -n numba-cuda python=3.9
        conda activate numba-cuda

        # 安装Numba和CUDA支持
        conda install numba cudatoolkit numpy scipy

        # 或指定CUDA版本
        conda install cudatoolkit=11.2
        ```

    * 使用 pip

        ```bash
        # 基础安装
        pip install numba numpy

        # CUDA支持（需要已安装CUDA Toolkit）
        pip install cudatoolkit  # 或使用系统CUDA

        # 可选：安装cuDNN加速库
        conda install cudnn
        ```

    验证安装

    ```python
    # test_cuda.py
    from numba import cuda
    import numpy as np
    import sys

    print("Python版本:", sys.version)
    print("Numba版本:", numba.__version__)

    # 检查CUDA是否可用
    print("CUDA可用:", cuda.is_available())

    if cuda.is_available():
        print("检测到的GPU数量:", cuda.gpus.len())
        
        # 获取当前设备信息
        with cuda.gpus[0]:
            device = cuda.get_current_device()
            print("设备名称:", device.name)
            print("计算能力:", device.compute_capability)
            print("多处理器数量:", device.MULTIPROCESSOR_COUNT)
            print("最大线程数/块:", device.MAX_THREADS_PER_BLOCK)
            
        # 简单的CUDA测试
        @cuda.jit
        def increment_by_one(an_array):
            idx = cuda.grid(1)
            if idx < an_array.size:
                an_array[idx] += 1
        
        data = np.ones(10)
        d_data = cuda.to_device(data)
        
        threads_per_block = 32
        blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block
        
        increment_by_one[blocks_per_grid, threads_per_block](d_data)
        result = d_data.copy_to_host()
        print("测试结果:", result)
    else:
        print("警告: CUDA不可用，请检查安装")
    ```

    1. 基础装饰器使用

        ```python
        from numba import cuda
        import numpy as np

        # 最简单的GPU加速函数
        @cuda.jit
        def gpu_add(a, b, result):
            idx = cuda.grid(1)  # 获取线程索引
            if idx < len(result):
                result[idx] = a[idx] + b[idx]
        ```

    2. 内核执行配置

        ```python
        # 配置线程和块
        threads_per_block = 256
        blocks_per_grid = (len(data) + threads_per_block - 1) // threads_per_block

        gpu_add[blocks_per_grid, threads_per_block](a, b, result)
        ```

    3. 设备函数

        ```python
        # 设备端函数（只能在内核中调用）
        @cuda.jit(device=True)
        def device_func(x):
            return x * x

        @cuda.jit
        def kernel(arr):
            idx = cuda.grid(1)
            if idx < len(arr):
                arr[idx] = device_func(arr[idx])
        ```

    4. 共享内存使用

        ```python
        @cuda.jit
        def shared_memory_example(arr):
            # 声明共享内存
            shared = cuda.shared.array(shape=256, dtype=np.float32)
            
            tid = cuda.threadIdx.x
            shared[tid] = arr[tid]
            cuda.syncthreads()  # 同步线程
            
            # 使用共享内存进行计算
            arr[tid] = shared[255 - tid]
        ```

    5. 原子操作

        ```python
        @cuda.jit
        def atomic_example(arr, output):
            idx = cuda.grid(1)
            if idx < len(arr):
                cuda.atomic.add(output, 0, arr[idx])  # 原子加操作
        ```

    6. 流和事件

        ```python
        # 使用流进行异步执行
        stream = cuda.stream()

        @cuda.jit
        def kernel_in_stream(data):
            idx = cuda.grid(1)
            # ... 计算 ...

        kernel_in_stream[blocks, threads, stream](data)
        ```

    7. 自动并行化

        ```python
        # 使用@vectorize自动并行化元素级运算
        from numba import vectorize

        @vectorize(['float32(float32, float32)'], target='cuda')
        def gpu_multiply(a, b):
            return a * b

        result = gpu_multiply(array1, array2)  # 自动在GPU上执行
        ```

    8. 常用特性

        ```python
        # 限制寄存器使用
        @cuda.jit(max_registers=32)

        # 内联函数
        @cuda.jit(inline=True)

        # 快速数学运算
        @cuda.jit(fastmath=True)
        ```

    9. 完整示例

        ```python
        import numpy as np
        from numba import cuda
        import time

        @cuda.jit
        def vector_add(a, b, c):
            idx = cuda.grid(1)
            if idx < len(c):
                c[idx] = a[idx] + b[idx]

        # 准备数据
        N = 10_000_000
        a = np.random.rand(N).astype(np.float32)
        b = np.random.rand(N).astype(np.float32)
        c = np.zeros(N, dtype=np.float32)

        # 传输数据到设备
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(c)

        # 执行配置
        threads = 256
        blocks = (N + threads - 1) // threads

        # 执行内核
        vector_add[blocks, threads](d_a, d_b, d_c)

        # 复制结果回主机
        c = d_c.copy_to_host()
        ```

    优势：

    * 简单易用：Python装饰器语法

    * 无需学习CUDA C：纯Python编写

    * 自动内存管理

    * 与NumPy无缝集成

    * 支持共享内存、原子操作等高级特性

    限制：

    * 主要适用于数据并行任务

    * 对复杂控制流支持有限

    * 调试相对困难

    Numba的@cuda.jit为Python开发者提供了快速入门CUDA编程的途径，特别适合科学计算和数据处理场景。

* RAPIDS 生态系统

    ```py
    # cuDF - GPU DataFrame（类似 pandas）
    import cudf
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    sorted_df = df.sort_values('a')

    # cuML - GPU 机器学习（类似 scikit-learn）
    from cuml.ensemble import RandomForestClassifier

    # cuGraph - GPU 图计算
    import cugraph

    # cuSignal - GPU 信号处理
    import cusignal
    ```

* cupy和pycuda都可以调用ElementwiseKernel吗

    不完全正确。让我详细解释一下两者的区别：

    * CuPy 和 ElementwiseKernel

        * CuPy：支持 ElementwiseKernel

            ```python
            import cupy as cp

            # CuPy 的 ElementwiseKernel
            kernel = cp.ElementwiseKernel(
                'float32 x, float32 y',
                'float32 z',
                'z = x + y',
                'add_kernel'
            )

            # 使用
            x = cp.array([1, 2, 3], dtype=cp.float32)
            y = cp.array([4, 5, 6], dtype=cp.float32)
            z = kernel(x, y)
            ```

        * CuPy 还支持 RawKernel：

            ```python
            # CuPy 的 RawKernel（更底层）
            add_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void add(const float* x, const float* y, float* z, int n) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < n) {
                    z[tid] = x[tid] + y[tid];
                }
            }
            ''', 'add')
            ```

    * PyCUDA 和 ElementwiseKernel

        PyCUDA：也有 ElementwiseKernel（但实现不同）

        ```python

        import pycuda.autoinit
        import pycuda.driver as drv
        from pycuda.elementwise import ElementwiseKernel
        import numpy as np

        # PyCUDA 的 ElementwiseKernel
        kernel = ElementwiseKernel(
            "float *x, float *y, float *z",
            "z[i] = x[i] + y[i]",
            "add_kernel"
        )

        # 使用
        x_gpu = drv.to_device(np.array([1, 2, 3], dtype=np.float32))
        y_gpu = drv.to_device(np.array([4, 5, 6], dtype=np.float32))
        z_gpu = drv.mem_alloc(12)  # 分配显存
        kernel(x_gpu, y_gpu, z_gpu)
        ```

    主要区别

    | 特性 | CuPy | PyCUDA |
    | - | - | - |
    | ElementwiseKernel | ✅ 支持 | ✅ 支持 |
    | 语法风格 | 类似 NumPy | 更接近 C |
    | 内存管理 | 自动 | 手动 |
    | 与 NumPy 兼容性 | 高 | 较低 |
    | 安装难度 | 相对简单 | 较复杂 |

    推荐选择

    * CuPy - 如果你：

        * 想要 NumPy 风格的 API

        * 希望更简单的内存管理

        * 需要与其他科学计算库集成

        * 代码可移植性更好

    * PyCUDA - 如果你：

        * 需要更底层的 CUDA 控制

        * 已经熟悉 CUDA C

        * 需要特定的 CUDA 特性

        * 不介意手动内存管理

    总结：两者都支持类似 ElementwiseKernel 的功能，但 CuPy 的 API 设计更接近 NumPy，更适合科学计算场景；PyCUDA 更底层，适合需要精细控制 CUDA 的情况。

* CuPy

    CuPy 是基于 CUDA 的 NumPy 兼容库，可以在 NVIDIA GPU 上实现高性能计算。它与 NumPy 共享大多数 API，学习成本低。

    CuPy 是 NumPy/SciPy 的 GPU 替代，API 高度兼容：

    ```py
    import cupy as cp

    # 类似 thrust::device_vector
    x = cp.array([1, 2, 3, 4, 5])
    y = cp.array([5, 4, 3, 2, 1])

    # 排序（类似 thrust::sort）
    sorted_arr = cp.sort(x)
    print(sorted_arr)

    # 归约（类似 thrust::reduce）
    sum_val = cp.sum(x)
    max_val = cp.max(x)
    print(f"Sum: {sum_val}, Max: {max_val}")

    # 变换（类似 thrust::transform）
    squared = cp.square(x)
    print(squared)

    # 前缀和（类似 thrust::inclusive_scan）
    prefix_sum = cp.cumsum(x)
    print(prefix_sum)

    # 流压缩（类似 thrust::copy_if）
    mask = x > 2
    compressed = x[mask]
    print(compressed)
    ```

    **安装**

    首先安装 cuda toolkit，然后安装 cupy:

    ```bash
    # 根据 CUDA 版本选择（常用）
    pip install cupy-cuda11x  # CUDA 11.x
    pip install cupy-cuda12x  # CUDA 12.x

    # 或使用 conda
    conda install -c conda-forge cupy
    ```

    验证安装

    ```py
    import cupy as cp
    print(cp.__version__)
    print(cp.cuda.runtime.getDeviceCount())  # 显示 GPU 数量
    ```

    二、基本用法

    1. 数组创建（类似 NumPy）

        ```python
        import cupy as cp

        # 创建数组
        x_gpu = cp.array([1, 2, 3, 4, 5])
        x_gpu = cp.arange(10)  # GPU 上的数组
        x_gpu = cp.random.randn(100, 100)  # 随机数组

        # 查看设备
        print(x_gpu.device)  # 显示所在 GPU
        ```

    2. 与 NumPy 互操作

        ```python
        import numpy as np

        # NumPy 到 CuPy
        np_arr = np.ones((5, 5))
        cp_arr = cp.asarray(np_arr)  # 数据复制到 GPU

        # CuPy 到 NumPy
        cp_arr = cp.ones((5, 5))
        np_arr = cp.asnumpy(cp_arr)  # 数据复制回 CPU

        # 原地转换（避免复制）
        with cp.cuda.Device(0):  # 指定 GPU
            cp_arr = cp.array([1, 2, 3])
        ```

    3. 常用计算示例

        ```python
        # 矩阵运算
        a = cp.random.randn(1000, 1000)
        b = cp.random.randn(1000, 1000)
        c = cp.dot(a, b)  # GPU 矩阵乘法

        # 逐元素运算
        d = cp.sin(a) + cp.exp(b)

        # 归约操作
        sum_a = cp.sum(a)
        max_a = cp.max(a)
        mean_a = cp.mean(a)

        # 索引和切片
        slice = a[100:200, 300:400]
        slice[:, :] = 0  # 修改数据
        ```

    4. 自定义核函数

        ```python
        # 使用 Elementwise 核
        square_kernel = cp.ElementwiseKernel(
            'float32 x',  # 输入参数
            'float32 y',  # 输出参数
            'y = x * x',  # 计算表达式
            'square'
        )

        x = cp.arange(10, dtype=cp.float32)
        y = square_kernel(x)  # y = x²

        # 使用 RawKernel（更灵活）
        kernel_code = '''
        extern "C" __global__
        void add(const float* a, const float* b, float* c) {
            int i = threadIdx.x;
            c[i] = a[i] + b[i];
        }
        '''
        add_kernel = cp.RawKernel(kernel_code, 'add')
        ```

    5. 内存管理

        ```python
        # 固定内存（提高传输速度）
        pinned_mem = cp.cuda.alloc_pinned_memory(1024)  # 1KB

        # 流处理（异步操作）
        stream = cp.cuda.Stream()
        with stream:
            a = cp.array([1, 2, 3])
            b = cp.array([4, 5, 6])
            c = a + b
        stream.synchronize()

        # 显存池
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        ```

    三、性能优化技巧

    1. 避免 CPU-GPU 频繁传输

        ```python
        # 不推荐：频繁复制
        for _ in range(100):
            np_arr = np.random.randn(1000, 1000)
            cp_arr = cp.asarray(np_arr)  # 每次循环都复制

        # 推荐：在 GPU 上生成数据
        for _ in range(100):
            cp_arr = cp.random.randn(1000, 1000)  # 直接在 GPU 生成
        ```

    2. 使用融合操作

        ```python
        # 分开计算
        y = cp.exp(x)
        z = y * 2

        # 融合内核（减少内核启动开销）
        @cp.fuse()
        def fused_op(x):
            return cp.exp(x) * 2
        z = fused_op(x)
        ```

    四、常见问题

    1. 版本兼容性

        ```bash
        # 查看 CUDA 版本
        nvidia-smi

        # 安装对应版本
        CUDA 11.8 → cupy-cuda11x
        CUDA 12.2 → cupy-cuda12x
        ```

    2. 显存不足处理

        ```python
        # 清理显存
        cp.get_default_memory_pool().free_all_blocks()

        # 监控显存使用
        mem_pool = cp.get_default_memory_pool()
        print(f"Used: {mem_pool.used_bytes()/1e9:.2f} GB")
        print(f"Free: {mem_pool.free_bytes()/1e9:.2f} GB")
        ```

    3. 多 GPU 操作

        ```python
        # 切换设备
        with cp.cuda.Device(0):  # GPU 0
            a = cp.array([1, 2, 3])

        with cp.cuda.Device(1):  # GPU 1
            b = cp.array([4, 5, 6])
        ```

    五、简单示例：向量加法

        ```python
        import cupy as cp
        import time

        # 大数据量
        n = 10**7
        a = cp.random.randn(n)
        b = cp.random.randn(n)

        # GPU 计算
        start = time.time()
        c = a + b
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        print(f"GPU time: {gpu_time:.4f}s")

        # 比较 CPU 版本
        import numpy as np
        a_cpu = cp.asnumpy(a)
        b_cpu = cp.asnumpy(b)

        start = time.time()
        c_cpu = a_cpu + b_cpu
        cpu_time = time.time() - start
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        ```

    六、资源

    * 官方文档: docs.cupy.dev

    * GitHub: cupy/cupy

    * 示例代码: cupy/examples 目录

    注意：CuPy 与 NumPy 高度兼容，但并非 100%。实际使用时建议先测试关键函数，并注意数据类型（GPU 上的 float32 更常见）。

* PyCUDA（直接调用 CUDA）

    Pycuda 是 Python 的 CUDA 并行计算接口，允许在 NVIDIA GPU 上直接运行 CUDA 内核。

    可以直接调用 CUDA API：

    ```python
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    import numpy as np

    # 创建 GPU 数组
    x = np.random.randn(100).astype(np.float32)
    x_gpu = gpuarray.to_gpu(x)

    # 基本操作
    y_gpu = 2 * x_gpu  # 元素级乘法
    sum_gpu = gpuarray.sum(x_gpu)
    dot_gpu = gpuarray.dot(x_gpu, x_gpu)

    # 自定义内核（类似 Thrust 的 transform）
    from pycuda.elementwise import ElementwiseKernel

    # 类似 thrust::transform 的平方操作
    square_kernel = ElementwiseKernel(
        "float *x, float *y",
        "y[i] = x[i] * x[i]",
        "square_kernel"
    )

    y_gpu = gpuarray.empty_like(x_gpu)
    square_kernel(x_gpu, y_gpu)

    # 归约内核
    from pycuda.reduction import ReductionKernel

    sum_kernel = ReductionKernel(
        np.float32, neutral="0",
        reduce_expr="a+b", map_expr="x[i]",
        arguments="float *x"
    )
    result = sum_kernel(x_gpu).get()
    ```

    1. 安装与导入

        ```python
        import pycuda.autoinit  # 自动初始化 CUDA 上下文
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        import numpy as np
        ```

    2. 基本使用流程

        A. 编写 CUDA 内核

        ```python
        # 直接在 Python 中编写 CUDA C 代码
        kernel_code = """
        __global__ void vector_add(float *a, float *b, float *c, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        ```

        B. 编译与调用内核

        ```python
        # 编译内核
        mod = SourceModule(kernel_code)
        vector_add = mod.get_function("vector_add")

        # 准备数据
        n = 1000
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        # 分配 GPU 内存
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(c.nbytes)

        # 数据传输
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)

        # 执行内核
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        vector_add(a_gpu, b_gpu, c_gpu, np.int32(n),
                   block=(block_size, 1, 1),
                   grid=(grid_size, 1))

        # 获取结果
        cuda.memcpy_dtoh(c, c_gpu)
        ```

    3. 高级用法

        A. 使用 ElementwiseKernel（便捷的内核）

        ```python
        from pycuda.elementwise import ElementwiseKernel

        add_kernel = ElementwiseKernel(
            "float *a, float *b, float *c",
            "c[i] = a[i] + b[i]",
            "vector_add")

        add_kernel(a_gpu, b_gpu, c_gpu)
        ```

        B. 使用 GPUArray（类似 NumPy 的接口）

        ```python
        import pycuda.gpuarray as gpuarray

        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = a_gpu + b_gpu  # 支持 NumPy 风格操作
        ```

        C. 纹理内存使用

        ```python
        # 创建纹理引用
        texref = mod.get_texref("texture_name")
        # 配置纹理参数...
        ```

    4. 性能优化技巧

        ```python
        # 1. 使用共享内存
        shared_kernel = """
        __global__ void shared_mem_example(float *data) {
            extern __shared__ float sdata[];
            int tid = threadIdx.x;
            sdata[tid] = data[tid];
            __syncthreads();
            // ... 处理共享内存数据
        }
        """

        # 2. 流并行
        stream = cuda.Stream()
        # 异步数据传输和执行
        cuda.memcpy_htod_async(a_gpu, a, stream)
        kernel_func(a_gpu, b_gpu, block=(256,1,1), grid=(100,1), stream=stream)
        ```

    5. 内存管理

        ```python
        # 手动分配
        ptr = cuda.mem_alloc(1024)  # 分配 1KB
        cuda.memcpy_htod(ptr, host_data)
        cuda.memcpy_dtoh(host_data, ptr)

        # 使用内存池
        from pycuda.tools import DeviceMemoryPool
        pool = DeviceMemoryPool()
        ptr = pool.allocate(1024)
        # 使用后自动回收
        ```

    注意事项：

    * 确保有 NVIDIA GPU 和 CUDA 工具包

    * Pycuda 是低级接口，需要理解 CUDA 编程模型

    * 注意 CPU-GPU 数据传输开销

    * 适当选择 block 和 grid 尺寸以获得最佳性能

    Pycuda 提供了与 CUDA C 几乎相同的控制能力，同时保持了 Python 的易用性，适合需要精细控制 GPU 计算的高级应用。

## note
