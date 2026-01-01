# Thrust Note

## cache

* thrust

    Thrust 是 CUDA 的 C++ 模板库，提供了类似 STL 的高级接口，简化了 GPU 编程。

    * 安装

        Thrust 是 CUDA Toolkit 的一部分，无需单独安装。

        如果确实需要仅安装 Thrust（独立版本），nv 也提供了纯头文件库：

        ```bash
        git clone https://github.com/NVIDIA/thrust.git
        ```

        doc: <https://nvidia.github.io/cccl/thrust/>

        注：

        1. thrust 目前已经被迁移到<https://github.com/nvidia/cccl>中。

    * 基本概念

        * 容器：`thrust::device_vector<T>`（GPU内存）、`thrust::host_vector<T>`（CPU内存）

        * 算法：并行算法（排序、归约、变换等）

        * 执行策略：`thrust::seq`（CPU串行）、`thrust::device`（GPU并行）

    * 核心用法示例

        * 基本容器操作

            ```cpp
            #include <thrust/device_vector.h>
            #include <thrust/host_vector.h>

            // 主机到设备的数据传输
            thrust::host_vector<int> h_vec = {5, 3, 1, 4, 2};
            thrust::device_vector<int> d_vec = h_vec;  // 复制到GPU

            // 设备到主机
            thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
            ```

        * 常用算法

            排序：

            ```cpp
            thrust::device_vector<int> d_vec = {5, 3, 1, 4, 2};
            thrust::sort(d_vec.begin(), d_vec.end());  // 升序排序
            thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());  // 降序
            ```

            归约操作：

            ```cpp
            int sum = thrust::reduce(d_vec.begin(), d_vec.end());  // 求和
            int max_val = thrust::reduce(d_vec.begin(), d_vec.end(), 
                                         0, thrust::maximum<int>());  // 最大值
            ```

            变换操作：

            ```cpp
            // 对每个元素执行操作
            thrust::device_vector<int> input(100, 2);
            thrust::device_vector<int> output(100);
            thrust::transform(input.begin(), input.end(), output.begin(),
                              [] __device__ (int x) { return x * x; });  // 平方
            ```

            前缀和（Scan）：

            ```cpp
            thrust::device_vector<int> data = {1, 2, 3, 4, 5};
            thrust::device_vector<int> result(5);
            thrust::inclusive_scan(data.begin(), data.end(), result.begin());
            // result = {1, 3, 6, 10, 15}
            ```

            流压缩：

            ```cpp
            // 过滤满足条件的元素
            thrust::device_vector<int> data = {1, 0, 2, 0, 3};
            thrust::device_vector<int> compressed(5);
            auto end = thrust::copy_if(data.begin(), data.end(), compressed.begin(),
                                       [] __device__ (int x) { return x != 0; });
            // compressed前3个元素为{1, 2, 3}
            ```

    * 迭代器高级用法

        ```cpp
        // 常量迭代器
        thrust::device_ptr<const int> const_ptr = d_vec.data();

        // 计数迭代器
        auto indices = thrust::make_counting_iterator(0);  // 生成序列0,1,2,...

        // 变换迭代器
        auto squared = thrust::make_transform_iterator(
            d_vec.begin(), 
            [] __device__ (int x) { return x * x; }
        );
        ```

    * 执行策略控制

        ```cpp
        // 明确指定执行位置
        thrust::sort(thrust::device, d_vec.begin(), d_vec.end());  // GPU执行
        thrust::sort(thrust::host, h_vec.begin(), h_vec.end());     // CPU执行

        // 自动选择（默认）
        thrust::sort(d_vec.begin(), d_vec.end());  // 根据容器类型自动选择
        ```

    * 实际应用示例

        向量点积：

        ```cpp
        thrust::device_vector<float> A(N, 2.0f);
        thrust::device_vector<float> B(N, 3.0f);

        float dot_product = thrust::inner_product(
            A.begin(), A.end(), B.begin(), 0.0f
        );
        ```

        直方图统计：

        ```cpp
        thrust::device_vector<int> data = {1, 3, 2, 1, 3, 2, 1};
        thrust::device_vector<int> histogram(4);
        thrust::sort(data.begin(), data.end());
        thrust::equal(data.begin(), data.end(), histogram.begin());
        ```

    * 注意事项

        * 头文件：`#include <thrust/*>`，无需链接额外库

        * Lambda表达式：CUDA 7.0+ 支持设备端lambda（需__device__注解）

        * 性能：小数据量可能因内核启动开销而性能不佳

        * 内存管理：避免主机-设备间频繁数据拷贝

        * 与CUDA互操作：

            ```cpp
            int* raw_ptr = thrust::raw_pointer_cast(d_vec.data());
            // 可与原生CUDA内核交互
            ```

    * 优点

        * 高开发效率，代码简洁

        * 与STL相似，学习曲线平缓

        * 自动内存管理

        * 支持复杂并行模式

    Thrust 适合数据并行操作，但对于复杂内核仍需使用 CUDA C/C++ 直接编写。

* thrust 独立头文件模式

    如果只有 Thrust 头文件（无 CUDA），Thrust 会回退到使用 CPU：

    ```cpp
    // 无需 CUDA 环境也能编译运行
    #define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
    #include <thrust/version.h>
    #include <thrust/sort.h>
    #include <thrust/host_vector.h>
    #include <iostream>

    int main() {
        std::cout << "Thrust v" << THRUST_VERSION / 100000 << "."
                  << (THRUST_VERSION % 100000) / 100 << "."
                  << THRUST_VERSION % 100 << std::endl;
        
        thrust::host_vector<int> h_vec = {5, 3, 1, 4, 2};
        thrust::sort(h_vec.begin(), h_vec.end());
        
        for (auto x : h_vec) std::cout << x << " ";
        return 0;
    }
    ```

    用 g++ 编译：

    ```bash
    g++ -std=c++14 -I/path/to/thrust test_cpu.cpp -o test_cpu
    ```

* thrust  常用 API 导航

    ```cpp
    算法 (Algorithms):
      - 排序: thrust::sort, thrust::stable_sort
      - 归约: thrust::reduce, thrust::transform_reduce
      - 扫描: thrust::inclusive_scan, thrust::exclusive_scan
      - 变换: thrust::transform, thrust::for_each
      - 搜索: thrust::binary_search, thrust::lower_bound

    容器 (Containers):
      - thrust::host_vector
      - thrust::device_vector
      - thrust::universal_vector

    迭代器 (Iterators):
      - thrust::constant_iterator
      - thrust::counting_iterator
      - thrust::transform_iterator
      - thrust::zip_iterator

    执行策略 (Execution Policies):
      - thrust::seq
      - thrust::device
      - thrust::host
    ```

    常用头文件映射:

    ```cpp
    基础算法: <thrust/algorithm.h>
    排序:     <thrust/sort.h>
    归约:     <thrust/reduce.h>
    变换:     <thrust/transform.h>
    扫描:     <thrust/scan.h>
    迭代器:   <thrust/iterator.h>
    设备管理: <thrust/device_ptr.h>
    ```

## note
