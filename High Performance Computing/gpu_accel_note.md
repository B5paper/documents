# GPU accel Note

这里存放可以使用 gpu / simd / 多线程加速的工程问题与技巧。

这里也存放 cuda 衍生出来的工具/库的笔记。

## cache

* CUB（CUDA Unbound）

    CUB（CUDA Unbound） 是 NVIDIA CUDA 平台上的一个开源模板库，主要作用是为 CUDA 开发者提供可重用、高性能的并行原语，用于 GPU 编程中的常见数据操作。它位于 CUDA 生态的中间层，介于底层 CUDA Runtime API 和上层算法库（如 Thrust）之间。

    主要作用与特性

    1. 提供并行原语（Parallel Primitives）

        * 排序（Sorting）：支持块内、设备级排序。

        * 规约（Reduction）：高效实现求和、最大值等聚合操作。

        * 扫描（Scan）：前缀和（Prefix Sum）等操作。

        * 直方图（Histogram）

        * 设备内存管理：缓存分配器（CachingAllocator），减少重复内存分配开销。

    2. 分层设计

        块级（Block-level）：线程块（Block）内的协作操作（如块内规约）。

        设备级（Device-level）：跨多个线程块的全局操作（如全局排序）。

        ** warp 级优化**：利用 warp 内线程的隐式同步高效执行。

    3. 性能优势

        * 高度优化：针对不同 GPU 架构（如 Volta、Ampere）进行内核调优。

        * 减少冗余计算：避免开发者重复实现通用操作。

        * 灵活配置：允许通过模板参数调整策略（如算法选择、线程数）。

    4. 轻量与可嵌入

        * 仅头文件（Header-only）：无需单独编译，直接包含头文件即可使用。

        * 与 CUDA 生态兼容：可与 Thrust、cuBLAS 等库结合使用。

    典型应用场景

    * 自定义内核优化：在手动编写的 CUDA 内核中直接调用 CUB 原语。

    * 数据预处理：在深度学习或科学计算中，对 GPU 数据并行处理。

    * 替代手写原语：避免重复实现复杂且易错的并行算法（如并行规约）。

    简单示例

    ```cpp
    #include <cub/cub.cuh>

    // 使用 CUB 的 BlockReduce 进行块内求和
    __global__ void kernel(float* data) {
        typedef cub::BlockReduce<float, 256> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;
        float thread_data = ...; // 每个线程的数据
        float block_sum = BlockReduce(temp).Sum(thread_data);
        // ...
    }
    ```

    与 Thrust 的区别

    | 特性 | CUB | Thrust |
    | - | - | - |
    | 抽象层级 | 中底层（面向线程块/设备） | 高层（类似 STL 的容器/算法） |
    | 使用方式 | 需手动管理内核启动与线程协作 | 隐式并行，更接近 CPU 的 STL |
    | 灵活性 | 更高，可嵌入自定义内核 | 较低，但开发效率高 |
    | 性能控制 | 细粒度调优 | 由库自动选择算法 |

    总结

    CUB 是 CUDA 开发中高性能并行原语的事实标准工具库，尤其适合需要精细控制 GPU 内核优化、避免重复造轮子的场景。它平衡了性能与可维护性，是许多 CUDA 高级库（如 Thrust）的底层基础。

    这个库目前已经归并入 CUDA Core Compute Libraries (CCCL)。

* libcudacxx

    libcudacxx 是 NVIDIA 提供的 CUDA C++ 标准库，它是 C++ 标准库在 GPU 上的实现和扩展，主要作用包括：

    核心作用

    * 为 GPU 提供 C++ 标准库支持

        * 实现部分 C++17/C++20 标准库组件（如 <atomic>、<type_traits>、<memory>），使开发者能在 CUDA 内核中使用熟悉的 C++ 抽象和工具。

    * 异构计算支持

        * 提供 主机（CPU）和设备（GPU）通用代码 的编程模型，同一套代码可同时在 CPU 和 GPU 上编译运行（通过 __host__ __device__ 修饰符）。

    * 高性能并行原语

        * 扩展 GPU 专用组件（如 cuda::barrier、cuda::atomic），优化 GPU 线程协作和内存访问。

    关键特性

    * 与标准 C++ 高度兼容

        例如，可在 GPU 内核中使用 std::atomic 进行线程同步，或使用 std::unique_ptr 管理设备内存。

    * 异构内存管理

        提供 cuda::memory_resource 等工具，支持自定义内存分配策略，统一管理主机与设备内存。

    * 编译时多态支持

        通过 constexpr 和模板元编程，在编译阶段优化 GPU 代码逻辑。

    使用示例

    ```cpp
    #include <cuda/std/atomic>

    __global__ void kernel(cuda::std::atomic<int>* counter) {
        atomic_fetch_add(counter, 1); // GPU 上的原子操作
    }
    ```

    与 Thrust 库的区别

    * Thrust：高层并行算法库（如 sort、reduce），类似 STL + OpenMP。

    * libcudacxx：底层基础库，提供标准 C++ 组件在 GPU 的底层实现，是 Thrust 等高层库的基础。

    应用场景

    * 需要 在 GPU 内核中使用复杂数据结构（如锁、原子操作）。

    * 开发 跨 CPU/GPU 的通用模板库。

    * 为 CUDA 生态库（如 Thrust、CUB）提供底层支持。

    总结

    libcudacxx 填补了 GPU 编程与标准 C++ 生态之间的缺口，让开发者能在 CUDA 内核中编写更现代、可移植的 C++ 代码，同时为高性能异构计算提供底层工具。

* 如何使用 gpu 求解最小二乘法的广义逆矩阵

    使用GPU求解最小二乘问题的广义逆矩阵（或伪逆）可以通过多种方法实现。以下是几种常见的方法和示例代码：

    1. 使用奇异值分解（SVD）

        方法原理

        对于矩阵 A∈Rm×nA∈Rm×n，其伪逆 A+=VΣ+UTA+=VΣ+UT，其中 Σ+Σ+ 是奇异值倒数矩阵。

        PyTorch 实现

        ```python
        import torch
        import time

        # 确保使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        def pinv_svd_gpu(A, rcond=1e-15):
            """
            使用SVD在GPU上计算伪逆
            """
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            
            # 截断小奇异值
            S_inv = torch.zeros_like(S)
            mask = S > rcond * S.max()
            S_inv[mask] = 1.0 / S[mask]
            
            return Vh.T @ torch.diag(S_inv) @ U.T

        # 示例使用
        m, n = 5000, 3000
        A = torch.randn(m, n, device=device)
        b = torch.randn(m, 1, device=device)

        # 预热GPU
        _ = pinv_svd_gpu(A[:100, :100])

        # 计时
        start = time.time()
        A_pinv = pinv_svd_gpu(A)
        x = A_pinv @ b
        torch.cuda.synchronize()
        end = time.time()

        print(f"矩阵大小: {m}×{n}")
        print(f"计算时间: {end-start:.3f}秒")
        print(f"解x的形状: {x.shape}")
        ```

        CuPy 实现（更适合大型矩阵）

        ```python
        import cupy as cp
        import numpy as np

        def pinv_gpu_cupy(A, rcond=1e-15):
            """
            使用CuPy在GPU上计算伪逆
            """
            # 转换到GPU
            A_gpu = cp.asarray(A)
            
            # 使用SVD
            U, S, Vh = cp.linalg.svd(A_gpu, full_matrices=False)
            
            # 计算伪逆
            S_inv = cp.zeros_like(S)
            mask = S > rcond * S.max()
            S_inv[mask] = 1.0 / S[mask]
            
            A_pinv = (Vh.T * S_inv) @ U.T
            
            return cp.asnumpy(A_pinv)  # 转回CPU如果需要

        # 示例
        A = np.random.randn(10000, 5000).astype(np.float32)
        A_pinv = pinv_gpu_cupy(A)
        ```

    2. 使用正规方程法

        对于最小二乘问题 min⁡∥Ax−b∥2min∥Ax−b∥2，解为 x=(ATA)−1ATbx=(ATA)−1ATb

        PyTorch 实现

        ```python
        def solve_normal_equations_gpu(A, b):
            """
            使用正规方程在GPU上求解最小二乘
            """
            ATA = A.T @ A
            ATb = A.T @ b
            
            # 使用Cholesky分解求解
            try:
                L = torch.linalg.cholesky(ATA)
                y = torch.linalg.solve_triangular(L, ATb, upper=False)
                x = torch.linalg.solve_triangular(L.T, y, upper=True)
            except:
                # 如果矩阵不正定，使用LU分解
                x = torch.linalg.solve(ATA, ATb)
            
            return x

        # 使用示例
        m, n = 10000, 2000
        A = torch.randn(m, n, device=device, dtype=torch.float32)
        b = torch.randn(m, 1, device=device, dtype=torch.float32)

        x = solve_normal_equations_gpu(A, b)
        print(f"残差范数: {torch.norm(A @ x - b):.6e}")
        ```

    3. 使用QR分解

        PyTorch 实现

        ```python
        def solve_qr_gpu(A, b):
            """
            使用QR分解在GPU上求解最小二乘
            """
            Q, R = torch.linalg.qr(A)
            x = torch.linalg.solve_triangular(R[:A.shape[1], :], Q.T[:A.shape[1], :] @ b)
            return x

        def pinv_qr_gpu(A):
            """
            使用QR分解计算伪逆
            """
            Q, R = torch.linalg.qr(A)
            R_inv = torch.linalg.inv(R[:A.shape[1], :])
            return R_inv @ Q.T[:A.shape[1], :]
        ```

    4. 直接使用 torch.linalg.pinv

        PyTorch 提供了内置的伪逆函数，支持GPU：

        ```python
        # 简单调用
        A_pinv = torch.linalg.pinv(A)

        # 带条件的伪逆
        A_pinv = torch.linalg.pinv(A, rcond=1e-15, hermitian=False)

        # 求解最小二乘
        x = torch.linalg.lstsq(A, b).solution
        ```

    5. 性能优化建议

        批量处理

        ```python
        def batch_pinv_gpu(A_batch):
            """
            批量计算伪逆
            A_batch: shape (batch_size, m, n)
            """
            batch_size = A_batch.shape[0]
            results = []
            
            for i in range(batch_size):
                U, S, Vh = torch.linalg.svd(A_batch[i], full_matrices=False)
                S_inv = torch.where(S > 1e-10, 1.0/S, torch.zeros_like(S))
                A_pinv = Vh.transpose(-2, -1) @ torch.diag_embed(S_inv) @ U.transpose(-2, -1)
                results.append(A_pinv)
            
            return torch.stack(results)

        # 或使用向量化操作（如果设备内存允许）
        def batched_pinv_vectorized(A_batch):
            U, S, Vh = torch.linalg.svd(A_batch, full_matrices=False)
            S_inv = torch.where(S > 1e-10, 1.0/S, torch.zeros_like(S))
            return Vh.transpose(-2, -1) @ (S_inv.unsqueeze(-1) * U.transpose(-2, -1))
        ```

        混合精度计算

        ```python
        def pinv_mixed_precision(A):
            """
            使用混合精度计算伪逆
            """
            # 转换为半精度计算
            A_half = A.half()
            
            # 执行SVD
            U, S, Vh = torch.linalg.svd(A_half, full_matrices=False)
            
            # 转换回单精度
            U, S, Vh = U.float(), S.float(), Vh.float()
            
            # 计算伪逆
            S_inv = torch.where(S > 1e-10, 1.0/S, torch.zeros_like(S))
            return Vh.T @ torch.diag(S_inv) @ U.T
        ```

    6. 完整示例

        ```python
        import torch
        import time

        class GPULinearSolver:
            def __init__(self, device='cuda'):
                self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
            def solve_pinv(self, A, b, method='svd'):
                """使用伪逆求解 Ax = b"""
                A = A.to(self.device)
                b = b.to(self.device)
                
                if method == 'svd':
                    A_pinv = torch.linalg.pinv(A)
                elif method == 'qr':
                    A_pinv = pinv_qr_gpu(A)
                
                return A_pinv @ b
            
            def solve_direct(self, A, b, method='lstsq'):
                """直接求解最小二乘"""
                A = A.to(self.device)
                b = b.to(self.device)
                
                if method == 'lstsq':
                    return torch.linalg.lstsq(A, b).solution
                elif method == 'normal':
                    return solve_normal_equations_gpu(A, b)
                elif method == 'qr':
                    return solve_qr_gpu(A, b)

        # 使用示例
        solver = GPULinearSolver()

        # 生成数据
        m, n = 10000, 5000
        A = torch.randn(m, n)
        b = torch.randn(m, 1)

        # 比较不同方法
        methods = ['lstsq', 'normal', 'qr']
        for method in methods:
            start = time.time()
            x = solver.solve_direct(A, b, method=method)
            torch.cuda.synchronize()
            end = time.time()
            
            residual = torch.norm(A.to(solver.device) @ x - b.to(solver.device))
            print(f"{method}: 时间={end-start:.3f}s, 残差={residual:.6e}")
        ```

    注意事项

    * 内存管理：GPU内存有限，大矩阵需要分批处理

    * 数值稳定性：SVD方法最稳定但计算量最大

    * 条件数：病态矩阵需要正则化（如Tikhonov正则化）

    * 数据类型：使用float32通常比float64快，但精度较低

    * 错误处理：添加适当的异常处理

    选择哪种方法取决于矩阵大小、条件数、精度要求以及可用硬件资源。对于非常大的矩阵，可能需要使用迭代方法（如共轭梯度法）而不是直接计算伪逆。

## topics
