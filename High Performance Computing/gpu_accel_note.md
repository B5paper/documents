# GPU accel Note

这里存放可以使用 gpu / simd / 多线程加速的工程问题与技巧。

## cache

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
