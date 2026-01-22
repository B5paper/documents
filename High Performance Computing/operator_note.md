# Operator Note

## cache

* 简述 flash attention

    Flash Attention 是一种高效计算注意力机制的算法，由斯坦福大学团队于2022年提出。它通过优化内存访问方式，显著减少了Transformer模型中注意力层的计算时间和内存占用。

    核心问题

    传统注意力计算（如标准的Softmax注意力）需要：

    * 计算所有查询（Q）和键（K）的点积，存储一个庞大的中间矩阵（大小为序列长度×序列长度）。

    * 对矩阵进行Softmax归一化。

    * 与值（V）相乘得到输出。

        这会导致内存复杂度和计算复杂度均为 O(N²)，其中 N 是序列长度。对于长序列（如文档、图像或长视频），内存和计算成本极高。

    **Flash Attention 的核心思想**

    Flash Attention 的关键是避免存储整个注意力矩阵，而是通过以下两种技术优化：

    * 分块计算（Tiling）：将 Q、K、V 分成小块，在 GPU 的 SRAM（高速缓存）中逐步计算注意力，避免将中间结果写入较慢的 GPU HBM（高带宽内存）。

    * 重计算（Recomputation）：在反向传播时，不存储注意力矩阵，而是通过存储的中间状态重新计算，以节省内存。

    主要优点

    * 内存高效：将内存复杂度从 O(N²) 降低到 O(N)，显著减少内存占用。

    * 速度快：通过减少内存读写，提升计算效率，尤其适合长序列任务。

    * 支持长序列：使得 Transformer 模型能够处理数千甚至数万个 token 的序列。

    应用场景

    * 大语言模型（LLM）的长上下文训练与推理。

    * 图像生成模型（如 Stable Diffusion）中的注意力层优化。

    * 科学计算或生物信息学中的长序列建模。

    后续发展

    * Flash Attention 2：进一步优化了算法，减少了非矩阵乘法运算，提升了 GPU 利用率。

    * Flash Attention 3：针对特定硬件（如 NVIDIA H100）做了更深度的优化。

    总结

    Flash Attention 是注意力计算的一项突破性优化，通过硬件感知的算法设计，显著提升了 Transformer 模型的可扩展性和效率，为处理更长序列的模型铺平了道路。

* 详细讲一下 flash attention 算子该怎么写

    详细讲解 Flash Attention 算子的实现是一个复杂的系统工程问题。我将从核心设计思路、关键算法步骤、代码结构框架和实现细节几个层面进行详细拆解。

    核心设计哲学：硬件感知优化

    Flash Attention 不是一个数学上的新注意力公式，而是一个硬件感知的算法重设计。其核心是围绕 GPU 的内存层次结构进行优化：

    * HBM：容量大（几十GB），但速度慢。

    * SRAM：容量小（几百KB到几MB），但速度快。
    算法目标：尽可能在 SRAM 中完成核心计算，减少与 HBM 的数据交换（IO）。

    关键算法：前向传播（FlashAttention V1/V2）

    标准注意力公式：

    O = softmax(QK^T) V，其中 Q, K, V 形状为 [N, d]，中间矩阵 S = QK^T 形状为 [N, N]。

    Flash Attention 将其重写为分块（Tiling）的增量式 Softmax。

    我们维护两个运行统计量：

    * m(x)：当前块为止，每行的最大值（最大值，用于数值稳定）。

    * l(x)：当前块为止，每行的指数和（用于归一化）。

    假设我们将序列长度 N 分成 Tr 块（对 Q）和 Tc 块（对 K, V）。对于 Q 的每一个块 i：

    * 从 HBM 加载 Q_i 到 SRAM。

    * 初始化该块对应的输出累加器 O_i = 0，行最大值 m_i = -inf，行指数和 l_i = 0。

    * 对于 K, V 的每一个块 j：

        a. 从 HBM 加载 K_j, V_j 到 SRAM。

        b. 在 SRAM 中计算小块注意力分数：S_ij = Q_i @ K_j^T。（形状：[block_size, block_size]）

        c. 根据当前的 m_i（旧）和 S_ij 的新最大值，计算校正因子。

        d. 更新运行统计量 m_i 和 l_i。这是一个关键步骤，它融合了新旧块的统计信息。

        e. 重缩放（Rescale）：由于 m_i 更新了，之前累加到 O_i 的“旧权重”和新计算的 P_ij = exp(S_ij - m_i_new) 需要重新调整比例，以确保数值一致性。

        f. 计算 P_ij @ V_j，并合并到 O_i 中。

    * 将最终计算好的 O_i, m_i, l_i 写回 HBM。

        伪代码（前向传播简化版）：

        ```text
        输入: Q, K, V (HBM)
        输出: O (HBM)， l, m (用于反向传播)

        将Q分成Tr块，K,V分成Tc块。
        O = zeros_like(Q)
        l = zeros(N)
        m = -inf * ones(N)

        for i in range(Tr):
            # 加载 Qi
            Qi = load_from_HBM(Q, i)
            Oi = 0
            mi_old = m[i*block_size:(i+1)*block_size]
            li_old = l[i*block_size:(i+1)*block_size]

            for j in range(Tc):
                # 加载 Kj, Vj
                Kj, Vj = load_from_HBM(K, V, j)

                # 计算小块Sij
                Sij = Qi @ Kj.T

                # 1. 计算新的行最大值 mij_new = max(mi_old, rowmax(Sij))
                # 2. 计算新旧统计量的指数差
                #    P_ij_raw = exp(Sij - mij_new)  # 新块的未归一化概率
                #    scale_old = exp(mi_old - mij_new) # 旧累加值的缩放因子
                # 3. 更新行指数和 li_new = scale_old * li_old + rowsum(P_ij_raw)
                # 4. 重缩放输出 Oi = scale_old * Oi + P_ij_raw @ Vj
                # 5. 更新运行统计量 mi_old, li_old = mij_new, li_new

            # 存储最终Oi, mi, li到HBM
            store_to_HBM(O, i, Oi)
            store_to_HBM(m, i, mi_old)
            store_to_HBM(l, i, li_old)
        ```

    **算子实现框架（以CUDA为例）**

    一个高性能的 Flash Attention CUDA 算子实现非常复杂，通常包含以下层次：

    1. 高层次调度 (Kernel Launch)

        ```cpp
        // 确定分块策略，启动内核
        void flash_attention_forward(
            const torch::Tensor& Q,
            const torch::Tensor& K,
            const torch::Tensor& V,
            torch::Tensor& O,
            float dropout_p,
            float softmax_scale) {

            int N = Q.size(0); // 序列长度
            int d = Q.size(1); // 头维度
            int batch_size = Q.size(2); // 批大小
            int num_heads = Q.size(3); // 头数

            // 确定块大小（Block Size）：基于SRAM容量、头维度d等启发式选择
            // 例如，对于 d=64，可能选择 block_size=128
            int block_size_m = 128; // Q的块大小
            int block_size_n = 128; // K的块大小

            // 计算网格（Grid）和线程块（Block）维度
            dim3 grid(num_heads, batch_size, ceil_div(N, block_size_m));
            dim3 block(CUDA_NUM_THREADS);

            // 启动内核
            flash_attention_kernel<<<grid, block>>>(
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                O.data_ptr<float>(),
                N, d,
                block_size_m, block_size_n,
                softmax_scale,
                dropout_p);
        }
        ```

    2. 内核函数 (Kernel Function) - 核心

        内核函数是一个 __global__ 函数，每个线程块负责计算 Q 的一个块（block_size_m 行）的最终输出。

        ```cpp
        template<typename scalar_t>
        __global__ void flash_attention_kernel(
            const scalar_t* __restrict__ Q,
            const scalar_t* __restrict__ K,
            const scalar_t* __restrict__ V,
            scalar_t* __restrict__ O,
            const int N, const int d,
            const int block_size_m, const int block_size_n,
            const float softmax_scale,
            const float dropout_p) {

            // 1. 分配共享内存（Shared Memory，模拟SRAM）
            //    用于存储当前的Q块、K块、V块以及中间Sij块
            extern __shared__ char shared_mem[];
            scalar_t* Qi = (scalar_t*)shared_mem;
            scalar_t* Kj = (scalar_t*)(Qi + block_size_m * d);
            scalar_t* Vj = (scalar_t*)(Kj + block_size_n * d);
            scalar_t* Sij = (scalar_t*)(Vj + block_size_n * d); // 形状 [block_size_m, block_size_n]

            // 2. 线程块ID确定自己负责的Q块索引（i）
            int q_block_idx = blockIdx.z;
            int start_m = q_block_idx * block_size_m;
            int end_m = min(start_m + block_size_m, N);

            // 3. 为当前Q块初始化运行统计量（在线程块的寄存器中或共享内存中）
            float mi[BLOCK_SIZE_M]; // 每线程负责一行，存储行最大值
            float li[BLOCK_SIZE_M]; // 存储行指数和
            scalar_t Oi[BLOCK_SIZE_M][HEAD_DIM]; // 累加的输出

            // 4. 从全局内存（HBM）加载当前Q块到共享内存
            load_q_block(Q, Qi, start_m, end_m, d, ...);

            // 5. 循环遍历K/V的块（j）
            for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                int start_n = k_block_idx * block_size_n;
                int end_n = min(start_n + block_size_n, N);

                // a. 加载Kj, Vj块到共享内存
                load_kv_block(K, V, Kj, Vj, start_n, end_n, d, ...);
                __syncthreads(); // 确保所有线程加载完毕

                // b. 在共享内存中计算Sij = Qi @ Kj^T
                //    使用WMMA（Tensor Core）或循环展开+向量化进行优化
                compute_scores(Sij, Qi, Kj, d, block_size_m, block_size_n, softmax_scale);

                // c. 应用Mask（如果有无注意力）、Dropout（可选）
                // d. **增量式Softmax核心**：
                //     - 计算Sij的局部行最大值
                //     - 更新全局行最大值mi_new = max(mi_old, local_max)
                //     - 计算缩放因子：exp(mi_old - mi_new), exp(Sij - mi_new)
                //     - 更新li_new
                //     - 重缩放Oi: Oi = Oi * exp(mi_old - mi_new) + Pij @ Vj
                //     - 更新mi, li = mi_new, li_new

                __syncthreads(); // 确保下次循环前共享内存数据就绪
            }

            // 6. 将最终计算好的Oi写回全局内存（HBM）
            write_output(O, Oi, start_m, end_m, d, ...);
        }
        ```

    3. 关键子函数优化

        * load_q_block / load_kv_block：使用向量化内存访问（如 float4 或 LDG.128）以最大化内存带宽。

        * compute_scores：

            * 对于 d 较小（<=64），可使用循环展开，每个线程负责计算多个点积。

            * 对于支持 Tensor Core 的 GPU（如 Volta+），使用 WMMA（Warp Matrix Multiply Accumulate）API 进行计算，这是获得极致性能的关键。

        * Softmax 更新：这部分逻辑需要仔细处理数值稳定性，且要尽可能在 Warp 级别进行规约（如使用 warpReduceMax， warpReduceSum）。

    **反向传播实现**

    反向传播遵循同样的分块原则，但更复杂。由于前向传播没有存储完整的 N x N 注意力矩阵 P，反向传播需要：

    * 重新计算：利用存储的 Q, K, V, O, l, m，重新按块计算注意力分数 S_ij 和概率 P_ij。

    * 分块梯度计算：计算 dQ, dK, dV 时，也采用分块策略，将梯度累加到正确的块上。

    核心挑战：反向传播需要访问前向传播中 Q_i 和 K_j 的所有组合，因此同样需要双重循环。但其计算模式是：对于每个 Q_i 的梯度 dQ_i，需要遍历所有 K_j 块；反之，对于每个 K_j 的梯度 dK_j，需要遍历所有 Q_i 块。这被称为“类似外层产品”的访问模式。

    现代实现建议

    除非你是为了深入研究和学习，不建议从零开始编写生产级别的 Flash Attention 算子。原因如下：

    * 极度复杂：需要考虑不同硬件（Ampere, Hopper）、不同数据类型（FP16, BF16, FP8）、不同序列长度和掩码模式。

    * 已有成熟库：

        * 官方实现：FlashAttention <https://github.com/Dao-AILab/flash-attention> 库提供了高度优化的 CUDA 实现，并与 PyTorch 集成。

        * xFormers：Meta 的 xFormers 库也集成了 Flash Attention 及其变体。

        * PyTorch 集成：PyTorch 2.0+ 已将 Flash Attention 作为 torch.nn.functional.scaled_dot_product_attention 的一个后端。

    如果你想学习实现：

    * 从 FlashAttention 官方仓库的代码开始阅读，重点关注 src/flash_attention 目录。

    * 先理解其 CPU 参考实现（Python/NumPy），它清晰地展示了分块和增量 Softmax 的算法逻辑。

    * 再研究 CUDA 内核的实现，重点关注其内存加载、WMMA 使用和 Warp 级规约。

    总结：算子编写要点

    * 以内存IO为中心设计：算法性能瓶颈在于 HBM 访问，而非浮点计算（FLOPS）。

    * 最大化共享内存使用：精心设计共享内存布局，用于缓存输入块和中间结果。

    * 利用硬件特性：如 Tensor Cores（WMMA）、向量化加载/存储、Warp 级操作。

    * 增量更新与重缩放：这是算法正确性的核心，需仔细推导 Softmax 的增量更新公式。

    * 分块与循环重排：将标准的 softmax(QK^T)V 计算，通过分块和双重循环，转化为内存友好的形式。

    实现一个高性能的 Flash Attention 算子是在 CUDA 编程、GPU 架构和数值计算方面的一次终极挑战。建议结合论文、官方代码和详细的 CUDA 文档进行深入学习。

## topics
