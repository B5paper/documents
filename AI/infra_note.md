# AI infra note

## cache

* 简述 linalg, mhlo, torchinductor, xla, Cooperative Vector

    这四个均是**机器学习/科学计算领域的核心计算框架/中间表示/编译器组件**，核心定位与作用如下：

    1.  **linalg**
        - 是**MLIR（Multi-Level Intermediate Representation）** 中的一个核心方言（Dialect），专注于**线性代数操作**的抽象与优化。
        - 提供统一的接口描述矩阵乘法、卷积、转置等操作，支持跨硬件（CPU/GPU/TPU）的代码生成与优化。

    2.  **MHLO（MLIR HLO）**
        - 是基于MLIR的**高阶优化（High-Level Optimization）** 方言，由Google主导，是XLA的核心中间表示。
        - 承接TensorFlow HLO的语义，支持复杂的深度学习计算图优化，是连接前端框架与底层硬件编译器的桥梁。

    3.  **TorchInductor**
        - 是**PyTorch**的官方深度学习编译器，作为TorchScript的替代方案，主打**即时编译（JIT）** 与高性能代码生成。
        - 支持将PyTorch模型转换为中间表示，再生成针对CPU/GPU的优化代码（如C++/CUDA），核心目标是提升模型推理与训练速度。

    4.  **XLA（Accelerated Linear Algebra）**
        - 是Google推出的**领域专用线性代数编译器**，最初为TensorFlow设计，现支持PyTorch等多框架。
        - 通过对计算图进行全局优化（如算子融合、常量折叠）和跨设备代码生成，提升深度学习模型在TPU/GPU/CPU上的执行效率。

    是否需要我为你梳理**这四个组件的技术关联与协作流程**，比如 PyTorch 模型如何通过 TorchInductor + XLA 实现高性能部署？

    **Cooperative Vector（CoopVec）** 是面向GPU的**子组/线程协作向量计算抽象**，用于在着色器/光线追踪等SIMT场景高效利用张量核，由NVIDIA在Vulkan与DirectX 12中推出，适配跨厂商硬件。

    **核心定位**

    - 解决传统张量核需**满线程束**与SIMT执行一致性的限制，支持线程束存在非活跃线程时的矩阵-向量运算。
    - 提供跨架构兼容的抽象接口，无需绑定特定CUDA版本，便于跨硬件部署。
    - 把单线程向量数据映射为张量核友好的计算模式，提升稀疏线程/发散执行下的硬件利用率。

    **关键特性**

    - **协作执行**：子组线程共同完成矩阵-向量乘加等操作，可处理部分线程不活跃的情况。
    - **高效映射**：自动适配张量核的矩阵运算能力，避免单线程向量运算导致的硬件浪费。
    - **兼容性**：跨NVIDIA/AMD/Intel等硬件的统一抽象，降低迁移成本。
    - **适用场景**：光线追踪中的MLP推理、神经渲染、神经网络加速等，尤其适合单线程向量运算密集型任务。

    **典型用法**

    - 在Vulkan通过**VK_NV_cooperative_vector**扩展，使用`coopVecMatMulAddNV`等函数实现矩阵-向量乘加并叠加偏置。
    - 在DX12通过CoopVec API，用HLSL指令直接调用硬件矩阵引擎，无需厂商专有接口。

    **性能要点**

    - 满线程束+统一权重时性能最佳；线程束不满或数据发散会有性能损失。
    - 结合Shader Execution Reordering（SER）可提升协作效率与硬件利用率。

    需要我给一个Vulkan中用VK_NV_cooperative_vector做MLP推理的最小可运行代码示例吗？

* 简述什么是 torchscale

    TorchScale是微软开源的**PyTorch库**，核心是高效扩展Transformer，面向基础模型与AGI，聚焦**稳定性、通用性、能力、效率**四大目标，几行代码即可构建和调整模型。

    **核心特性**

    - **稳定性**：DeepNet（含DeepNorm）支持稳定训练**1000层+**Transformer，改进残差与初始化，提升训练鲁棒性。
    - **通用性**：Foundation Transformers（Magneto）支持语言、视觉、语音等**多模态通用建模**，适配BERT、GPT等架构。
    - **能力扩展**：长度外推Transformer，支持更长序列；集成RetNet、LongNet等创新架构，解锁更长上下文（如**10亿token**）。
    - **效率**：X-MoE提供可扩展、可微调的**稀疏混合专家**，显著提升容量且控制计算成本。

    **典型应用**

    - 大语言模型（LLM）高效训练与深度扩展
    - 多模态基础模型开发
    - 长文本/长序列建模（如LongNet）
    - 稀疏专家模型（MoE）快速落地

    **安装与使用**

    ```bash
    pip install torchscale
    ```
    一行安装，快速构建编码器、解码器或序列到序列模型，适配各类任务。

    需要我给你一段快速上手的最小示例代码（比如用TorchScale构建一个基础Transformer编码器并跑通前向计算）吗？

    * 简述如何使用

        TorchScale的核心使用逻辑是**基于其预定义模块快速构建/扩展Transformer类模型**，无需从零实现复杂组件，步骤简洁且可灵活定制。

        **核心使用步骤**

        1. **安装依赖**
           执行 `pip install torchscale` 完成安装，需确保已安装PyTorch。

        2. **选择模型架构**
           根据任务选择三大核心模块：
           - `Encoder`：适用于分类、表示学习等单向任务
           - `Decoder`：适用于生成、自回归任务
           - `EncoderDecoder`：适用于翻译、摘要等序列到序列任务

        3. **配置模型参数**
           关键参数包括模型维度（`d_model`）、头数（`n_head`）、层数（`n_layer`），以及TorchScale特有参数（如`deepnorm`控制是否使用DeepNorm提升稳定性）。

        4. **前向计算与训练**
           实例化模型后，传入张量格式的输入数据（含token_ids、attention_mask等），即可完成前向传播，后续可接入PyTorch常规训练流程。

        **极简示例（Decoder模型）**

        ```python
        import torch
        from torchscale.architecture import Decoder
        from torchscale.architecture.config import DecoderConfig

        # 1. 配置模型参数（启用DeepNorm提升稳定性）
        config = DecoderConfig(
            d_model=512,
            n_head=8,
            n_layer=6,
            deepnorm=True
        )

        # 2. 实例化解码器模型
        model = Decoder(config)

        # 3. 构造输入（token_ids: [batch_size, seq_len], attention_mask: 同形状）
        token_ids = torch.randint(0, 10000, (2, 32))
        attention_mask = torch.ones(2, 32)

        # 4. 前向计算
        output = model(token_ids, attention_mask)
        print(output.shape)  # 输出: [2, 32, 512]
        ```

        需要我给你一个**带训练循环的完整示例**，展示如何用TorchScale训练一个简单的文本生成模型吗？

    * 给我一个带训练循环的完整示例，展示如何用TorchScale训练一个简单的文本生成模型。

        以下是一个**带完整训练循环**的 TorchScale 文本生成模型示例，基于 Decoder 架构，实现简单的自回归文本生成任务。示例中使用**随机模拟数据**（可直接运行），核心流程包括模型定义、数据加载、训练循环、验证与生成推理。

        **完整代码示例**

        ```python
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from torchscale.architecture import Decoder
        from torchscale.architecture.config import DecoderConfig

        # 设备配置：优先使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # ---------------------- 1. 超参数配置 ----------------------
        VOCAB_SIZE = 10000  # 词汇表大小
        D_MODEL = 256       # 模型维度
        N_HEAD = 8          # 注意力头数
        N_LAYER = 4         # Decoder层数
        MAX_SEQ_LEN = 32    # 最大序列长度
        BATCH_SIZE = 16     # 批次大小
        EPOCHS = 10         # 训练轮数
        LEARNING_RATE = 1e-3# 学习率

        # ---------------------- 2. 模拟数据集定义 ----------------------
        class TextGenerationDataset(Dataset):
            """简单的文本生成数据集（自回归任务：输入序列的前n-1个token，预测第n个token）"""
            def __init__(self, vocab_size, max_seq_len, num_samples=1000):
                self.vocab_size = vocab_size
                self.max_seq_len = max_seq_len
                self.num_samples = num_samples
                # 生成随机token序列：[num_samples, max_seq_len]
                self.data = torch.randint(0, vocab_size, (num_samples, max_seq_len))

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                seq = self.data[idx]
                # 自回归任务：input = seq[:-1], target = seq[1:]
                input_ids = seq[:-1]
                target_ids = seq[1:]
                return input_ids, target_ids

        # 构建数据集和数据加载器
        train_dataset = TextGenerationDataset(VOCAB_SIZE, MAX_SEQ_LEN, num_samples=1000)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # ---------------------- 3. 模型定义（Decoder + 输出头） ----------------------
        class TextGenerationModel(nn.Module):
            def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len):
                super().__init__()
                # 1. 配置TorchScale Decoder
                self.config = DecoderConfig(
                    d_model=d_model,
                    n_head=n_head,
                    n_layer=n_layer,
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    deepnorm=True  # 启用DeepNorm提升训练稳定性
                )
                # 2. 实例化Decoder
                self.decoder = Decoder(self.config)
                # 3. 输出头：将模型输出映射到词汇表大小
                self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            def forward(self, input_ids, attention_mask=None):
                """前向传播：input_ids -> decoder -> 词汇表概率分布"""
                # Decoder前向计算（输出形状: [batch_size, seq_len, d_model]）
                decoder_output = self.decoder(input_ids, attention_mask=attention_mask)
                # 映射到词汇表（输出形状: [batch_size, seq_len, vocab_size]）
                logits = self.lm_head(decoder_output)
                return logits

        # 实例化模型并移动到设备
        model = TextGenerationModel(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, MAX_SEQ_LEN).to(device)
        print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

        # ---------------------- 4. 损失函数与优化器 ----------------------
        # 自回归任务：交叉熵损失（忽略padding token，此处模拟数据无padding，可省略ignore_index）
        criterion = nn.CrossEntropyLoss()
        # 优化器：AdamW
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        # 学习率调度器（可选）
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # ---------------------- 5. 训练循环 ----------------------
        def train_one_epoch(model, loader, criterion, optimizer, device):
            model.train()
            total_loss = 0.0
            for batch_idx, (input_ids, target_ids) in enumerate(loader):
                # 数据移动到设备
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                # 构造注意力掩码（此处为全1，因为无padding）
                attention_mask = torch.ones_like(input_ids)

                # 前向计算
                logits = model(input_ids, attention_mask=attention_mask)
                # 损失计算：需要将logits和target_ids展平（CrossEntropyLoss要求输入为[batch*seq_len, vocab_size]）
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    target_ids.reshape(-1)
                )

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累计损失
                total_loss += loss.item()
                # 打印批次信息
                if (batch_idx + 1) % 20 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")

            # 计算平均损失
            avg_loss = total_loss / len(loader)
            return avg_loss

        # 开始训练
        print("\nStarting training...")
        for epoch in range(EPOCHS):
            avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            # 学习率调度
            scheduler.step()
            print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # ---------------------- 6. 自回归生成推理（贪心搜索） ----------------------
        def generate_text(model, start_token, max_len, vocab_size, device):
            """
            贪心搜索生成文本
            :param start_token: 起始token（int）
            :param max_len: 生成的最大长度
            :return: 生成的token序列
            """
            model.eval()
            # 初始化输入序列：[1, 1]（batch_size=1, seq_len=1）
            generated = torch.tensor([[start_token]], dtype=torch.long).to(device)

            with torch.no_grad():
                for _ in range(max_len - 1):
                    # 构造注意力掩码
                    attention_mask = torch.ones_like(generated)
                    # 前向计算
                    logits = model(generated, attention_mask=attention_mask)
                    # 取最后一个token的logits，预测下一个token（贪心选择概率最大的）
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    # 拼接至生成序列
                    generated = torch.cat([generated, next_token], dim=1)
                    # 如果生成到最大长度，停止
                    if generated.shape[1] >= max_len:
                        break

            return generated.squeeze(0).cpu().numpy()

        # 测试生成：从起始token=0开始，生成长度为MAX_SEQ_LEN的序列
        start_token = 0
        generated_seq = generate_text(model, start_token, MAX_SEQ_LEN, VOCAB_SIZE, device)
        print(f"\nGenerated sequence: {generated_seq}")
        ```

        **核心要点说明**

        1. **模型结构**
           - 基于 TorchScale `Decoder` 构建核心模型，搭配 `nn.Linear` 作为语言模型头（`lm_head`）。
           - 启用 `deepnorm=True` 确保深层模型训练稳定，即使层数增加也不易发散。

        2. **任务逻辑**
           - 自回归文本生成：输入序列为 `seq[:-1]`，目标序列为 `seq[1:]`，模型学习“根据前n个token预测第n+1个token”。
           - 损失计算：使用交叉熵损失，需将 `logits` 和 `target_ids` 展平，适配 PyTorch 损失函数要求。

        3. **生成推理**
           - 采用**贪心搜索**：每一步选择概率最大的token拼接至序列末尾。
           - 实际应用中可替换为**束搜索（Beam Search）**或**采样（Sampling）**以提升生成多样性。

        4. **数据说明**
           - 示例使用随机模拟数据，可直接运行；实际应用中需替换为真实文本数据集（如PTB、WikiText），并通过分词器（如BPE、WordPiece）将文本转换为token序列。

        **运行结果说明**

        - 训练过程中会打印每轮的平均损失，损失应逐步下降。
        - 训练结束后会输出一个贪心生成的token序列。

        需要我帮你修改这个示例，**接入真实文本数据集（如WikiText-2）**并整合分词器（如Hugging Face Tokenizers）吗？

* Transformer Lab is a Machine Learning Research Platform designed for frontier AI/ML workflows. Local, on-prem, or in the cloud. Open source.

    <https://lab.cloud/>

* alpaka - Abstraction Library for Parallel Kernel Acceleration

    <https://github.com/alpaka-group/alpaka>

* RAPIDS: GPU Accelerated Data Science

    <https://rapids.ai/>

    <https://github.com/rapidsai>

* byteps: A high performance and generic framework for distributed DNN training 

    <https://github.com/bytedance/byteps>

* 有哪些常用的 ai infra 的编译前端或图优化工具？

    以下是 AI 基础设施（AI Infra）中常用的编译前端和图优化工具，它们主要用于深度学习模型的编译、优化和部署：

    一、编译前端（Compilation Frontends）

    这些工具将深度学习框架（如 PyTorch、TensorFlow）的模型转换为中间表示（IR），以便进行后续优化和编译。

    1. ONNX (Open Neural Network Exchange)

        * 作用：通用的模型交换格式，支持跨框架转换（PyTorch → ONNX → TensorRT 等）。

        * 特点：

            * 支持多种运行时后端（TensorRT、OpenVINO、ONNX Runtime 等）。

            * 提供简单的图优化（常量折叠、算子融合等）。

    2. TorchScript (PyTorch)

        * 作用：将 PyTorch 模型转换为静态图（通过 torch.jit.trace 或 torch.jit.script）。

        * 特点：

            * 支持 PyTorch 模型的序列化和部署。

            * 可与 TorchDynamo（PyTorch 2.0+）结合使用，增强动态图捕获能力。

    3. TensorFlow GraphDef / SavedModel

        * 作用：TensorFlow 的图表示格式，可通过 tf.function 生成静态图。

        * 特点：

            * 支持 TensorFlow 原生优化（Grappler 优化器）。

            * 可导出为 TensorFlow Lite 或 TensorRT 支持的格式。

    4. MLIR (Multi-Level IR)

        * 作用：谷歌推出的多级中间表示框架，支持多种计算图（HLO、Affine、LLVM IR 等）。

        * 特点：

            * 被 TensorFlow、PyTorch（通过 Torch-MLIR）等用作编译基础设施。

            * 支持自定义算子、硬件特定优化。

    5. XLA (Accelerated Linear Algebra)

        * 作用：主要用于 TensorFlow 和 JAX 的编译器，将计算图编译为硬件特定代码。

        * 特点：

            * 支持 JIT 和 AOT 编译。

            * 通过 HLO（High-Level Optimizer）进行图优化。

    6. Apache TVM

        * 作用：端到端深度学习编译器，支持多种前端框架（PyTorch、TensorFlow、ONNX 等）。

        * 特点：

            * 自动调度和优化（AutoTVM、Ansor）。

            * 支持多种硬件后端（CPU、GPU、NPU 等）。

    7. IREE (Intermediate Representation Execution Environment)

        * 作用：基于 MLIR 的编译器，专注于移动端和边缘设备的推理部署。

        * 特点：

            * 支持从 TensorFlow、PyTorch（通过 Torch-MLIR）导入模型。

            * 提供轻量级运行时和 Vulkan/CPU 后端。

    二、图优化工具（Graph Optimization Tools）

    这些工具对计算图进行优化，如算子融合、内存优化、量化等，以提高推理性能。

    1. ONNX Runtime

        * 作用：高性能推理引擎，支持 ONNX 模型的图优化和加速。

        * 优化技术：

            * 算子融合、常量折叠。

            * 量化（静态/动态量化）。

            * 硬件特定优化（CUDA EP、TensorRT EP 等）。

    2. TensorRT (NVIDIA)

        * 作用：针对 NVIDIA GPU 的深度学习推理优化器和运行时。

        * 优化技术：

            * 层融合、精度校准（FP16/INT8）。

            * 内核自动调优、动态张量内存管理。

    3. OpenVINO (Intel)

        * 作用：英特尔推出的推理工具包，支持 CPU、GPU、VPU 等硬件。

        * 优化技术：

            * 图优化（算子融合、冗余消除）。

            * 量化、硬件特定内核优化。

    4. TFLite (TensorFlow Lite)

        * 作用：移动端和边缘设备推理框架。

        * 优化技术：

            * 算子融合、量化（全整型/浮点混合）。

            * 硬件委托（Delegate）机制（如 GPU、Hexagon DSP）。

    5. PyTorch Glow (Facebook)

        * 作用：PyTorch 的深度学习编译器，专注于推理优化。

        * 优化技术：

            * 高级优化（如数据布局转换）。

            * 低级 IR 优化（基于 LLVM）。

    6. nGraph (Intel)

        * 作用：跨框架编译器，支持 TensorFlow、PyTorch 等。

        * 优化技术：

            * 图优化（常量传播、死代码消除）。

            * 后端支持 CPU、GPU、IPU 等。

    7. AITemplate (Meta)

        * 作用：针对 NVIDIA 和 AMD GPU 的高性能推理编译器。

        * 优化技术：

            * 自动生成高性能内核（基于模板）。

            * 支持动态形状和算子融合。

    三、综合工具链

    1. PyTorch 2.x (torch.compile + TorchDynamo)

        * 作用：PyTorch 原生编译工具链，通过动态图捕获和优化提升训练/推理性能。

        * 特点：

            * 支持图级优化（Inductor 编译器）。

            * 与 Triton 集成，自动生成 GPU 内核。

    2. TensorFlow Model Optimization Toolkit

        * 作用：TensorFlow 官方模型优化工具包。

        * 功能：

            * 量化（训练后量化、量化感知训练）。

            * 剪枝、聚类等模型压缩技术。

    四、趋势和选择建议

    * 跨平台部署：考虑 ONNX + ONNX Runtime 或 TVM。

    * NVIDIA GPU：优先使用 TensorRT。

    * 移动端：TFLite（Android）或 Core ML（iOS）。

    * PyTorch 生态：使用 TorchScript 或 PyTorch 2.x torch.compile。

    * 编译器研究：关注 MLIR 和 TVM 等开源编译器框架。

    这些工具可以根据硬件平台、框架偏好和性能需求进行组合使用。实际应用中，通常采用多级优化流水线（如 PyTorch → ONNX → TensorRT）来达到最佳性能。

