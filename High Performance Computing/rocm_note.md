# rocm note

## cache

* ROCm 的主要模块

    1. 内核级驱动 (Kernel Driver)

        * AMDGPU 驱动: 这是 Linux 内核的一部分，负责最底层的硬件资源管理，如内存管理、GPU 调度、电源管理、硬件初始化等。它是 ROCm 的基石。

        * HSA Kernel Driver (amdkfd): 负责支持 HSA 的特性，如 GPU 的均匀内存访问。它使得 CPU 和 GPU 能够共享同一个页表，对同一块内存进行访问，这是实现 hipHostMalloc 等零拷贝内存操作的基础。

    2. 用户态运行时和编译器 (User-Space Runtime & Compiler)

        这是 ROCm 栈的核心，开发者直接或间接与之交互的部分。

        * HIP: 这是最重要的入口点。HIP 是 C++ 运行时 API 和内核语言，它允许开发者编写可以同时在 AMD 和 NVIDIA GPU 上运行的代码。其接口设计非常类似 CUDA。对于大多数从 CUDA 转来的开发者，HIP 是最熟悉的地方。

            * hipcc: HIP 的编译器驱动，它背后会调用 Clang/LLVM。

        * ROCclr: 这是一个关键抽象层。它作为 HIP 运行时和不同平台特定后端之间的“通用语言运行时”。HIP 的调用最终会通过 ROCclr 转发到对应的后端，比如在 Linux 上使用 ROCr 运行时。

        * LLVM 编译器后端:

            * AMDGPU 后端: 这是 LLVM 项目的一部分，负责将高级语言生成的 LLVM IR 编译成 AMD GPU 的机器代码。

            * clang: ROCm 修改和扩展了 Clang，使其能够解析和编译 HIP 语言代码。

    3. 运行时和驱动接口 (Runtime & Driver Interface)

        * ROCr (ROCm Runtime): 这是 ROCclr 在 Linux 上使用的底层运行时。它提供了更底层的、与 HSA 标准兼容的 API，用于管理队列、信号、内存等。ROCr 通过 libhsa-runtime64.so 库体现。

        * ROCr Driver (ROCr Driver): 与 ROCr 运行时交互的底层用户态驱动接口。

    4. 数学库和通信库 (Libraries)

        * rocBLAS, rocFFT, rocSPARSE, rocRAND, etc.: 针对 AMD GPU 优化的基础线性代数子程序、快速傅里叶变换、稀疏矩阵、随机数生成等库。它们通常使用 HIP 或汇编编写，以提供极致性能。

        * MIOpen: AMD 的深度学习原语库，提供高度优化的卷积、池化、归一化等操作，是 PyTorch 和 TensorFlow 等框架在 ROCm 上的基石。

        * RCCL: AMD 的集合通信库，类似于 NVIDIA 的 NCCL，用于多卡和多机训练。

    5. 开发工具 (Developer Tools)

        * ROCgdb: AMD GPU 的调试器。

        * ROCprofiler: 性能分析工具，用于收集和分析 GPU 内核执行的性能数据。

        * ROCtracer: 底层 API 调用跟踪工具。

        * Omniperf: 更先进的性能分析工具，可以提供硬件性能计数器的详细数据。

## topics