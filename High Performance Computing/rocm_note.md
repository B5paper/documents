# rocm note

## cache

* `ROCm/ROCgdb/gas/testsuite/gas/arm/maverick.c`

    ROCm/ROCgdb/gas/testsuite/gas/arm/maverick.c 是一个自动化测试生成器。它本身不是一个可以直接运行的C程序，而是用来生成大量ARM汇编测试用例的源代码模板。

    它的主要目的是为了测试 GNU汇编器（GAS） 对一款名为 “Maverick” 的特定ARM协处理器的指令支持是否正确。

    GAS： GNU Assembler， 是 GNU Binutils 工具集的一部分，用于将汇编代码（比如 .s 文件）编译成目标文件（.o 文件）。

    这个文件位于 GAS 的测试套件中，专门针对 ARM 架构。

        测试目标： 测试汇编器是否能正确识别、解析并为 ARM Maverick Crunch Coprocessor 生成正确的机器码。

            Maverick 是什么？ 它是 Cirrus Logic（后来被 Intel 收购）为 ARM 处理器设计的一款 DSP/数学协处理器。它有自己的特殊指令集，用于高效处理音频、视频等多媒体数据。虽然这款芯片已经比较老旧且不常见，但 GNU 工具链为了保持兼容性，仍然需要支持它。

        为什么是 C 文件？ 这个 .c 文件并不是用来编译成一个普通的可执行程序。当你构建 Binutils/GAS 时，这个测试套件的一部分构建过程可能会运行这个 C 程序。

        运行结果： 当这个 C 程序被编译并执行时，它的输出是一系列 .s 汇编文件。它会系统地生成 Maverick 协处理器所有可能指令的排列组合。

        “指令集排列组合”的原因：

            ARM 汇编指令的格式通常是：<指令助记符> <操作数1>, <操作数2>, ...

            操作数可以是寄存器（如 mvf0, mvf4）、立即数、或内存地址模式。

            为了全面测试汇编器，需要确保每一条指令与每一种合法的操作数组合都能被正确汇编。手动编写所有这些测试用例极其繁琐且容易出错。

            因此，这个 maverick.c 程序通过循环和逻辑，自动生成所有有效的指令-操作数组合，从而创建出成千上万个独立的测试文件。

    工作流程

        构建阶段： 在编译 ROCm/ROCgdb 或上游的 Binutils 时，构建系统会处理测试套件。

        生成测试用例： 系统会编译并运行 maverick.c 这个“生成器程序”。

        输出： 该程序在 testsuite 的输出目录中生成大量的 .s 文件（例如 maverick_instruction1.s, maverick_instruction2.s, ...）。

        执行测试： GAS 的测试框架（如 run_dump_test）会逐个对这些生成的 .s 文件进行汇编，并检查：

            汇编过程是否成功（不报错）。

            生成的机器码是否正确（与预期的操作码匹配）。

            反汇编结果是否与原始汇编代码一致。

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