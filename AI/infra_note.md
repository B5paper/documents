# AI infra note

## cache

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

