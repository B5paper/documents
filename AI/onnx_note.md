# onnx note

## cache

* 简述 ONNX 的用法

    ONNX（Open Neural Network Exchange）是一种开放格式，用于表示机器学习模型，支持框架之间的互操作性。以下是其主要用法简述：

    1. **模型转换**

        - **导出为ONNX**：从深度学习框架（如PyTorch、TensorFlow、MXNet等）将训练好的模型转换为ONNX格式。

            ```python
            # PyTorch示例
            torch.onnx.export(model, input_sample, "model.onnx")
            ```

        - **框架支持**：使用各框架内置工具或第三方库（如`tf2onnx`、`keras2onnx`）进行转换。

    2. **模型优化**

        - **使用ONNX Runtime优化**：通过ONNX Runtime提供的工具简化模型结构、融合运算节点。

        ```bash
        python -m onnxruntime.tools.optimize_model model.onnx -o optimized.onnx
        ```

        - **图形优化**：利用ONNX库的图形优化功能（如常量折叠、冗余节点消除）。

    3. **跨平台推理**

        - **ONNX Runtime推理**：使用ONNX Runtime（高性能推理引擎）在多平台（CPU/GPU/移动设备）运行模型。

            ```python
            import onnxruntime
            session = onnxruntime.InferenceSession("model.onnx")
            results = session.run(["output_name"], {"input_name": input_data})
            ```

        - **其他推理引擎**：支持TensorRT、OpenVINO等引擎加速推理。

    4. **模型可视化与验证**

        - **可视化工具**：使用Netron（图形化工具）查看ONNX模型结构。

        - **API验证**：通过ONNX Python API检查模型格式和一致性。

            ```python
            import onnx
            model = onnx.load("model.onnx")
            onnx.checker.check_model(model)
            ```

    5. **跨框架工作流**

        - **框架互操作**：在多个框架间传递模型（例如，PyTorch训练 → ONNX转换 → TensorRT部署）。

        - **中间表示**：作为中间格式统一不同框架的模型表达。

    主要优势

    - **互操作性**：解决框架锁定问题，便于模型迁移。
    - **高性能推理**：ONNX Runtime针对不同硬件优化。
    - **标准化**：减少重复开发，简化部署流程。

    典型应用场景

    - 将PyTorch/TensorFlow模型部署到生产环境（如服务器、移动端）。
    - 多框架混合开发时转换模型。
    - 利用硬件专用加速器（如NPU）部署模型。

    通过以上步骤，ONNX简化了从训练到部署的流程，提升了模型的可移植性和效率。

## topics
