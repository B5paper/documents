# TensorRT Note

## cache

* 简述 tensorrt 的用法

    TensorRT是NVIDIA推出的高性能深度学习推理优化器，以下是其核心用法简述：

    核心流程

    * 模型转换：将训练好的模型（PyTorch/TensorFlow等）转换为TensorRT优化格式

    * 推理加速：使用转换后的引擎进行高效推理

    主要使用方式

    1. 通过ONNX转换（推荐）

        ```python
        import tensorrt as trt

        # 创建logger和builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # 构建网络
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        # 解析ONNX模型
        with open("model.onnx", "rb") as f:
            parser.parse(f.read())

        # 构建配置
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        # 构建引擎
        engine = builder.build_engine(network, config)

        # 序列化保存
        with open("model.engine", "wb") as f:
            f.write(engine.serialize())
        ```

    2. 直接使用PyTorch（torch2trt）

        ```python
        from torch2trt import torch2trt

        # 转换PyTorch模型
        model_trt = torch2trt(model, [dummy_input], 
                              fp16_mode=True,  # FP16量化
                              max_workspace_size=1<<30)

        # 保存和加载
        torch.save(model_trt.state_dict(), 'model_trt.pth')
        ```

    3. Triton Inference Server部署

        ```bash
        # 配置模型仓库
        model_repository/
        └── your_model
            ├── config.pbtxt  # 配置文件
            └── 1
                └── model.plan  # TensorRT引擎
        ```

    关键优化技术

    * 层融合：合并连续操作减少内存访问

    * 精度校准：FP16/INT8量化加速

    * 内核自动调优：选择最优实现

    * 动态张量内存：减少内存分配开销

    典型工作流

    ```text
    训练模型 → 导出ONNX → TensorRT优化 → 部署推理
             (PyTorch/TF)    (构建引擎)   (TensorRT Runtime)
    ```

    实用工具

    * trtexec: 命令行工具，快速测试和基准测试

    * polygraphy: 调试和验证工具

    * NSight Systems: 性能分析

    注意事项

    * 确保CUDA/cuDNN版本兼容

    * INT8量化需要校准数据集

    * 动态维度需要显式配置

    * 不同TensorRT版本间引擎可能不兼容

    TensorRT通过硬件感知优化，通常能实现比原始框架快2-10倍的推理速度，特别适合生产环境部署。

    **给点学习资源，网站、博客、书籍、项目仓库都可以**

    以下是我精选的TensorRT学习资源，从入门到进阶全覆盖：

    官方核心资源

    * 官方文档 🌟

        * TensorRT Developer Guide - 最权威

            <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html>

        * TensorRT API Reference

            <https://docs.nvidia.com/deeplearning/tensorrt/api/index.html>

        * TensorRT Sample Support Guide

            <https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html>

    * GitHub官方仓库

        * TensorRT - 主仓库

            <https://github.com/NVIDIA/TensorRT>

        * TensorRT OSS - 开源组件

            <https://github.com/NVIDIA/TensorRT>

        * torch2trt - PyTorch转换工具

            <https://github.com/NVIDIA-AI-IOT/torch2trt>

        * onnx-tensorrt - ONNX解析器

            <https://github.com/onnx/onnx-tensorrt>

    中文优质资源

    * 博客教程

        * TensorRT详细入门指南 - 知乎专栏

            <https://zhuanlan.zhihu.com/p/487905343>

        * 从零开始学TensorRT - CSDN系列

            <https://blog.csdn.net/qq_40672115/category_11667879.html>

        * TensorRT部署实战 - 极市平台

            <https://www.cvmart.net/topics/175>

    * 视频课程

        * B站：TensorRT部署实战教程

            <https://www.bilibili.com/video/BV1QY4y1W7hX>

        * B站：NVIDIA官方TensorRT教程

            <https://www.bilibili.com/video/BV1GV4y1q7M9>

    实战项目仓库

    * 综合示例

        * TensorRT-For-YOLO-Series - YOLO系列部署

            <https://github.com/Linaom1214/TensorRT-For-YOLO-Series>

        * TensorRTx - 各类模型TensorRT实现

            <https://github.com/wang-xinyu/tensorrtx>

        * trt-samples-for-hackathon-cn - 中文黑客松示例

            <https://github.com/NVIDIA/trt-samples-for-hackathon-cn>

    * 部署框架

        * TensorRT-Pro - 封装完善的部署框架

            <https://github.com/shouxieai/tensorRT_Pro>

        * MMDeploy - OpenMMLab官方部署工具

            <https://github.com/open-mmlab/mmdeploy>

        * FastDeploy - 飞桨多后端部署

            <https://github.com/PaddlePaddle/FastDeploy>

    进阶学习

    * 性能优化

        * TensorRT Best Practices

            <https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html>

        * Performance Tuning Guide

            <https://developer.nvidia.com/blog/optimizing-tensorrt-performance/>

    * 论文与技术解析

        * NVIDIA开发者博客

            <https://developer.nvidia.com/blog/tag/tensorrt/>

        * INT8量化原理详解

            <https://arxiv.org/abs/1712.05877>

    实用工具链

    * 调试分析

        * Polygraphy - 模型调试神器

            <https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy>

        * NSight Systems - 性能分析

            <https://developer.nvidia.com/nsight-systems>

        * trt-pose - 姿态估计示例

            <https://github.com/NVIDIA-AI-IOT/trt_pose>

    * 在线资源

        * NVIDIA NGC Catalog - 预训练模型和容器

            <https://catalog.ngc.nvidia.com/>

        * TensorRT论坛

            <https://forums.developer.nvidia.com/c/accelerated-computing/intelligent-video-analytics/tensorrt/132>

    学习路径建议

    ```text
    第一阶段：基础入门 (1-2周)
      官方文档 + B站视频 + 简单示例

    第二阶段：实战部署 (2-3周)
      TensorRTx项目 + YOLO部署实战

    第三阶段：深度优化 (3-4周)
      性能调优 + INT8量化 + 自定义插件

    第四阶段：生产实践
      Triton部署 + 多模型服务化
    ```

    小贴士

    * 从简单模型开始：先从ResNet、MobileNet等标准模型入手

    * 善用trtexec：快速验证模型是否支持

    * 关注版本兼容：CUDA、cuDNN、TensorRT版本匹配很关键

    * 社区活跃：遇到问题多看GitHub Issues和NVIDIA论坛

    这些资源覆盖了从零基础到进阶的全过程，建议按照学习路径逐步深入，边学边实践效果最好！

## topics
