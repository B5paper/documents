# Torch Note

## cache

* 简述 pytorch 中，a.storage().data_ptr() 的作用

    在 PyTorch 中，`a.storage().data_ptr()` 的作用是**返回张量 `a` 的底层数据存储区的内存地址**。

    具体解释如下：

    1. **`a.storage()`**：

       - 返回张量 `a` 的底层 `Storage` 对象

       - PyTorch 的张量由两部分组成：一个元信息头（shape、stride、dtype 等）和一个实际的连续数据存储区（Storage）

       - 多个张量可以共享同一个 Storage（如切片操作、`.view()`、`.detach()` 等）

    2. **`.data_ptr()`**：

       - 返回该 Storage 对象所管理的内存的起始地址（Python 整数）

       - 这是实际存储张量数据的内存位置

    3. **主要用途**：

       - **检查内存共享**：两个张量如果 `storage().data_ptr()` 相同，说明它们共享同一份底层数据

       - **与外部库交互**：将 PyTorch 张量的内存地址传递给 C/CUDA 扩展或其他库（如 NumPy、OpenCV）

       - **调试**：确认内存是否被重新分配，或验证操作是原地执行还是新建内存

    **示例**：

    ```python
    import torch

    a = torch.tensor([1, 2, 3])
    print(a.storage().data_ptr())  # 输出内存地址，如 139823456789012

    b = a[:2]  # 切片操作，共享存储
    print(b.storage().data_ptr())  # 与 a 相同

    c = a.clone()  # 深拷贝，新分配内存
    print(c.storage().data_ptr())  # 与 a 不同
    ```

    **注意**：

    - 在 CPU 上返回的是系统内存地址，CUDA 张量返回的是 GPU 显存地址
    - 0.4.0 版本之前也常用 `.data_ptr()`，现在更推荐通过 `storage()` 访问


* `torch.allclose()`

    torch.allclose 是 PyTorch 中用于判断两个张量是否在给定容忍度范围内接近相等的函数。

    原型

    ```python
    torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)
    ```

    参数说明

    * input (Tensor): 第一个张量

    * other (Tensor): 第二个张量

    * rtol (float): 相对容忍度，默认 1e-05

    * atol (float): 绝对容忍度，默认 1e-08

    * equal_nan (bool): 是否将 NaN 视为相等，默认 False

    作用

    判断两个张量是否满足以下条件：

    ```text
    abs(input - other) ≤ atol + rtol * abs(other)
    ```

    对于所有元素都成立时返回 True，否则返回 False。

    用法示例

    基本使用

    ```python
    import torch

    # 创建两个接近的张量
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.00001, 2.00001, 3.00001])

    # 判断是否接近
    result = torch.allclose(a, b, rtol=1e-4)  # True
    print(result)  # True
    ```

    设置容忍度

    ```python
    x = torch.tensor([100.0, 200.0])
    y = torch.tensor([100.1, 200.1])

    # 绝对容忍度
    result1 = torch.allclose(x, y, atol=0.2)  # True
    # 相对容忍度
    result2 = torch.allclose(x, y, rtol=1e-3)  # True
    ```

    NaN 处理

    ```python
    import torch

    a = torch.tensor([1.0, float('nan'), 3.0])
    b = torch.tensor([1.0, float('nan'), 3.0])

    result1 = torch.allclose(a, b)  # False (默认 equal_nan=False)
    result2 = torch.allclose(a, b, equal_nan=True)  # True
    ```

    与 torch.equal 的区别

    ```python
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.000001, 2.000001])

    print(torch.equal(a, b))      # False (严格相等)
    print(torch.allclose(a, b))   # True (在容忍度范围内)
    ```

    应用场景

    * 数值比较：比较浮点数计算结果

    * 模型测试：验证模型输出是否符合预期

    * 梯度检查：比较梯度值是否接近

    * 单元测试：验证张量运算的正确性

    注意事项

    * 两个张量必须有相同的形状

    * 主要用于浮点数比较，整数类型会比较严格

    * rtol 和 atol 可以同时起作用，满足其中一个条件即可

* pytorch 中 dataloader 在打包 batch 时，如何返回 list 类型，而不是 tensor 类型？

    在PyTorch的DataLoader中，默认情况下会将batch数据转换为Tensor。如果你需要返回list类型而不是Tensor类型，有以下几种方法：

    方法1：使用自定义的collate_fn（推荐）

    ```python
    import torch
    from torch.utils.data import Dataset, DataLoader

    # 自定义collate_fn，直接返回list
    def list_collate_fn(batch):
        # batch是一个包含多个样本的列表，每个样本是__getitem__返回的元组
        # 将每个字段的数据收集到list中
        transposed = zip(*batch)
        return [list(samples) for samples in transposed]

    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # 返回list而不是Tensor
            return [self.data[idx], self.data[idx] * 2]

    # 创建数据集和数据加载器
    data = list(range(10))
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=list_collate_fn)

    # 测试
    for batch in dataloader:
        print(f"Batch type: {type(batch)}, Content: {batch}")
        # batch将是list类型
    ```

    方法2：在collate_fn中更精细地控制

    ```python
    def custom_collate_fn(batch):
        """
        处理多种数据类型，只对某些字段转换为Tensor
        """
        elem = batch[0]
        if isinstance(elem, (list, tuple)):
            # 如果是列表或元组，递归处理每个字段
            return [custom_collate_fn([d[i] for d in batch]) for i in range(len(elem))]
        elif isinstance(elem, dict):
            # 如果是字典
            return {key: custom_collate_fn([d[key] for d in batch]) for key in elem}
        else:
            # 对于数字类型，可以选择转换为Tensor或保持原样
            # return torch.tensor(batch)  # 转换为Tensor
            return list(batch)  # 保持为list

    # 使用示例
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=custom_collate_fn)
    ```

    方法3：在Dataset的__getitem__中处理

    ```python
    class MyDataset2(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # 返回tuple，让collate_fn分别处理
            return self.data[idx], str(self.data[idx])  # 混合数据类型

    def mixed_collate_fn(batch):
        # 将batch解包为不同字段
        field1, field2 = zip(*batch)
        
        # 对field1转换为Tensor，field2保持为list
        return torch.tensor(field1), list(field2)

    # 使用
    dataloader = DataLoader(MyDataset2(data), batch_size=3, 
                           collate_fn=mixed_collate_fn)
    ```

    方法4：最简单的方法 - 返回Python原生类型

    ```python
    # 如果只需要避免自动转换为Tensor，最简单的方法是确保__getitem__返回Python原生类型
    class SimpleDataset(Dataset):
        def __getitem__(self, idx):
            # 返回Python原生类型，而不是Tensor
            return float(self.data[idx])  # 而不是 torch.tensor(self.data[idx])
    ```

    示例：处理图像和标签

    ```python
    from PIL import Image
    import numpy as np

    class ImageDataset(Dataset):
        def __init__(self, image_paths, labels):
            self.image_paths = image_paths
            self.labels = labels
        
        def __getitem__(self, idx):
            # 加载图像但不转换为Tensor
            img = Image.open(self.image_paths[idx])
            img = np.array(img)  # 保持为numpy数组
            label = self.labels[idx]
            
            return img, label

    def image_collate_fn(batch):
        images, labels = zip(*batch)
        
        # 图像保持为list，标签转换为Tensor（可选）
        return list(images), torch.tensor(labels)

    # 使用
    dataloader = DataLoader(dataset, batch_size=32, 
                           collate_fn=image_collate_fn)
    ```

    注意事项：

    * 性能考虑：使用list而不是Tensor可能会降低性能，因为PyTorch的Tensor操作经过了优化

    * GPU加速：如果要在GPU上训练，最终需要转换为Tensor

    * 灵活性：collate_fn提供了最大的灵活性，可以根据需要处理不同类型的数据

    推荐使用方法1或方法2，通过自定义collate_fn函数来控制返回类型。这是最灵活和清晰的方式。

* `tensor.detach()`

    作用

    * 断开计算图：返回一个新张量，与原始张量共享数据但不参与梯度计算

    * 阻止梯度回传：在反向传播时，从该张量开始的运算不会被追踪梯度

    * 内存共享：不复制数据，仅创建新的引用

    典型应用场景

    * GAN训练中分离不同网络的梯度计算

    * 在计算损失时冻结部分参数

    * 将张量转换为numpy数组前

    * 模型推理时减少内存占用

* torch 分级优化参数

    ```py
    import torch.nn as nn
    import torch.optim as optim

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    # 假设我们想对网络不同部分使用不同的学习率
    backbone_params = list(model[0].parameters())  # 第一层参数
    head_params = list(model[2].parameters())      # 第三层参数

    # 不同参数组使用不同的学习率
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': 1e-4},  # 基础层学习率小
        {'params': head_params, 'lr': 1e-3}       # 头部层学习率大
    ])

    # 或者：只优化最后一层
    optimizer = optim.SGD(model[2].parameters(), lr=1e-3)
    ```

* `optim.SGD([train_param], lr=1e-3)`

    我们可以把想优化的数据做成 list，sgd 可以只优化 list 中的对象。

    example:

    Y = A x + b

    我们可以写成`optim.SGD([A[0], A[3]], lr=1e-3)`，这样可以只优化`A[0]`, `A[3]`这两个数字。

    在进行 layer A -> layer B 的反向传播，计算 A 中参数的梯度时，不需要用到 layer B 中参数的**梯度**，但是有可能用到 B 中参数的**值**。

    在整介神经网络中，只要某一层的输出全为恒定值（比如 Y = A x + b 中，A = 0，此时 Y 恒等于 b），那么整个神经网络的输出以及梯度流就会在这里被截断，梯度优化会失效。

    只要神经网絡的每一层输出都至少存在一个值随 input x 的变化而变化，那么梯度优化仍然奏效。

    这意味着 freeze 神经网络中其中某一层（或几层）的梯度（`requires_grad = False`），或者在调用 sgd 时，只传入指定几层 layer 的梯度（`optim.SGD([layer_1_param, layer_3_param])`），整体仍是 work 的。

* 虚拟拼接

    你说得非常对！这正是现代深度学习框架正在探索的优化方向。你描述的这种设计实际上涉及到 "虚拟拼接" 或 "延迟拼接" 的概念。

    理想的指针式 cat 设计

    1. 概念实现（伪代码）

        ```python
        class VirtualCatTensor:
            def __init__(self, tensors, dim):
                self.tensors = tensors  # 只存储引用
                self.dim = dim
                self._shape = self._compute_shape()
            
            def _compute_shape(self):
                # 计算虚拟形状而不实际复制数据
                total_dim = sum(t.shape[self.dim] for t in self.tensors)
                shape = list(self.tensors[0].shape)
                shape[self.dim] = total_dim
                return tuple(shape)
            
            # 惰性计算：只在需要时访问数据
            def __getitem__(self, indices):
                # 智能索引，定位到正确的子张量
                pass
        ```

    2. 实际应用场景

        ```python
        # 传统方式：立即复制
        x = torch.randn(1000, 1000).cuda()  # 200MB
        y = torch.randn(1000, 1000).cuda()  # 200MB
        z = torch.cat([x, y], dim=0)  # 立即分配400MB新内存并复制

        # 虚拟方式：延迟处理
        v = VirtualCat([x, y], dim=0)  # 只存储指针，几乎零开销

        # 需要时再实际计算
        result = v @ weight  # 此时才需要实际数据
        ```

    现实中的实现挑战

    1. 计算图兼容性

        ```python
        # 在自动求导中，指针设计变得复杂
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        z = virtual_cat([x, y], dim=0)  # 反向传播时梯度如何传播？
        ```

    2. 内存访问模式

        ```python
        # 不连续内存访问影响GPU效率
        # 传统cat: [xxxxxxxxxx][yyyyyyyyyy] -> [xxxxxxxxxxyyyyyyyyyy] 连续访问
        # 虚拟cat: [xxxxxxxxxx]...[yyyyyyyyyy] 跳转访问，影响缓存效率

        # GPU更喜欢连续的内存访问模式
        for i in range(n):
            # 连续访问更高效
            output[i] = continuous_data[i] * weight
        ```

    PyTorch 中的现有近似方案

    1. torch.nested

        ```python
        import torch

        # 嵌套张量支持不规则结构
        nt = torch.nested.nested_tensor([
            torch.randn(2, 3),
            torch.randn(3, 3),
            torch.randn(4, 3)
        ])

        # 内部存储为独立数据块
        print(nt.shape)  # torch.Size([3, None, 3])
        ```

    2. 使用稀疏张量

        ```python
        # 稀疏表示可以减少数据移动
        indices = torch.cat([
            torch.arange(0, 1000).unsqueeze(0),
            torch.arange(0, 1000).unsqueeze(0)
        ], dim=0)
        values = torch.cat([x.flatten(), y.flatten()])
        sparse_cat = torch.sparse_coo_tensor(indices, values)
        ```

    3. 自定义操作符

        ```python
        class LazyCatFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *tensors):
                ctx.save_for_backward(*tensors)
                ctx.dim = dim
                # 返回一个包装器而不是实际数据
                return LazyCatWrapper(tensors, dim)
            
            @staticmethod
            def backward(ctx, grad_output):
                # 处理梯度传播
                pass
        ```

    实际可行的折中方案

    1. 批量延迟处理

        ```python
        class BatchCollector:
            def __init__(self, dim=0, buffer_size=1024):
                self.buffers = []  # 存储数据块
                self.dim = dim
                self.buffer_size = buffer_size
            
            def add(self, tensor):
                self.buffers.append(tensor)
                if self.total_size() >= self.buffer_size:
                    return self.flush()  # 达到阈值时实际cat
                return None
            
            def flush(self):
                if self.buffers:
                    result = torch.cat(self.buffers, dim=self.dim)
                    self.buffers = []
                    return result
        ```

    2. 内存池技术

        ```python
        class TensorPool:
            def __init__(self, size):
                self.pool = torch.empty(size, device='cuda')
                self.offset = 0
            
            def cat_view(self, tensors, dim):
                # 在内存池中分配空间并返回视图
                total_size = sum(t.numel() for t in tensors)
                view = self.pool[self.offset:self.offset+total_size].view(new_shape)
                # 手动复制数据
                current = self.offset
                for t in tensors:
                    self.pool[current:current+t.numel()] = t.flatten()
                    current += t.numel()
                self.offset += total_size
                return view
        ```

    前沿研究方向

    * Unified Virtual Memory (UVM)

        * GPU可以直接访问CPU内存

        * 减少显存间的数据复制

    * Zero-copy Tensor Operations

        * 如NVIDIA的cudaMemcpyAsync

        * 异步内存传输

    * Compressed Sparse Tensor Formats

        * 只存储非零元素和索引

    总结

    你的想法在理论上是理想的，但实践中面临挑战：

    * 计算效率：不连续访问降低GPU利用率

    * 软件复杂性：破坏现有API和自动求导系统

    * 内存管理：生命周期管理复杂

    目前的折中方案：

    * 小张量：直接cat

    * 大张量：预分配+分批处理

    * 特定场景：使用稀疏表示或自定义内核

    未来随着硬件和软件的发展，这类"零复制"操作可能会变得更加普遍。

* torch.cat()

    torch.cat() 是 PyTorch 中用于张量拼接的核心函数。

    原型

    ```python
    torch.cat(tensors, dim=0, *, out=None) → Tensor
    ```

    参数：

    * tensors：需要拼接的张量序列（列表或元组）

    * dim：沿着哪个维度进行拼接（默认为0）

    * out：输出张量（可选）

    作用

    将多个张量沿着指定维度连接起来，要求非拼接维度的大小必须相同。

    **用法示例**

    基础用法

    ```python
    import torch

    # 创建两个张量
    x = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
    y = torch.tensor([[5, 6], [7, 8]])  # shape: (2, 2)

    # 沿着第0维拼接（上下堆叠）
    result = torch.cat([x, y], dim=0)
    # shape: (4, 2)
    # [[1, 2],
    #  [3, 4],
    #  [5, 6],
    #  [7, 8]]

    # 沿着第1维拼接（左右拼接）
    result = torch.cat([x, y], dim=1)
    # shape: (2, 4)
    # [[1, 2, 5, 6],
    #  [3, 4, 7, 8]]
    ```

    不同维度的示例

    ```python
    # 三维张量拼接
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 4)

    # 沿着dim=0拼接：shape变为 (4, 3, 4)
    # 沿着dim=1拼接：shape变为 (2, 6, 4)
    # 沿着dim=2拼接：shape变为 (2, 3, 8)
    ```

    注意事项

    ```python
    # 错误示例：非拼接维度大小不同会报错
    x = torch.randn(2, 3)
    y = torch.randn(3, 3)  # 第0维大小不同
    # torch.cat([x, y], dim=0)  # 报错

    # 正确：保证非拼接维度一致
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 5)
    result = torch.cat([x, y], dim=2)  # 正确：沿着第2维拼接
    ```

    与 stack() 的区别

    * cat()：扩展现有维度，要求其他维度大小相同

    * stack()：创建新维度，要求所有维度大小都相同

        ```python
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])

        cat_result = torch.cat([x, y], dim=0)  # [1, 2, 3, 4, 5, 6], shape: (6,)
        stack_result = torch.stack([x, y], dim=0)  # [[1, 2, 3], [4, 5, 6]], shape: (2, 3)
        ```

    torch.cat() 在神经网络中常用于特征拼接、多分支网络结果的合并等场景。

* 简述 torch 中 tensor 与 numpy 的转换方式

    PyTorch Tensor 与 NumPy 数组转换

    PyTorch 与 NumPy 之间的转换非常方便，因为它们共享底层内存（默认情况下），避免了数据复制。

    1. Tensor → NumPy

        ```python
        import torch
        import numpy as np

        # 创建 PyTorch Tensor
        tensor = torch.tensor([1, 2, 3])
        tensor = torch.randn(3, 4)  # 随机张量

        # 转换为 NumPy 数组
        numpy_array = tensor.numpy()
        # 或使用更明确的方式
        numpy_array = tensor.cpu().numpy()  # 推荐：确保在 CPU 上
        ```

        注意事项：

        * CPU Tensor：共享内存（修改一方会影响另一方）

        * GPU Tensor：不能直接转换，需要先移到 CPU

        * 当 Tensor 有梯度时（requires_grad=True），需要先分离

            ```python
            # GPU Tensor 转换
            if tensor.is_cuda:
                numpy_array = tensor.cpu().numpy()  # 必须先移到 CPU

            # 有梯度的 Tensor
            tensor_with_grad = torch.tensor([1., 2., 3.], requires_grad=True)
            numpy_array = tensor_with_grad.detach().numpy()  # 必须先 detach()
            ```

    2. NumPy → Tensor

        ```python
        import numpy as np
        import torch

        # 创建 NumPy 数组
        np_array = np.array([1, 2, 3])
        np_array = np.random.randn(3, 4)

        # 转换为 PyTorch Tensor
        tensor = torch.from_numpy(np_array)  # 共享内存
        # 或使用构造函数
        tensor = torch.tensor(np_array)      # 创建副本（不共享内存）
        ```

        重要区别：

        * torch.from_numpy()：共享内存，修改 NumPy 数组会影响 Tensor

        * torch.tensor()：创建副本，两者独立

            ```python
            np_array = np.array([1, 2, 3])
            tensor1 = torch.from_numpy(np_array)  # 共享内存
            tensor2 = torch.tensor(np_array)      # 创建副本

            np_array[0] = 999
            print(tensor1)  # tensor([999, 2, 3])  改变了！
            print(tensor2)  # tensor([1, 2, 3])    没改变
            ```

    3. 数据类型对应关系

        ```python
        # NumPy 类型 → PyTorch 类型
        np.float32  ↔  torch.float32 / torch.float
        np.float64  ↔  torch.float64 / torch.double
        np.int32    ↔  torch.int32
        np.int64    ↔  torch.int64 / torch.long
        np.bool_    ↔  torch.bool

        # 指定数据类型转换
        tensor = torch.from_numpy(np_array).float()  # 转换为 float32
        tensor = torch.from_numpy(np_array).long()   # 转换为 int64
        ```

    4. GPU Tensor 的特殊处理

        ```python
        # NumPy → GPU Tensor
        np_array = np.array([1, 2, 3])
        gpu_tensor = torch.from_numpy(np_array).cuda()          # 先创建 CPU Tensor 再移到 GPU
        gpu_tensor = torch.tensor(np_array, device='cuda')     # 直接在 GPU 创建

        # GPU Tensor → NumPy
        gpu_tensor = torch.randn(3, 4, device='cuda')
        cpu_tensor = gpu_tensor.cpu()          # 先移到 CPU
        numpy_array = cpu_tensor.numpy()       # 再转换为 NumPy
        ```

    5. 实用函数封装

        ```python
        def numpy_to_tensor(np_array, device='cpu', dtype=None):
            """安全地将 NumPy 数组转换为 Tensor"""
            tensor = torch.from_numpy(np_array)
            if dtype:
                tensor = tensor.to(dtype)
            return tensor.to(device)

        def tensor_to_numpy(tensor):
            """安全地将 Tensor 转换为 NumPy 数组"""
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            return tensor.numpy()
        ```

    6. 内存共享示例

        ```python
        # 共享内存（高效，但需小心）
        np_array = np.ones((3, 3))
        tensor = torch.from_numpy(np_array)  # 共享内存

        tensor[0, 0] = 999
        print(np_array[0, 0])  # 输出 999，原始数组也被修改了

        # 避免共享内存（创建副本）
        tensor = torch.tensor(np_array)      # 创建副本
        tensor[0, 0] = 888
        print(np_array[0, 0])  # 输出 999，原始数组不受影响
        ```

    总结要点

    * 转换方法：

        * Tensor → NumPy：.numpy() 或 .cpu().numpy()

        * NumPy → Tensor：torch.from_numpy() 或 torch.tensor()

    * 内存共享：

        * torch.from_numpy() 共享内存

        * torch.tensor() 创建副本

        * CPU Tensor 与 NumPy 数组默认共享内存

    * 特殊处理：

        * GPU Tensor 需要先移到 CPU

        * 有梯度的 Tensor 需要先 detach()

        * 注意数据类型的一致性

    * 性能建议：

        * 训练时尽量使用 Tensor（支持 GPU 和自动微分）

        * 数据预处理可使用 NumPy

        * 避免频繁转换以减少开销

* torch set_num_threads() 与 set_num_interop_threads() 区别

    torch.set_num_threads() 和 torch.set_num_interop_threads() 控制的是 不同层面的并行化，理解它们的区别对于性能调优很重要。

    主要区别

    | 特性 | torch.set_num_threads() | torch.set_num_interop_threads() |
    | - | - | - |
    | 作用对象 | 单个操作内部并行（如矩阵乘法） | 多个独立操作间的并行（如多个独立的矩阵乘法） |
    | 并行层级 | 操作内并行（intra-op） | 操作间并行（inter-op） |
    | 典型场景 | 大型矩阵运算、卷积等 | 数据加载、多个小操作的并行执行 |
    | 默认值 | CPU 核心数 | 通常为 1（保守默认） |
    | 线程池 | 不同的线程池 | 不同的线程池 |

    详细解释

    1. torch.set_num_threads() - 操作内并行

        ```python
        import torch
        import time

        # 设置操作内并行线程数
        torch.set_num_threads(4)  # 这个操作内部最多用4个线程

        # 单一大操作会被拆分成多个子任务并行执行
        x = torch.randn(5000, 5000)
        y = torch.randn(5000, 5000)

        # 这个矩阵乘法会在内部使用最多4个线程
        z = x @ y  # 操作内并行
        ```

        工作方式：

        * 单个复杂操作（如大矩阵乘法）被分解成多个子任务

        * 这些子任务在多个线程上并行执行

        * 所有线程共同完成这一个操作

    2. torch.set_num_interop_threads() - 操作间并行

        ```python
        import torch
        import concurrent.futures

        # 设置操作间并行线程数
        torch.set_num_interop_threads(2)  # 最多同时执行2个独立操作
        torch.set_num_threads(2)  # 每个操作内部最多用2个线程

        def compute(i):
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            return (x @ y).mean()

        # 多个独立的矩阵乘法可以并行执行
        # 由于设置了 interop_threads=2，最多2个操作同时进行
        results = []
        for i in range(4):
            results.append(compute(i))
        ```

    实际应用场景对比

    场景1：单个大任务

    ```python
    # 适合用 set_num_threads() 优化
    torch.set_num_threads(8)  # 让单个大操作用8个核心
    torch.set_num_interop_threads(1)  # 只有一个操作，无需操作间并行

    # 单个大型矩阵运算
    large_matrix = torch.randn(10000, 10000)
    result = large_matrix @ large_matrix.T  # 这个操作内部会并行
    ```

    场景2：多个小任务

    ```python
    # 适合用 set_num_interop_threads() 优化
    torch.set_num_threads(2)  # 每个小操作用2个核心
    torch.set_num_interop_threads(4)  # 同时执行4个小操作

    # 多个独立的小运算
    def process_batch(batch_data):
        return torch.mm(batch_data, batch_data.T)

    # 如果有4个batch，可以并行处理
    batches = [torch.randn(500, 500) for _ in range(4)]
    results = [process_batch(batch) for batch in batches]
    ```

    场景3：混合场景

    ```python
    # 数据加载（操作间并行） + 计算（操作内并行）
    torch.set_num_interop_threads(2)  # 同时加载2个batch
    torch.set_num_threads(4)          # 每个batch计算用4个核心

    # 假设的数据加载和计算流程
    def load_and_compute(i):
        # 模拟数据加载
        data = torch.randn(1000, 1000)
        # 计算
        return data @ data.T

    # 可以同时加载和计算多个batch
    ```

    性能影响示例

    ```python
    import torch
    import time

    # 测试不同配置的性能
    def benchmark():
        # 创建多个小矩阵
        matrices = [torch.randn(500, 500) for _ in range(8)]
        
        # 配置1：只有操作内并行
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        
        start = time.time()
        results = [m @ m.T for m in matrices]
        print(f"只有操作内并行: {time.time()-start:.3f}s")
        
        # 配置2：只有操作间并行
        torch.set_num_threads(1)
        torch.set_num_interop_threads(4)
        
        start = time.time()
        results = [m @ m.T for m in matrices]
        print(f"只有操作间并行: {time.time()-start:.3f}s")
        
        # 配置3：两者结合
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        
        start = time.time()
        results = [m @ m.T for m in matrices]
        print(f"两者结合(2x2): {time.time()-start:.3f}s")

    benchmark()
    ```

    最佳实践建议

    * 确定工作负载类型：

        * 计算密集型大操作 → 增加 num_threads

        * 多个独立小操作 → 增加 num_interop_threads

    * 通用推荐配置：

        ```python
        import os

        total_cores = os.cpu_count()

        # 通用场景：平衡配置
        torch.set_num_threads(total_cores // 2)
        torch.set_num_interop_threads(2)

        # 数据预处理流水线
        # torch.set_num_threads(2)  # 每个预处理任务用2核心
        # torch.set_num_interop_threads(4)  # 同时预处理4个batch
        ```

    * 注意资源竞争：

        ```python
        # 不要设置过大，避免线程竞争
        # 总线程数 ≈ num_threads * num_interop_threads
        # 建议不超过物理核心数

        # 错误示例（可能导致性能下降）：
        # torch.set_num_threads(8)
        # torch.set_num_interop_threads(8)  # 最多64个线程！
        ```

    与 DataLoader 的关系

    ```python
    from torch.utils.data import DataLoader

    # DataLoader 有自己的并行机制（num_workers）
    # 与 torch 的线程设置是独立的

    # 典型配置：
    loader = DataLoader(dataset, 
                       batch_size=32,
                       num_workers=4,      # 数据加载进程数
                       pin_memory=True)

    # PyTorch 计算线程设置
    torch.set_num_threads(4)          # 每个GPU操作用4线程
    torch.set_num_interop_threads(1)  # 通常GPU操作是串行的
    ```

    总结：

    * torch.set_num_threads()：垂直扩展，让单个任务更快

    * torch.set_num_interop_threads()：水平扩展，让多个任务同时进行

    * 根据具体场景调整，两者乘积不要超过CPU物理核心数

    * 对于大多数深度学习训练，调整 num_threads 更重要

* torch 的 cpu backend 会自动多线程吗，还是使用 cpu 的 simd 相关指令集？

    PyTorch 的 CPU backend 会同时使用多线程和 SIMD 指令集，但两者的触发机制和配置方式不同。

    1. 自动多线程

        是的，PyTorch CPU 运算默认会自动使用多线程，具体表现：

        配置方式：

        ```python
        import torch

        # 查看和设置线程数
        print(torch.get_num_threads())    # 获取当前线程数
        print(torch.get_num_interop_threads())  # 获取并行操作线程数

        # 设置线程数
        torch.set_num_threads(4)         # 设置计算线程数
        torch.set_num_interop_threads(2) # 设置并行操作线程数
        ```

        环境变量控制：

        ```bash
        # 常用的环境变量
        export OMP_NUM_THREADS=4        # OpenMP 线程数（影响底层运算）
        export MKL_NUM_THREADS=4        # MKL 线程数（如果有用 MKL）
        ```

    2. SIMD 指令集优化

        PyTorch 会自动利用 CPU 的 SIMD 指令集：

        支持的指令集：

        * SSE4.2

        * AVX2

        * AVX-512（如果 CPU 支持）

        * ARM NEON（ARM 架构）

        检查是否启用：

        ```python
        import torch

        # 检查构建时启用的指令集（需要从源码编译的信息）
        print(torch.__config__.show())  # 显示构建配置

        # 运行时检测
        print("是否使用MKL:", torch.backends.mkl.is_available())
        print("是否使用OpenMP:", torch.backends.openmp.is_available())
        ```

        自动优化机制：

        * PyTorch 在编译时会根据 CPU 架构生成多版本代码

        * 运行时通过 CPU 特性检测选择最优的代码路径

        * 底层使用 Eigen、MKL、OpenBLAS 等优化的数学库

    3. 实际示例

        ```python
        import torch
        import time

        # 创建一个较大的张量
        x = torch.randn(10000, 10000)
        y = torch.randn(10000, 10000)

        # 查看默认线程数
        print(f"线程数: {torch.get_num_threads()}")

        # 矩阵乘法（会自动并行化和SIMD优化）
        start = time.time()
        z = torch.matmul(x, y)
        end = time.time()

        print(f"计算时间: {end - start:.3f}秒")
        print(f"张量形状: {z.shape}")
        ```

    4. 性能调优建议

        推荐配置：

        ```python
        # 在程序开始时设置
        import torch
        import os

        # 通常设置为物理核心数
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)

        # 或者让 OpenMP 自动管理
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        os.environ['MKL_NUM_THREADS'] = str(num_cores)
        ```

        注意事项：

        * 线程数不要超过物理核心数，否则可能因上下文切换降低性能

        * SIMD 优化是自动的，但需要 PyTorch 编译时支持对应的指令集

        * 小矩阵运算可能不会充分并行化（开销大于收益）

        * I/O 密集型任务与计算密集型任务的线程配置可能不同

    5. 验证优化效果

        ```python
        import torch
        import numpy as np

        # 比较有无多线程的性能差异
        torch.set_num_threads(1)  # 单线程
        x = torch.randn(5000, 5000)
        start = time.time()
        _ = x @ x
        print(f"单线程时间: {time.time() - start:.3f}s")

        torch.set_num_threads(8)  # 多线程
        start = time.time()
        _ = x @ x
        print(f"8线程时间: {time.time() - start:.3f}s")
        ```

    总结：

    * ✅ 多线程是默认开启的，可通过 torch.set_num_threads() 配置

    * ✅ SIMD 优化是自动的，PyTorch 会检测并选择最优指令集

    * ⚠️ 最佳线程数取决于具体任务和硬件

    * 📊 大矩阵运算最能体现并行化和 SIMD 的优势

    PyTorch 在 CPU 上的性能很大程度上依赖于底层的数学库（MKL、OpenBLAS 等），这些库本身就包含了高度优化的多线程和 SIMD 实现。

* 简介一下 torchmetrics 的基本用法

    TorchMetrics 是一个专门为 PyTorch 深度学习模型提供评估指标的库。它能够帮你标准化、简化模型性能的评估过程，尤其擅长处理分布式训练场景，并有效减少样板代码。

    下面这个表格汇总了 TorchMetrics 的一些核心组件和基本用法：

    | 组件/概念 | 说明 | 使用场景/说明 |
    | - | - | - |
    | 核心方法 | - | - |
    | update() | 累积单个批次的预测和目标值，更新指标内部状态。 | 通常在训练循环中每个批次结束后调用，仅更新状态，不计算最终结果，效率较高。 |
    | compute() | 在所有批次数据上计算最终指标值。 | 通常在一个训练周期（Epoch）结束后调用。 |
    | reset() | 重置指标内部状态，准备下一轮评估。 | 在新的评估周期开始前调用。 |
    | forward() | 相当于 update() + compute()，同时更新状态并返回当前批次的指标。 | 如需要当前批次结果可使用，但注意与最终周期结果可能不同. |
    | 常用指标 | - | - |	
    | Accuracy | 分类准确率。 | 支持多分类、多标签等不同任务。 |
    | Precision | 精确率。 | 支持多分类，可通过 num_classes 和 average 参数调整。 |
    | Recall | 召回率。 | 支持多分类，可通过 num_classes 和 average 参数调整。 |
    | F1Score | F1分数（精确率和召回率的调和平均数）。 | 支持多分类，可通过 num_classes 和 average 参数调整。 |
    | MeanMetric | 计算平均值，例如平均损失. |
    | 工具类 | - | - |
    | MetricCollection | 将多个指标合并为单个可调用单元，同时计算并返回所有结果。 | 简化多指标管理，接口与单指标一致。 |

    🔧 安装与基本使用流程

    安装很简单，通常通过pip安装即可：

    ```bash
    pip install torchmetrics
    ```

    使用 TorchMetrics 评估模型性能的基本流程如下：

    * 初始化指标：选择与你的任务相匹配的指标，例如多分类任务的Accuracy。

    * 将指标移至设备：确保指标与模型、数据在同一设备上（CPU或GPU）。

    * 更新指标状态：在每个批次结束后，使用update()方法将模型的预测输出和真实标签传递给指标对象。

    * 计算最终指标：在一个训练或验证周期结束后，使用compute()方法获取所有批次上的最终指标值。

    * 重置指标状态：在下一个评估周期开始前，使用reset()方法清除历史状态。

    💡 使用技巧与注意事项

    * 利用MetricCollection管理多个指标：如果你需要同时计算多个评估指标，MetricCollection会非常方便。

    * 警惕性能开销，合理使用compute()：避免在每一个训练批次中都调用compute()方法，因为这可能会导致性能下降，特别是在GPU训练时。建议在批次中使用update()，仅在周期结束时调用compute()。

    * 注意指标的设备位置：确保指标与输入数据位于同一设备上，否则可能导致运行时错误。在大部分情况下，TorchMetrics 会自动处理设备问题，但明确指定设备是一个好习惯。

    💎 总结

    TorchMetrics 的核心优势在于其模块化设计、分布式训练支持以及与PyTorch生态系统的无缝集成。

    希望这些信息能帮助你快速上手 TorchMetrics。如果你在使用特定指标时遇到问题，或者想了解更高级的用法，欢迎随时提问。

* PyTorch数据集划分方法总结

    1. 使用 torch.utils.data.random_split（推荐）

        这是最直接的方式，可以按任意比例划分：

        ```python
        import torch
        from torch.utils.data import Dataset, DataLoader, random_split
        from torchvision import datasets, transforms

        # 示例：加载完整数据集
        dataset = datasets.MNIST(
            root='./data', 
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )

        # 手动划分比例（7:3）
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size

        # 随机划分
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        ```

    2. 使用 Subset 手动选择索引

        如果你想更精确地控制哪些数据进入哪个集合：

        ```python
        from torch.utils.data import Subset
        import numpy as np

        # 创建索引
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        # 7:3 划分
        split = int(0.7 * len(dataset))
        train_indices = indices[:split]
        val_indices = indices[split:]

        # 创建子集
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        ```

    3. 使用 sklearn 的 train_test_split

        ```python
        from sklearn.model_selection import train_test_split

        # 获取所有索引
        indices = list(range(len(dataset)))

        # 使用 sklearn 划分（可设置随机种子）
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=0.3,  # 验证集比例
            random_state=42,  # 随机种子
            shuffle=True
        )

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        ```

    4. 自定义 Dataset 类实现划分

        如果需要更复杂的划分逻辑：

        ```python
        from torch.utils.data import Dataset

        class CustomDataset(Dataset):
            def __init__(self, data, targets, is_train=True, train_ratio=0.7, seed=42):
                self.data = data
                self.targets = targets
                
                # 固定随机种子确保可重复性
                torch.manual_seed(seed)
                indices = torch.randperm(len(data))
                
                split = int(train_ratio * len(data))
                if is_train:
                    self.indices = indices[:split]
                else:
                    self.indices = indices[split:]
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                return self.data[actual_idx], self.targets[actual_idx]

        # 使用示例
        full_data = ...  # 你的数据
        full_targets = ...  # 你的标签

        train_dataset = CustomDataset(full_data, full_targets, is_train=True, train_ratio=0.7)
        val_dataset = CustomDataset(full_data, full_targets, is_train=False, train_ratio=0.7)
        ```

    5. 使用 K-fold 交叉验证（更复杂的划分）

        ```python
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold + 1}")
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            
            # 训练和验证...
        ```

    推荐实践

    * 对于简单划分：使用 random_split 最简单直接

    * 需要可重复性：使用 random_split 或 train_test_split 并设置 generator 或 random_state

    * 需要分层采样：使用 sklearn 的 StratifiedShuffleSplit

    * 需要复杂逻辑：自定义 Dataset 类

    注意事项

    * 设置随机种子确保划分可重复：

        ```python
        torch.manual_seed(42)  # PyTorch
        np.random.seed(42)     # NumPy
        ```

    * 保持类平衡：如果数据集类别不平衡，考虑使用分层划分：

        ```python
        from sklearn.model_selection import StratifiedShuffleSplit

        # targets 是标签数组
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, val_idx = next(sss.split(data, targets))
        ```

    PyTorch 本身提供了足够灵活的工具来处理各种数据集划分需求，结合 sklearn 可以满足更复杂的场景。

* 为什么在含有fc layer, relu, softmax的三层简单网络中，训练mnist时，fc1的 weight的grad有时会出现一大片的0？

    这是一个很好的问题，你观察到的现象在训练简单神经网络时并不少见。FC1层权重梯度出现大片零值，通常是由ReLU激活函数的特性和网络初始化/数据共同导致的。下面我来详细解释原因和解决方案。

    核心原因：ReLU的“Dead Neuron”问题

    这是最可能的原因。让我们一步步分析这个过程：

    * ReLU的函数特性：

        ReLU函数的定义为 f(x) = max(0, x)。它的梯度是：

        * 当 x > 0 时，梯度为 1。

        * 当 x <= 0 时，梯度为 0。

    * 前向传播：

        输入数据经过第一层全连接层 fc1 后，产生输出 z1 = W1 * X + b1。然后 z1 被送入ReLU函数：a1 = ReLU(z1)。

        如果在 z1 的某个维度（对应某个神经元）上，对于当前整个batch的输入，其值都小于或等于0，那么ReLU就会将这个神经元的输出 a1 的对应维度全部置为0。

    * 反向传播：

        根据链式法则，损失函数 L 对 fc1 的权重 W1 的梯度为：

        ∂L/∂W1 = (∂L/∂a1) * (∂a1/∂z1) * (∂z1/∂W1)

        其中 (∂a1/∂z1) 就是ReLU的梯度。

        * 如果一个神经元在本次前向传播中“死亡”了（即 z1 <= 0），那么它的 (∂a1/∂z1) = 0。

        * 这会导致 ∂L/∂W1 中与该神经元相连的所有权重梯度都变为 0。

    结果：在权重梯度矩阵 W1.grad 中，对应那些“死亡神经元”的整行都会是0。这就解释了为什么你看到的是一片0，而不是零星几个0。

    **其他可能的原因和加剧因素**

    * 不合适的权重初始化

        * 如果 fc1 的权重初始值过大或过小，会更容易导致神经元输出 z1 的绝对值很大。

        * 过大的负值会直接让ReLU输出0并导致梯度为0。

        * 过大的正值在初期可能没问题，但如果后续某个batch的数据使其输出变为负，也可能陷入“死亡”。

    * 过大的学习率

        * 如果学习率设置得过高，权重更新步长会很大。

        * 这可能导致某些神经元的权重被“震荡”到一个区域，使得对于大多数输入，其输出 z1 都小于0，从而永久性死亡。即使这个神经元本来是有用的，一次过大的更新也可能使其再也无法恢复。

    * 偏置初始化为0

        * 如果将偏置项 b1 初始化为0，那么在训练初期，神经元的活性完全由 W1 * X 决定。如果 W1 也初始化得不好，会加剧神经元死亡的问题。

    * 输入数据问题

        * MNIST数据是归一化到 [0, 1] 的，这本身是好的。但如果你的网络结构非常深（虽然你的是三层，但原理相通），或者数据中存在大量相同的特征值，也可能在某些层导致输出为0。

    如何验证和解决

    验证方法

    * 检查神经元输出：在反向传播之前，打印出 fc1 层输出经过ReLU后的值 a1。你会很可能发现 a1 中有很多列（对应一个神经元）全部是0。

        ```python
        # 在你的模型前向传播中
        a1 = F.relu(self.fc1(x))
        print(“ReLU输出中值为0的比例：”, (a1 == 0).float().mean().item())
        ```

        如果这个比例很高（比如超过50%），就证实了“死亡ReLU”问题。

    解决方案

    * 使用Leaky ReLU或其变体

        * Leaky ReLU: f(x) = max(0.01x, x)。当 x < 0 时，它有一个很小的斜率（如0.01），而不是0。这确保了梯度永远不会完全为0，给了“死亡”神经元复活的机会。

        * PReLU: 将负区间的斜率作为一个可学习的参数。

        * ELU: 在负数区间是一个指数曲线，能缓解梯度消失问题并使输出均值接近0。

    * 调整权重初始化

        * 使用 He初始化，这是专门为配合ReLU设计的。它会根据前一层的神经元数量来调整初始权重的方差。

        * 在PyTorch中，可以这样实现：

            ```python
            torch.nn.init.kaiming_normal_(model.fc1.weight, mode=‘fan_in’, nonlinearity=‘relu’)
            ```

        * 同时，将偏置初始化为一个小的常数（如0.01）有时也有帮助，但这在当今的实践中不那么常见。

    * 降低学习率

        * 尝试使用更小的学习率，例如从 0.01 降到 0.001 或 0.0001，看看问题是否缓解。配合学习率调度器（如 ReduceLROnPlateau）效果更好。

    * 使用批归一化

        * 在 fc1 层和ReLU之间加入一个 BatchNorm1d 层。BN层可以稳定网络的激活值分布，使其均值和方差保持在一个稳定的范围内，从而大大减少“死亡ReLU”的发生。

            ```python
            self.net = nn.Sequential(
                nn.Linear(784, 256),
                nn.BatchNorm1d(256), # 加入BN层
                nn.ReLU(),
                nn.Linear(256, 10)
            )
            ```

    总结

    你看到的FC1权重梯度出现大片零值的现象，根本原因是ReLU激活函数导致了一批神经元在本次前向传播中“死亡”，从而在反向传播时，与这些神经元相连的权重无法获得梯度更新。

    推荐的解决组合拳是：

    * 将ReLU替换为Leaky ReLU（最简单直接的解决方案）。

    * 确保使用了正确的权重初始化（如He初始化）。

    * 如果问题依然存在，考虑加入批归一化层和适当调低学习率。

    对于MNIST这样的简单数据集，通常前两步就足以解决问题了。

* optim.SGD() 只能接收 Parameter 类型的数据吗，还是普通的 tensor 也可以？

    torch.optim.SGD() 只能接收 nn.Parameter 类型的数据，不能直接接收普通的 tensor。

    验证示例

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 测试1: 使用 nn.Parameter
    class ModelWithParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))
            
    model1 = ModelWithParam()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)  # ✅ 正常工作

    # 测试2: 使用普通 tensor
    class ModelWithTensor(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(3, 3)  # 普通 tensor
            
    model2 = ModelWithTensor()
    try:
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
        print("普通 tensor 也可以")
    except:
        print("普通 tensor 不行")  # ❌ 这里不会报错，但参数列表为空

    # 检查参数列表
    print("ModelWithParam 参数:", list(model1.parameters()))  # 有参数
    print("ModelWithTensor 参数:", list(model2.parameters()))  # 空列表
    ```

    为什么需要 Parameter

    ```python
    # 演示问题
    model = ModelWithTensor()

    # 虽然不会报错，但优化器没有参数可优化
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print("可优化参数数量:", len(list(optimizer.param_groups[0]['params'])))  # 0

    # 训练时不会更新任何参数
    x = torch.randn(1, 3)
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()  # 不会更新任何参数
    ```

    正确的用法

    ```python
    class CorrectModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 正确：使用 nn.Parameter
            self.weight = nn.Parameter(torch.randn(3, 3))
            self.bias = nn.Parameter(torch.zeros(3))
            
            # 错误：普通 tensor 不会被优化
            self.non_learnable = torch.randn(3, 3)  # 这不会被训练
            
        def forward(self, x):
            return x @ self.weight + self.bias

    model = CorrectModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 查看实际会被优化的参数
    for name, param in model.named_parameters():
        print(f"可学习参数: {name}, 形状: {param.shape}")
    ```

    特殊情况：手动传递参数

    ```python
    # 如果你有非 Parameter 的 tensor 也想优化，可以手动包装
    tensor1 = torch.randn(3, 3, requires_grad=True)
    tensor2 = torch.randn(2, 2, requires_grad=True)

    # 方法1: 转换为 Parameter
    param1 = nn.Parameter(tensor1)
    param2 = nn.Parameter(tensor2)

    optimizer = optim.SGD([param1, param2], lr=0.01)

    # 方法2: 直接传递，但需要 requires_grad=True
    optimizer = optim.SGD([tensor1, tensor2], lr=0.01)  # 这样也可以，但不推荐
    ```

    总结

    * optim.SGD() 主要设计用于 nn.Parameter 对象

    * 普通 tensor 不会被 model.parameters() 包含，因此优化器看不到它们

    * 即使手动传递普通 tensor 给优化器，也不如使用 nn.Parameter 规范

    * 在自定义模块中，所有需要学习的参数都应该用 nn.Parameter 包装

    最佳实践是始终使用 nn.Parameter 来定义可学习参数，这样可以确保：

    * 参数被正确注册到模块中

    * 优化器能够识别和更新参数

    * 参数能够随模型一起移动到不同设备（GPU/CPU）

    * 便于参数管理和保存

* `nn.Parameter()`

    在 PyTorch 中，nn.Parameter() 是一个用于将张量包装为模型参数的类，它是 torch.Tensor 的子类。

    原型

    ```python
    torch.nn.Parameter(data=None, requires_grad=True)
    ```

    参数：

    * data (Tensor): 要包装为参数的张量

    * requires_grad (bool, 可选): 是否需要在反向传播中计算梯度，默认为 True

    用法

    1. 基本用法

        ```python
        import torch
        import torch.nn as nn

        # 创建一个张量并包装为参数
        tensor = torch.randn(3, 3)
        param = nn.Parameter(tensor)

        print(type(param))  # <class 'torch.nn.parameter.Parameter'>
        print(param.requires_grad)  # True
        ```

    2. 在自定义模块中使用

        ```python
        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                # 使用 nn.Parameter 定义可学习参数
                self.weight = nn.Parameter(torch.randn(10, 5))
                self.bias = nn.Parameter(torch.zeros(5))
                
            def forward(self, x):
                return x @ self.weight + self.bias

        model = MyModel()
        ```

    3. 与普通张量的区别

        ```python
        class CompareModel(nn.Module):
            def __init__(self):
                super(CompareModel, self).__init__()
                # 使用 nn.Parameter - 会被自动注册为参数
                self.param_weight = nn.Parameter(torch.randn(3, 3))
                
                # 普通张量 - 不会被注册为参数
                self.tensor_weight = torch.randn(3, 3)
                
            def forward(self, x):
                return x @ self.param_weight

        model = CompareModel()

        # 查看模型参数
        for name, param in model.named_parameters():
            print(name)  # 只输出 "param_weight"，不会输出 "tensor_weight"
            ```

    4. 参数访问和管理

        ```python
        model = MyModel()

        # 访问所有参数
        print(list(model.parameters()))

        # 获取参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数数量: {total_params}")

        # 参数梯度管理
        with torch.no_grad():
            # 在不计算梯度的情况下更新参数
            model.weight += 0.1
        ```

    主要特点

    * 自动注册: 在 nn.Module 中使用时，nn.Parameter 会被自动添加到模块的参数列表中

    * 梯度计算: 默认需要梯度计算，参与反向传播

    * 设备同步: 当模块移动到 GPU 时，参数也会自动移动

    * 优化器识别: 优化器能够自动识别并更新这些参数

    注意事项

    * 只有 nn.Parameter 包装的张量才会被 model.parameters() 包含

    * 在自定义模块中，应该使用 nn.Parameter 来定义所有需要学习的参数

    * 参数默认需要梯度，如果不需要可以设置 requires_grad=False

    nn.Parameter 是构建可训练神经网络模型的基础组件，它确保了参数能够正确地被优化器识别和更新。

* dataloader 返回的是`[inputs, gts]`

    而 dataset 返回的是`(input, gt)`

    如果 dataset 返回的是`(x_1, x_2, x_3)`，dataloader 返回的会不会是`(x_1s, x_2s, x_3s)`？

* Negative Log Likelihood Loss

    After the output of the softmax layer is calculated (i.e. a value between 0 and 1), negative log is calculated of that value. The final layer combined is called as log-softmax layer. Generally, it is used in multi-class classification problems.

    Formula:

    $$\mathrm{NegativeLogLikelihoodLoss}(x, \mathrm{target}) = − \frac 1 N \sum_i \log⁡(x_{target_i})$$

    Here,

    * $x$ represents the predicted values,

    * target represents the ground truth or target values

    syntax:

    ```py
    torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean')
    ```

    ```py
    import torch
    import torch.nn as nn

    # size of input (N x C) is = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # every element in target should have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    #initialising loss function
    m = nn.LogSoftmax(dim=1)
    nll_loss = nn.NLLLoss()
    output = nll_loss(m(input), target)
    #backpropagation
    output.backward()
    #printing output
    print(output)
    ```

* 为什么计算卷积时，需要翻转卷积核

    连续卷积数学定义：

    $$(f * g)(t) = \int f(\tau) g(t - \tau) \mathrm d \tau$$

    注意公式中的 g(t - τ) - τ 是负号，这意味着卷积核需要翻转（翻转180度）。

    实际上，$g(t)$ 是对一个冲击的响应，在现实中是一个电路 module，它会处理先到达的信号，然后处理后到达的信号，因此需要先处理 $f(t)$ 左边的数据，然后从左到右依次处理所有数据，因此我们翻转的应该是 $f(t)$。

    但是如果我们把原信号看作静止的，让一个 filter 在上面从左到右滑过，为了实现和上面相同的效果，这个时候就需要翻转 $g(t)$ 了。

    example:

    ```py
    import numpy as np

    # 原始信号
    signal = np.array([1, 2, 3, 4, 5])
    # 卷积核
    kernel = np.array([1, 2, 1])

    # 1. 数学卷积（需要翻转）
    kernel_flipped = kernel[::-1]  # 翻转：[1, 2, 1] → [1, 2, 1]
    # np.convolve() 是否会自动翻转卷积核？
    conv_result = np.convolve(signal, kernel_flipped, mode='valid')

    # 2. 互相关（不翻转）
    corr_result = np.correlate(signal, kernel, mode='valid')

    print("翻转后的卷积核：", kernel_flipped)
    print("数学卷积结果：", conv_result)  # [1*1 + 2*2 + 3*1, 2*1 + 3*2 + 4*1, ...]
    print("互相关结果：", corr_result)    # [1*1 + 2*1 + 3*2, 2*1 + 3*1 + 4*2, ...]
    ```

    为什么数学定义要翻转？

    只有包含翻转的卷积才满足：

    * 交换律：$f * g = g * f$

    * 结合律：$(f * g) * h = f * (g * h)$

    * 平移不变性等数学性质

* 7点移动平均

    指滑动窗口中共有 7 个数据。每次计算时，取当前数据点及其前后各 3 个点（共 7 个点）。将这 7 个数据点相加，然后除以 7。相当于一个低通滤波器，平滑掉高频噪声。

    卷积核代码：`np.ones(7) / 7`

    相当于：`[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]`

    7 点并没有什么特殊的意义，通常与 5 点，9 点，15 点等做比较，结合实际问题赋予意义。

    通常选择奇数点，而不是偶数点。因为奇数点中心对称，输出与输入时间对齐，偶数点会产生半个时间单位的相位偏移。

    example:

    ```py
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    x = np.arange(0, 4 * np.pi, 0.1)
    y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=x.shape)

    kernel_size = 7
    mean_kernel = np.ones(kernel_size) / kernel_size
    mean_kernel = mean_kernel[::-1]

    y_1 = np.pad(y, kernel_size // 2)  # [0, 0, 0, y, 0, 0, 0]
    y_2 = np.zeros_like(y)
    for i in range(len(y)):
        y_2[i] = np.sum(y_1[i:i+kernel_size] * mean_kernel)

    fig, axes = plt.subplots(2, 1)
    ax: Axes = axes[0]
    ax.plot(x, y, 'b')
    ax = axes[1]
    ax.plot(x, y_2, 'r')
    plt.show()
    ```

    output:

    ![ref_42/pic_1](../../Reference_resources/ref_42/pic_1.png)

    此时在前后各 padding `kernel_size // 2`个 0 元素，正好把每个时刻放在 kernel window 的正中间。

* 高斯分布（正态分布）的推导

    高斯函数（正态分布）的发现是科学史上的重要里程碑，由高斯和拉普拉斯分别独立发现。最初的动机是解决测量误差问题。

    * 基本假设（公理系统）

        高斯函数的推导基于以下合理假设：

        1. 误差对称性假设

            误差围绕真值对称分布：

            $p(\varepsilon) = p(- \varepsilon)$

            其中$\varepsilon = 测量值 - 真值$

        2. 最大似然原理

            函数中最可能的参数值应该是使所有观测值出现的概率乘积最大的值。

        3. 独立同分布假设

            多次测量误差相互独立。

    * 推导过程

        1. 设定问题框架

            设：

            * 真值：$\mu$

            * 测量值：$x_1$, $x_2$, $\dots$, $x_n$

            * 误差：$\varepsilon_i = x_i - \mu$

            * 误差概率密度函数：$\varphi(\varepsilon)$

            根据独立性，我们可以计算出观测到这些数据的联合概率：

            $$L(\mu) = \varphi(x_1 - \mu) \cdot \varphi(x_2 - \mu) \cdot \dots \cdot \varphi(x_n - \mu)$$

        2. 最大似然估计

            对$L(\mu)$取对数（取对数后，求到的极值和原问题等价吗？是否存在取对数后，极值与原函数不相同的情况？）：

            $$\ln L(\mu) = \sum_i \ln \varphi(x_i - \mu)$$

            最大化条件（为什么是最大化，而不是最小化？）：

            $$\frac{\mathrm d \ [\ln L(\mu)]} {\mathrm d \ \mu} = 0$$

            即：

            $$\sum \frac{\varphi'(x_i - \mu)} {\varphi(x_i - \mu)} = 0$$

        3. 引入关键函数

            令：

            $$\Psi(\varepsilon) = \frac{\varphi'(\varepsilon)} {\varphi(\varepsilon)}$$

            则方程变为：

            $$\sum_i \Psi(x_i - \mu) = 0$$

        4. 高斯的关键洞察

            高斯意识到，如果取算术平均作为 $\mu$ 的估计：

            $$\hat \mu = \frac{x_1 + x_2 + \dots + x_n} {n}$$

            那么对于任意 $a$, $b$ （为什么？不懂）：

            $$\sum \Psi(x_i - (a \cdot x_j + b \cdot x_k)) = 0$$

            这要求 $\Psi$ 必须是线性函数 (依然不懂)：

            $$\Psi (\varepsilon) = k \cdot \varepsilon$$

        5. 求解微分方程

            由

            $$\Psi(\varepsilon) = \varphi'(\varepsilon) / \varphi(\varepsilon) = k \cdot \varepsilon$$

            解这个微分方程：

            $$\mathrm d \, \varphi / \varphi = k \varepsilon \ \mathrm d \, \varepsilon$$

            两边积分：

            $$\ln \varphi(\varepsilon) = (k / 2) \varepsilon^2 + C$$

            所以：

            $$φ(ε) = A \cdot \exp(kε²/2)$$

        6. 确定常数

            因为概率密度函数必须满足：

                归一化：∫φ(ε)dε = 1

                对称性：φ(ε) = φ(-ε)

                衰减性：当|ε|→∞时，φ(ε)→0

            这要求k必须为负数，令 k = -1/σ²

            则：

            φ(ε) = A · exp(-ε²/(2σ²))

        7. 归一化计算

            计算归一化常数A：

            ∫_{-∞}^{∞} A · exp(-ε²/(2σ²)) dε = 1

            利用高斯积分公式：

            ∫_{-∞}^{∞} exp(-αx²) dx = √(π/α)

            令 α = 1/(2σ²)，则：

            ∫ exp(-ε²/(2σ²)) dε = √(2πσ²)

            所以：

            A = 1/√(2πσ²)

        8. 最终形式

            得到标准的高斯分布：

            φ(ε) = 1/√(2πσ²) · exp(-ε²/(2σ²))

* 梯度权重

    在 PyTorch 中，y.backward() 的参数表示的是梯度权重（gradient weights），也称为 v 或 grad_output。

    y.backward(gradient) 中的 gradient 参数表示 y 对自身的梯度，即 ∂y/∂y。在标量情况下通常默认为 1，但在张量情况下需要显式指定。

    如果 `y` 是向量（多维）时，torch 会自动构造一个 sum 表达式，`y' = w_1 * y_1 + w_2 * y_2 + ... + w_n * y_n`，最终求导是对 `y'` 求导。

    `y.backward(v)`计算的是`∂(v·y)/∂x = vᵀ · ∂y/∂x`。

    当 y 是多维张量时，PyTorch 会：

    1. 自动构造一个标量函数：$y' = w_1 \cdot y_1 + w_2 \cdot y_2 + \dots + w_n \cdot y_n$

    2. 使用用户提供的权重：$w = [w_1, w_2, \dots, w_n]$ 就是 `backward()` 中的参数

    3. 对 $y'$ 求导：最终计算的是 $\partial y' / \partial x$，而不是直接计算 $\partial y / \partial x$

    example:

    ```py
    from hlc_utils import *

    x = t.tensor([1, 2, 3], dtype=t.float, requires_grad=True)
    y = x**2
    y.backward(t.tensor([1, 1, 1]))
    print('y:', y)
    print('x.grad:', x.grad)

    x.grad.zero_()
    y = x**2
    y[0].backward()
    # y[1].backward()  # error, 计算图只能 backward 一次
    # y[2].backward()
    print('y:', y)
    print('x.grad:', x.grad)
    ```

    output:

    ```
    y: tensor([1., 4., 9.], grad_fn=<PowBackward0>)
    x.grad: tensor([2., 4., 6.])
    y: tensor([1., 4., 9.], grad_fn=<PowBackward0>)
    x.grad: tensor([2., 0., 0.])
    ```

    还可以调用`t.autograd.backward()`:

    ```py
    x = t.tensor([1, 2, 3], dtype=t.float, requires_grad=True)
    y = x**2
    t.autograd.backward([y[0], y[1]])
    print('y:', y)
    print('x.grad:', x.grad)
    ```

    output:

    ```
    y: tensor([1., 4., 9.], grad_fn=<PowBackward0>)
    x.grad: tensor([2., 4., 6.])
    y: tensor([1., 4., 9.], grad_fn=<PowBackward0>)
    x.grad: tensor([2., 4., 0.])
    ```

* 进化算法 es

    核心思想：仿生“优胜劣汰”

    进化算法是一种受生物进化论（物竞天择，适者生存）启发而设计的优化算法。它的核心思想是：通过模拟自然选择、交叉（杂交）和变异等过程，让一个“种群”在代代繁衍中不断进化，最终找到复杂问题的最优解或满意解。

    一个生动的比喻：寻找最高峰

    假设你的任务是在一个完全漆黑、地形复杂（有很多山丘和山谷）的区域里找到最高点。你没有地图，只能靠派出一支“探险队”去摸索。

        初始化种群（第一代探险队）：

            你随机地在地图上撒下一把“探险者”（这就是初始种群）。每个探险者都有一个位置坐标（这就是一个“染色体”或“解”）。

        评估适应度（判断谁站得高）：

            你让每个探险者报告他们所在位置的海拔高度。海拔越高，代表他的“适应度”越好。

        选择（优胜劣汰）：

            你更倾向于选择那些站得高的探险者作为“父母”，让他们繁衍下一代。站得越低的人，被选中的几率就越小。这保证了优秀的基因（位置信息）能传递下去。

        交叉/重组（父母生孩子）：

            你让选出的“父母”两两配对，交换他们的一部分位置信息（比如，取父亲的一半坐标和母亲的一半坐标，组合成一个新的坐标）。这样就产生了新的“孩子”探险者，他们可能站在父母位置之间的某个新地方。

        变异（随机的小变化）：

            在新生的“孩子”中，你随机地对极少数孩子的坐标进行一个微小的、随机的变动。比如，让某个孩子向左或向右随机移动一小步。变异非常重要，它能引入新的可能性，比如让孩子偶然发现一个父母从未探索过的、可能更高的新山丘。

        形成新一代，并循环：

            现在，你用这些新生的“孩子”们（通过选择和交叉、变异产生的）组成新一代的探险队，替换掉大部分老一代的成员。

            然后回到第2步，重复这个过程：评估新队伍的适应度 -> 选择 -> 交叉 -> 变异。

    经过很多代这样的循环后，你会发现，你的整个探险队会逐渐向地图上最高的几座山峰聚集，最终找到最高点（或一个非常高的点）。

    进化算法的关键组成部分

    根据上面的比喻，一个标准的进化算法包含以下要素：

        种群：一组潜在解的集合。

        染色体/个体：对问题的一个潜在解的编码（比如一串数字）。

        适应度函数：用于评估一个解好不好的标准。算法的目标就是最大化或最小化这个函数值。

        选择：根据适应度高低，从当前种群中挑选出优秀的个体作为父母。

        交叉：将两个父代个体的部分结构加以替换重组，生成新个体。这是产生新解的主要手段。

        变异：以一定的概率随机改变个体编码的某些部分，以增加种群的多样性，避免陷入局部最优。

    主要特点与优势

        适用于黑箱问题：你不需要知道问题的精确数学模型，只需要能评估每个解的好坏（适应度）即可。

        全局搜索能力强：由于同时处理一个种群（多个解），并且有变异操作，它不太容易像传统方法那样陷入局部最优解。

        鲁棒性好：对问题的数学性质（如是否可微、是否连续）要求不高，能处理各种奇形怪状的搜索空间。

        高度并行：种群中每个个体的评估可以同时进行，非常适合并行计算。

    常见的进化算法类型

        遗传算法：最经典和著名的一种，通常使用二进制字符串编码。

        遗传规划：用于进化计算机程序（通常表示为树形结构）。

        进化策略：主要用于连续数值优化，特别强调变异操作。

        进化规划：与进化策略类似，但通常没有交叉操作。

    总结

    进化算法是一类通过模拟生物进化过程中的“选择、交叉、变异”机制，来引导一个候选解种群朝着更优方向发展的随机优化算法。它特别擅长解决那些传统数学方法难以处理的、复杂的、非线性的优化问题。

    简单来说，它就是让计算机通过 “随机生成 -> 评估好坏 -> 优胜劣汰 -> 混合变异” 的循环，自己“摸索”出问题答案的一种强大方法。

* 进化算法中的交叉是否是必要的？

    这是一个非常深刻的问题，答案是：交叉不是严格必要的，但它极其重要。 没有交叉的进化算法（即只依赖选择和变异）仍然是完整且理论上可行的算法，但在大多数情况下，它的性能会大打折扣。

    下面我们从理论和实践两个角度来详细解释。
    1. 没有交叉的进化算法：进化策略(1+1)-ES

    一个著名的例子是 (1+1)-进化策略：

        种群大小： 1个个体。

        操作：

            变异： 从当前个体产生一个变异的后代。

            选择： 比较父代和子代，保留两者中适应度更高的一个。

        这个算法完全没有交叉操作。它就像一个独行侠在解空间里通过随机扰动（变异）进行摸索。

    它能工作吗？ 能！对于许多问题，它都能找到不错的解。它证明了变异是维持种群多样性和进行探索的必要操作。
    
    2. 为什么交叉如此重要？理论依据

    交叉的核心作用不是探索，而是开发。它将已有的优良“基因模块”进行重组，从而高效地构建出更优的解。其理论依据主要来自以下两个经典理论：
    a) 建筑块假说

    这是遗传算法最核心的理论基础。

        核心思想： 优秀的解通常是由一些短的、性能良好的“基因模块”组合而成。这些模块本身具有较高的平均适应度。

        交叉的作用： 交叉操作允许这些在不同个体中独立进化出来的优良模块（建筑块）组合到一起，从而像搭积木一样，快速构建出包含多个优良模块的、全局更优的解。

        比喻：

            只有变异： 就像你试图一个字一个字地随机修改来写出一篇好文章，过程极其缓慢。

            加入交叉： 就像两位作家（父代）交换了他们文章中最精彩的段落（建筑块），然后组合成一篇可能更精彩的新文章（子代）。这大大加速了创造过程。

    b) 模式定理

    这是对建筑块假说的数学化描述，由遗传算法之父John Holland提出。

        模式： 一个模式是描述一组具有特定基因相似性的字符串的模板。例如，在二进制编码中，模式 1**0*1 代表了所有第一位为1、第四位为0、第六位为1的字符串（*是通配符）。

        定理内容： 模式定理定量地证明了：短定义的、低阶的、高于平均适应度的模式（即建筑块）在种群中会以指数级增长。

        交叉的角色： 交叉虽然会破坏某些长的模式，但它对短的、优良的模式破坏概率很低。因此，总体上，这些“建筑块”能够通过选择被保留，并通过交叉被传播和重组，从而在种群中迅速占据主导地位。

    简单来说，模式定理从数学上解释了为什么交叉能有效地让“好点子”在种群中传播和组合。

    3. 交叉 vs. 变异：分工明确

    为了更好地理解，我们可以对比一下两者的角色：

    特性	变异	交叉
    主要角色	探索	开发
    操作对象	单个个体	两个或多个个体
    创造新基因	能。通过随机改变，可以产生种群中从未有过的基因值。	不能。它只能重新组合现有基因，无法创造全新的基因信息。
    作用方式	局部、随机的微调。	全局、结构化的重组。
    类比	一个发明家在自己的实验室里偶然发现了一个新材料。	两个公司合并，将各自的核心技术（建筑块）整合成一个更强大的新公司。
    结论与总结

        非必要性： 从存在性上讲，交叉不是必需的。一个只包含变异和选择的进化算法是完整的，并且可以解决问题。

        关键重要性： 从效率和性能上讲，交叉通常是至关重要的。它是进化算法区别于其他随机搜索算法的关键特征。

        理论依据： 建筑块假说和模式定理为交叉的重要性提供了坚实的理论依据。它们解释了交叉如何通过重组优良的“基因模块”来指数级地加速搜索过程，实现“1+1>2”的效果。

        协同工作： 变异和交叉是相辅相成的。变异负责“开疆拓土”，探索新的可能性和维持多样性；交叉负责“精耕细作”，高效地整合已有的成果。 没有变异，算法会过早收敛，失去发现新机会的能力；没有交叉，算法的收敛速度会非常缓慢，难以解决复杂问题。

    因此，在实际应用中，绝大多数进化算法都会同时包含交叉和变异这两个操作员，让它们在搜索过程中各司其职，协同工作。

* 进化算法引入表观遗传学机制

    在计算机模型中，表观遗传可以被模拟为：

        可遗传的标记：在基因型（染色体编码）之上，增加一个“标记层”。这个标记层可以决定某个基因是“开启”还是“关闭”（表达或不表达），而这个标记本身也可以以一定的概率遗传给后代。

        对环境的学习与继承：

            拉马克进化的引入：表观遗传在一定程度上支持了“获得性遗传”的可能性。父母一生中因环境因素（如饮食、压力）导致的表观标记变化，有可能传递给后代。

            在算法中的体现：个体在生命周期内可以通过局部搜索、学习等策略来“优化”自己的表现型，然后将这种优化成果通过某种机制（例如，改变基因的显性/隐性，或直接修改编码）部分地遗传给后代。这被证明可以显著加速收敛。

* 其他被引入进化算法的复杂生物学机制

    发育生物学：

        问题：传统进化算法中，基因型到表现型是直接转换（如，二进制字符串直接解码为一个数字）。

        更生物学的模型：引入一个发育过程。基因型作为“配方”，通过一个模拟胚胎发育的过程（如基因调控网络）逐步“生长”成复杂的表现型。这使得小的基因变化能通过发育过程产生巨大而结构化的表现型变化，从而创造出更复杂、更鲁棒的解决方案。

    生态位与共生：

        问题：传统算法中个体间主要是竞争关系。

        更生物学的模型：模拟生态系统，个体可以占据不同的“生态位”，避免直接竞争。还可以引入共生关系，即不同个体通过合作产生单独无法实现的适应度优势。

    性选择与宿主-寄生虫协同进化：

        不仅仅基于生存能力进行选择，还引入基于“吸引力”的选择。

        通过模拟宿主与寄生虫之间的“军备竞赛”，来维持种群的多样性和避免过早收敛。

* `torch.randint()`要求 size 参数必须为 tuple 类型

    比如：

    * 对于一维张量: `(batch_size,)`

    * 对于二维张量: `(batch_size, seq_len)`

* 多模态推理（Multimodal Reasoning）

    1. 多模态表征学习

        * 模态对齐（Alignment）：将不同模态的数据映射到统一的语义空间，使相似语义的内容（如图像中的狗和文本中的“狗”）在表征空间中接近。例如：

            对比学习（如CLIP）：通过对比损失函数拉近匹配的图文对，推开不匹配的对。

            跨模态编码器（如ViLBERT、UniT）：用Transformer架构联合编码多模态输入。

        * 模态融合（Fusion）：将不同模态的特征合并为统一的表征。常见方法包括：

            早期融合：在输入层直接拼接不同模态的原始特征。

            晚期融合：分别处理各模态后合并高层特征（如注意力机制加权融合）。

    2. 跨模态关联与推理

        * 互补性利用：不同模态提供的信息可能互补（如视频中的动作+音频中的声音可更准确识别场景）。

        * 冗余性消除：通过跨模态注意力机制（如Cross-Modal Attention）动态选择重要信息，忽略重复或噪声。

        * 符号-感知结合：将神经网络的感知能力（如图像分类）与符号推理（如逻辑规则）结合，实现高层推理（如Visual Question Answering中回答“图片中是否有比猫更大的物体？”）。

    3. 多模态预训练模型

        现代多模态推理常基于大规模预训练模型，其核心思想是：

        * 自监督学习：利用海量无标注多模态数据（如互联网图文对）进行预训练，学习通用表征。

            * 任务示例：图文匹配、掩码语言建模（MLM）、掩码区域建模（MRM）等。

        * 微调（Fine-tuning）：在特定下游任务（如视觉推理、多模态情感分析）上微调模型。

        典型模型：

        * CLIP（OpenAI）：通过对比学习对齐图文表征。

        * Flamingo（DeepMind）：处理交错图文序列，支持少样本学习。

        * GPT-4V（OpenAI）：扩展大语言模型至多模态输入，实现复杂推理。

    4. 推理机制的具体实现

        * 注意力机制：通过跨模态注意力权重动态聚焦关键信息（如文本描述中的关键词与图像区域的关联）。

        * 图神经网络（GNN）：将多模态数据表示为图结构（如对象关系图），通过消息传递进行推理。

        * 神经符号系统：结合神经网络（处理感知）和符号推理（处理逻辑），例如：

            * Neuro-Symbolic Concept Learner（NS-CL）：从图像中提取符号化概念后进行逻辑推理。

    5. 应用与挑战

        * 应用场景：

            * 视觉问答（VQA）、视频理解、医疗诊断（结合影像和报告）、自动驾驶（融合激光雷达、摄像头、地图）。

        * 关键挑战：

            * 模态异构性：不同模态的数据分布差异大（如文本离散、图像连续）。

            * 数据稀缺性：高质量对齐的多模态数据较少。

            * 可解释性：复杂模型的决策过程难以透明化。

    示例：多模态问答的推理流程

    1. 输入：问题（文本）“图中戴帽子的人手里拿着什么？” + 图像。

    2. 表征：文本用BERT编码，图像用CNN提取区域特征。

    3. 对齐：通过注意力机制找到“戴帽子的人”对应的图像区域。

    4. 推理：结合区域特征（检测“手”和“物体”）和问题语义预测答案（如“杯子”）。

* `torch.relu()`

    定义：relu(x) = max(0, x)

    通俗解释：它像一个“过滤器”，把所有输入进来的负数都变成 0，而正数则保持不变。

    在神经网络中的意义：

    * 引入非线性。如果没有激活函数，无论神经网络有多少层，它都等价于一个线性模型，表达能力非常有限。ReLU 的加入使得网络可以学习并拟合复杂的数据模式。

    * 计算简单，只有比较和取0的操作，因此训练速度比 Sigmoid、Tanh 等函数更快。

    * 有助于缓解梯度消失问题（在正数区域，梯度恒为1

    example:

    ```py
    import torch

    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 5.0])
    y = torch.relu(x)
    print(y)
    # 输出：tensor([0., 0., 0., 1., 5.])
    ```

    ReLU 的导数:

    当 x < 0 时：函数值是常数 0，所以导数为 0

    当 x > 0 时：函数是 f(x) = x，所以导数为 1

    关键问题：在 x = 0 处的导数

    在 x = 0 这个点，ReLU 函数是不可微的，或者说是一个次梯度点。

    * 左导数 = 0

    * 右导数 = 1

    * 左右导数不相等，因此在 x=0 处导数不存在

    在实际应用中（如深度学习框架 PyTorch, TensorFlow），通常采用以下约定之一：

    1. 将 x=0 处的导数定义为 0（这是最常见的选择）

    2. 或者定义为 1

    3. 或者随机选择 0 或 1

    在 PyTorch 中，torch.relu 在 x=0 处的导数被定义为 0。

    example:

    ```py
    import torch

    x = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    y = torch.relu(x)

    # 假设上游梯度为 1
    y.backward(torch.tensor([1.0, 1.0, 1.0]))
    print(x.grad)  # 输出：tensor([0., 0., 1.])
    ```

    关于 nn.ReLU：

    ```py
    # 对于 Sequential 模型，使用 nn.ReLU 模块
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),      # 这是一个模块，有状态，可以训练参数（虽然ReLU没有参数）
        nn.Linear(20, 1)
    )
    ```

    在 nn.Sequential 中必须使用 nn.ReLU() 模块

* 如果在`batch_loss.backward()`之前就尝试拿 grad `net.fc1.weight.grad`，那么 grad 为 None

* 不能使用`sgd = torch.optim.sgd.SGD()`, 但是可以使用`from torch.optim.sgd import SGD`。不清楚为什么。

* pytorch model save(), load()

    ```py
    # 保存模型
    t.save(net.state_dict(), 'model_weights.pth')

    # 加载模型
    net.load_state_dict(t.load('model_weights.pth'))
    ```

    注意这种方法没有保存 model 的结构，只保存了参数。

* dataloader

    syntax:

    ```py
    DataLoader(dataset, shuffle=True, sampler=None, batch_size=32)
    ```

    一个简单的 example:

    ```py
    import torch as t
    from torch.utils.data import Dataset, DataLoader

    class MyDataset(Dataset):
        def __init__(self):
            self.m_arr = list(range(8))
            self.m_len = len(self.m_arr)
        
        def __len__(self):
            return self.m_len
        
        def __getitem__(self, index):
            return self.m_arr[index]
        
    my_dataset = MyDataset()
    print("first elm: {}".format(my_dataset[0]))
    print("dataset len: {}".format(len(my_dataset)))

    my_dataloader = DataLoader(my_dataset, batch_size=2, shuffle=True)
    for batch_data in my_dataloader:
        print("batch_data: {}, type: {}, shape: {}".format(batch_data, type(batch_data), batch_data.shape))
    ```

    output:

    ```
    first elm: 0
    dataset len: 8
    batch_data: tensor([4, 1]), type: <class 'torch.Tensor'>, shape: torch.Size([2])
    batch_data: tensor([6, 7]), type: <class 'torch.Tensor'>, shape: torch.Size([2])
    batch_data: tensor([3, 0]), type: <class 'torch.Tensor'>, shape: torch.Size([2])
    batch_data: tensor([5, 2]), type: <class 'torch.Tensor'>, shape: torch.Size([2])
    ```

    DataLoaders on Built-in Datasets:

    ```py
    # importing the required libraries
    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import seaborn as sns
    from torch.utils.data import TensorDataset

    # defining the dataset consisting of 
    # two columns from iris dataset
    iris = sns.load_dataset('iris')
    petal_length = torch.tensor(iris['petal_length'])
    petal_width = torch.tensor(iris['petal_width'])
    dataset = TensorDataset(petal_length, petal_width)

    # implementing dataloader on the dataset 
    # and printing per batch
    dataloader = DataLoader(dataset, 
                            batch_size=5, 
                            shuffle=True)

    for i in dataloader:
        print(i)
    ```

* `torch.nn.Module`

    * `__init__()`: The __init__ method is used to initialize the module's parameters. This method is called when the module is created, and it allows we to set up any internal state that the module needs. For example, we might use this method to initialize the weights of a neural network or to create other modules that the module needs in order to function.

    * `forward()`: The forward method is used to perform the computation that the module represents. This method takes in one or more input tensors, performs computations on them, and returns the output tensors. It is a forward pass of the module.

    example:

    ```py
    class MyModule(nn.Module):
        
        # Initialize the parameter
        def __init__(self, num_inputs, num_outputs, hidden_size):
            super(MyModule, self).__init__()
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.linear2 = nn.Linear(hidden_size, num_outputs)
        
        # Forward pass
        def forward(self, input):
            lin    = self.linear1(input)
            output = nn.functional.relu(lin)
            pred   = self.linear2(output)
            return pred

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_module.parameters(), lr=0.005)

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    ```

    complete code:

    ```py
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from sklearn.metrics import classification_report

    class MyModule(nn.Module):
        def __init__(self, num_inputs, num_outputs, hidden_size):
            super(MyModule, self).__init__()
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.linear2 = nn.Linear(hidden_size, num_outputs)

        def forward(self, input):
            lin    = self.linear1(input)
            output = nn.functional.relu(lin)
            pred   = self.linear2(output)
            return pred

    # Instantiate the custom module
    my_module = MyModule(num_inputs=28*28, num_outputs=10, hidden_size=20)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_module.parameters(), lr=0.01)

    # Define the transformations for the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Define the data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train the model
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            optimizer.zero_grad()
            output = my_module(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print('Epoch -->',epoch,'-->',loss)

        

    #Test the model
    with torch.no_grad():
        y_true = []
        y_pred = []
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            output = my_module(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            y_true += labels.tolist()
            y_pred += predicted.tolist()

        # Accuracy
        print('Accuracy: {} %'.format(100 * correct / total))
        
        # Classification Report
        report = classification_report(y_true, y_pred)
        print(report)
    ```

* 一个可以跑通的 lstm example

    ```py
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # ==== 超参数 ====
    seq_len = 20       # 每个输入序列长度
    hidden_size = 64   # LSTM 隐层维度
    num_layers = 1
    num_epochs = 200
    lr = 0.01
    torch.manual_seed(0)
    np.random.seed(0)

    # ==== 生成数据 ====
    x = np.linspace(0, 100, 1000)
    y = np.sin(x)
    y = (y - y.min()) / (y.max() - y.min())

    # 构造序列
    def create_dataset(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i+seq_len])
            ys.append(data[i+seq_len])
        return np.array(xs), np.array(ys)

    X, Y = create_dataset(y, seq_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, 1]
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # [batch, 1]

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # ==== 模型定义 ====
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # 取最后时刻输出
            out = self.fc(out)
            return out

    model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ==== 训练 ====
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.6f}")

    # ==== 测试 ====
    model.eval()
    with torch.no_grad():
        pred = model(X_test)   # shape [test_len, 1]
        loss = criterion(pred, Y_test)
        print(f"Test MSE: {loss.item():.6f}")

    # ==== 未来趋势预测（修正后的循环） ====
    future_steps = 50  # 预测未来 50 个点
    future_preds = []

    # 取测试集最后一个序列作为起点（注意是最后一个 X_test）
    last_seq = X_test[-1].clone().detach().unsqueeze(0)  # shape [1, seq_len, 1]

    model.eval()
    with torch.no_grad():
        for _ in range(future_steps):
            # next_pred: shape [1, 1]
            next_pred = model(last_seq)
            future_preds.append(next_pred.item())
            # 把 next_pred 扩成 [1, 1, 1]，然后滑动窗口拼接成新的 last_seq
            next_pred_expanded = next_pred.unsqueeze(-1)     # [1, 1, 1]
            last_seq = torch.cat((last_seq[:, 1:, :], next_pred_expanded), dim=1)  # [1, seq_len, 1]

    # ==== 可视化（用解析 sin 生成 future GT 并只标 10 个 x） ====
    plt.figure(figsize=(12,4))

    # 背景完整 sin 曲线（淡化）
    plt.plot(y, label='True sin (background)', alpha=0.15, zorder=1)

    # 测试区间预测（实线）
    test_pred_idx_start = train_size + seq_len
    test_pred_idx_end = test_pred_idx_start + len(pred)
    plt.plot(range(test_pred_idx_start, test_pred_idx_end),
             pred.squeeze().cpu().numpy(), label='Predicted (test)', zorder=2)

    # 未来预测（红线）
    future_idx_start = test_pred_idx_end
    future_idx_end = future_idx_start + future_steps
    plt.plot(range(future_idx_start, future_idx_end),
             np.array(future_preds), label='Future Prediction', color='red', linewidth=2, zorder=3)

    # --- 用解析式继续生成 future x 与 GT（并用和训练时相同的归一化） ---
    # 原始 x 数组名为 x；我们假设它还在作用域内
    step = x[1] - x[0]
    future_x = x[-1] + step * np.arange(1, future_steps + 1)  # length future_steps
    future_y_raw = np.sin(future_x)

    # 使用训练时的归一化参数（与之前 y 的归一化保持一致）
    # 注意：在你的脚本里 y = np.sin(x); 然后 y = (y - y.min()) / (y.max() - y.min())
    # 所以我们用同样的 min/max 来归一化 future_y_raw
    orig_y_raw = np.sin(x)  # 原始未归一化的 y（基于原 x）
    y_min, y_max = orig_y_raw.min(), orig_y_raw.max()
    future_y = (future_y_raw - y_min) / (y_max - y_min)

    # 在未来段均匀选取 10 个点标出 'x'
    n_marks = 10
    if future_steps >= n_marks:
        mark_indices = np.linspace(0, future_steps - 1, n_marks, dtype=int)
    else:
        # 如果 future_steps 少于 10，标全部点
        mark_indices = np.arange(future_steps, dtype=int)

    x_gt_marks = np.array(range(future_idx_start, future_idx_end))[mark_indices]
    y_gt_marks = future_y[mark_indices]

    plt.scatter(x_gt_marks, y_gt_marks, marker='x', color='darkred',
                s=80, linewidths=2.5, label='Ground Truth (sampled x)', zorder=4)

    # 计算并打印未来预测与解析 GT 的误差（全量比较）
    pred_array = np.array(future_preds)
    mse_future = np.mean((pred_array - future_y) ** 2)
    print(f"Future MSE against analytic sin (future {future_idx_start}..{future_idx_end-1}): {mse_future:.6f}")

    plt.legend()
    plt.title("LSTM: sin(x) prediction + future trend (10 sampled GT 'x' marks)")
    plt.xlabel("Time step")
    plt.ylabel("Normalized sin(x)")
    plt.tight_layout()
    plt.show()
    ```

    output:

    ```
    Epoch [20/200]  Loss: 0.030417
    Epoch [40/200]  Loss: 0.002496
    Epoch [60/200]  Loss: 0.000462
    Epoch [80/200]  Loss: 0.000082
    Epoch [100/200]  Loss: 0.000028
    Epoch [120/200]  Loss: 0.000012
    Epoch [140/200]  Loss: 0.000006
    Epoch [160/200]  Loss: 0.000003
    Epoch [180/200]  Loss: 0.000002
    Epoch [200/200]  Loss: 0.000002
    Test MSE: 0.000002
    ```

    还会输出一个 sin 曲线的图像。

* Long Short-Term Memory (LSTM) 

    * Hidden State (h_n)

        The hidden state in an LSTM represents the short-term memory of the network.

        Shape: The hidden state h_n has the shape (num_layers * num_directions, batch, hidden_size). This shape indicates that the hidden state is maintained for each layer and direction in the LSTM.

    * Output (output)

        The output of an LSTM is the sequence of hidden states from the last layer for each time step. 

* Natural language processing (NLP) 常见的任务

    * Automatic Text Generation: Deep learning model can learn the corpus of text and new text like summaries, essays can be automatically generated using these trained models.

    * Language translation: Deep learning models can translate text from one language to another, making it possible to communicate with people from different linguistic backgrounds. 

    * Sentiment analysis: Deep learning models can analyze the sentiment of a piece of text, making it possible to determine whether the text is positive, negative or neutral.

    * Speech recognition: Deep learning models can recognize and transcribe spoken words, making it possible to perform tasks such as speech-to-text conversion, voice search and voice-controlled devices. 

* Batch Normalization

    Batch Normalization (BN) is a critical technique in the training of neural networks, designed to address issues like vanishing or exploding gradients during training.

    Batch Normalization(BN) is a popular technique used in deep learning to improve the training of neural networks by normalizing the inputs of each layer.

    How Batch Normalization works?

    1. During each training iteration (epoch), BN takes a mini batch of data and normalizes the activations (outputs) of a hidden layer. This normalization transforms the activations to have a mean of 0 and a standard deviation of 1.

    2. While normalization helps with stability, it can also disrupt the network's learned features. To compensate, BN introduces two learnable parameters: gamma and beta. Gamma rescales the normalized activations, and beta shifts them, allowing the network to recover the information present in the original activations.

    It ensures that each element or component is in the right proportion before distributing the inputs into the layers and each layer is normalized before being passed to the next layer.

    PyTorch provides the nn.BatchNormXd module (where X is 1 for 1D data, 2 for 2D data like images, and 3 for 3D data) for convenient BN implementation.

    example:

    ```py
    # Define your neural network architecture with batch normalization
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Flatten(),                   # Flatten the input image tensor
                nn.Linear(28 * 28, 64),         # Fully connected layer from 28*28 to 64 neurons
                nn.BatchNorm1d(64),             # Batch normalization for stability and faster convergence
                nn.ReLU(),                      # ReLU activation function
                nn.Linear(64, 32),              # Fully connected layer from 64 to 32 neurons
                nn.BatchNorm1d(32),             # Batch normalization for stability and faster convergence
                nn.ReLU(),                      # ReLU activation function
                nn.Linear(32, 10)               # Fully connected layer from 32 to 10 neurons (for MNIST classes)
            )

        def forward(self, x):
            return self.layers(x)
    ```

    BN 放在 ReLU 之前和之后的区别：

    * BN 在 ReLU 之前（更常见的情况）：

        * 数据分布更对称

            ```py
            # BN 先将输入规范化为 ~N(0,1)
            # 这样 ReLU 激活时，约50%的神经元会被激活
            normalized = BN(linear_output)  # ~N(0,1)
            activated = ReLU(normalized)    # 一半为0，一半为正
            ```

        * 避免ReLU的Dead Neuron问题

            如果某些神经元输出总是负值，ReLU会使其完全失活, BN先进行归一化，减少这种情况

        * 与原始论文一致

            Batch Normalization 原始论文推荐放在激活函数之前

    * BN在ReLU之后:

        * 激活值直接归一化

            ```py
            activated = ReLU(linear_output)  # 都是非负数
            normalized = BN(activated)       # 归一化非负分布
            ```

        * 直接对激活后的值进行归一化, 可能在某些情况下更稳定

    两种顺序性能差异通常很小，可能因网络架构、数据集而异。


    对于某些激活函数，比如 Sigmoid/Tanh，顺序可能更重要，BN 在前可以防止饱和。对于 Leaky ReLU：两种顺序差异可能更小

    BN 可能有害的情况:

    1. 小批量大小（Small Batch Size）

        ```py
        # 当 batch_size 很小时
        batch_size = 2  # 或者 4, 8
        nn.BatchNorm1d(64)  # 这时候BN的统计估计不可靠，可能损害性能
        ```

    2. RNN/LSTM 等序列模型

        在RNN中BN很难用，通常用LayerNorm代替, 因为序列长度变化，BN统计不稳定, 数据分布一直在变，BN的running stats跟不上

    3. 噪声敏感的任务

        在一些对噪声敏感的任务中, BN引入的随机性（来自batch统计）可能有害

    4. 某些生成模型

        GANs中BN有时会导致模式崩溃, 很多现代GAN用LayerNorm或InstanceNorm代替

    BN 更好用的情况：

    - 大型数据集（ImageNet等）
    - 足够大的batch_size（32+）
    - 卷积网络/MLP
    - 稳定的数据分布

    Benefits of Batch Normalization

    * Faster Convergence: By stabilizing the gradients, BN allows you to use higher learning rates, which can significantly speed up training.
    
    * Reduced Internal Covariate Shift: As the network trains, the distribution of activations within a layer can change (internal covariate shift). BN helps mitigate this by normalizing activations before subsequent layers, making the training process less sensitive to these shifts.

    * Initialization Insensitivity: BN makes the network less reliant on the initial weight values, allowing for more robust training and potentially better performance.

* Apply a 2D Max Pooling in PyTorch

    There are two main types of pooling used in deep learning: Max Pooling and Average Pooling.

    Max Pooling: Max Pooling selects the maximum value from each set of overlapping filters and passes this maximum value to the next layer. This helps to retain the most important feature information while reducing the size of the representation.

    Average Pooling: Average Pooling computes the average value of each set of overlapping filters, and passes this average value to the next layer. This helps to retain a more general form of the feature information, but with a reduced spatial resolution.

    Pooling is usually applied after a convolution operation and helps to reduce overfitting and improve the generalization performance of the model.

* transforms.Compose 能够接收 PIL.Image 类型的对象，是因为它内部组合的各个变换（transform）都实现了对 PIL.Image 的处理逻辑。

    内部的可能实现如下：

    ```py
    # 在 torchvision/transforms/functional.py 中
    def to_tensor(pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        """
        if not(_is_pil_image(pic) or _is_numpy(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if _is_pil_image(pic):
            # 处理PIL.Image的代码路径
            # ... 将PIL图像转为numpy，再转为tensor
        elif _is_numpy(pic):
            # 处理numpy数组的代码路径
            # ... 直接处理numpy数组
        
        return result
    ```

* `SubsetRandomSampler()`

    从一个完整的数据集中，随机地选取一个子集，并且在这个子集上进行无放回地随机采样。

    syntax:

    ```py
    torch.utils.data.SubsetRandomSampler(indices, generator=None)
    ```

    参数详解

    1. indices

        类型: Sequence (序列)

        说明: 这是一个整数索引的序列，用于指定要从原始数据集中抽取哪些样本。

        详细信息:

            它可以是任何 Python 序列类型，如 list, range, numpy.array, torch.Tensor 等。

            索引对应的是原始数据集中的样本位置（从 0 开始）。

            采样器会从这些指定的索引中随机抽取，不会重复抽取同一个索引（无放回抽样）。

            索引的顺序不需要是排序的，也不需要是连续的。

    2. generator

        类型: torch.Generator

        默认值: None

        说明: 用于控制随机数生成的生成器。

        详细信息:

            如果指定了 generator，采样器将使用这个特定的生成器来进行随机打乱。

            如果为 None（默认），采样器将使用默认的随机数生成器。

            这个参数主要用于确保结果的可重现性。当你希望每次运行代码时都能得到相同的随机顺序时，可以传入一个固定种子的生成器。

        example:

        ```py
        # 使用固定种子的生成器以确保可重现性
        generator = torch.Generator().manual_seed(42)
        sampler = SubsetRandomSampler(indices, generator=generator)
        ```

    返回值

        返回一个 SubsetRandomSampler 迭代器对象。

        当在 DataLoader 中迭代时，这个采样器会按照随机顺序逐个返回 indices 中的索引。

        当遍历完所有 indices 后，一个 epoch 就结束了。

    example:

    ```py
    import torch
    from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

    # 1. 创建示例数据集
    data = torch.randn(10, 3)  # 10个样本，每个样本3个特征
    labels = torch.arange(10)  # 10个标签
    dataset = TensorDataset(data, labels)

    # 2. 定义要使用的索引
    indices = [2, 5, 1, 8, 3, 9]  # 只使用这6个样本

    # 3. 创建采样器（带固定生成器以确保可重现性）
    generator = torch.Generator().manual_seed(42)  # 固定随机种子
    sampler = SubsetRandomSampler(indices, generator=generator)

    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        sampler=sampler,  # 使用自定义采样器
        # shuffle=True    # 注意：这里不能设置 shuffle=True！
    )

    # 5. 测试输出
    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Indices: {target.tolist()}")  # 这里target正好是原始索引
        print(f"  Data shape: {data.shape}")
    ```

    使用时要避免在 DataLoader 中设置 shuffle=True。同时，通常也不指定 batch_sampler。

    example: 划分训练集和验证集

    ```py
    import torch
    from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

    # 1. 创建一个示例数据集 (10000个样本)
    dataset = TensorDataset(torch.randn(10000, 10), torch.randint(0, 2, (10000,)))

    # 2. 定义数据集总大小和划分比例
    dataset_size = len(dataset)
    indices = list(range(dataset_size)) # 生成 [0, 1, 2, ..., 9999] 的索引列表
    split = int(0.8 * dataset_size) # 计算划分点：8000

    # 3. 随机打乱索引，以确保划分是随机的
    torch.manual_seed(42) # 设置随机种子以保证结果可复现
    indices.shuffle() # 就地打乱索引列表

    # 4. 创建训练集和验证集的索引子集
    train_indices = indices[:split]   # 前8000个索引作为训练集
    val_indices = indices[split:]     # 后2000个索引作为验证集

    # 5. 创建 SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 6. 创建对应的 DataLoader
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)

    # 现在就可以在训练循环中使用 train_loader，在验证中使用 val_loader 了
    # for data, target in train_loader:
    #     ...
    ```

    与 Subset 的区别：

    * SubsetRandomSampler 是一个 采样器，它作用于 DataLoader 级别。DataLoader 仍然会遍历整个数据集，但采样器告诉它只从指定的索引中取数据。

    * torch.utils.data.Subset 是一个 数据集，它直接返回原始数据集的一个子集。当你使用 Subset 后，得到的就是一个全新的、更小的数据集对象。

    * 如何选择：如果你需要随机打乱，用 SubsetRandomSampler。如果你只是想静态地获取一个子集（不打乱），可以用 Subset。

* PIL 显示 np.array 的图片

    使用 Image.fromarray()

    ```py
    from PIL import Image
    import numpy as np

    # 创建或加载numpy数组
    # 假设你的数组形状为 (height, width, channels) 或 (height, width)
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # 转换为PIL图像
    img = Image.fromarray(array)

    # 显示图像
    img.show()
    ```

    PIL 只接受 uint8 类型：

    ```py
    # 对于不同数据类型的处理
    # uint8 类型 (0-255)
    array_uint8 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img1 = Image.fromarray(array_uint8)

    # float 类型 (0.0-1.0)
    array_float = np.random.rand(100, 100, 3)
    # 需要转换为uint8
    array_float_uint8 = (array_float * 255).astype(np.uint8)
    img2 = Image.fromarray(array_float_uint8)
    ```

    处理灰度图像：

    ```py
    # 灰度图像 (2D数组)
    gray_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    gray_img = Image.fromarray(gray_array)
    gray_img.show()
    ```

    注意：

    * 数据类型：确保numpy数组的数据类型是np.uint8

    * 数值范围：RGB值应该在0-255范围内

    * 数组形状：

        * 彩色图像：(height, width, 3) 或 (height, width, 4)

        * 灰度图像：(height, width)

* 在 jupyter 中显示 PIL 图片

    ```py
    from PIL import Image
    from IPython.display import display

    img = Image.open('example.jpg')
    display(img)  # 在 Jupyter 中直接显示
    ```

* PIL 结合 matplotlib 显示图片

    ```py
    from PIL import Image
    import matplotlib.pyplot as plt

    file = '/home/hlc/Pictures/SAVE_20250313_134654.jpg'

    img = Image.open(file)
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    ```

* 使用 PIL 结合 tk 显示图片

    ```py
    from PIL import Image, ImageTk
    import tkinter as tk

    file = '/home/hlc/Pictures/SAVE_20250313_134654.jpg'

    # 创建主窗口
    root = tk.Tk()
    root.title("PIL Image Display")

    # 打开图片
    img = Image.open(file)

    # 转换为 Tkinter 兼容的格式
    tk_img = ImageTk.PhotoImage(img)

    # 创建标签显示图片
    label = tk.Label(root, image=tk_img)
    label.pack()

    # 运行主循环
    root.mainloop()
    ```

* PIL (Python Imaging Library) 显示图片

    安装：`pip install Pillow`

    ```py
    from PIL import Image

    img = Image.open('/home/hlc/Pictures/SAVE_20250313_134654.jpg')
    img.show()
    ```

    这个方法会将图片保存为一个临时文件（通常是 PNG 格式）, 然后使用操作系统默认的图片查看器打开该文件.

    在 Windows 上通常用"照片"应用，在 macOS 上用"预览"，在 Linux 上用 xdg-open

* 使用多种优化式的 training 过程

    ```py
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(784, 128)
            self.fc2 = torch.nn.Linear(128, 128)
            self.fc3 = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.softmax(self.fc3(x), dim=1)
            return x

    net = Net()

    criterion = torch.nn.CrossEntropyLoss()

    # SGD optimizer
    optimizer_sgd = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Adam optimizer
    optimizer_adam = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer_adam = torch.optim.Adam(net.parameters(), lr=0.001)

    # Adagrad optimizer
    optimizer_adagrad = torch.optim.Adagrad(net.parameters(), lr=0.01)

    # Adadelta optimizer
    optimizer_adadelta = torch.optim.Adadelta(net.parameters(), rho=0.9)

    device = 'cpu'

    # Train the neural network using different optimization algorithms
    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # optimizer_sgd.zero_grad()
            optimizer_adam.zero_grad()
            # optimizer_adagrad.zero_grad()
            # optimizer_adadelta.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # optimizer_sgd.step()
            optimizer_adam.step()
            # optimizer_adagrad.step()
            # optimizer_adadelta.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch: %d | Loss: %.3f | Accuracy: %.3f %%' %
              (epoch + 1, running_loss / len(trainloader), 100 * correct / total))
    ```

    output:

    ```
    Epoch: 1 | Loss: 1.618 | Accuracy: 85.717 %
    Epoch: 2 | Loss: 1.545 | Accuracy: 91.893 %
    Epoch: 3 | Loss: 1.526 | Accuracy: 93.702 %
    Epoch: 4 | Loss: 1.517 | Accuracy: 94.523 %
    Epoch: 5 | Loss: 1.511 | Accuracy: 95.153 %
    Epoch: 6 | Loss: 1.506 | Accuracy: 95.550 %
    Epoch: 7 | Loss: 1.503 | Accuracy: 95.872 %
    Epoch: 8 | Loss: 1.501 | Accuracy: 96.028 %
    Epoch: 9 | Loss: 1.500 | Accuracy: 96.173 %
    Epoch: 10 | Loss: 1.497 | Accuracy: 96.412 %
    ```

* adam 优化器的学习率应该设置为 sgd 的 1/10，比如 sgd 为 0.01，adam 应该设置为 0.001

* 下载 mnist 数据集

    ```py
    import torchvision
    from torchvision.datasets.utils import download_url
    import os

    mirror = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    root = "./data/MNIST/raw"
    os.makedirs(root, exist_ok=True)

    files = {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }

    for filename, md5 in files.items():
        download_url(mirror + filename, root, filename, md5)
    ```

    手动解压：

    ```bash
    cd ./data/MNIST/raw
    gunzip *.gz
    ```

    此时再运行：

    ```py
    import torchvision

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False)
    ```

    即可正常使用。

* convolution

    $$(f ∗ g) (t)= \int_{-\infty}^{\infty} ​f(\tau) g(t − \tau) d\tau$$

    Where f and g are functions representing the image and the filter respectively, and * denotes the convolution operator.

* Batch Processing for Efficient Training

    ```py
    for epoch in range(2):  
        for inputs, labels in dataloader:
            
            outputs = inputs + 1  
            print(f"Epoch {epoch + 1}, Inputs: {inputs}, Labels: {labels}, Outputs: {outputs}")
    ```

    不清楚为啥 outputs 会是 inputs + 1。这个看上去只是个矩阵所有元素加一，而且也并不是序列数据，比如 target = input + 1。而且这个也不像 c 语言的 ptr -> ptr + 1 就可以拿到下个数据。

    这一步可能和上一步的 data aug 结合的，如果能找到上一步 data aug 的代码，可以跑跑看，创建出来 dataloader 后，就可以看到 outputs 和 inputs 的内容了。

* imdb 二分类 example

    ```py
    from datasets import load_dataset
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              TrainingArguments,
                              Trainer)
    import numpy as np
    from sklearn.metrics import accuracy_score

    # 1. 加载数据集和分词器
    dataset = load_dataset("imdb")
    model_checkpoint = "distilbert-base-uncased" # 选择一个轻量且高效的模型，例如 DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 2. 对数据集进行分词处理
    def tokenize_function(examples):
        # 对文本进行分词 truncation 和 padding
        # 这里设置最大长度，超过的部分会被截断
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    # 使用 map 函数批量处理整个数据集
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 为了节省时间和内存，我们创建一个更小的子集进行演示（可选）
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # 3. 加载预训练模型
    # num_labels=2 表示二分类
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    # 4. 定义评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./my_imdb_model",      # 输出目录，模型和检查点会保存在这里
        evaluation_strategy="epoch",       # 每个 epoch 结束后进行评估
        learning_rate=2e-5,                # 学习率
        per_device_train_batch_size=16,    # 训练批次大小
        per_device_eval_batch_size=16,     # 评估批次大小
        num_train_epochs=3,                # 训练轮数
        weight_decay=0.01,                 # 权重衰减
    )

    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset, # 使用子集，完整训练请用 tokenized_datasets["train"]
        eval_dataset=small_eval_dataset,   # 使用子集，完整评估请用 tokenized_datasets["test"]
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # 确保分词器在保存模型时也被保存
    )

    # 7. 开始训练！
    trainer.train()

    # 8. 在测试集上评估模型（使用我们创建的小子集）
    final_metrics = trainer.evaluate(small_eval_dataset)
    print(f"\n最终评估结果: {final_metrics}")

    # 9. 保存模型（可选）
    # trainer.save_model("./my_final_imdb_model")
    ```

* RNN (循环神经网络) 

    RNN是一种专门用于处理序列数据的神经网络。它的核心思想是：网络能对序列中的元素进行循环操作，且能够通过内部状态（隐藏状态）记住之前的信息，并利用这些信息来影响后续的输出。

    核心特征：

    * “循环”与“记忆”：RNN单元不仅接收当前的输入（如句子中的一个词），还接收来自上一个时间步的隐藏状态（Hidden State）。这个隐藏状态充当了网络的“记忆”，它包含了之前所有时间步的序列信息。

    * 参数共享：RNN在每个时间步上使用相同的权重参数（U, W, V）。这使得模型可以处理不同长度的序列，并减少需要训练的参数数量。

    * 计算过程：

        * 在任意时间步 $t$：

            * 新的隐藏状态 $h_t$ 由当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 共同计算得出：$h_t = \tanh(W \cdot h_{t-1} + U \cdot x_t + b)$

            * 输出 $o_t$ 由当前隐藏状态 $h_t$ 计算得出：$o_t = \mathrm{softmax}(V \cdot h_t + c)$

    * 常见问题：

        梯度消失/爆炸（Vanishing/Exploding Gradients）：在处理长序列时，RNN难以学习到远距离时间步之间的依赖关系，因为梯度在反向传播过程中会指数级地减小或增大。

    example:

    ```py
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. 生成正弦波序列数据
    def generate_sine_wave_data(seq_length=50, num_samples=1000):
        """
        生成训练数据：用前seq_length个点预测第seq_length+1个点
        X: [num_samples, seq_length, 1]
        y: [num_samples, 1]
        """
        time_steps = np.linspace(0, 100, num_samples + seq_length)
        data = np.sin(time_steps)
        data = data.reshape(-1, 1) # 转换为特征维度为1

        X = []
        y = []
        for i in range(num_samples):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)

    # 生成数据
    seq_length = 10
    X, y = generate_sine_wave_data(seq_length, 1000)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # 划分训练集和测试集
    train_ratio = 0.8
    train_size = int(train_ratio * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # 3. 定义简单的RNN模型
    class SinePredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, output_size=1):
            super(SinePredictor, self).__init__()
            self.hidden_size = hidden_size
            # 使用一个RNN层
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            # 全连接层用于输出预测
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # x的形状: (batch_size, seq_length, input_size)
            # out: 所有时间步的隐藏状态 (batch_size, seq_length, hidden_size)
            # hidden: 最后一个时间步的隐藏状态 (1, batch_size, hidden_size)
            out, hidden = self.rnn(x)
            # 我们只使用最后一个时间步的隐藏状态来进行预测
            out = self.fc(out[:, -1, :]) # 取序列的最后一个输出
            return out

    # 初始化模型、损失函数和优化器
    model = SinePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练模型
    num_epochs = 100
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    # 5. 评估模型并可视化
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train)
        test_predictions = model(X_test)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    # 绘制一部分测试集上的真实值和预测值
    plt.subplot(1, 2, 2)
    # 取前100个测试点进行绘制
    plt.plot(y_test[:100].numpy(), label='True Value', alpha=0.7)
    plt.plot(test_predictions[:100].numpy(), label='Prediction', alpha=0.7)
    plt.title('Sine Wave Prediction on Test Set')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印最终训练损失和测试损失
    with torch.no_grad():
        test_loss = criterion(test_predictions, y_test)
    print(f'Final Training Loss: {train_losses[-1]:.6f}')
    print(f'Final Test Loss: {test_loss.item():.6f}')
    ```

    output:

    ```
    Epoch [10/100], Loss: 0.302159
    Epoch [20/100], Loss: 0.080454
    Epoch [30/100], Loss: 0.047464
    Epoch [40/100], Loss: 0.025511
    Epoch [50/100], Loss: 0.005334
    Epoch [60/100], Loss: 0.002867
    Epoch [70/100], Loss: 0.001352
    Epoch [80/100], Loss: 0.001240
    Epoch [90/100], Loss: 0.000831
    Epoch [100/100], Loss: 0.000813
    Final Training Loss: 0.000813
    Final Test Loss: 0.000799
    ```

    代码说明：

    * 数据生成：我们生成了一个正弦波，并创建了输入-输出对。每个输入是一个长度为seq_length的序列，输出是序列后的下一个值。

    * 模型定义：

        * `nn.RNN`层是核心，它处理输入序列并返回所有时间步的输出和最后一个隐藏状态。

        * 我们只使用了最后一个时间步的隐藏状态（out[:, -1, :]）并通过一个全连接层(nn.Linear)来生成最终的预测值。这是一种常见的做法，适用于“多对一”的序列任务。

    * 训练：使用均方误差（MSE）作为损失函数，Adam作为优化器。

    * 评估：模型在测试集上进行预测，并绘制结果图。你会看到预测曲线（橙色）能够很好地跟随真实正弦曲线（蓝色）。

    注：

    1. 画出来的图是 [0, 100]，实际上给出的是 y_test 和 y_pred 的最后 100 个数据，并不是 x 数据的 0 到 100，所以 sin 图像只有 1 个半的波长

    1. `time_steps = np.linspace(0, 100, num_samples + seq_length)`，其中的`num_samples + seq_length`表示，x 一共有`num_samples`个，但`x_i`并不是标量，而是一个长度为`seq_length`的向量，`y_i`则为`x_i[0]`后的第`seq_length + 1`个数，是个标量。

        因此为了 x 起始位置共有`num_sample`个，而 y 的最大值则需要比 y 再多`seq_length`个。这就是所有需要用到的数据。

* 随机梯度下降（SGD）

    沿着损失函数梯度的反方向更新参数，从而最小化损失函数。

    基本SGD（无动量）:

    对于一组可学习的参数（权重）$\theta$，损失函数为 $J(\theta)$，学习率为 $\eta$。

    在每一步（每个batch）$t$，基本的SGD更新规则为：

    $\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J_t (\theta_t)$

    其中：

    * $\theta_t$ 是第 `t` 步（迭代）时的参数值。

    * $\nabla_\theta J_t(\theta_t)$ 是第 `t `步损失函数 $J_t$ 关于参数 $\theta$ 的梯度（在当前 batch 上计算得出）。

    * $\eta$ 是学习率（learning rate），控制每次更新的步长。

* `tens.index_copy_()`

    将指定维度上的指定索引（可以是多个）复制到`tens`的对应位置。

    syntax:

    ```py
    index_copy_(dim, index, tensor) -> Tensor
    ```

    example:

    ```py
    import torch

    tens_1 = torch.ones(4, 4)
    tens_2 = torch.randn(2, 4)
    my_indices = torch.tensor([1,3])

    tens_1.index_copy_(0, my_indices, tens_2)
    print("tens_1: {}".format(tens_1))
    ```

    output:

    ```
    tens_1: tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 0.3654,  0.9840, -0.4651,  1.4270],
            [ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 0.0722, -1.2526, -0.8574, -1.2249]])
    ```

    注意：

    * `my_indices`元素的数量必须和`tens_2`在`dim`维度上的长度对应，即`my_indices.size() == tens_2.shape[dim]`。上面例子中，如果`tens_2 = torch.randn(3, 4)`，则会报错。

    `index_copy()`是其 out-of-place 版本。

* torch 拟合 xor 函数

    ```py
    import torch
    import torch.nn as nn
    from torch import optim

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
    X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # Instantiate the Model, Define Loss Function and Optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        model.train()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        predictions = model(test_data)
        print(f'Predictions:\n{predictions}')
    ```

    output:

    ```
    Epoch [10/100], Loss: 0.2205
    Epoch [20/100], Loss: 0.1844
    Epoch [30/100], Loss: 0.1600
    Epoch [40/100], Loss: 0.1357
    Epoch [50/100], Loss: 0.1115
    Epoch [60/100], Loss: 0.0890
    Epoch [70/100], Loss: 0.0671
    Epoch [80/100], Loss: 0.0481
    Epoch [90/100], Loss: 0.0320
    Epoch [100/100], Loss: 0.0199
    Predictions:
    tensor([[0.1897],
            [0.9428],
            [0.8315],
            [0.0905]])
    ```

    说明：

    1. `super(SimpleNN, self).__init__()`与`super().__init__()`是等价的

    1. `model.train()`将模型切换为训练模式，不需要写成`model = model.train()`

        特点：

        * Dropout层会随机丢弃神经元

        * BatchNorm层使用当前批次的统计量（均值和方差）

        * 启用梯度计算（autograd）

        * 适合训练阶段使用

    1. `model.eval()`将模型切换为评估模式

        * Dropout层不会丢弃神经元（所有神经元都参与计算）

        * BatchNorm层使用训练阶段学到的运行统计量

        * 通常与torch.no_grad()一起使用来禁用梯度计算

        * 适合测试、验证和推理阶段使用

* 带自回归的 Encoder-Decoder 架构

    一种用于处理序列到序列（Seq2Seq） 任务的深度学习模型框架。它的核心思想是将一个输入序列（如一句英文句子）转换为一个输出序列（如对应的中文句子），并且输出序列的生成是逐步、自回归地进行的。

    * Encoder（编码器）：

        * 作用：读取并理解整个输入序列。

        * 工作方式：它接收整个输入序列（例如 “I love machine learning”），并通过神经网络（通常是 RNN, LSTM, GRU 或 Transformer）将其压缩成一个固定维度的上下文向量（Context Vector） 或一组隐藏状态。这个向量/状态集旨在包含输入序列的全部语义信息。

    * Decoder（解码器）：

        * 作用：根据编码器的信息和已生成的部分输出，逐步生成完整的输出序列。

        * 工作方式：解码器的生成过程是自回归的（Autoregressive）。这是最关键的一点。

            * 自回归：意味着在生成输出序列的每一个新词（或 token）时，都会将之前已经生成的所有词作为额外输入。

            * 具体步骤：

                1. 解码器从编码器得到的上下文向量和一個特殊的开始符（如 <start>） 开始。
                
                2. 它产生第一个输出词（如 “我”）。

                3. 然后，它将这个刚刚生成的词“我”（而不是真实的目标词）和当前的隐藏状态一起作为输入，来生成下一个词“爱”。

                4. 如此循环，每次生成都依赖于之前的输出，直到生成一个特殊的结束符（如 `<end>`） 表示生成为止。

    简单比喻：

    就像一个同声传译员。

    * Encoder：听完整句英文，并理解其含义。

    * Decoder：开始用中文翻译，每说一个词（“我”），都会参考自己刚才说的词和听到的英文原意，来决定下一个词说什么（“爱”），直到翻译完整个句子。

    相关的模型：

    * RNN-based Seq2Seq (2014)

        由 Sutskever 等人和 Bahdanau 等人提出。

        使用RNN或LSTM作为Encoder和Decoder的核心。最初的模型将整个输入序列压缩成一个固定的上下文向量，这在处理长序列时会造成信息瓶颈。

        改进：注意力机制（Attention Mechanism） 被引入（Bahdanau et al.），允许解码器在生成每个词时“回头看”编码器的所有隐藏状态，从而动态地获取最相关的信息，极大提升了长序列的处理能力。（注意：带注意力的Seq2Seq是极其重要的变体）

    * transformer (2017)

        由 Vaswani 等人在论文《Attention Is All You Need》中提出。

        完全基于自注意力机制（Self-Attention） 的模型，彻底抛弃了RNN。它仍然是Encoder-Decoder架构，但其编码和解码的方式发生了革命性变化。

        Encoder：由多层自注意力和前馈网络组成，并行处理整个输入序列。

        Decoder：同样是自回归的，但在自注意力层中加入了掩码（Mask），确保在生成位置 i 的词时，只能看到位置 1 到 i-1 的词，而不能看到“未来”的信息。

    * 基于Transformer的著名模型（都属于此架构）

        * GPT 系列：严格来说，GPT是只有Decoder的模型。它通过掩码自注意力实现自回归生成，可以看作是Decoder-only架构，但其核心思想——自回归生成——与Encoder-Decoder中的Decoder部分完全相同。

        * BART 和 T5：这些是经典的、真正的带自回归Decoder的Encoder-Decoder模型。它们在预训练时专门为此架构设计（如通过去噪、文本填充等任务），在摘要、翻译、问答等任务上表现卓越。

        * 现代大语言模型（LLMs）：如 ChatGPT 背后的模型，虽然其基础（GPT）是Decoder-only，但其通过指令微调（Instruction Tuning）和人类反馈强化学习（RLHF）学会了很多“理解-生成”的对话能力，其生成回复的过程就是典型的自回归方式。

    * 奠基性论文：

        * Seq2Seq 开创：Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In NeurIPS. [必读]

        * 注意力机制：Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473. [必读]

        * Transformer：Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In NeurIPS. [必读中的必读]

* 可以跑通的 pytorch example

    ```py
    import torch as t
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def main():
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        batch_size = 64

        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        # Define model
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(28*28, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                )

            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits

        model = NeuralNetwork().to(device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("Done!")

        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load("model.pth", weights_only=True))

        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    Using cuda device
    Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    Shape of y: torch.Size([64]) torch.int64
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    Epoch 1
    -------------------------------
    loss: 2.301282  [   64/60000]
    loss: 2.282217  [ 6464/60000]
    loss: 2.261925  [12864/60000]
    loss: 2.257128  [19264/60000]
    loss: 2.247277  [25664/60000]
    loss: 2.211387  [32064/60000]
    loss: 2.218867  [38464/60000]
    loss: 2.184250  [44864/60000]
    loss: 2.178685  [51264/60000]
    loss: 2.146116  [57664/60000]
    Test Error: 
     Accuracy: 52.2%, Avg loss: 2.137231 

    Epoch 2
    -------------------------------
    loss: 2.150523  [   64/60000]
    loss: 2.139497  [ 6464/60000]
    loss: 2.077158  [12864/60000]
    loss: 2.098047  [19264/60000]
    loss: 2.051788  [25664/60000]
    loss: 1.977449  [32064/60000]
    loss: 2.012526  [38464/60000]
    loss: 1.926008  [44864/60000]
    loss: 1.933322  [51264/60000]
    loss: 1.853627  [57664/60000]
    Test Error: 
     Accuracy: 60.0%, Avg loss: 1.850576 

    Epoch 3
    -------------------------------
    loss: 1.884275  [   64/60000]
    loss: 1.859825  [ 6464/60000]
    loss: 1.733056  [12864/60000]
    loss: 1.781410  [19264/60000]
    loss: 1.680241  [25664/60000]
    loss: 1.617407  [32064/60000]
    loss: 1.645341  [38464/60000]
    loss: 1.538832  [44864/60000]
    loss: 1.571115  [51264/60000]
    loss: 1.457203  [57664/60000]
    Test Error: 
     Accuracy: 62.4%, Avg loss: 1.475583 

    Epoch 4
    -------------------------------
    loss: 1.537457  [   64/60000]
    loss: 1.513721  [ 6464/60000]
    loss: 1.354834  [12864/60000]
    loss: 1.441262  [19264/60000]
    loss: 1.327532  [25664/60000]
    loss: 1.310910  [32064/60000]
    loss: 1.334382  [38464/60000]
    loss: 1.248879  [44864/60000]
    loss: 1.292152  [51264/60000]
    loss: 1.186263  [57664/60000]
    Test Error: 
     Accuracy: 64.9%, Avg loss: 1.212287 

    Epoch 5
    -------------------------------
    loss: 1.276597  [   64/60000]
    loss: 1.273734  [ 6464/60000]
    loss: 1.098410  [12864/60000]
    loss: 1.221964  [19264/60000]
    loss: 1.097947  [25664/60000]
    loss: 1.114543  [32064/60000]
    loss: 1.145893  [38464/60000]
    loss: 1.072613  [44864/60000]
    loss: 1.119054  [51264/60000]
    loss: 1.029024  [57664/60000]
    Test Error: 
     Accuracy: 66.1%, Avg loss: 1.050324 

    Done!
    Saved PyTorch Model State to model.pth
    Predicted: "Ankle boot", Actual: "Ankle boot"
    ```

## topics

### 内存布局，GPU 与 CPU

* 使用 permute 导致 tensor 变成 continuous 的例子

    ```py
    import torch as t

    a = t.rand(3, 4)
    print('a shape: {}'.format(a.shape))
    a = a.permute(1, 0)
    print('after permute, a shape: {}'.format(a.shape))
    print('is continuous: {}'.format(a.is_contiguous()))
    a = a.view(2, 6)
    ```

    output:

    ```
    a shape: torch.Size([3, 4])
    after permute, a shape: torch.Size([4, 3])
    is continuous: False
    Traceback (most recent call last):
      File "/home/hlc/Documents/Projects/torch_test/main.py", line 8, in <module>
        a = a.view(2, 6)
    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    ```

* 将 tensor 从 cpu 转移到 gpu

    * 推荐接口`.to()`

        ```py
        import torch

        # 假设有一个在 CPU 上的 tensor
        cpu_tensor = torch.tensor([1, 2, 3])
        print(cpu_tensor.device) # 输出：cpu

        # 检查 GPU 是否可用
        if torch.cuda.is_available():
            device = torch.device("cuda") # 指定目标设备为 GPU
            gpu_tensor = cpu_tensor.to(device) # 转移到 GPU
            print(gpu_tensor.device) # 输出：cuda:0

            # 你也可以直接使用字符串
            gpu_tensor_2 = cpu_tensor.to('cuda')
        ```

    * 旧兼容接口`.cuda()`

        ```py
        if torch.cuda.is_available():
            gpu_tensor = cpu_tensor.cuda() # 转移到默认 GPU (cuda:0)
            gpu_tensor = cpu_tensor.cuda(0) # 明确转移到第一个 GPU
        ```

    在创建时指定设备：

    ```py
    # 直接在 GPU 上创建 tensor，省去转移步骤
    gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
    # 或者
    gpu_tensor = torch.tensor([1, 2, 3]).to('cuda')
    ```

* list 在 append tensor 时，需要 tensor clone()，否则 append 的都是 tensor 的引用，值都是一模一样的

    `params_record.append(param.clone().detach())`

* 将 tensor 数据放到 gpu 里

    ```py
    # 检查设备
    print("Tensor 设备:", torch_tensor.device)

    # 如果需要，移动到 GPU
    if torch.cuda.is_available():
        torch_tensor = torch_tensor.cuda()
    ```

* `tensor.view()`和`tensor.reshape()`都是浅拷贝，`reshape()`可能是深拷贝

    ```py
    import torch

    # 原始张量
    original_tensor = torch.arange(6)  # tensor([0, 1, 2, 3, 4, 5])
    reshaped_tensor = original_tensor.view(2, 3)

    # 修改reshape后的张量
    reshaped_tensor[0, 0] = 100

    print(original_tensor)  # tensor([100,   1,   2,   3,   4,   5])
    print(reshaped_tensor)  # tensor([[100,   1,   2],
                            #         [  3,   4,   5]])
    ```

    output:

    ```
    tensor([100,   1,   2,   3,   4,   5])
    tensor([[100,   1,   2],
            [  3,   4,   5]])
    ```

    可以看到，修改 reshaped_tensor 也会影响 original_tensor，因为它们共享底层数据存储。

    如果原始张量在内存中不是连续的，view() 可能会失败，此时需要使用 reshape()：

    ```py
    # 转置操作会创建不连续的张量
    non_contiguous = original_tensor.t()  # 转置

    # 可能会报错
    reshaped = non_contiguous.view(2, 3)
    print('view reshaped: {}'.format(reshaped))

    # 应该使用reshape()
    reshaped = non_contiguous.reshape(2, 3)  # 同样也是浅拷贝
    print('reshape reshaped: {}'.format(reshaped))
    ```

    output:

    ```
    view reshaped: tensor([[0, 1, 2],
            [3, 4, 5]])
    reshape reshaped: tensor([[0, 1, 2],
            [3, 4, 5]])
    ```

    目前看到使用 view 也没有报错，不清楚为什么。

    如果需要深拷贝，可以使用 clone() 方法：

    ```py
    # 创建真正的深拷贝
    deep_copy = original_tensor.view(2, 3).clone()

    # 修改深拷贝不会影响原始张量
    deep_copy[0, 0] = 999
    print(original_tensor)  # 不会被修改
    ```

    首先`.view()`一定是浅拷贝。对于`.reshape()`，如果张量是 连续的，reshape() 内部直接调用 view()；如果张量是 非连续的（例如经过 transpose），reshape() 会先调用 .contiguous()，把数据整理成标准布局（开辟新内存、复制数据），此时会发生深拷贝，然后再调用 view()。

* permute 和 transpose 都是只交换维度，不改变底层数据，所以会造成 tensor 不连续

* 关于`tensor.view()`与内存的讨论

    * view() 在 PyTorch 中只是改变张量的 视图，不做实际的数据拷贝，因此要求底层内存是 连续的 (contiguous)。如果原始张量不是连续的（例如经过 transpose、permute 等操作），直接调用 view() 就会报错。

    * reshape() 更灵活：它会尝试返回一个 view，但如果数据在内存中不连续，它会自动做一次拷贝，把数据整理成连续的，再返回结果。因此 reshape() 一定能成功（只要新形状是合法的）。

    example:

    ```py
    import torch

    # 创建一个 2x3 张量
    a = torch.arange(6).reshape(2, 3)
    print("原始 a:\n", a)

    # 转置，得到非连续内存的张量
    b = a.t()   # transpose
    print("转置 b:\n", b)
    print("b 是否连续:", b.is_contiguous())  # False

    # 尝试 view
    try:
        aaa = b.view(-1)
        print('aaa: {}'.format(aaa))
    except RuntimeError as e:
        print("view 报错:", e)

    # 使用 reshape 则没问题
    c = b.reshape(-1)
    print("reshape 成功:", c)
    ```

    output:

    ```
    原始 a:
     tensor([[0, 1, 2],
            [3, 4, 5]])
    转置 b:
     tensor([[0, 3],
            [1, 4],
            [2, 5]])
    b 是否连续: False
    view 报错: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    reshape 成功: tensor([0, 3, 1, 4, 2, 5])
    ```
    
    PyTorch Tensor 底层由两个关键部分组成：

    1. Storage（存储区）

        使用一块连续的内存（1D array），存放所有元素。不使用链表或分散块存储。

        即使是多维张量，本质上还是在一维数组里。

    2. Tensor 元信息：size + stride

        * size：每一维的长度。

        * stride：每一维跨越的步长（在内存里隔多少元素算一步）。

        例子：

        shape 为 (2, 3) 的张量，stride = (3, 1)。

        如果我们对其进行转置（transpose），那么 torch 会实行一个 trick，即只交换维度信息，不改变底层数据，此时 stride 会变成 (1, 3)，我们通过索引`arr[m][n]`可以正确访问到转置后的数据，但是此时它已经不再是先行后列的含义了，因此不连续。

        如果我们改变底层数据，使它是连续的，那么转置后的 tensor，shape 为 (3, 2)，stride 为 (2, 1)。

        `stride[i]`表示在第 i 维上 索引加 1，在底层 1D 存储里需要移动多少个元素。

    下面的代码解释了 torch 中 transpose() 的 trick:

    ```py
    import numpy as np

    class Arr:
        def __init__(self, arr, m: int, n: int):
            self.arr = arr
            self.shape = [m, n]
            self.stride = [n, 1]

        def view(self, m: int, n: int):
            self.shape = [m, n]
            self.stride = [n, 1]

        def transpose(self):
            self.shape = [self.shape[1], self.shape[0]]
            self.stride = [1, self.stride[0]]

        def get(self, i, j):
            return self.arr[i * self.stride[0] + j * self.stride[1]]

    def print_arr(arr: Arr):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                print('{}, '.format(arr.get(i, j)), end='')
            print()
        print()
        return

    def main():
        data = np.arange(3 * 4)
        arr = Arr(data, 3, 4)

        print('arr (3 x 4):')
        print_arr(arr)

        arr.view(4, 3)
        print('arr (4 x 3):')
        print_arr(arr)

        arr.view(3, 4)  # back to original state
        arr.transpose()
        print('arr transposed (4 x 3):')
        print_arr(arr)

        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    arr (3 x 4):
    0, 1, 2, 3, 
    4, 5, 6, 7, 
    8, 9, 10, 11, 

    arr (4 x 3):
    0, 1, 2, 
    3, 4, 5, 
    6, 7, 8, 
    9, 10, 11, 

    arr transposed (4 x 3):
    0, 4, 8, 
    1, 5, 9, 
    2, 6, 10, 
    3, 7, 11,
    ```

    如果我们需要将这种非连续的底层数据变成连续的，那么可以调用`.contiguous()`方法将其变成连续的。

### tensor 运算 / tensor 操作

* index_fill_

    'Val' value is filled with the elements of 'x' along with the order of indices given in the vector 'index'.

    syntax:

    ```py
    index_fill_(dim, index, val) → Tensor
    ```

    这个函数中的`val`是个 scalar。

    对应的 out of place 版本：

    `index_fill()`

    `index_put_()`, `index_put()`:

    This operation puts the value of 'val' into the self tensor using the indices of the given 'index'.

    syntax:

    ```py
    index_put_(indices, values, accumulate=False) → Tensor
    ```

    将 value 放到 indices 指定的位置。这里的 value 是个 vector，indices 则是 tensor 中要修改的数据的索引（可能是多维的）。

    example:

    ```py
    #importing libraries
    import torch
     
    target=torch.zeros([4,4])
    indices = torch.LongTensor([[0,1],[1,2],[3,1],[1,0]])#indices to which values to be put
    value = torch.ones(indices.shape[0])
    #tuple of the index tensor is passed along with the value
    target.index_put_(tuple(indices.t()), value)
    ```

    output:

    ```
    tensor([[0., 1., 0., 0.],
           [1., 0., 1., 0.],
           [0., 0., 0., 0.],
           [0., 1., 0., 0.]])
    ```

    如果`accumulate`为 true，那么新元素会叠加到旧元素上。

    `index_select()`:

    A tensor is returned with indices as mentioned, by selecting from the target tensor.

    syntax:

    ```py
    torch.index_select(input, dim, index, out=None) 
    ```
    
    选取指定维度的几行/几列。

    这个操作可以直接用`[:, [y_1, y_2], :]`这种索引方式完成，感觉比较鸡肋。

* `squeeze()`

    移除所有长度为 1 的维度（或者只移除指定维度，如果其长度为 1）。

    example:

    ```py
    # 接上面的例子
    x = torch.randn(1, 4, 1, 2)
    print(f"Original shape: {x.shape}") # torch.Size([1, 4, 1, 2])

    y = x.squeeze() # 移除所有长度为1的维度
    print(f"After squeeze(): {y.shape}") # torch.Size([4, 2])

    z = x.squeeze(0) # 只移除第0维，如果其长度为1
    print(f"After squeeze(0): {z.shape}") # torch.Size([4, 1, 2])

    w = x.squeeze(2) # 只移除第2维，如果其长度为1
    print(f"After squeeze(2): {w.shape}") # torch.Size([1, 4, 2])
    ```

* `unsqueeze()`

    在张量的指定维度上增加一个长度为 1 的维度。这个操作通常也被称为“升维”。

    syntax:

    ```py
    torch.unsqueeze(input, dim) → Tensor
    ```

    * input: 输入张量。

    * dim: 一个整数，指定在哪个位置插入新的维度。这个新维度的长度将为 1。

        dim 的取值范围是 [-input.dim()-1, input.dim()]。

        * 正索引: 从前往后数，0 表示在最前面插入。

        * 负索引: 从后往前数，-1 表示在最后一个维度之后插入。

    这是一个“视图操作”，意味着它通常不会复制底层数据，而只是改变了看待数据的“视角”，因此效率很高。

    例如：

    对于一个 3 维张量 (C, H, W)：

    * dim=0 -> 新形状为 (1, C, H, W)

    * dim=1 -> 新形状为 (C, 1, H, W)

    * dim=-1 -> 新形状为 (C, H, W, 1)

    * dim=-2 -> 新形状为 (C, H, 1, W)

* torch 中的`@`

    Python 中的矩阵乘法运算符，A @ B 等价于 torch.matmul(A, B)。

    PyTorch 通过实现 Python 的特殊方法来自定义运算符行为：

    | 运算符 | Python 特殊方法 |
    | - | - |
    | `@` | `__matmul__`, `__rmatmul__` |
    | `+` | `__add__` |
    | `-` | `__sub__` |
    | `*` | `__mul__` |	
    | `/` | `__truediv__` |

* 使用`random_()`可以将数据初始化为随机值

    example:

    `target = torch.empty(2, dtype=t.long).random_(4)`

    创建一个 shape 为`(2, )`的数组，将其数据初始化为`[0, 4)`的随机值。

* `np.pad()`

    np.pad() 是 NumPy 中用于数组填充（padding）的函数，主要用于在数组边界扩展指定宽度的元素。

    syntax:

    ```py
    numpy.pad(array, pad_width, mode='constant', **kwargs)
    ```

    主要参数：

    * array：要填充的数组

    * pad_width：填充宽度，格式为 ((before_1, after_1), ..., (before_N, after_N))

        对于多维数组，pad_width 的每个元组对应一个维度

    * mode：填充模式，默认为 'constant'

    * constant_values：当模式为 'constant' 时使用的填充值

    常用填充模式

    * constant：常数填充（默认）

    * edge：使用边缘值填充

    * linear_ramp：线性斜坡填充

    * maximum/minimum：使用数组最大/最小值填充

    * mean：使用数组平均值填充

    * median：使用中位数填充

    * reflect/symmetric：反射/对称填充

    example:

    ```py
    import numpy as np

    # 1. 一维数组常数填充
    arr_1d = np.array([1, 2, 3])
    padded = np.pad(arr_1d, (2, 3), mode='constant', constant_values=0)
    # 结果：[0 0 1 2 3 0 0 0]

    # 2. 二维数组不同方向填充
    arr_2d = np.array([[1, 2], [3, 4]])
    # 上下各填充1行，左右各填充2列
    padded = np.pad(arr_2d, ((1, 1), (2, 2)), mode='constant', constant_values=0)

    # 3. 使用不同填充值
    arr = np.array([1, 2, 3])
    # 左侧填充5，右侧填充10
    padded = np.pad(arr, (2, 3), mode='constant', constant_values=(5, 10))

    # 4. 边缘值填充
    arr = np.array([1, 2, 3, 4])
    padded = np.pad(arr, (2, 2), mode='edge')
    # 结果：[1 1 1 2 3 4 4 4]

    # 5. 反射填充
    arr = np.array([1, 2, 3, 4])
    padded = np.pad(arr, (2, 2), mode='reflect')
    # 结果：[3 2 1 2 3 4 3 2]

    # 6. 不同维度不同填充宽度
    arr_2d = np.ones((3, 3))
    pad_width = ((1, 2), (3, 4))  # 第一维：上1行下2行，第二维：左3列右4列
    padded = np.pad(arr_2d, pad_width, mode='constant', constant_values=0)
    ```

    注：

    1. 如果`pad_width`只写一份，那么会在最外层维度上被广播

        example:

        ```py
        import numpy as np

        arr = np.ones((3, 2))
        arr = np.pad(arr, (2, 3))
        print(arr)
        ```

        output:

        ```
        [[0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 1. 1. 0. 0. 0.]
        [0. 0. 1. 1. 0. 0. 0.]
        [0. 0. 1. 1. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0.]]
        ```

        可以看到，在行上是前面添加两行，后面添加三行；在列上是左边添加两列，右边添加三列。

        相当于把`(2, 3)`广播成了`((2, 3), (2, 3))`。

        如果希望被广播成`((2, 2), (3, 3))`，那么可以写成`((2, ), (3, ))`

    1. 对于`constant_value`，后 padding 的会覆盖先 padding 的

        ```py
        import numpy as np

        arr = np.ones((3, 2))
        arr = np.pad(arr, ((2, ), (3, )), constant_values=((2, ), (3, )))
        print(arr)
        ```

        output:

        ```
        [[3. 3. 3. 2. 2. 3. 3. 3.]
        [3. 3. 3. 2. 2. 3. 3. 3.]
        [3. 3. 3. 1. 1. 3. 3. 3.]
        [3. 3. 3. 1. 1. 3. 3. 3.]
        [3. 3. 3. 1. 1. 3. 3. 3.]
        [3. 3. 3. 2. 2. 3. 3. 3.]
        [3. 3. 3. 2. 2. 3. 3. 3.]]
        ```

* tensor 的 indexing, slicing, reshaping 操作

    ```py
    import torch

    tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

    element = tensor[1, 0]
    print(f"Indexed Element (Row 1, Column 0): {element}")
    
    slice_tensor = tensor[:2, :]
    print(f"Sliced Tensor (First two rows): \n{slice_tensor}")

    reshaped_tensor = tensor.view(2, 3)
    print(f"Reshaped Tensor (2x3): \n{reshaped_tensor}")
    ```

    output:

    ```
    Indexed Element (Row 1, Column 0): 3
    Sliced Tensor (First two rows): 
    tensor([[1, 2],
            [3, 4]])
    Reshaped Tensor (2x3): 
    tensor([[1, 2, 3],
            [4, 5, 6]])
    ```

* tensor 的 Broadcasting 和 Matrix Multiplication 操作

    ```py
    import torch

    tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print('tensor a shape: {}'.format(tensor_a.shape))

    tensor_b = torch.tensor([[10, 20, 30]]) 
    print('tensor b shape: {}'.format(tensor_b.shape))

    broadcasted_result = tensor_a + tensor_b 
    print(f"Broadcasted Addition Result: \n{broadcasted_result}")

    matrix_multiplication_result = torch.matmul(tensor_a, tensor_a.T)
    print(f"Matrix Multiplication Result (tensor_a * tensor_a^T): \n{matrix_multiplication_result}")
    ```

    output:

    ```
    tensor a shape: torch.Size([2, 3])
    tensor b shape: torch.Size([1, 3])
    Broadcasted Addition Result: 
    tensor([[11, 22, 33],
            [14, 25, 36]])
    Matrix Multiplication Result (tensor_a * tensor_a^T): 
    tensor([[14, 32],
            [32, 77]])
    ```

* `index_add()`

    It is the out-of place version of the function `index_add_()`.

    example:

    ```py
    import torch

    y = torch.ones(5,5)
    index2 = torch.tensor([0,1,1,1,2])
    ten = torch.randn(5,5)

    print("Indexed Matrix:\n",y.index_add(1,index2,ten))
    print ("Printing Indexed Matrix again:\n",y)
    ```

    output:

    ```
    Indexed Matrix:
     tensor([[ 1.1614,  2.1703,  1.5247,  1.0000,  1.0000],
            [-0.2930,  4.1282,  0.3124,  1.0000,  1.0000],
            [ 0.5624,  0.3906,  3.0302,  1.0000,  1.0000],
            [ 1.7235,  2.3990,  2.5070,  1.0000,  1.0000],
            [ 1.9170,  1.0716, -0.3112,  1.0000,  1.0000]])
    Printing Indexed Matrix again:
     tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])
    ```

    可以看出`index_add()`不修改原 tensor 的数据。

* Index-based Operation

    * `index_add_()`

        Adds the given tensor elements to the self tensor along the order given in the matrix.

        syntax:

        ```py
        index_add_(dim, index, tensor) ---> Tensor
        ```

        params:

        * dim: dimension along which index to add. '0' stands for column and '1' stands for row.

        * index: indices of the tensor to select from. It can be LongTensor or IntTensor.

        * tensor: tensor containing the values to add.

        example:

        ```py
        import torch

        x = torch.zeros(5,5)
        te = torch.tensor([[1,3,5,7,9], [1,3,5,7,9], [1,3,5,7,9]], dtype=torch.float32)
        print('te shape: {}\n'.format(te.shape))
        index0 = torch.tensor([0, 2, 4])

        x.index_add_(0, index0, te) #adding tensor te to x along row of the given order
        print('x:\n{}'.format(x))
        ```

        output:

        ```
        te shape: torch.Size([3, 5])

        x:
        tensor([[1., 3., 5., 7., 9.],
                [0., 0., 0., 0., 0.],
                [1., 3., 5., 7., 9.],
                [0., 0., 0., 0., 0.],
                [1., 3., 5., 7., 9.]])
        ```

        可以看出，是让`te`中的三行数据分别叠加到`x`的`[0, 2, 4]`行上。

        example 2:

        ```py
        import torch

        y = torch.ones(5, 5) # unit vector
        index2 = torch.tensor([0, 1, 1, 1, 2])
        ten = torch.randn(1, 5)

        # adding values to y along the column with given order
        y.index_add_(1, index2, ten)
        print('y is: {}'.format(y))
        ```

        output:

        ```
        Traceback (most recent call last):
          File "/home/hlc/Documents/Projects/torch_test/main.py", line 8, in <module>
            y.index_add_(1, index2, ten)
        RuntimeError: source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = [5, 5] source.shape = [1, 5]
        ```

        可以看出并没有发生 broadcasting。

        可以改成这样：

        ```py
        import torch

        y = torch.ones(5,5) # unit vector
        index2 = torch.tensor([0, 1, 1, 1, 2])
        ten = torch.randn(1, 5)
        ten = ten.expand(5, 5)
        print('ten is: {}'.format(ten))

        # adding values to y along the column with given order
        y.index_add_(1, index2, ten)
        print('y is: {}'.format(y))
        ```

        output:

        ```
        ten is: tensor([[ 0.1083, -0.3369, -0.7591, -0.2532, -0.4060],
                [ 0.1083, -0.3369, -0.7591, -0.2532, -0.4060],
                [ 0.1083, -0.3369, -0.7591, -0.2532, -0.4060],
                [ 0.1083, -0.3369, -0.7591, -0.2532, -0.4060],
                [ 0.1083, -0.3369, -0.7591, -0.2532, -0.4060]])
        y is: tensor([[ 1.1083, -0.3493,  0.5940,  1.0000,  1.0000],
                [ 1.1083, -0.3493,  0.5940,  1.0000,  1.0000],
                [ 1.1083, -0.3493,  0.5940,  1.0000,  1.0000],
                [ 1.1083, -0.3493,  0.5940,  1.0000,  1.0000],
                [ 1.1083, -0.3493,  0.5940,  1.0000,  1.0000]])
        ```

        可以看出，`[0, 1, 1, 1, 2]`表示将`ten`中的五列分别叠加到`y`的第 0, 1, 1, 1, 2 列。

* Tensor 中的转置（Transpose）

    转置是一种改变张量维度（轴）顺序的操作。

    矩阵（一个 2D 张量），它的转置就是沿着主对角线翻转的操作。将矩阵 A 的行和列互换，就得到了它的转置 Aᵀ。

    如果原矩阵 A 的形状是 (m, n)，那么转置后的矩阵 Aᵀ 的形状就是 (n, m)。

    元素的位置关系为：A[i, j] = Aᵀ[j, i]。

    对于维度大于 2 的张量（例如 3D、4D），转置指任意地重新排列张量的所有维度。

    PyTorch 中转置操作是一种“视图操作”，由于不复制数据，原张量和转置后的张量共享同一块内存。修改其中一个的值，另一个也会随之改变。

    1. 默认转置（`.T` 或 `transpose()`）

        在很多框架中，如果不提供参数，.T 属性会默认反转所有维度的顺序。

        `y = x.T`

        新的维度顺序是原顺序的反转：`(2, 1, 0)`

        因此，转置后的形状为：`(original_shape[2], original_shape[1], original_shape[0]) = (4, 3, 2)`

    2. 自定义转置（指定 perm 参数）

        * example 1: 交换最后两个维度

            ```py
            # 假设 x.shape = (2, 3, 4)
            y = x.transpose(0, 2, 1) # 或者 x.permute(0, 2, 1) in PyTorch
            # 新的维度顺序: (0, 2, 1)
            # 新形状: (original_shape[0], original_shape[2], original_shape[1])
            #        = (2, 4, 3)
            ```

        * example 2: 复杂的重新排列

            ```py
            # 假设 x.shape = (2, 3, 4, 5)
            # 我们想要一个新的顺序：将原来的维度 2 放到最前面，然后是维度 0，维度 3，最后是维度 1。
            perm = (2, 0, 3, 1)
            y = x.transpose(perm)
            # 新形状: (original_shape[2], original_shape[0], original_shape[3], original_shape[1])
            #        = (4, 2, 5, 3)
            ```

    numpy 与 torch 的接口函数：

    * numpy

        ```py
        import numpy as np
        x = np.random.rand(2, 3, 4)
        y = x.transpose(0, 2, 1) # 使用 transpose 函数
        z = x.T # 反转所有维度
        ```

    * torch

        ```py
        import torch
        x = torch.randn(2, 3, 4)
        y = x.permute(0, 2, 1) # 常用 permute 函数
        z = x.transpose(1, 2)  # transpose 通常一次只交换两个指定维度，这里是交换维度1和2
        w = x.T # 反转所有维度
        ```

### 数据增广，数据预处理

* torchvision.transforms 中常用的 augmentation 方法：

    * 图像预处理 & 基本变换

        ```py
        # Resize：调整图像尺寸
        transforms.Resize((256, 256))

        # CenterCrop / RandomCrop：中心/随机裁剪
        transforms.RandomCrop(224)

        # Pad：边缘填充
        transforms.Pad(50, fill=255)
        ```

    * 颜色 & 亮度变换

        ```py
        # ColorJitter：随机调整亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        # Grayscale / RandomGrayscale：转灰度图
        transforms.RandomGrayscale(p=0.1)

        # RandomAdjustSharpness / RandomAutocontrast：调整锐度、自动对比度
        ```

    * 几何变换

        ```py
        # RandomHorizontalFlip / RandomVerticalFlip：随机水平/垂直翻转
        transforms.RandomHorizontalFlip(p=0.5)

        # RandomRotation：随机旋转
        transforms.RandomRotation(degrees=30)

        # RandomAffine：随机仿射变换（平移、旋转、缩放、剪切）
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))

        # RandomPerspective：随机透视变换
        ```

    * 模糊 & 噪声

        ```py
        # GaussianBlur：高斯模糊
        transforms.GaussianBlur(kernel_size=5)

        # RandomErasing：随机擦除（CutOut）
        transforms.RandomErasing(p=0.5)
        ```

    * 标准化 & 张量转换

        ```py
        # ToTensor：将PIL图像或NumPy数组转换为张量，并缩放到 [0,1]
        transforms.ToTensor()

        # Normalize：标准化（减均值、除标准差）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ```

    * 组合变换

        使用 Compose 将多个变换组合：

        ```py
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ```

* `ToTensor()`

    1. 数据类型转换：ToTensor() 将 PIL Image 或 numpy.ndarray 转换为 PyTorch Tensor，后续的 transforms 都需要在 Tensor 上操作

    2. 通道顺序：将 H×W×C 转换为 C×H×W，符合 PyTorch 的期望格式

    3. 数值范围：将 [0, 255] 的整数或 [0, 1] 的浮点数转换为 [0.0, 1.0] 的浮点数

    变换前后数据 shape 对比：

    ```py
    # 对于 RGB 图像
    (H, W, C) = (224, 224, 3)
    # 变换后
    (C, H, W) = (3, 224, 224)

    # 对于灰度图像  
    (H, W) = (224, 224)  # 或 (H, W, 1)
    # 变换后
    (1, H, W) = (1, 224, 224)
    ```

* `transforms.Normalize()`

    example:

    `transforms.Normalize((0.5,), (0.5,))`作用如下：

    ```py
    # 对于每个像素值：
    normalized_pixel = (pixel - mean) / std

    # 具体到你的例子：
    normalized_pixel = (pixel - 0.5) / 0.5
    ```

    如果 RGB 三个通道的 mean 和 std 相同，那么可以写成：

    ```py
    transforms.Normalize(mean, std)
    ```

    如果是多通道图像，那么可以写成：

    ```py
    # RGB 图像归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # R, G, B 通道的均值
                             std=[0.5, 0.5, 0.5])   # R, G, B 通道的标准差
    ])
    ```

    为什么要归一化？

    * 训练稳定性：将数据缩放到相似的范围，避免梯度爆炸

    * 收敛速度：帮助优化器更快收敛

    * 模型性能：很多模型假设输入数据是零均值的

    * 数值精度：在 [-1, 1] 范围内计算更稳定

* 在 transform 时，numpy 只能先 to tensor，再 resize，不能先 resize。PIL 图像既可以先 resize，也可以先 to tensor

    * numpy ndarray 只能先 to tensor;

        ```py
        from torchvision import transforms
        import numpy as np

        img = np.random.random((256, 256))

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ])

        img_trans = trans(img)

        print("img shape: {}".format(img.shape))
        print("img trans shape: {}".format(img_trans.shape))
        ```

        output:

        ```
        img shape: (256, 256)
        img trans shape: torch.Size([1, 512, 512])
        ```

        三维的数据也可以处理：

        `img = np.random.random((256, 256, 3))`

        output:

        ```
        img shape: (256, 256, 3)
        img trans shape: torch.Size([3, 512, 512])
        ```

        如果我们设置先 resize，那么会报错：

        ```py
        trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        ```

        output:

        ```
        ...
          File "/home/hlc/miniconda3/envs/torch/lib/python3.10/site-packages/torchvision/transforms/_functional_pil.py", line 31, in get_dimensions
            raise TypeError(f"Unexpected type {type(img)}")
        TypeError: Unexpected type <class 'numpy.ndarray'>
        ```

    * PIL 图片既可以先 resize，也可以先 to tensor:

        ```py
        from torchvision import transforms
        from PIL import Image

        img = Image.open('../example.jpg')

        trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        img_trans = trans(img)

        # print("img shape: {}".format(img.shape))  # PIL Image object has no shape attribute
        print("img trans shape: {}".format(img_trans.shape))
        ```

        output:

        ```
        img trans shape: torch.Size([3, 512, 512])
        ```

        先 to tensor 也是可以的：

        ```py
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ])
        ```

        output:

        ```
        img trans shape: torch.Size([3, 512, 512])
        ```

    如果先做了 to tensor，那么后续操作会在 GPU 里完成（是 CPU 吧？）。如果先做 resize，那么 resize 操作会调用 PIL 提供的 resize 函数。

* PyTorch Functional Transforms for Computer Vision

    Most of the functional transforms accept both PIL images and tensor images. A tensor image is a tensor with shape (C, H, W),

    if the input is a PIL image output is also a PIL image and the same for Tensor image.

    * `adjust_brightness()`

        adjusts the brightness of an image. It accepts both PIL image and Tensor Image.

        syntax:

        ```py
        torchvision.transforms.functional.adjust_brightness(
            img: Tensor,
            brightness_factor: float
        ) -> Tensor
        ```

        * img (Tensor): 输入图像，形状为 (..., H, W) 或 (C, H, W) 或 (H, W)

        * brightness_factor is any non-negative floating-point number:

            * brightness_factor = 1, the original image.

            * brightness_factor < 1, a darker output image.

            * brightness_factor > 1, a brighter output image.

        example:

        ```py
        import torchvision.transforms.functional as F
        import torch
        from PIL import Image

        image = Image.open('nature.jpg')

        output = F.adjust_brightness(image, brightness_factor=3.0)
        output.show()
        ```

        注意事项：

        * 亮度调整是通过将每个像素值乘以 brightness_factor 实现的

        * 结果会被裁剪到图像的原始值范围内（通常是 [0, 1]）

        * 如果输入是 PIL 图像，F.adjust_brightness() 的输出是 Tensor，而不是 PIL 图像。

        与``transforms.Compose`合用的例子：

        ```py
        transform = transforms.Compose([
            transforms.ToTensor(),  # PIL -> Tensor
            lambda x: F.adjust_brightness(x, brightness_factor=1.5),
            transforms.ToPILImage()  # Tensor -> PIL
        ])
        ```

* image augmentation

    ```py
    import torchvision.transforms as transforms
    from PIL import Image

    image = Image.open('example.jpg')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    augmented_image = transform(image)
    print("Augmented Image Shape:", augmented_image.shape)
    ```

    output:

    ```
    Augmented Image Shape: torch.Size([3, 500, 500])
    ```

### 可视化 visualization

* 在`fig, axes = subplots()`时，如果是一行或者一列，那么`axes`是一维的，如果是多行多列，`axes`是二维的。

* matplotlib 画 3d surface 的 example

    ```py
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.font_manager as fm

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    # 创建数据
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # 添加颜色条
    fig.colorbar(surf)

    # 设置标签 - 现在中文可以正常显示
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('3D曲面图示例')

    plt.show()
    ```

* Axes3D

    模块：`mpl_toolkits.mplot3d`

    基本功能 routine：

    1. 创建三维坐标轴

        使用 projection='3d' 参数将一个普通的二维坐标轴转换为三维坐标轴。

        ```py
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # 虽然显式导入有时不需要，但建议保留以确保环境正常

        # 创建图形和三维坐标轴
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 111 表示 1x1 网格的第1个子图

        # 在较新的 Matplotlib 版本中，也可以这样创建：
        # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ```

    2. 基本三维绘图方法

        创建了 Axes3D 对象（通常命名为 ax）后，你可以使用类似二维绘图的方法，但它们接受三维坐标（x, y, z）作为参数。

        * 三维散点图 (Scatter Plot)

            使用 `.scatter(xs, ys, zs)` 方法。

            ```py
            import numpy as np

            # 生成随机数据
            n = 100
            x = np.random.rand(n)
            y = np.random.rand(n)
            z = np.random.rand(n)

            ax.scatter(x, y, z, c=z, cmap='viridis', marker='o') # c=z 表示用 z 值映射颜色
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
            ```

        * 三维线图 (Line Plot)

            使用 .plot(xs, ys, zs) 方法。

            ```py
            # 生成螺旋线数据
            theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
            z = np.linspace(-2, 2, 100)
            r = z**2 + 1
            x = r * np.sin(theta)
            y = r * np.cos(theta)

            ax.plot(x, y, z, label='3D Curve', linewidth=2)
            ax.legend()
            plt.show()
            ```

        * 三维曲面图 (Surface Plot)

            使用 .plot_surface(X, Y, Z) 方法。注意： X, Y, Z 必须是二维网格数据。

            ```py
            # 创建网格数据
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2))  # 计算每个网格点上的 Z 值（一个曲面）

            # 绘制曲面
            surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)

            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5)
            plt.show()
            ```

        * 三维线框图 (Wireframe Plot)

            使用 .plot_wireframe(X, Y, Z) 方法，类似于曲面图但只显示网格线。

            ```py
            ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.5)
            plt.show()
            ```

        * 三维柱状图 (Bar Plot)

            使用 .bar3d(x, y, z, dx, dy, dz) 方法。

            * x, y, z: 柱子的底部坐标。

            * dx, dy, dz: 柱子在 x, y, z 方向上的长度（宽度、深度、高度）。

            ```py
            # 定义柱子的位置和大小
            x_pos = [0, 1, 2]
            y_pos = [0, 1, 2]
            z_pos = np.zeros(3)  # 所有柱子从 z=0 开始

            dx = dy = 0.5 * np.ones(3)  # 所有柱子的宽度和深度都是 0.5
            dz = [1, 2, 3]              # 三个柱子的高度分别为 1, 2, 3

            ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=['r', 'g', 'b'], alpha=0.7)
            plt.show()
            ```

    3. 自定义视图

        调整三维图形的视角：

        ```py
        # 设置视角 (仰角, 方位角)
        ax.view_init(elev=30,  azim=45)  # elev: 仰角（上下看）, azim: 方位角（左右转）

        # 设置坐标轴比例（使其等比例显示，避免图形扭曲）
        ax.set_box_aspect([1, 1, 1])  # [x, y, z] 方向的比例
        ```

    example:

    ```py
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 创建图形和三维坐标轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 2. 生成并绘制数据（一个曲面和一条曲线）
    # 曲面数据
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z_surf = np.sin(np.sqrt(X**2 + Y**2))
    ax.plot_surface(X, Y, Z_surf, cmap='viridis', alpha=0.7)

    # 曲线数据（一条螺旋线）
    theta = np.linspace(0, 6*np.pi, 100)
    z_line = np.linspace(0, 2, 100)
    x_line = np.cos(theta)
    y_line = np.sin(theta)
    ax.plot(x_line, y_line, z_line, 'r-', linewidth=3, label='Spiral')

    # 3. 设置标签、标题和图例
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Surface and Line Plot')
    ax.legend()

    # 4. 调整视角
    ax.view_init(elev=20, azim=35)

    plt.tight_layout()
    plt.show()
    ```

    Axes3D 的基本用法可以概括为：

    1. 创建：通过 fig.add_subplot(projection='3d') 创建。

    2. 绘图：使用与二维绘图类似的方法（如 plot, scatter），但传入三个坐标参数（x, y, z）。对于曲面和线框图，需要二维网格数据。

    3. 定制：使用 set_xlabel, view_init 等方法定制坐标轴和视图。

    4. 显示：最后用 plt.show() 显示图形。

### 稀疏矩阵

* 稀疏矩阵乘法

    加速算法简述（以 CSR x CSC 为例）：

    1. 外层循环：遍历矩阵A的每一行 i（利用CSR的 row_ptr）。

    2. 中层循环：对于A的第 i 行，遍历该行的每一个非零元素 A(i,k)（利用CSR的 col_indices 和 values）。这个 k 是A的列号，同时也是B的行号。

    3. 内层循环：对于每个 k，找到矩阵B的第 k 行（即CSC格式下的第 k 列）。遍历B的第 k 行上的每一个非零元素 B(k,j)（利用CSC的 row_indices 和 values）。

    4. 累加：将乘积 A(i,k) * B(k,j) 累加到结果矩阵 C(i,j) 上。

    我们只处理那些可能产生非零结果的计算。

* `scipy.sparse.csr_matrix`

    Compressed Sparse Row matrix

    是 SciPy 库中用于表示稀疏矩阵的一种数据结构。它专门用于高效地存储和操作那些大部分元素为零的矩阵。

    CSR 格式只存储非零元素的值及其位置，极大地节省了内存和计算时间。

    适用场景:

    * 词袋模型（Bag-of-Words）中的文档-词项矩阵

    * 图的邻接矩阵

    * 有限元分析中的刚度矩阵

    CSR 格式通过三个一维数组来表示整个矩阵：

    1. data：存储所有非零元素的值。

    2. indices：存储每个非零元素所在的列索引。

    3. indptr（索引指针）：存储每一行第一个非零元素在 data 和 indices 中的起始位置。

    这种结构使得按行访问和操作（如矩阵-向量乘法）非常高效。

* `csc_matrix`常用属性和操作

    * 查看矩阵信息

        ```py
        print(sparse_matrix.shape)   # 矩阵形状: (3, 4)
        print(sparse_matrix.nnz)     # 非零元素个数: 4
        print(sparse_matrix.dtype)   # 数据类型: int64
        print(sparse_matrix.has_sorted_indices) # 索引是否已排序: True
        ```

    * 转换格式

        ```py
        # 转换为其他稀疏格式
        csc_matrix = sparse_matrix.tocsc() # 转为CSC格式（按列压缩，列操作快）
        coo_matrix = sparse_matrix.tocoo() # 转为COO格式（坐标格式，构建快）

        # 转换为密集NumPy数组
        dense_array = sparse_matrix.toarray()
        ```

    * 数学运算

        ```py
        # 标量运算
        result = sparse_matrix * 2   # 所有非零元素乘以2

        # 矩阵运算（结果通常也是稀疏矩阵）
        vector = np.array([1, 2, 3, 4])
        result_vector = sparse_matrix.dot(vector) # 矩阵-向量乘法

        other_sparse_matrix = csr_matrix([[1], [0], [1], [0]])
        result_matrix = sparse_matrix.dot(other_sparse_matrix) # 矩阵-矩阵乘法
        ```

        csr_matrix 支持大多数常见的矩阵运算。

    * 切片和索引

        ```py
        # 获取第1行（返回一个1xN的CSR矩阵）
        row_1 = sparse_matrix[1, :]

        # 获取第2列（效率较低，考虑用CSC格式做列操作）
        col_2 = sparse_matrix[:, 2]
        ```

        对 CSR 矩阵进行切片通常不如对密集矩阵高效，尤其是列切片。

* `csr_matrix`的创建方法

    * 从密集矩阵（Dense Array）创建

        从一个普通的 2D NumPy 数组或列表的列表创建。

        ```py
        import numpy as np
        from scipy.sparse import csr_matrix

        dense_matrix = np.array([[1, 0, 0, 0],
                                 [0, 0, 2, 0],
                                 [0, 3, 0, 4]])
                                 
        sparse_matrix = csr_matrix(dense_matrix)
        print(sparse_matrix)
        print(sparse_matrix.toarray()) # 转回密集矩阵查看
        ```

        output:

        ```
        <Compressed Sparse Row sparse matrix of dtype 'int64'
        	with 4 stored elements and shape (3, 4)>
          Coords	Values
          (0, 0)	1
          (1, 2)	2
          (2, 1)	3
          (2, 3)	4
        [[1 0 0 0]
         [0 0 2 0]
         [0 3 0 4]]
        ```

    * 使用 (data, (row, col)) 坐标格式创建

        明确指定每个非零元素的值及其所在的行和列坐标

        ```py
        import numpy as np
        from scipy.sparse import csr_matrix

        # 数据： [1, 2, 3, 4]
        # 行索引：[0, 1, 2, 2] -> 第一个元素在第0行，第二个在第1行，第三、四个在第2行
        # 列索引：[0, 2, 1, 3] -> 第一个元素在第0列，第二个在第2列，第三个在第1列，第四个在第3列

        data = [1, 2, 3, 4]
        row = [0, 1, 2, 2]
        col = [0, 2, 1, 3]

        sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 4))
        print(sparse_matrix.toarray())
        ```

        output:

        ```
        [[1 0 0 0]
         [0 0 2 0]
         [0 3 0 4]]
        ```

    * 使用 (data, indices, indptr) 直接创建（高级）

        直接使用 CSR 格式的三个内部数组来创建。

        ```py
        # 假设矩阵为：
        # [[1, 0, 2, 0]
        #  [0, 0, 3, 4]
        #  [5, 0, 0, 6]]

        data = [1, 2, 3, 4, 5, 6]    # 所有非零值
        indices = [0, 2, 2, 3, 0, 3] # 每个值对应的列号
        indptr = [0, 2, 4, 6]        # 第i行的非零值范围是 data[indptr[i]:indptr[i+1]]

        # indptr 解释：
        # 第0行：有 indptr[1]-indptr[0] = 2 个元素，是 data[0:2] -> [1,2]，列号为 indices[0:2] -> [0,2]
        # 第1行：有 indptr[2]-indptr[1] = 2 个元素，是 data[2:4] -> [3,4]，列号为 indices[2:4] -> [2,3]
        # 第2行：有 indptr[3]-indptr[2] = 2 个元素，是 data[4:6] -> [5,6]，列号为 indices[4:6] -> [0,3]

        sparse_matrix = csr_matrix((data, indices, indptr), shape=(3, 4))
        print(sparse_matrix.toarray())
        # [[1 0 2 0]
        #  [0 0 3 4]
        #  [5 0 0 6]]
        ```

* COO

    COO 是 “Coordinate Format” 的缩写，即坐标格式。它的设计理念非常直观：分别存储非零元素所在的行索引、列索引以及元素的值。

    coo_matrix 就是由这三个等长的数组构成的：

    * data： 存储所有非零元素的值，例如 [5, 9, 1, 4]

    * row： 存储每个非零元素对应的行索引，例如 [0, 1, 2, 2]

    * col： 存储每个非零元素对应的列索引，例如 [2, 0, 1, 2]

    COO 格式本身并不适合直接进行矩阵乘法、加法等科学计算。它的主要职责是作为一种高效的构建格式。

    一旦用 COO 格式构建好矩阵，你可以非常快速地将它转换为其他更适合计算的格式，例如：

    * CSR (Compressed Sparse Row)： 用于高效的矩阵运算（如乘法）。

    * CSC (Compressed Sparse Column)： 用于高效的列操作和求解线性方程组。

    coo_matrix 的 tocsr() 和 tocsc() 方法就是用来做这个转换的。

    example:

    ```py
    import numpy as np
    from scipy.sparse import coo_matrix

    # 1. 创建 COO 矩阵的三大核心数组
    data = np.array([5, 9, 1, 4])    # 非零元素的值
    row  = np.array([0, 1, 2, 2])    # 这些元素的行索引
    col  = np.array([2, 0, 1, 2])    # 这些元素的列索引

    # 2. 创建 COO 矩阵
    # 参数 shape 指定矩阵的总大小，这里是一个 3x3 的矩阵
    coo_sparse_matrix = coo_matrix((data, (row, col)), shape=(3, 3))

    # 3. 查看矩阵（转换为稠密矩阵显示，便于观察）
    print("COO矩阵（以稠密形式显示）:")
    print(coo_sparse_matrix.toarray())

    # 输出结果：
    # [[0 0 5]
    #  [9 0 0]
    #  [0 1 4]]

    # 4. 转换为 CSR 格式以进行高效运算
    csr_sparse_matrix = coo_sparse_matrix.tocsr()
    print("\n已转换为CSR格式。")
    ```

* `Eigen::SparseMatrix`

    Eigen::SparseMatrix 是 Eigen 库中用于表示和操作稀疏矩阵的模板类。

    install:

    `sudo apt install libeigen3-dev`

    头文件被安装在：`/usr/include/eigen3`

    这似乎是一个 header-only 的模板库，所以没有库文件。

    example:

    ```cpp
    #include <eigen3/Eigen/Sparse>
    #include <cstdio>
    #include <vector>

    int main() {
        // 创建稀疏矩阵
        Eigen::SparseMatrix<double> mat(1000, 1000);

        // 使用 triplet 插入非零元素
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.push_back({0, 0, 3.14});  // (行, 列, 值)
        triplets.push_back({1, 2, 2.71});

        mat.setFromTriplets(triplets.begin(), triplets.end());

        // 稀疏矩阵运算
        Eigen::SparseMatrix<double> mat2 = mat * mat.transpose();
        
        return 0;
    }
    ```

    当矩阵密度 < 5% 时，Eigen::SparseMatrix 在内存和计算效率上显著优于稠密矩阵。

* `scipy.sparse.lil_matrix`

    scipy.sparse.lil_matrix 是 SciPy 中用于存储稀疏矩阵的一种数据结构，特别适用于逐步构建和修改稀疏矩阵的场景。

    LIL (List of Lists) 格式将稀疏矩阵存储为：

    * 行列表：每个元素对应矩阵的一行

    * 每行存储：两个列表，分别存储非零元素的列索引和值

    这种结构使得按行操作（添加、删除、修改元素）非常高效。

    **基本用法:**

    * 创建 LIL 矩阵

        ```py
        import numpy as np
        from scipy.sparse import lil_matrix

        # 方法1：指定形状创建空矩阵
        matrix = lil_matrix((3, 3))  # 3x3 矩阵

        # 方法2：从稠密数组创建
        dense_array = np.array([[1, 0, 0], [0, 0, 2], [0, 3, 0]])
        matrix = lil_matrix(dense_array)

        # 方法3：从其他稀疏格式转换
        from scipy.sparse import csr_matrix
        csr_mat = csr_matrix((3, 3))
        lil_mat = csr_mat.tolil()
        ```

    * 元素赋值和修改

        ```py
        # 创建 3x3 矩阵
        matrix = lil_matrix((3, 3))

        # 逐个元素赋值
        matrix[0, 0] = 1
        matrix[1, 2] = 2
        matrix[2, 1] = 3

        # 批量赋值
        matrix[0, [1, 2]] = [4, 5]  # 第0行，第1、2列
        matrix[[1, 2], 0] = [6, 7]  # 第1、2行，第0列

        print(matrix.toarray())
        # 输出：
        # [[1. 4. 5.]
        #  [6. 0. 2.]
        #  [7. 3. 0.]]
        ```

    * 访问矩阵数据

        ```py
        # 访问单个元素
        print(matrix[0, 0])  # 1.0

        # 访问整行
        print(matrix[0].toarray())  # [[1. 4. 5.]]

        # 获取非零元素信息
        print("行指针:", matrix.rows)     # 每行的列索引列表
        print("数据值:", matrix.data)     # 每行的数值列表

        # 转换为稠密数组
        dense = matrix.toarray()
        ```

    * 实际应用示例

        ```py
        # 示例：构建邻接矩阵
        n_nodes = 5
        adj_matrix = lil_matrix((n_nodes, n_nodes))

        # 添加边（无向图）
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # 无向图对称

        print("邻接矩阵:")
        print(adj_matrix.toarray())

        # 转换为其他格式进行高效运算
        csr_adj = adj_matrix.tocsr()  # 转换为CSR格式进行矩阵运算
        ```

    * 格式转换

        ```py
        # 转换为其他稀疏格式
        csr_matrix = matrix.tocsr()   # 压缩稀疏行格式（高效计算）
        csc_matrix = matrix.tocsc()   # 压缩稀疏列格式（高效列操作）
        coo_matrix = matrix.tocoo()   # 坐标格式（快速构建）

        # 转换回稠密矩阵
        dense_matrix = matrix.toarray()
        ```

    **使用建议**

    * 构建阶段：使用 LIL 格式进行频繁的元素修改

    * 计算阶段：转换为 CSR/CSC 格式进行数学运算

    * 内存敏感：对于超大矩阵，考虑使用 COO 格式

### 数据集获取、划分与加载

* DataLoader 中的 sampler

    sampler 只负责生成索引，dataloader 则按照索引生成 batch。伪代码描述这个过程：

    ```py
    # 伪代码，解释 dataloader 内部逻辑
    for epoch in range(...):
        for batch_indices in sampler: # 采样器生成一个batch的索引列表，如 [3, 1, 4, 9]
            batch_data = [dataset[i] for i in batch_indices] # 根据索引从数据集中获取数据
            # ... 后续的 collate 等操作
            yield batch_data
    ```

    默认的 sampler 有`SequentialSampler`和`RandomSampler`。

    package 与使用方法：

    ```py
    import torch
    from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

    # 创建一个简单的数据集
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    labels = torch.tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(data, labels)

    # 使用 SequentialSampler
    sequential_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sequential_sampler)

    # 遍历 DataLoader
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Data: {batch_data}")
        print(f"  Labels: {batch_labels}")
        print("---")
    ```

* dataset 似乎支持 slice 访问

    ```py
    my_dataset = MyDataset()
    print(my_dataset[:3])
    ```

    output:

    ```
    [0, 1, 2]
    ```

* 获取 hugging face 的 imdb 数据集

    ```py
    from datasets import load_dataset
    dataset = load_dataset('imdb')
    print(dataset['train'][0])
    ```

    数据会被下载到`~/.cache/huggingface/datasets`中。imdb 数据集大小为 128 M。

* hugging face 中的数据集

    <https://huggingface.co/datasets>

    使用 python 代码查询：

    ```py
    from huggingface_hub import list_datasets

    # 这是一个生成器，要获取总数需要将其转换为列表，但对于数万个数据集这会很慢且耗内存。
    # all_datasets = list(list_datasets())
    # print(f"Total datasets: {len(all_datasets)}")

    # 更高效的方法是使用分页并计数（但依然需要遍历所有数据集）
    count = 0
    for ds in list_datasets():
        count += 1
    print(f"Total datasets: {count}") # 注意：这会运行一段时间，因为要遍历数万个数据集
    ```

    常见的NLP任务和相关数据集:

    * 文本分类（如情感分析、主题分类）：imdb, ag_news, yelp_review_full

    * 问答（Question Answering）：squad, natural_questions

    * 文本摘要（Summarization）：cnn_dailymail, xsum

    * 文本生成（Text Generation）：wikitext-2, story_cloze

    * 机器翻译（Translation）：wmt14, wmt16, opus_books

    * 命名实体识别（Named Entity Recognition, NER）：conll2003, wnut_17

    * 语义相似度（Semantic Textual Similarity）：stsb_multi_mt

    * 自然语言推理（Natural Language Inference）：mnli, snli

    * 指令微调数据集（用于训练Chat模型）：alpaca, dolly-15k

    使用代码按标签筛选:

    ```py
    from huggingface_hub import list_datasets

    # 查找所有打上 "text-classification" 标签的数据集
    nlp_datasets = list(list_datasets(filter="task_categories:text-classification"))
    print(f"Number of text-classification datasets: {len(list(nlp_datasets))}")

    # 您可以尝试其他标签，如 "text-generation", "question-answering", "translation" 等。
    ```

* `nn.MSELoss()`

    Mean Squared Error（均方误差）, 衡量模型预测值 $\hat{y}$ 与真实值 $y$ 之间差的平方的平均值。

    公式：

    $L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

    其中：

    * $L$ 是最终的损失值（一个标量）。

    * $N$ 是样本的数量（或者是需要计算损失的元素的总个数）。

    * $y_i$ 是第 $i$ 个数据的真实值（ground truth）。

    * $\hat{y}_i$ 是模型对第 $i$ 个数据的预测值（prediction）。

    * $\sum_{i=1}^{N}$ 表示对所有 $N$ 个数据点的差值平方进行求和。

    平方的作用：

    * 消除正负误差相互抵消的问题（例如，-2 和 +2 的误差如果直接相加会变成 0，但这显然不对）。

    * 放大较大误差的贡献。误差越大，平方后的惩罚越大，这使得模型会对大的错误更加敏感。

    PyTorch 的 nn.MSELoss 还提供了一个重要的参数 reduction，它可以改变计算最终损失的方式：

    * `reduction='mean'` (默认值): 计算所有元素平方差的平均值。 $\rightarrow L = \frac{1}{N} \sum (y_i - \hat{y}_i)^2$

    * `reduction='sum'`: 计算所有元素平方差的总和。 $\rightarrow L = \sum (y_i - \hat{y}_i)^2$

    * `reduction='none'`: 不进行汇总（sum 或 mean），直接返回一个与输入形状相同的、每个位置都是一个平方差的损失张量。 $\rightarrow L_i = (y_i - \hat{y}_i)^2$

    example:

    ```py
    import torch
    import torch.nn as nn

    # 1. 创建损失函数实例
    # reduction 可以是 'mean', 'sum', 'none'
    criterion = nn.MSELoss() # 默认 reduction='mean'
    # criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.MSELoss(reduction='none')

    # 2. 准备示例数据
    # 假设我们有4个样本的预测值和真实值
    predictions = torch.tensor([3.0, 5.0, 2.5, 4.0])
    targets = torch.tensor([2.5, 4.8, 2.0, 3.8])

    # 3. 计算损失
    loss = criterion(predictions, targets)

    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"MSE Loss:    {loss.item()}")
    ```

    output:

    ```
    Predictions: tensor([3.0000, 5.0000, 2.5000, 4.0000])
    Targets:     tensor([2.5000, 4.8000, 2.0000, 3.8000])
    MSE Loss:    0.14499999582767487
    ```

    手动代码实现：

    ```py
    def my_mse_loss(pred, targ, reduction='mean'):
        # 1. 计算所有元素的平方差
        squared_diff = (pred - targ) ** 2
        
        # 2. 根据 reduction 参数进行汇总
        if reduction == 'mean':
            loss = torch.mean(squared_diff)
        elif reduction == 'sum':
            loss = torch.sum(squared_diff)
        elif reduction == 'none':
            loss = squared_diff
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        return loss

    # 使用我们自己实现的函数
    my_loss_mean = my_mse_loss(predictions, targets, 'mean')
    my_loss_sum = my_mse_loss(predictions, targets, 'sum')
    my_loss_none = my_mse_loss(predictions, targets, 'none')

    print(f"Manual MSE Loss (mean): {my_loss_mean.item()}")
    print(f"Manual MSE Loss (sum):  {my_loss_sum.item()}")
    print(f"Manual MSE Loss (none): {my_loss_none}")
    ```

* torch dataset and dataloader

    ```py
    import torch
    from torch.utils.data import Dataset, DataLoader

    class MyDataset(Dataset):
        def __init__(self):
            self.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.labels = torch.tensor([0, 1, 0])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print("Batch Data:", batch[0])  
        print("Batch Labels:", batch[1])
    ```

    output:

    ```
    Batch Data: tensor([[1., 2.],
            [3., 4.]])
    Batch Labels: tensor([0, 1])
    Batch Data: tensor([[5., 6.]])
    Batch Labels: tensor([0])
    ```

* IMDb 电影评论数据集

    res: <http://ai.stanford.edu/~amaas/data/sentiment/>

    IMDb 数据集是一个用于二元情感分类的经典基准数据集。它包含来自互联网电影数据库（IMDb）的 50,000 条高度极化的电影评论。

    内容： 每条评论都被标记为 正面（positive） 或 负面（negative）。

    规模： 数据集通常被分为 25,000 条带标签的训练评论和 25,000 条测试评论。此外，还有 50,000 条无标签的额外评论（在此任务中通常不使用）。

    任务： 根据评论文本预测其情感极性（正面/负面）。这是一个典型的文本分类任务。

    explore example:

    ```py
    from datasets import load_dataset
    import numpy as np

    # 1. 加载 IMDb 数据集
    imdb_dataset = load_dataset("imdb")

    # 2. 探索数据集结构
    print("数据集结构:", imdb_dataset)
    print("\n训练集特征:", imdb_dataset["train"].features)
    print("\n测试集第一条样本:", imdb_dataset["test"][0])

    # 3. 查看一些基本统计信息
    # 查看训练集和测试集的大小
    print(f"\n训练集大小: {len(imdb_dataset['train'])}")
    print(f"测试集大小: {len(imdb_dataset['test'])}")

    # 查看标签分布
    train_labels = imdb_dataset["train"]["label"]
    test_labels = imdb_dataset["test"]["label"]

    print(f"\n训练集 - 正面评论: {np.sum(train_labels)}, 负面评论: {len(train_labels) - np.sum(train_labels)}")
    print(f"测试集 - 正面评论: {np.sum(test_labels)}, 负面评论: {len(test_labels) - np.sum(test_labels)}")

    # 4. 随机查看几条样本
    def show_samples(dataset, split="train", num_samples=3):
        sampled_data = dataset[split].shuffle(seed=42).select(range(num_samples))
        for i in range(num_samples):
            print(f"\n--- 样本 {i+1} ---")
            print(f"文本预览: {sampled_data[i]['text'][:200]}...") # 只打印前200个字符
            print(f"标签: {sampled_data[i]['label']} ({'正面' if sampled_data[i]['label'] == 1 else '负面'})")

    show_samples(imdb_dataset, "train")
    ```

* `torch.utils.data`

    There are two types of datasets:

    * map-style datasets: This data set provides two functions  `__getitem__( )`, `__len__( )` that returns the indices of the sample data referred to and the numbers of samples respectively. In the example, we will use this type of dataset.

    * iterable-style datasets: Datasets that can be represented in a set of iterable data samples, for this we use `__iter__( )` function.

    Dataloader syntax:

    ```py
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *, prefetch_factor=2, persistent_workers=False)
    ```

    example:

    ```py
    # importing libraries
    import torch
    import torchvision
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import math

    # class to represent dataset
    class HeartDataSet():

        def __init__(self):
          
            # loading the csv file from the folder path
            data1 = np.loadtxt('heart.csv', delimiter=',',
                               dtype=np.float32, skiprows=1)
            
            # here the 13th column is class label and rest 
            # are features
            self.x = torch.from_numpy(data1[:, :13])
            self.y = torch.from_numpy(data1[:, [13]])
            self.n_samples = data1.shape[0] 
        
        # support indexing such that dataset[i] can 
        # be used to get i-th sample
        def __getitem__(self, index):
            return self.x[index], self.y[index]
          
        # we can call len(dataset) to return the size
        def __len__(self):
            return self.n_samples


    dataset = HeartDataSet()

    # get the first sample and unpack
    first_data = dataset[0]
    features, labels = first_data
    print(features, labels)
    ```

    output:

    ```
    tensor([ 63.0000,   1.0000,   3.0000, 145.0000, 233.0000,   1.0000,   0.0000,
            150.0000,   0.0000,   2.3000,   0.0000,   0.0000,   1.0000]) tensor([1.])
    ```

    dataloader example:

    ```py
    # Loading whole dataset with DataLoader
    # shuffle the data, which is good for training
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    # total samples of data and number of iterations performed
    total_samples = len(dataset)
    n_iterations = total_samples//4
    print(total_samples, n_iterations)
    for i, (targets, labels) in enumerate(dataloader):
        print(targets, labels)
    ```

    traning example:

    ```py
    num_epochs = 2

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):

            # here: 303 samples, batch_size = 4, n_iters=303/4=75 iterations
            # Run our training process
            if (i+1) % 5 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}|\
                    Inputs {inputs.shape} | Labels {labels.shape}')
    ```

* CIFAR-10

    This contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

    使用 torch 下载和加载 cifar 10:

    ```py
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.optim as optim

    # Step 1: Loading the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for data in trainloader:
        input_data: torch.Tensor
        gt: torch.Tensor
        input_data, gt = data
        print('input_data:')
        print(input_data)
        print('input_data shape: {}'.format(input_data.shape))
        print('gt:')
        print(gt)
        print('gt shape: {}'.format(gt.shape))
        break
    ```

    output:

    ```
    Files already downloaded and verified
    Files already downloaded and verified
    input_data:
    tensor([[[[-0.5843, -0.5765, -0.5608,  ..., -0.6314, -0.6784, -0.8118],
              [-0.6392, -0.5843, -0.5765,  ..., -0.6706, -0.6941, -0.7804],
              [-0.6471, -0.6078, -0.6392,  ..., -0.7020, -0.7176, -0.7725],
              ...,
              [-0.4431, -0.4196, -0.3725,  ..., -0.6000, -0.6392, -0.6157],
              [-0.4118, -0.3804, -0.3647,  ..., -0.5216, -0.4980, -0.6235],
              [-0.3333, -0.3333, -0.3255,  ..., -0.5216, -0.4980, -0.6157]],

             [[-0.4902, -0.5059, -0.5294,  ..., -0.6000, -0.6471, -0.7804],
              [-0.5373, -0.5137, -0.5373,  ..., -0.6392, -0.6627, -0.7490],
              [-0.5373, -0.5294, -0.5922,  ..., -0.6706, -0.6863, -0.7412],
              ...,
              [-0.3490, -0.3490, -0.3333,  ..., -0.5765, -0.6157, -0.6078],
              [-0.3569, -0.3333, -0.3333,  ..., -0.4902, -0.4745, -0.6078],
              [-0.3490, -0.3412, -0.3255,  ..., -0.4902, -0.4745, -0.6078]],

             [[-0.5843, -0.5922, -0.6078,  ..., -0.6078, -0.6549, -0.7882],
              [-0.6784, -0.6471, -0.6549,  ..., -0.6471, -0.6706, -0.7569],
              [-0.7020, -0.6784, -0.7333,  ..., -0.6784, -0.6941, -0.7490],
              ...,
              [-0.4824, -0.4824, -0.4745,  ..., -0.7412, -0.7333, -0.6784],
              [-0.4745, -0.4588, -0.4745,  ..., -0.6784, -0.6235, -0.6784],
              [-0.4431, -0.4431, -0.4510,  ..., -0.6941, -0.6392, -0.6784]]],


            [[[-0.4353, -0.4431, -0.4275,  ..., -0.4510, -0.4275, -0.3569],
              [-0.7490, -0.7882, -0.8196,  ..., -0.8039, -0.7569, -0.7098],
              [-0.6941, -0.7804, -0.8118,  ..., -0.8431, -0.8039, -0.7804],
              ...,
              [-0.6706, -0.7725, -0.7176,  ..., -0.7333, -0.7412, -0.7412],
              [-0.6549, -0.7882, -0.7569,  ..., -0.7804, -0.7882, -0.7569],
              [-0.5451, -0.6392, -0.6706,  ..., -0.6941, -0.7098, -0.5843]],

             [[-0.6235, -0.5843, -0.5922,  ..., -0.6157, -0.5765, -0.6392],
              [-0.7333, -0.7098, -0.7255,  ..., -0.7569, -0.6863, -0.7882],
              [-0.7647, -0.7490, -0.7647,  ..., -0.7647, -0.7020, -0.7804],
              ...,
              [-0.6627, -0.6549, -0.6392,  ..., -0.7725, -0.7647, -0.7647],
              [-0.6471, -0.6392, -0.6000,  ..., -0.7882, -0.7882, -0.8039],
              [-0.6000, -0.5922, -0.5765,  ..., -0.7490, -0.7569, -0.7804]],

             [[-0.6941, -0.6549, -0.7020,  ..., -0.6549, -0.6000, -0.7725],
              [-0.5922, -0.4353, -0.5137,  ..., -0.4667, -0.3725, -0.5922],
              [-0.5529, -0.3020, -0.3882,  ..., -0.4118, -0.3255, -0.5059],
              ...,
              [-0.4667, -0.3098, -0.3725,  ..., -0.5294, -0.5451, -0.5608],
              [-0.4667, -0.2863, -0.3176,  ..., -0.4980, -0.4980, -0.5843],
              [-0.5216, -0.4039, -0.4353,  ..., -0.5608, -0.5451, -0.7098]]],


            [[[-0.7098, -0.6627, -0.7725,  ..., -0.5765, -0.5529, -0.5294],
              [-0.7098, -0.6549, -0.7412,  ..., -0.5686, -0.5373, -0.4980],
              [-0.7098, -0.6549, -0.7412,  ..., -0.5608, -0.5451, -0.5059],
              ...,
              [-0.4275, -0.3255,  0.4431,  ..., -0.1922, -0.5765, -0.6157],
              [-0.5137, -0.3255,  0.4588,  ..., -0.0980, -0.5843, -0.6314],
              [-0.6706, -0.3412,  0.4431,  ...,  0.0118, -0.5843, -0.6627]],

             [[-0.7725, -0.7490, -0.8118,  ..., -0.7176, -0.6941, -0.6863],
              [-0.7725, -0.7490, -0.8039,  ..., -0.7176, -0.6941, -0.6863],
              [-0.7647, -0.7490, -0.8118,  ..., -0.7098, -0.7020, -0.6863],
              ...,
              [-0.5686, -0.4745,  0.3176,  ..., -0.3647, -0.7098, -0.7176],
              [-0.6392, -0.4510,  0.3333,  ..., -0.2863, -0.7176, -0.7333],
              [-0.7569, -0.4510,  0.3176,  ..., -0.1765, -0.7176, -0.7490]],

             [[-0.8196, -0.8039, -0.8588,  ..., -0.7961, -0.7882, -0.7804],
              [-0.8196, -0.8118, -0.8431,  ..., -0.8039, -0.7961, -0.7882],
              [-0.8118, -0.7961, -0.8431,  ..., -0.7882, -0.7882, -0.7804],
              ...,
              [-0.6706, -0.5922,  0.1922,  ..., -0.5216, -0.7882, -0.7804],
              [-0.7098, -0.5529,  0.2078,  ..., -0.4510, -0.7961, -0.7961],
              [-0.8039, -0.5294,  0.1922,  ..., -0.3490, -0.8039, -0.8118]]],


            [[[-0.0353,  0.0118,  0.0667,  ..., -0.2627, -0.3098, -0.2863],
              [ 0.0196,  0.0588,  0.1137,  ..., -0.3098, -0.1765, -0.1373],
              [ 0.0353,  0.0745,  0.1216,  ..., -0.1216, -0.0510, -0.1451],
              ...,
              [-0.7490, -0.7255, -0.6471,  ..., -0.8118, -0.8431, -0.8510],
              [-0.7647, -0.7176, -0.6784,  ..., -0.8275, -0.8431, -0.8667],
              [-0.7804, -0.7412, -0.7176,  ..., -0.8275, -0.8431, -0.8667]],

             [[ 0.3490,  0.3961,  0.4510,  ..., -0.1686, -0.2000, -0.1608],
              [ 0.3882,  0.4353,  0.4824,  ..., -0.2392, -0.0902, -0.0353],
              [ 0.3961,  0.4275,  0.4745,  ..., -0.0588,  0.0275, -0.0510],
              ...,
              [-0.7255, -0.7412, -0.7098,  ..., -0.8118, -0.8431, -0.8510],
              [-0.7412, -0.7255, -0.7176,  ..., -0.8275, -0.8431, -0.8667],
              [-0.7569, -0.7412, -0.7333,  ..., -0.8275, -0.8431, -0.8667]],

             [[ 0.7647,  0.8039,  0.8667,  ..., -0.0431, -0.0667, -0.0353],
              [ 0.8196,  0.8588,  0.9137,  ..., -0.1137,  0.0431,  0.0980],
              [ 0.8196,  0.8588,  0.9059,  ...,  0.0745,  0.1686,  0.0824],
              ...,
              [-0.7020, -0.7020, -0.6549,  ..., -0.8118, -0.8431, -0.8510],
              [-0.7176, -0.6863, -0.6627,  ..., -0.8275, -0.8431, -0.8667],
              [-0.7333, -0.7098, -0.6941,  ..., -0.8275, -0.8431, -0.8667]]]])
    input_data shape: torch.Size([4, 3, 32, 32])
    gt:
    tensor([4, 5, 5, 9])
    gt shape: torch.Size([4])
    ```

    数据会被下载到当前文件夹的`./data`目录里。

    ```
    cifar-10-batches-py  cifar-10-python.tar.gz
    ```

### loss

* Cross Entropy Loss

    用于计算两个概率分布之间的差值。

    $$\mathrm{CrossEntropyLoss}(x, \mathrm{target}) = - \frac 1 N \sum_i (\mathrm{target}_i \cdot \log x_i)$$

    * x represents the predicted values,

    * target represents the ground truth or target values.

    注：

    1. 这个数学公式中的 $\mathrm{target}_i$ 是向量中的元素，与下面 torch 实现的标签编码不一样。

        在实际任务中，$\mathrm{target}_i$ 大部分为 0，只有一个为 1，其实相当于一个 indicator。

    1. 这里的 $N$ 指的并不是 batch size，而是一个向量中的 N 个元素，相当于下面的`N_class`。

    syntax:

    ```py
    torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
    ```

    example:

    ```py
    from hlc_utils import *

    ce_loss = nn.CrossEntropyLoss()

    batch_size = 2
    N_class = 4

    input = torch.randn(batch_size, N_class)
    print('input, shape: {}, data:\n{}\n'.format(input.shape, input))

    target = torch.randint(0, N_class, (batch_size,))
    print('target, shape: {}, data:\n{}\n'.format(target.shape, target))

    output = ce_loss(input, target)
    print('output, shape: {}. data:\n{}'.format(output.shape, output))
    ```

    output:

    ```
    input, shape: torch.Size([2, 4]), data:
    tensor([[ 1.0211,  2.0191, -0.9489, -1.2573],
            [ 1.2270,  1.9557, -0.6735, -0.9454]])

    target, shape: torch.Size([2]), data:
    tensor([3, 2])

    output, shape: torch.Size([]). data:
    3.379208564758301
    ```

    注：

    1. `input` 是**未经过**“概率化”的向量，所谓概率化指的是一个向量中的 `N_class` 个值加起来和为 1. `CrossEntropyLoss` 内置了对输入值进行 softmax 预处理的操作。

    1. `target` 的值是标签编码（Label Encoding，与 one-hot 编码相对应）

    1. 如果 batch size 大于 1，那么 CrossEntropyLoss 求的是 batch 的均值。

        在上面的例子中，batch size 的值为 2.

    1. `input` 必须是二维的，如果是一维的，会报错

    Advantages:

    * Invariant to scaling and shifting of the predicted probabilities.

    Disadvantages:

    * Sensitive to outliers and imbalanced data (can be biased towards majority class).

    * It does not provide a similarity between classes which can be required in some cases.

* L1 loss

    The L1 loss function also called Mean Absolute Error (MAE) computes the average of the sum of absolute differences between the predicted and the actual values.

    Formula: 

    $\mathcal L_{L1} (y, \hat y) = \frac 1 n \sum_{i=1}^n \lvert y_i - \hat y_i\rvert$

    Here,

    * $n$ represents the total number of observations or samples

    * $y_i$ represents the actual or observed value for the i-th sample,

    * $\hat y_i$ represents the predicted or estimated value for the i-th sample.

    L1 loss is mostly used for regression problems and is more robust to outliers.

    syntax:

    ```py
    torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    ```

    example:

    ```py
    import torch
    from torch import nn

    #initialising the loss function
    loss = nn.L1Loss()
    #randomly initialising the input and the target value...input is considered as predicted value here.
    input = torch.randn(2, 4, requires_grad=True)
    target = torch.randn(2, 4)
    #passing both the values inside the loss function.
    output = loss(input, target)
    #backpropagation
    output.backward()
    print(output)
    ```

    output:

    ```
    tensor(1.1041, grad_fn=<MeanBackward0>)
    ```

    Advantage:

    * MAE is more robust to outliers compared to Mean Squared Error (MSE) because it takes the absolute difference, reducing the impact of extremely large errors.

    * The MAE loss is straightforward to interpret as it represents the average magnitude of errors, making it easier to communicate the model's performance to stakeholders.

    Disadvantage:

    * MAE treats all errors equally, regardless of their magnitude. This can be a disadvantage in cases where distinguishing between small and large errors is important.

    * The gradient of MAE is a constant value, which can slow down convergence during optimization, especially in comparison to MSE, where the gradient decreases as the error decreases.

* Mean Square Error (L2 loss)

    L2 computes the average of the squared differences between the predicted and actual values.

    The main idea behind squaring is to penalise the model for large difference so that the model avoid larger differences. 

    $$MSE = \frac 1 n \sum_{i=1}^n (y_i - \hat y_i)^2$$

    Here,

    * $n$ represents the total number of observations or samples,

    * $y_i$ represents the actual or observed value for the ith sample,

    * $\hat y_i$ represents the predicted or estimated value for the ith sample.

    syntax:

    ```py
    torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    ```

    example:

    ```py
    import torch
    from torch import nn
    #initialising the loss function
    loss = nn.MSELoss()
    #randomly initialising the input and the target value...input is considered as predicted value here.
    input = torch.randn(2, 4, requires_grad=True)
    target = torch.randn(2, 4)
    #passing both the values inside the loss function.
    output = loss(input, target)
    #backpropagation
    output.backward()
    print(output)
    ```

    output:

    ```
    tensor(1.6697, grad_fn=<MseLossBackward0>)
    ```

    example 2:

    ```py
    #import nn module
    import torch.nn as nn
    mse_loss_fn = nn.MSELoss()

    loss = mse_loss_fn(predicted_value, target)
    #predicted value is what the model is predicting 
    #target is the actual value
    ```

    Disadvantages:

    Sensitive to outliers due to the squaring operation, which deviates the results in the optimization process.

* Huber Loss

    This loss is used while tackling regression problems especially when dealing with outliers.

    $$\mathrm{HuberLoss}(x, \mathrm{target}, \delta) = 
    \frac 1 N \sum_i
    \left\{
    \begin{aligned}
        &\frac 1 2 (x_i - \mathrm{target}_i)^2 \quad \text{if } \lvert x_i - \mathrm{target}_i \rvert \leq \delta \\
        &\delta \left( \lvert x_i - \mathrm{target}_i \rvert - \frac 1 2 \delta \right) \quad \text{otherwise}
    \end{aligned}    
    \right.$$

    Here,

    * x represents the predicted values,target represents the ground truth or target values,

    * δ is a parameter controlling the threshold for switching between quadratic and linear loss

    It combines both MAE( Mean Absolute Error ) and MSE( Mean Squared Error) and which loss will be used depends upon the delta value.

    syntax:

    ```py
    torch.nn.HuberLoss(reduction='mean', delta=1.0)
    ```

    Advantage:

    * Less sensitive to outliers than MSE but still provide a more balanced approach to evaluating the performance of regression models compared to MAE.

    Disadvantage:

    * Introduces a new hyper parameter and the optimization of that leads to more complexity in the model.

    MAE, MSE and Huber loss are used in regression problems but, which one should we use. MSE can be used when you want to penalize larger errors more heavily. It's useful when the data does not have significant outliers and you assume that the errors are normally distributed. MAE can be used when you want robust loss function that is less affected by outliers. And Huber loss can be used when you want to compromise the benefits of both MAE and MSE. 

### 网络参数

* `net.named_parameters()`

    遍历神经网络中的所有可学习参数（权重和偏置），并返回参数名称和参数值本身的迭代器。

    example:

    ```py
    from hlc_utils import *

    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(784, 64)
            self.fc2 = Linear(64, 10)
        
        def forward(self, x):
            x = self.fc1(x)
            x = F.sigmoid(x)
            x = self.fc2(x)
            x = F.softmax(x)

    net = MyModel()

    for name, param in net.named_parameters():
        param: Parameter
        print('param: {}'.format(param))
        print("name: {}".format(name))
        print('shape: {}'.format(param.shape))
        print('data: {}'.format(param.data))
        print('grad: {}'.format(param.grad))
        break
    ```

    output:

    ```
    param: Parameter containing:
    tensor([[ 0.0224, -0.0285, -0.0134,  ...,  0.0081,  0.0048, -0.0166],
            [ 0.0114, -0.0229, -0.0186,  ...,  0.0354, -0.0218, -0.0119],
            [ 0.0211, -0.0086,  0.0258,  ..., -0.0265,  0.0103, -0.0192],
            ...,
            [ 0.0037,  0.0333, -0.0095,  ...,  0.0202, -0.0237, -0.0126],
            [-0.0068, -0.0324, -0.0191,  ...,  0.0220,  0.0154,  0.0047],
            [ 0.0280,  0.0258, -0.0333,  ...,  0.0143, -0.0299,  0.0020]],
           requires_grad=True)
    name: fc1.weight
    shape: torch.Size([64, 784])
    data: tensor([[ 0.0224, -0.0285, -0.0134,  ...,  0.0081,  0.0048, -0.0166],
            [ 0.0114, -0.0229, -0.0186,  ...,  0.0354, -0.0218, -0.0119],
            [ 0.0211, -0.0086,  0.0258,  ..., -0.0265,  0.0103, -0.0192],
            ...,
            [ 0.0037,  0.0333, -0.0095,  ...,  0.0202, -0.0237, -0.0126],
            [-0.0068, -0.0324, -0.0191,  ...,  0.0220,  0.0154,  0.0047],
            [ 0.0280,  0.0258, -0.0333,  ...,  0.0143, -0.0299,  0.0020]])
    grad: None
    ```

    可以看到`Parameter`继承自 Tensor，可以使用`param.data`获取到 tensor。并且 parameter 本身没有 name 属性。

    已知一个 param，无法快速找到它对应的 layer，必须通过 name 去匹配。

    设置不同的学习率:

    ```py
    optimizer_params = []
    for name, param in net.named_parameters():
        if 'bias' in name:
            # 偏置项使用双倍学习率
            optimizer_params.append({'params': param, 'lr': 0.02})
        else:
            optimizer_params.append({'params': param, 'lr': 0.01})

    optimizer = torch.optim.SGD(optimizer_params)
    ```

    参数冻结:

    ```py
    # 冻结前几层的参数
    for name, param in net.named_parameters():
        if 'fc1' in name:
            param.requires_grad = False  # 冻结该参数
    ```

    参数统计:

    ```py
    total_params = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    print(f"可训练参数总数: {total_params}")
    ```

    相关方法对比

    * parameters(): 只返回参数值，不包含名称

    * state_dict(): 返回包含参数名称和值的字典，用于模型保存

    * named_parameters(): 返回包含名称和参数的迭代器，适合遍历操作

* nn.Parameter()

    主要做两件事情：

    1. 为 tensor 增加 grad

    2. 将 tensor 注册到 model 的参数列表中

    example:

    * add grad

        ```py
        # 自动设置 requires_grad=True
        param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        print(param.requires_grad)  # 输出: True
        ```

    * register as model parameter

        ```py
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 5))
                self.bias = nn.Parameter(torch.zeros(5))
            
            def forward(self, x):
                return x @ self.weight + self.bias

        model = MyModel()
        # 自动包含在模型参数中
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
        ```

* `nn.Parameter()`

    nn.Parameter() 是一个用于将张量包装为模型参数的类，它是 torch.Tensor 的子类。

    syntax:

    ```py
    torch.nn.Parameter(data=None, requires_grad=True)
    ```

    params:

    * `data` (Tensor): 要包装为参数的张量

    * `requires_grad` (bool, 可选): 是否需要在反向传播中计算梯度，默认为 True

### tensor 创建与转换

* `np.linspace()`

    syntax:

    ```py
    np.linspace(start, stop, num=50, endpoint=True, dtype=None, retstep=False)
    ```

    retstep：如果为True，返回（数组，步长）；如果为False（默认），只返回数组

    example:

    ```py
    import numpy as np

    lin_1, step_1 = np.linspace(0, 2, 5, endpoint=True, retstep=True)
    lin_2, step_2 = np.linspace(0, 2, 5, endpoint=False, retstep=True)

    print("{}, step: {}".format(lin_1, step_1))
    print("{}, step: {}".format(lin_2, step_2))
    ```

    output:

    ```
    [0.  0.5 1.  1.5 2. ], step: 0.5
    [0.  0.4 0.8 1.2 1.6], step: 0.4
    ```

    可以看到，当包含 endpoint 时，`step = (end - start) / (num - 1)`；当不包含 endpoint 时，`step = (end - start) / num`。

    其他常见的创建数组的方法：

    ```py
    # np.zeros() - 全零数组
    np.zeros(5)                    # [0., 0., 0., 0., 0.]
    np.zeros((2, 3))               # 2x3的全零矩阵

    # np.ones() - 全1数组
    np.ones(4)                     # [1., 1., 1., 1.]
    np.ones((2, 2))                # 2x2的全1矩阵

    # np.full() - 填充指定值
    np.full(3, 7)                  # [7, 7, 7]
    np.full((2, 2), 5)             # 2x2的填充5的矩阵

    # np.eye() - 单位矩阵
    np.eye(3)                      # 3x3单位矩阵

    # np.arange() - 类似range，但返回数组
    np.arange(5)                   # [0, 1, 2, 3, 4]
    np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]

    # np.logspace() - 对数等间距
    np.logspace(0, 2, 5)           # [1., 3.16, 10., 31.62, 100.]

    # np.random.rand() - 均匀分布
    np.random.rand(3)              # 3个[0,1)的随机数
    np.random.rand(2, 2)           # 2x2的随机矩阵

    # np.random.randn() - 标准正态分布
    np.random.randn(3)             # 3个标准正态分布随机数

    # np.random.randint() - 整数随机数
    np.random.randint(0, 10, 5)    # 5个[0,10)的随机整数

    # np.array() - 从列表/元组创建
    np.array([1, 2, 3])            # 从列表创建
    np.array([[1, 2], [3, 4]])     # 二维数组

    # np.asarray() - 转换为数组
    np.asarray(existing_list)      # 将现有序列转为数组

    # np.empty() - 未初始化数组（速度快）
    np.empty(3)                    # 内容随机，不初始化

    # np.copy() - 创建副本
    arr_copy = np.copy(original_arr)

    # np.meshgrid() - 坐标矩阵
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    X, Y = np.meshgrid(x, y)       # 创建网格坐标
    ```

* `np.meshgrid()`

    np.meshgrid() 的主要作用是 从一维坐标向量生成网格坐标矩阵。它接受多个（通常是两个）一维数组，这些数组分别代表不同坐标轴上的点。然后，它会生成一个网格，并返回这个网格中 每一个点 的横坐标和纵坐标。

    syntax:

    ```py
    numpy.meshgrid(*xi, copy=True, sparse=False, indexing='xy')
    ```

    参数解释：

    * `*xi`： 一个或多个一维数组，代表网格的坐标。通常是等间距的数值序列（例如，由 np.linspace 或 np.arange 生成）。

    * `copy`： 布尔值，默认为 True。如果为 False，则返回原始数组的视图以节省内存。通常保持默认即可。

    * `sparse`： 布尔值，默认为 False。如果为 True，则返回稀疏网格以节省内存和计算时间。在数组很大时有用。

    * `indexing`： 字符串，'xy' 或 'ij'，默认为 'xy'。这是一个非常关键的参数，决定了输出的顺序。

        indexing='xy'： 返回的第一个数组是 纵坐标（Y） 的矩阵，第二个数组是 横坐标（X） 的矩阵。这与我们通常的数学和图像处理习惯（行对应Y，列对应X）一致。

        indexing='ij'： 返回的第一个数组是 横坐标（X） 的矩阵，第二个数组是 纵坐标（Y） 的矩阵。这与矩阵索引一致。

    返回值：

    返回一个 `list` of ndarray（Numpy数组的列表）。对于二维网格，返回两个二维数组；对于三维网格，返回三个三维数组，依此类推。

    example:

    ```py
    import numpy as np

    x = np.array([1, 2, 3])
    y = np.array([4, 5])

    # 使用默认的 indexing='xy'
    X, Y = np.meshgrid(x, y)

    print("X (坐标矩阵):")
    print(X)
    print("\nY (坐标矩阵):")
    print(Y)
    ```

    output:

    ```
    X (坐标矩阵):
    [[1 2 3]
     [1 2 3]]

    Y (坐标矩阵):
    [[4 4 4]
     [5 5 5]]
    ```

    结果分析：

        X 矩阵：每一 行 都是相同的，是 x 数组的复制。它代表了网格中每个点的 横坐标。

        Y 矩阵：每一 列 都是相同的，是 y 数组的复制。它代表了网格中每个点的 纵坐标。

    这样，网格中的点 (X[i, j], Y[i, j]) 就是所有 (x[j], y[i]) 的组合。例如：

        (X[0,0], Y[0,0]) = (1, 4)

        (X[0,1], Y[0,1]) = (2, 4)

        (X[1,0], Y[1,0]) = (1, 5)

        ...以此类推

    example:

    ```py
    import numpy as np
    import matplotlib.pyplot as plt

    # 创建一维坐标向量
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)

    # 生成网格坐标矩阵
    X, Y = np.meshgrid(x, y)

    # 定义二维函数，例如 R = sqrt(X^2 + Y^2)
    R = np.sqrt(X**2 + Y**2)
    # 计算每个网格点的Z值，例如 Z = sin(R)
    Z = np.sin(R)

    # 绘制三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()
    ```

* 将 tensor 从 numpy 转换到 torch

    * `torch.from_numpy()`

        这种方案会共享内存。

        ```py
        import torch
        import numpy as np

        # 创建 NumPy 数组
        numpy_array = np.array([1, 2, 3, 4, 5])

        # 转换为 Torch Tensor
        torch_tensor = torch.from_numpy(numpy_array)

        print("NumPy 数组:", numpy_array)
        print("Torch Tensor:", torch_tensor)
        print("Tensor 类型:", torch_tensor.dtype)
        ```

    * `torch.as_tensor()`

        这种方案会尽可能共享内存，但不保证。

        ```py
        torch_tensor = torch.as_tensor(numpy_array)
        ```

    * `torch.tensor()`

        这种方案会创建数据的副本。

        ```py
        torch_tensor = torch.tensor(numpy_array)
        ```

* torch 创建 tensor 的常见方法

    ```py
    import torch

    tensor_1d = torch.tensor([1, 2, 3])
    print("1D Tensor (Vector):")
    print(tensor_1d)
    print()

    tensor_2d = torch.tensor([[1, 2], [3, 4]])
    print("2D Tensor (Matrix):")
    print(tensor_2d)
    print()

    random_tensor = torch.rand(2, 3)
    print("Random Tensor (2x3):")
    print(random_tensor)
    print()

    zeros_tensor = torch.zeros(2, 3)
    print("Zeros Tensor (2x3):")
    print(zeros_tensor)
    print()

    ones_tensor = torch.ones(2, 3)
    print("Ones Tensor (2x3):")
    print(ones_tensor)
    ```

    output:

    ```
    1D Tensor (Vector):
    tensor([1, 2, 3])

    2D Tensor (Matrix):
    tensor([[1, 2],
            [3, 4]])

    Random Tensor (2x3):
    tensor([[0.9134, 0.1796, 0.5852],
            [0.8830, 0.9940, 0.2796]])

    Zeros Tensor (2x3):
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    Ones Tensor (2x3):
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    ```

* 可以在创建 tensor 时使用`device=`参数来指定是否使用 gpu

    ```py
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tensor_size = (10000, 10000)  
    a = torch.randn(tensor_size, device=device)  
    b = torch.randn(tensor_size, device=device)  

    c = a + b  

    print("Result shape (moved to CPU for printing):", c.cpu().shape)

    print("Current GPU memory usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    ```

    output:

    ```
    Using device: cpu
    Result shape (moved to CPU for printing): torch.Size([10000, 10000])
    Current GPU memory usage:
    Allocated: 0.00 MB
    Cached: 0.00 MB
    ```

* 将 numpy ndarray 转换为 torch tensor

    * 使用 torch.from_numpy()

        ```py
        import torch
        import numpy as np

        # 创建 NumPy 数组
        numpy_array = np.array([1, 2, 3, 4, 5])

        # 转换为 Torch Tensor
        torch_tensor = torch.from_numpy(numpy_array)

        print("NumPy 数组:", numpy_array)
        print("Torch Tensor:", torch_tensor)
        print("Tensor 类型:", torch_tensor.dtype)
        ```

    * 使用 torch.as_tensor()

        ```py
        torch_tensor = torch.as_tensor(numpy_array)
        ```

    * 使用 torch.tensor()

        这个方法会创建数据的副本

        ```py
        torch_tensor = torch.tensor(numpy_array)
        ```

    关于内存的共享性：

    * `torch.from_numpy()`: 共享内存

    * `torch.as_tensor()`: 如果可能的话，共享内存

    * `torch.tensor()`: 不共享内存，会创建副本

    example:

    ```py
    import numpy as np
    import torch

    # 创建 NumPy 数组
    numpy_array = np.array([1, 2, 3])

    # 使用 from_numpy（共享内存）
    torch_tensor = torch.from_numpy(numpy_array)

    # 修改 NumPy 数组
    numpy_array[0] = 100

    print("修改后的 NumPy 数组:", numpy_array)
    print("Torch Tensor（也改变了）:", torch_tensor)  # 也会显示 100

    # 使用 torch.tensor（不共享内存）
    torch_tensor_copy = torch.tensor(numpy_array)
    numpy_array[1] = 200
    print("Torch Tensor 副本（未改变）:", torch_tensor_copy)  # 不会改变
    ```

    output:

    ```
    修改后的 NumPy 数组: [100   2   3]
    Torch Tensor（也改变了）: tensor([100,   2,   3])
    Torch Tensor 副本（未改变）: tensor([100,   2,   3])
    ```

* 关于 torch tensor 创建数据副本的几种情况

    * torch.tensor(任何Python数据) → 总是创建副本

    * torch.from_numpy(np_array) → 共享内存（仅对NumPy数组）

    * torch.as_tensor() → 尽可能共享内存（智能选择）

* 将 numpy 转换为 tensor 时指定类型

    ```py
    # 转换为 float32
    torch_tensor_float = torch.from_numpy(numpy_array).float()

    # 或者在转换时指定
    torch_tensor_float = torch.from_numpy(numpy_array.astype(np.float32))

    # 使用 dtype 参数
    torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
    ```

    最佳实践

        推荐使用 torch.from_numpy() - 效率高，内存共享

        如果需要独立副本 - 使用 torch.tensor()

        注意数据类型 - 确保使用适合深度学习的数据类型（通常是 float32）

        检查设备 - 确保 tensor 在正确的设备上（CPU/GPU）

    注：

    1. `.float()`会创建副本

        ```py
        import torch
        import numpy as np

        # 创建 NumPy 数组
        numpy_array = np.array([1, 2, 3], dtype=np.int32)

        # 转换过程
        torch_tensor_int = torch.from_numpy(numpy_array)  # 共享内存，dtype=int32
        torch_tensor_float = torch.from_numpy(numpy_array).float()  # 创建新副本，dtype=float32
        ```

    1. 只有提前把 numpy ndarray 的数据类型转换过来，才能共享数据

        ```py
        # 方法1：先转换 NumPy 数组的数据类型
        numpy_array_float = numpy_array.astype(np.float32)
        torch_tensor = torch.from_numpy(numpy_array_float)  # 共享内存，float32

        # 方法2：使用 astype 并保持共享
        torch_tensor = torch.from_numpy(numpy_array.astype(np.float32, copy=False))
        ```

### metric

* f measure 延伸

    这里的 "F" 通常被认为是代表 F-measure（F 度量），源自统计学中的 F-test 概念。

    f1-score 有时也被解释为平衡 Precision 和 Recall 的 Harmonic Mean（调和平均）。

    $\beta$ 参数的意义：

    * $\beta$ 参数控制着 Precision 和 Recall 的相对重要性

    * $\beta = 1$：Precision 和 Recall 同等重要 → F1-score

    * $\beta > 1$：更重视 Recall（如 $\beta = 2$ 时，Recall 的权重是 Precision 的 4 倍）

    * $\beta < 1$：更重视 Precision（如 $\beta = 0.5$ 时，Precision 的权重是 Recall 的 4 倍）

* F1-score

    F1 指的是 F-score 或 F-measure 家族中的第一个成员，具体来说是当参数 β = 1 时的特殊情况。

    F-score 的通用公式是：

    $$F_\beta = (1 + \beta^2) \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{(\beta^2 \cdot \mathrm{Precision}) + \mathrm{Recall}}$$

    当 $\beta = 1$ 时，公式简化为：

    $$F_1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$$

    这就是 F1-score 的由来 - 它是 F-measure with $\beta = 1$。

* 为什么 macro 不使用调和平均值？

    F1-score 已经是考虑过类别平衡的数据了，直接对 F1-socre 使用 macro 就可以。F1-socre 对 Precision 和 Recall 使用调和平均是因为 Precision 和 Recall 是同一类别不同维度的指标。而 macro 是对同一个维度的指标进行调和平均，没有必要。

    对已经平衡过的指标（F1）再进行一次平衡，这可能导致过度惩罚。

    如果我们考虑到不同类别的平衡，可以使用

    1. 加权F1（Weighted-F1）

        `Weighted-F1 = Σ(weight_i × F1_i)`

        其中 weight_i 通常是该类别的样本比例

    2. 几何平均（Geometric Mean）

        对极端值比算术平均更敏感，但比调和平均温和：

        `G-Mean = (F1_1 × F1_2 × ... × F1_N)^(1/N)`

    3. 使用专门的不平衡学习指标

        如 G-Mean（几何平均）或 Balanced Accuracy。

    如果特别关注最差类别，考虑报告 最小F1（Min-F1）

* 如何选择 micro 与 macro

    选择哪种平均方式完全取决于你的业务目标和数据集特点。

    选择 'micro' 当：

        你关心模型的整体性能，并且每个样本的错误代价是相同的。

        数据存在不平衡，但大类的性能更重要。例如，在电商产品分类中，热销商品的准确率远比冷门商品重要。

        你希望得到一个单一的、概括性的性能指标，并且这个指标与准确率等价。

    选择 'macro' 当：

        所有类别都同等重要，无论其样本数量多少。

        你特别关心模型在小类/稀有类上的表现。这在很多关键领域至关重要：

            医疗： 诊断一个稀有病。

            金融风控： 检测极少数但危害巨大的欺诈交易。

            工业： 预测罕见的设备故障。

        你的数据集类别相对平衡。

        你想评估模型的稳健性和泛化能力，看它是否在所有类别上都“学得不错”。


    最佳实践

    不要只看一个数字！ 一个负责任的实践是：

        同时报告 Micro 和 Macro 值，以提供更全面的视图。

        查看每个类别的单独指标（即不平均），这能最直接地发现问题所在。

        分析混淆矩阵，直观地看到哪些类别被混淆了。

* precision 的三种模式 micro, macro 与 none

    * `'micro'`： 全局视角。先汇总所有类别（或所有样本）的 TP, FP, FN，再用汇总后的总数计算一个全局指标。

    * `'macro'`： 平均视角。先独立计算每个类别的指标，然后对所有类别的指标值求算术平均。

    * `'none'`: 计算每个类别的 precision，不进行平均

    example:

    ```py
    from torchmetrics import Precision

    pred = t.tensor([0, 1, 2, 3])
    gt = t.tensor([0, 1, 0, 0])

    pre = Precision('multiclass', num_classes=10, average='micro')
    pre.update(pred, gt)
    pre_score = pre.compute()
    print('micro pre: {}'.format(pre_score))

    pre = Precision('multiclass', num_classes=10, average='macro')
    pre.update(pred, gt)
    pre_score = pre.compute()
    print('macro pre: {}'.format(pre_score))

    pre = Precision('multiclass', num_classes=10, average='none')
    pre.update(pred, gt)
    pre_score = pre.compute()
    print('none pre: {}'.format(pre_score))
    ``` 

    output:

    ```
    micro pre: 0.5
    macro pre: 0.5
    none pre: tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    ```

    * `'micro'`模式详解

        假设我们有一个多类分类问题，有 C 个类别。

        1. 逐类统计：

            对于每个类别 i，计算其真正例（TP_i）和假正例（FP_i）。

            * 真正例（TP_i）： 真实标签为 i 且被预测为 i 的样本数。

            * 假正例（FP_i）： 真实标签不是 i 但被预测为 i 的样本数。

        2. 全局汇总：

            * 计算所有类别的 TP 之和： total_TP = TP_1 + TP_2 + ... + TP_C

            * 计算所有类别的 FP 之和： total_FP = FP_1 + FP_2 + ... + FP_C

        3. 计算 Micro Precision：

            使用汇总后的 total_TP 和 total_FP 来计算 Precision，公式和标准的二分类 Precision 一模一样。

            $\mathrm{Precision_{micro}} = \frac{\mathrm{total\_TP}}{\mathrm{total\_TP + total\_FP}}$

        重要特性与注意事项

        * 与 Accuracy 的关系： 在多类分类中，Micro Precision 的值等于 准确率（Accuracy）。这是因为：

            * total_TP 就是所有被正确分类的样本总数。

            * total_TP + total_FP 就是所有被预测为正例的样本总数，在多类分类中，这等于总样本数（因为每个样本必须被分到一个类别）。

            * 所以，Micro Precision = total_TP / N = Accuracy。

        * 样本不平衡： Micro 平均对每个样本“一视同仁”，因此它更适合样本不平衡的数据集，因为大类的性能会主导最终结果。如果你关心小类的性能，应该使用 'macro' 平均。

        * 多标签任务： 在多标签任务中（一个样本可以有多个标签），Micro Precision 的计算逻辑完全相同（汇总所有标签的 TP 和 FP），但此时它不等于 Accuracy，因为一个样本可以有多个预测和多个真实标签。

    * `'micro'`与`'macro'`模式的对比

        * 对类别不平衡的敏感性

            * `micro`

                不敏感（默认偏向大类）
                大类的性能主导了最终结果。因为大类的 TP/FP 数量远多于小类，在汇总时贡献最大。在我们的例子中，Micro Precision (0.786) 更接近大类 A 的 Precision (0.947)。

            * `macro`

                敏感（平等对待每个类）
                将所有类别视为同等重要，无论其样本多少。小类的差劲性能会直接拉低平均值。在我们的例子中，Macro Precision (0.831) 被小类 C 的 Precision (0.714) 拉低了。
        * 优点

            * micro

                1. 综合性能： 很好地衡量了模型在整体数据集上的性能。
                2. 等于Accuracy： 在多类分类中，Micro-Precision/Recall/F1 等于准确率，易于理解。
                3. 适用于样本不平衡但关心整体性能的场景。

            * macro

                1. 公平性： 给予所有类别同等权重，能揭示模型在小类上的短板。
                2. 稳定性： 不受类别分布影响，适合比较不同数据集或不同采样策略下的模型。
                3. 适用于需要关注小类的场景（如医疗诊断、故障检测）。

        * 缺点

            * micro

                1. 掩盖小类问题： 如果模型完全忽略小类，但只要大类表现好，Micro指标依然会很高，从而误导你认为模型很好。
                2. 对数据分布敏感： 结果严重依赖于数据集的类别分布。

            * macro

                1. 可能低估性能： 如果一个模型在大类上表现极好，但在一个样本极少的小类上表现稍差，Macro指标可能会给出一个相对较低的评价，这可能不完全符合业务直觉。
                2. 对噪声敏感： 一个在某个小类上的极端差值（如 Precision=0）会严重拉低整体平均值。

* torchmetrics

    `acc.update(pred, gt)`

    在`.update()`函数中，第一个参数必须是 pred，第二个参数必须是 gt。

    pred 和 gt 必须是 torch 的 tensor 类型，不能是 numpy 的 ndarray。

    如果 pred 是一维的，那么其编码方式为标签编码，即预测的类别的索引，而不是概率。
    
    如果 pred 是二维的，那么 pred 的类型必须是 float，不能是 int，其代表的含义为输出的概率。问题：是否需要经过 softmax？问题：如果使用 max() 取概率最大值，那么 threshold = 0.5 有什么意义？

* Accuracy（准确率）, Precision（精确率/查准率）, Recall（召回率/查全率）

    * accuracy

        含义：所有预测结果中，预测正确的比例。

        公式：

        `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

        意义：衡量模型整体的正确率。它是一个非常直观的指标。

        优缺点：

        * 优点：容易理解。

        * 缺点：在数据不平衡的数据集上，准确率会严重失真。

            例子：在一个有1000个样本的数据集中，有990个负样本（0），只有10个正样本（1）。如果一个模型简单地将所有样本都预测为负，那么它的准确率是 (0 + 990) / 1000 = 99%。虽然准确率很高，但这个模型完全没有识别正例的能力，是一个无用的模型。

    * precision

        含义：在所有被模型预测为正例的样本中，真正的正例有多少。

        公式：

        `Precision = TP / (TP + FP)`

        意义：衡量模型的“精准度”或“宁缺毋滥”的程度。它关注的是预测结果。

        核心问题：当模型说某个东西是“正例”时，它有多可信？

        应用场景：注重减少误报（FP）的场景。

        * 垃圾邮件检测：我们非常不希望把正常邮件误判为垃圾邮件（FP）。宁可放过一些垃圾邮件（FN），也不能误杀正常邮件。因此，我们需要高精确率。

        * 推荐系统：给用户推送的内容，希望尽量都是他感兴趣的。如果推送了不感兴趣的内容（FP），会影响用户体验。

    * recall

        含义：在所有实际为正例的样本中，模型成功预测出来的有多少。

        公式：

        `Recall = TP / (TP + FN)`

        意义：衡量模型的“覆盖率”或“宁错杀不漏放”的程度。它关注的是真实情况。

        核心问题：在所有真正的正例中，模型找出了多少？

        应用场景：注重减少漏报（FN）的场景。

        * 疾病检测：我们非常不希望把一个患病的人误判为健康（FN）。宁可让一些健康的人做进一步检查（FP），也不能漏掉一个病人。因此，我们需要高召回率。

        * 逃犯识别：在安检系统中，绝对不能漏掉一个逃犯（FN）。即使需要误警一些普通人（FP）进行二次检查，也要确保高召回率。

    * Precision和Recall的“跷跷板”关系

        在大多数情况下，精确率（Precision）和召回率（Recall）是相互矛盾的。提高一个，通常会导致另一个的降低。

        * 如果你想提高Precision（减少FP）：

            你需要提高预测正例的门槛。例如，只有模型有99%的把握时才预测为正。这样，被预测为正的样本确实很可能是正的（Precision高），但很多“没那么确定”的正例会被判为负例，从而导致漏报增加（FN增加），Recall降低。

        * 如果你想提高Recall（减少FN）：

            你需要降低预测正例的门槛。例如，只要模型有50%的把握就预测为正。这样，你能抓住几乎所有的正例（Recall高），但也会混入很多其实是负例的样本，导致误报增加（FP增加），Precision降低。

    * 与Accuracy的关系

        Accuracy提供了一个宏观的、整体的性能视图。

        Precision和Recall提供了更细粒度的、针对特定类别（正例）的性能视图。

        在数据平衡且FP和FN的成本相似的问题中，Accuracy是一个不错的指标。

        在数据不平衡或FP与FN的成本明显不同的问题中，必须结合Precision和Recall（以及F1-Score）来分析。

* F1-Score：调和平均数

    为了同时考虑Precision和Recall，我们引入了 F1-Score。

    公式：

    `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

    意义：F1-Score 是 Precision 和 Recall 的调和平均数。它只有在 Precision 和 Recall 都较高时才会高。因此，它是一个综合性的指标，特别适用于不平衡数据集的评价。

    * 为什么取调和平均数，而不是代数平均数，或者几何平均数？

        因为调和平均数对较低值施加了更严厉的惩罚。

        三种平均数：

        假设我们有 Precision (P) 和 Recall (R) 两个值。

        * 算术平均数：(P + R) / 2

            特点：对所有值一视同仁，是普通的“平均值”。

        * 几何平均数：sqrt(P * R)

            特点：受极端值影响较小，更适合衡量比例或增长率。

        * 调和平均数：2 * P * R / (P + R)

            特点：强烈惩罚不平衡的数值。当P和R中有一个非常低时，调和平均数会接近这个低值。

        我们希望一个模型在Precision和Recall上都表现良好，而不是用其中一个的高分来“掩盖”另一个的低分。

        example:

        场景：我们有一个疾病检测模型。

            模型A： Precision = 1.0， Recall = 0.1

                它预测有病的人，100%确实有病（非常准，绝不误诊）。

                但实际有病的人，它只找出了10%（漏掉了90%的病人，非常危险）。

            模型B： Precision = 0.5， Recall = 0.5

                它预测有病的人，一半确实有病。

                实际有病的人，它找出了一半。

        问题：哪个模型更好？

        计算它们的平均数：

        | 模型 | A | B |
        | - | - | - |
        |　Precision (P) | 1.0 | 0.5 |
        | Recall (R) | 0.1 | 0.5 |
        | 算术平均 | (1.0 + 0.1) / 2 = 0.55 | (0.5+0.5)/2 = 0.50 |
        | 几何平均 sqrt(1.0 * 0.1) ≈ 0.32 | sqrt(0.5 * 0.5) = 0.50 |
        | 调和平均 | (F1) 2(1.0 * 0.1)/(1.0+0.1) ≈ 0.18 | 2(0.5* 0.5)/(0.5+0.5) = 0.50 |

        分析结果：

        * 从算术平均数看：模型A (0.55) > 模型B (0.50)。这显然是不合理的。模型A是一个“懒惰”的模型，它为了保持100%的准确率，只敢对极少数非常有把握的病例做出阳性预测，导致大量病人被漏诊。在医学上，这是一个灾难性的模型。然而，算术平均数却被它极高的Precision所“欺骗”，给出了更高的分数。

        * 从几何平均数看：模型A (0.32) < 模型B (0.50)。这个结果已经比算术平均数合理了，它识别出了模型A的不平衡性。

        * 从调和平均数 (F1-Score) 看：模型A (0.18) << 模型B (0.50)。调和平均数对模型A的“偏科”行为施加了最严厉的惩罚，给出了一个极低的分数，清晰地表明模型B的综合性能远优于模型A。

        结论与总结:

        平均数类型	对不平衡的惩罚力度	在评估模型中的适用性
        算术平均	最弱	不适用。容易被一个高指标和另一个低指标的模型所误导。
        几何平均	中等	比算术平均好，在某些场景下（如Fβ-Score的变体）有应用。
        调和平均 (F1)	最强	最常用。能有效惩罚“偏科”的模型，确保模型在P和R之间取得有意义的平衡。

    F1-Score特别适合于类别不平衡的数据集，以及那些没有明确倾向是更需要Precision还是Recall的场景，它提供了一个稳健的、单一的综合性评估指标。

    当你有明确倾向时，可以使用Fβ-Score。

    Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)

    当β=1时，就是F1。

    当β>1时，Recall的权重更高（更看重查全）。

    当β<1时，Precision的权重更高（更看重查准）。

* 混淆矩阵（Confusion Matrix）

    有时也被称为 Error Matrix（错误矩阵），它是一个2x2的表格，总结了分类模型对二分类问题的预测结果。


    | | 实际为正例 | 实际为负例 |
    | - | :-: | :-: |
    | 预测为正例 | TP (True Positive) | FP (False Positive) |
    | 预测为负例 | FN (False Negative) | TN (True Negative) |

    * TP（真阳性）：模型预测为正，实际也是正。预测正确。

    * FP（假阳性）：模型预测为正，但实际是负。误报。

    * FN（假阴性）：模型预测为负，但实际是正。漏报。

    * TN（真阴性）：模型预测为负，实际也是负。预测正确。

* torchmetrics

    install: `pip install torchmetrics`

    ```py
    import torch
    from torchmetrics import Accuracy, Precision

    # accuracy
    accuracy = Accuracy(task="multiclass", num_classes=10)
    accuracy.reset()

    batch1_preds = torch.tensor([0, 1, 2, 3]) # 模型预测的类别索引
    batch1_target = torch.tensor([0, 1, 1, 3]) # 真实的类别索引

    batch2_preds = torch.tensor([1, 0, 2])
    batch2_target = torch.tensor([1, 0, 1])

    accuracy.update(batch1_preds, batch1_target)
    accuracy.update(batch2_preds, batch2_target)

    final_accuracy = accuracy.compute()
    print(f"最终准确率: {final_accuracy}") # 例如：tensor(0.7143)


    # precision
    pre = Precision('multiclass', num_classes=10, average='macro')
    pre.reset()
    pre.update(batch1_preds, batch1_target)
    pre.update(batch2_preds, batch2_target)
    final_pre = pre.compute()
    print('final pre: {}'.format(final_pre))
    ```

    output:

    ```
    最终准确率: 0.7142857313156128
    final pre: 0.75
    ```

    注：

    1. 如果 precision 的 average 设置为`micro`，那么最后得到的结果和 accuracy 相同。
