# PyTorch Note

## cache

tten()的作用

    好的，我们来简单明了地解释一下 `nn.Unflatten()` 的作用。

    **核心作用**

    **`nn.Unflatten()` 的作用是将一个“压平”的张量重新恢复成具有特定维度的张量。**

    你可以把它理解为 `nn.Flatten()` 的逆操作。在神经网络中，我们经常需要将多维数据（如图片）压平成一维向量，以便输入到全连接层。而在某些时候（例如在生成模型、自编码器或从全连接层转换回卷积层时），我们需要将这个一维向量再“重塑”回多维形式，这时 `nn.Unflatten()` 就派上了用场。

    ---

    **主要参数**

    - `dim`：指定在哪个维度上进行“反压平”操作。
    - `unflattened_size`：指定反压平后该维度应该变成的形状。

    ---

    **举个例子**

    假设我们有一个神经网络，它先通过一个全连接层，然后需要将输出重新变成一个图片般的三维结构（通道，高度，宽度），以便后续进行卷积操作。

    ```python
    import torch
    import torch.nn as nn

    # 1. 假设我们有一个从全连接层出来的向量，形状为 (batch_size, 120)
    flattened_tensor = torch.randn(4, 120) # 批量大小为4，每个样本是120维的向量

    # 2. 我们希望将它恢复成一个三维张量，例如 (通道, 高, 宽) = (6, 5, 4)
    # 因为 6 * 5 * 4 = 120，所以元素总数是匹配的。
    unflatten_layer = nn.Unflatten(dim=1, unflattened_size=(6, 5, 4))

    # 3. 进行反压平操作
    unflattened_tensor = unflatten_layer(flattened_tensor)

    print(f"原始压平后的张量形状: {flattened_tensor.shape}")
    print(f"反压平后的张量形状: {unflattened_tensor.shape}")
    ```

    **输出：**

    ```
    原始压平后的张量形状: torch.Size([4, 120])
    反压平后的张量形状: torch.Size([4, 6, 5, 4])
    ```

    **解释：**

    - `dim=1`：表示我们要对第1个维度（索引从0开始，也就是120这个维度）进行反压平操作。
    - `unflattened_size=(6, 5, 4)`：表示我们希望把120这个维度，拆分成 `(6, 5, 4)` 这三个维度。
    - 最终，张量的形状从 `[4, 120]` 变成了 `[4, 6, 5, 4]`。批量大小 `4` 保持不变。

    **与 `torch.reshape()` 的区别**

    你可能会想，这个操作和 `x.reshape(4, 6, 5, 4)` 有什么区别？

    **主要区别在于：`nn.Unflatten()` 是一个 `nn.Module`。**

    这意味着：
    1.  **它可以作为神经网络模型的一部分被定义和保存。** 当你使用 `torch.jit.script` 或 `torch.jit.trace` 来序列化模型时，使用 `nn.Unflatten()` 比在 `forward` 函数里直接写 `reshape` 更可靠。
    2.  **它更清晰、更结构化。** 在模型定义中，看到 `nn.Unflatten` 能立刻明白这是一个改变数据维度的层，意图明确。

    **总结**

    | 特性 | `nn.Unflatten()` | `torch.reshape()` / `view()` |
    | :--- | :--- | :--- |
    | **本质** | 一个神经网络层（`nn.Module`） | 一个张量操作函数 |
    | **用途** | 主要在设计模型结构时使用，作为模型的一部分 | 在模型前向传播或数据处理时进行灵活的维度变换 |
    | **序列化** | 与 PyTorch 的 JIT 等工具兼容性更好 | 在某些序列化场景下可能不被直接支持 |

    简单来说，**当你在构建一个 `nn.Sequential` 模型或需要一个明确的层来改变维度时，就使用 `nn.Unflatten()`。如果只是临时需要改变一下张量形状，用 `reshape` 或 `view` 就足够了。**

    * 那么它是 view 操作，还是对内存进行布局重排？

        这是一个非常好的问题，触及了PyTorch底层操作的核心区别。

        **`nn.Unflatten()` 本质上是一个 `view` 操作，而不是对内存进行布局重排（即拷贝）的操作。**

        让我们来详细解释一下：

        ### 1. 什么是 `view` 操作？

        - **`view` 操作**：它只是**改变张量的解释方式**，而不改变其底层内存中的数据存储顺序。
        - 它要求张量在内存中是**连续**的，并且新形状必须与原始张量的元素总数兼容。
        - 因为它不拷贝数据，所以**速度极快**，并且几乎不消耗额外内存。
        - 常见的 `view` 操作包括：`tensor.view()`, `tensor.reshape()`（当可能时）, `torch.reshape()`（当可能时）。

        ### 2. 什么是内存重排操作？

        - **内存重排操作**：如 `tensor.permute()`, `tensor.transpose()` 等，它们会**改变张量在内存中的存储顺序**。
        - 这些操作通常需要**创建新的内存布局**，并可能涉及数据拷贝。
        - 例如，对一个行优先存储的矩阵进行转置后，新矩阵在内存中的排列方式就完全不同了。

        ### 3. 验证 `nn.Unflatten()` 是 `view` 操作

        我们可以通过一个简单的实验来验证：

        ```python
        import torch
        import torch.nn as nn

        # 创建一个原始张量
        original_tensor = torch.randn(2, 6)  # 形状 [2, 6]
        print("原始张量:")
        print(original_tensor)
        print(f"原始张量内存地址: {id(original_tensor.data_ptr())}")

        # 使用 Unflatten 进行变换
        unflatten = nn.Unflatten(dim=1, unflattened_size=(2, 3))
        unflattened_tensor = unflatten(original_tensor)

        print("\n反压平后的张量:")
        print(unflattened_tensor)  # 形状 [2, 2, 3]
        print(f"反压平张量内存地址: {id(unflattened_tensor.data_ptr())}")

        # 检查它们是否共享内存
        print(f"\n是否共享内存: {original_tensor.data_ptr() == unflattened_tensor.data_ptr()}")
        ```

        **输出结果可能类似：**
        ```
        原始张量:
        tensor([[ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6],
                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]])
        原始张量内存地址: 140153461374784

        反压平后的张量:
        tensor([[[ 0.1,  0.2,  0.3],
                [ 0.4,  0.5,  0.6]],

                [[-0.1, -0.2, -0.3],
                [-0.4, -0.5, -0.6]]])
        反压平张量内存地址: 140153461374784

        是否共享内存: True
        ```

        **关键证据**：两个张量的 `data_ptr()`（指向底层内存的指针）是相同的，这证明它们共享同一块内存，只是解释方式不同。

        ### 4. 源码层面的证据

        查看PyTorch源码，`nn.Unflatten` 的实现最终调用的是 `torch.Tensor.view()`：

        ```python
        # 这是 nn.Unflatten 前向传播的简化逻辑
        def forward(self, input: Tensor) -> Tensor:
            return input.view(input.size(0), *self.unflattened_size)  # 使用 view 操作
        ```

        ### 重要提醒：何时会失效？

        虽然 `nn.Unflatten()` 本身是 `view` 操作，但在某些情况下它会失败，并提示需要先使用 `.contiguous()`：

        ```python
        # 如果对非连续张量使用 Unflatten
        x = torch.randn(2, 3, 4)
        y = x.transpose(1, 2)  # 这会使得 y 在内存中不连续
        # z = nn.Unflatten(1, (2, 3))(y)  # 可能会报错
        z = y.contiguous().view(2, -1, 2, 3)  # 需要先使其连续
        ```

        总结

        | 特性 | `nn.Unflatten()` | 内存重排操作（如 `permute`） |
        |------|------------------|----------------------------|
        | **内存使用** | 共享底层内存 | 通常需要新的内存布局 |
        | **速度** | 很快（几乎零开销） | 相对较慢（可能涉及拷贝） |
        | **本质** | 改变张量的解释方式 | 改变数据在内存中的物理排列 |
        | **底层实现** | 基于 `tensor.view()` | 涉及数据重排和可能的数据拷贝 |

        所以，**`nn.Unflatten()` 是一个高效的内存共享操作，它通过 `view` 机制来重新解释张量的维度，而不进行实际的数据拷贝。**

* 详细讲讲 pytorch 中 Subset 的作用和用法

    好的，我们来详细讲解 PyTorch 中 `torch.utils.data.Subset` 的用法和作用。

    ---

    1. Subset 的本质

        `Subset` 是一个**数据集包装器（Dataset Wrapper）**，它不存储数据本身，而是持有一个**索引列表**，用于从原始数据集中按索引取数。

        ```python
        class Subset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset      # 原始数据集
                self.indices = indices      # 索引列表（可以是列表 (list)、范围 (range) 或张量 (tensor)）
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
            
            def __len__(self):
                return len(self.indices)
        ```

        创建 Subset 的方式

        ```python
        from torch.utils.data import Subset, TensorDataset
        import torch

        # 创建一个虚拟数据集（10个样本，每个样本是[特征, 标签]）
        X = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))
        dataset = TensorDataset(X, y)
        
        indices_list = [0, 2, 4, 6, 8]
        subset1 = Subset(dataset, indices_list)  # 方式1：使用 list
        subset2 = Subset(dataset, range(3, 7))  # 方式2：使用 range
        indices_tensor = torch.tensor([1, 3, 5, 7, 9])
        subset3 = Subset(dataset, indices_tensor)  # 方式3：使用 torch.tensor
        ```

    ---

    4. 典型用法示例

        **示例 1：划分训练集和验证集（最经典）**

        ```python
        from torch.utils.data import Subset, DataLoader
        from torchvision import datasets, transforms

        # 加载完整数据集（如 CIFAR-10）
        full_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.ToTensor()
        )

        # 划分比例（80% 训练，20% 验证）
        total_len = len(full_dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len

        # 生成随机打乱的索引
        indices = torch.randperm(total_len).tolist()

        # 创建 Subset
        train_dataset = Subset(full_dataset, indices[:train_len])
        val_dataset = Subset(full_dataset, indices[train_len:])

        # 包装为 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        print(f"训练集大小: {len(train_dataset)}")  # 40000
        print(f"验证集大小: {len(val_dataset)}")    # 10000
        ```

        ---

        **示例 2：快速实验（取小样本调试）**

        ```python
        # 调试模式：只用 100 张图片快速测试代码
        debug_indices = list(range(100))
        debug_dataset = Subset(full_dataset, debug_indices)

        debug_loader = DataLoader(debug_dataset, batch_size=16, shuffle=True)

        for batch_idx, (images, labels) in enumerate(debug_loader):
            # 快速验证模型前向传播
            print(f"Batch {batch_idx}: {images.shape}")
            if batch_idx >= 5:
                break  # 只跑几个 batch 检查代码
        ```

        ---

        **示例 3：分层采样保持类别分布**

        `Subset` 本身不支持分层，但可以配合 `sklearn` 实现：

        ```python
        from sklearn.model_selection import train_test_split
        import numpy as np

        # 获取原始数据集的标签（假设 dataset 是 CIFAR-10）
        labels = [full_dataset[i][1] for i in range(len(full_dataset))]

        # 使用 sklearn 的 stratify 得到索引
        train_indices, val_indices = train_test_split(
            np.arange(len(full_dataset)),
            test_size=0.2,
            stratify=labels,          # 保持类别比例
            random_state=42
        )

        # 用这些索引创建 Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        ```

        ---

        **示例 4：多个 Subset 组合（ConcatDataset）**

        ```python
        from torch.utils.data import ConcatDataset

        # 从不同数据集中取子集并合并
        subset_a = Subset(dataset_a, range(100))
        subset_b = Subset(dataset_b, range(50, 150))

        combined = ConcatDataset([subset_a, subset_b])
        # 相当于从 dataset_a 取前100个，从 dataset_b 取50-149个
        ```

        ---

        **示例 5：K-Fold 交叉验证**

        ```python
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        full_dataset = ...  # 完整数据集

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
            
            print(f"Fold {fold+1}: 训练 {len(train_subset)}, 验证 {len(val_subset)}")
            # 训练模型...
        ```

    ---

    5. 注意事项与常见陷阱

        | 注意事项 | 说明 |
        |---------|------|
        | **索引有效性** | 索引必须在 `[0, len(dataset)-1]` 范围内，否则会报 `IndexError`。 |
        | **随机打乱** | `Subset` 本身不改变数据顺序。如需打乱，在 `DataLoader` 中设置 `shuffle=True`，或先打乱索引列表。 |
        | **数据复制问题** | Subset 不会复制数据，只存储索引。修改原始数据集会影响所有 Subset，反之亦然（但 PyTorch Dataset 通常不修改数据）。 |
        | **嵌套 Subset** | 可以嵌套使用，但会增加查找开销：`Subset(Subset(dataset, [1,2]), [0])`。 |
        | **重复索引** | 索引列表可以包含重复值，这会生成包含重复样本的 Subset（有时用于过采样）。 |
        | **标签提取** | 如果原始 Dataset 是 `TensorDataset`，可通过 `dataset.tensors[1][indices]` 取标签，但更规范的是遍历 Subset。 |
        | **与 DataLoader 配合** | Subset 本身就实现了 `__getitem__`，可以直接传入 DataLoader。 |

    ---

    6. 与其他划分方法的对比

        | 方法 | 适用场景 | 优点 | 缺点 |
        |------|---------|------|------|
        | **Subset + random split** | 一次性划分 | 灵活、内存高效、易用 | 需手动处理随机索引 |
        | **`torch.utils.data.random_split`** | 随机划分（底层就是 Subset） | 更简洁的一行代码 | 只能按比例随机划分 |
        | **自定义 Dataset** | 复杂逻辑（如动态过滤） | 完全可控 | 代码量大 |
        | **Sampler（如 SubsetRandomSampler）** | DataLoader 级别的采样 | 不创建新 Dataset | 不能直接获得子集 Dataset |

        ---

    7. `random_split` vs `Subset`

        `random_split` 其实就是 `Subset` 的便捷封装：

        ```python
        from torch.utils.data import random_split

        # 使用 random_split（简洁）
        train_dataset, val_dataset = random_split(
            full_dataset, [40000, 10000],  # 长度列表
            generator=torch.Generator().manual_seed(42)
        )

        # 等价于 Subset 写法
        indices = torch.randperm(len(full_dataset)).tolist()
        train_dataset = Subset(full_dataset, indices[:40000])
        val_dataset = Subset(full_dataset, indices[40000:])
        ```

        **选择建议**：
        - 简单随机划分 → 用 `random_split`
        - 需要自定义索引逻辑（分层、按条件筛选、多折）→ 用 `Subset`

        ---

    8. 高级技巧：从 Subset 恢复原始索引

        有时需要知道 Subset 中的样本在原数据集中的位置：

        ```python
        # 保存原始索引
        class IndexedSubset(Subset):
            def __getitem__(self, idx):
                data = self.dataset[self.indices[idx]]
                # 返回数据 + 原始索引
                if isinstance(data, tuple):
                    return (*data, self.indices[idx])
                return data, self.indices[idx]

        # 使用时
        subset = IndexedSubset(dataset, [5, 10, 15])
        image, label, original_idx = subset[0]
        print(f"取的是原数据集的第 {original_idx} 个样本")
        ```

        ---

    9. 性能考虑

        - **内存**：几乎为 0（只存索引列表）
        - **速度**：单次访问多一层函数调用，但通常可忽略
        - **DataLoader 并行**：Subset 支持多进程加载（因为只读操作）

        ---

    10. 总结

        | 方面 | 要点 |
        |------|------|
        | **本质** | 索引列表包装器，不复制数据 |
        | **核心用途** | 数据集划分、快速实验、交叉验证 |
        | **创建方式** | `Subset(dataset, indices)` |
        | **优点** | 内存高效、灵活、与所有 Dataset 兼容 |
        | **配合工具** | DataLoader、ConcatDataset、random_split |
        | **最佳实践** | 用 `random_split` 做简单划分；需要精细控制时用 Subset |

        ---

    **快速记忆口诀**：
    > Subset 不存数据存索引，划分采样省内存。  
    > random_split 是它的糖，复杂逻辑自己写。

    如果你有更具体的场景（比如多标签分层划分、大规模数据集的采样策略），欢迎继续提问！

* torch.nn.CrossEntropyLoss() 中，input 里的数可以小于 0
    
    `torch.nn.CrossEntropyLoss` 是 **交叉熵损失** 和 **Softmax 函数** 的结合。为了理解为什么输入值可以为负，我们需要拆解它的计算过程：

    1. **输入**：你的 `input` 张量（通常也叫 `logits`）是**未经过归一化**的原始分数。这些分数可以是任何实数（正数、负数、零），它们代表了模型对每个类别的“置信度”或“证据”。

    2. **内部 Softmax**：损失函数内部首先会对这些 `logits` 应用 **Softmax 函数**。Softmax 的核心作用就是将任意范围的实数**转换成一个概率分布**。

        `Softmax(x_i) = exp(x_i) / Σ(exp(x_j))`

        **关键点在这里**：`exp(x_i)`（e 的 x_i 次方）这个操作，**无论 `x_i` 是正还是负，其结果永远是一个正数**。Softmax 关心的是所有输入值之间的**相对差距**，而不是它们的绝对值。

    3. **计算交叉熵**：然后，损失函数会使用这个由 Softmax 计算出的概率分布与真实标签（`target`）来计算交叉熵损失。

* 为什么这段代码里没有第一步先 to tensor?

    ```py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      # 调整大小
        transforms.ToTensor(),              # 转为Tensor [0,1]
        transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1,1]
    ])
    ```


    您提出了一个很好的观察点！实际上，这段代码**确实包含了 to tensor 步骤**，让我来解释一下这个流程：

    ## 完整的预处理流程

    ```python
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      # 第1步：调整大小 (处理 PIL Image)
        transforms.ToTensor(),              # 第2步：转为Tensor [0,1] (从 PIL Image 到 Tensor)
        transforms.Normalize((0.5,), (0.5,)) # 第3步：归一化到 [-1,1] (处理 Tensor)
    ])
    ```

    ## 关键点说明

    1. **`Resize` 处理的是 PIL Image 对象**
       - 输入通常是 PIL Image 或图像文件路径
       - `Resize` 操作在 PIL Image 格式上进行

    2. **`ToTensor()` 就是转换步骤**
       - 将 PIL Image 或 numpy array 转换为 PyTorch Tensor
       - 同时将像素值从 [0, 255] 缩放到 [0.0, 1.0]

    3. **`Normalize` 必须在 `ToTensor` 之后**
       - 因为归一化操作是针对 Tensor 进行的
       - 公式：`normalized = (tensor - mean) / std`

    ## 如果您想先转换为 Tensor

    如果您确实希望先转换为 Tensor，可以这样写：

    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),              # 先转换为 Tensor
        transforms.Resize((224, 224)),      # 在 Tensor 上调整大小
        transforms.Normalize((0.5,), (0.5,)) # 归一化
    ])
    ```

    不过需要注意的是，**`Resize` 在 PIL Image 上通常更高效**，所以常见的做法是先在 PIL 格式下进行几何变换，然后再转换为 Tensor 进行数值处理。

    * to tensor 后，shape 变为 (c, h, w)，但是 pil image 的 shape 为 (h, w, c)， resize() 会根据这两者采取不同的 resize 方法吗？

        您提出了一个非常关键的问题！答案是：**`Resize()` 的行为确实会根据输入类型的不同而有所不同**。

        ## 输入类型决定 Resize 行为

        ### 1. 当输入是 PIL Image 时 (H, W, C)
        ```python
        # PIL Image 格式: (Height, Width, Channels)
        from PIL import Image
        import torchvision.transforms as transforms

        # 创建示例 PIL Image (100, 150, 3)
        pil_img = Image.new('RGB', (150, 100), color='red')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 处理 (H, W, C)
            transforms.ToTensor(),          # 转换为 (C, H, W)
        ])
        result = transform(pil_img)  # 输出: (3, 224, 224)
        ```

        ### 2. 当输入是 Tensor 时 (C, H, W)
        ```python
        # Tensor 格式: (Channels, Height, Width)
        tensor_img = torch.randn(3, 100, 150)  # (C, H, W)

        transform = transforms.Compose([
            transforms.ToTensor(),          # 如果是PIL输入，先转换
            transforms.Resize((224, 224)),  # 处理 (C, H, W)
        ])
        ```

        ## Resize 的内部逻辑

        `transforms.Resize` 会根据输入类型自动调整：

        ```python
        def resize_function(input_data, size):
            if isinstance(input_data, Image.Image):  # PIL Image
                # 使用 PIL 的插值方法，处理 (H, W, C)
                return input_data.resize(size, Image.BILINEAR)
            
            elif isinstance(input_data, torch.Tensor):  # PyTorch Tensor
                # 使用 torch.nn.functional.interpolate，处理 (C, H, W)
                return F.interpolate(input_data.unsqueeze(0), size=size, mode='bilinear').squeeze(0)
        ```

        ## 实际影响

        1. **插值方法不同**：
        - PIL：使用 PIL 的插值算法
        - Tensor：使用 PyTorch 的插值算法

        2. **性能差异**：
        - PIL resize 通常更快（优化更好）
        - Tensor resize 在 GPU 上可能更有优势

        3. **推荐做法**：
        ```python
        # 推荐：在 PIL 阶段进行几何变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),      # 在PIL上resize（更快）
            transforms.ToTensor(),              # 然后转换
            transforms.Normalize((0.5,), (0.5,))
        ])
        ```

        所以您的观察是正确的 - `Resize()` 确实会根据输入是 PIL Image 还是 Tensor 来采用不同的内部实现！

    * 那么 to tensor 可以放到 normalize 后面吗

        **不可以！`ToTensor()` 绝对不能放到 `Normalize()` 后面。**

        ## 原因分析

        ### 1. 数据类型不匹配
        ```python
        # 错误的顺序 - 会报错！
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5,), (0.5,)),  # ❌ 错误：Normalize 需要 Tensor 输入
            transforms.ToTensor()                   # 太晚了！
        ])
        ```

        ### 2. `Normalize` 的输入要求
        ```python
        # Normalize 的源码大致如下：
        def normalize(tensor, mean, std):
            # 假设 tensor 是 PyTorch Tensor，形状为 (C, H, W)
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
            return tensor
        ```

        **`Normalize` 期望：**
        - 输入是 `torch.Tensor` 类型
        - 像素值范围在 [0, 1]（这是 `ToTensor()` 的输出）
        - 形状为 (C, H, W)

        ## 正确的数据流

        ```python
        # 正确流程
        transform = transforms.Compose([
            transforms.Resize((224, 224)),      # 输入: PIL Image (H, W, C)
            transforms.ToTensor(),              # 输出: Tensor [0,1] (C, H, W)
            transforms.Normalize((0.5,), (0.5,)) # 输入: Tensor [0,1] → 输出: Tensor [-1,1]
        ])
        ```

        ## 数值变化示例

        假设原始像素值 = 128（在 0-255 范围内）：

        ```
        PIL Image (128) 
            → ToTensor() → 0.502 (在 [0,1] 范围内)
            → Normalize((0.5,), (0.5,)) → (0.502 - 0.5) / 0.5 = 0.004 (在 [-1,1] 范围内)
        ```

        ## 唯一可能的变体

        ```python
        # 如果使用其他转换库先得到 numpy array
        transform = transforms.Compose([
            transforms.ToTensor(),              # 从 numpy 到 tensor
            transforms.Normalize((0.5,), (0.5,))
        ])
        ```

        **总结：`ToTensor()` 必须在 `Normalize()` 之前，因为 `Normalize` 操作是针对 Tensor 数据类型设计的。**

    * 还有哪些 transform 函数像 resize() 一样，既可以处理  tensor 也可以处理非 tensor，哪些 transform 函数只能处理 tensor？

        这是一个很好的问题！torchvision.transforms 中的函数确实有不同的输入要求。让我为您详细分类：

        ## 🌟 既可以处理 PIL Image 也可以处理 Tensor

        这些转换通常比较"智能"，能自动检测输入类型：

        ```python
        import torchvision.transforms as transforms

        # 双向兼容的转换
        universal_transforms = [
            transforms.Resize((224, 224)),      # 尺寸调整
            transforms.CenterCrop(224),         # 中心裁剪
            transforms.RandomCrop(224),         # 随机裁剪
            transforms.RandomResizedCrop(224),  # 随机缩放裁剪
            transforms.Pad(10),                 # 填充
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),    # 随机垂直翻转
            transforms.RandomRotation(30),      # 随机旋转
            transforms.Grayscale(),             # 灰度化
            transforms.ColorJitter(),           # 颜色抖动
            transforms.RandomGrayscale(),       # 随机灰度化
            transforms.FiveCrop(224),           # 五裁剪
            transforms.TenCrop(224),            # 十裁剪
        ]
        ```

        ## 🔧 只能处理 Tensor 的转换

        这些通常涉及数值运算，需要 Tensor 格式：

        ```python
        # 只能处理 Tensor 的转换
        tensor_only_transforms = [
            transforms.Normalize((0.5,), (0.5,)),  # 归一化
            transforms.Lambda(lambda x: x * 2),    # Lambda 函数（自定义操作）
            transforms.RandomErasing(),            # 随机擦除
            # 任何涉及数学运算的转换
        ]
        ```

        ## 🔄 格式转换类（特殊用途）

        ```python
        # 格式转换
        format_transforms = [
            transforms.ToTensor(),      # PIL/numpy → Tensor
            transforms.ToPILImage(),    # Tensor → PIL Image
            transforms.ConvertImageDtype(torch.float16),  # 数据类型转换
        ]
        ```

        ## 实际使用示例

        ```python
        from PIL import Image
        import torch

        # 示例1：混合使用（推荐）
        transform_mixed = transforms.Compose([
            transforms.Resize((256, 256)),          # 在PIL上处理（高效）
            transforms.RandomCrop(224),             # 在PIL上处理
            transforms.RandomHorizontalFlip(),      # 在PIL上处理
            transforms.ToTensor(),                  # 转换为Tensor
            transforms.Normalize((0.5,), (0.5,)),   # 在Tensor上处理
            transforms.RandomErasing(),             # 在Tensor上处理
        ])

        # 示例2：纯Tensor流程
        transform_tensor_only = transforms.Compose([
            transforms.ToTensor(),                  # 必须先转换
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(),
            # 注意：这里的几何变换在Tensor上进行可能不如PIL高效
            transforms.Resize((224, 224)),          # 在Tensor上resize
        ])
        ```

        ## 判断方法

        您可以通过查看源码或文档来判断：

        ```python
        # 查看转换的输入要求
        print(transforms.Resize.__doc__)  # 通常会说明支持的类型
        ```

        ## 经验法则

        1. **几何变换**（尺寸、裁剪、翻转、旋转）→ 通常双向兼容
        2. **颜色变换**（亮度、对比度、饱和度）→ 通常双向兼容  
        3. **数值运算**（归一化、数学运算）→ 通常需要 Tensor
        4. **数据增强**（RandomErasing等）→ 通常需要 Tensor

        **最佳实践**：在 PIL 阶段完成几何变换，转换为 Tensor 后再进行数值运算和数据增强。


* 详细讲讲 pytorch 中 nn.ConvTranspose2d() 的原型、作用和用法

    在深度学习的图像处理中，`nn.ConvTranspose2d` 常被称为 **转置卷积**（Transposed Convolution），有时也被误称为“反卷积”（Deconvolution）。它是实现图像**上采样**（Upsampling）的核心工具。

    ---

    1. 函数原型

        在 PyTorch 中，该类的构造函数如下：

        ```python
        torch.nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            groups=1, 
            bias=True, 
            dilation=1, 
            padding_mode='zeros'
        )
        ```

        **关键参数解析：**
        *   **in_channels / out_channels**: 输入和输出的通道数。
        *   **kernel_size**: 卷积核大小。
        *   **stride**: 步长。**注意：** 在转置卷积中，stride 决定了输出放大的倍数。
        *   **padding**: 填充。这里的 padding 会从输出结果中“裁剪”掉一部分。
        *   **output_padding**: 额外的填充。用于解决当 $stride > 1$ 时，多个输入形状可能对应同一个输出形状的歧义问题。

    ---

    2. 核心作用

        `nn.ConvTranspose2d` 的主要作用是将较小的特征图放大到较大的特征图，同时通过**可学习的参数**来填充细节，而不仅仅是像插值那样简单的像素复制。

        *   **场景应用：**
            *   **生成对抗网络 (GANs)：** 从随机噪声向量生成图像。
            *   **语义分割 (U-Net)：** 在解码器（Decoder）部分恢复图像的分辨率。
            *   **超分辨率：** 提升图像的清晰度。

    ---

    3. 工作原理（直观理解）

        转置卷积可以理解为普通卷积的“逆向过程”，但计算逻辑略有不同：

        1.  **插值：** 在输入特征图的像素之间填充 $stride - 1$ 个零。
        2.  **填充：** 在四周填充额外的零（受 `padding` 参数控制）。
        3.  **卷积：** 执行正常的正向卷积。

        因此，当你设置 `stride=2` 时，输出的分辨率大约会翻倍。

    ---

    4. 尺寸计算公式

        如果你需要精确控制输出形状，可以使用以下公式：

        $$H_{out} = (H_{in} - 1) \times stride - 2 \times padding + dilation \times (kernel\_size - 1) + output\_padding + 1$$

        > **快速记忆法：** 如果 `output_padding=0` 且 `padding` 设置合理，输出尺寸大约是 $H_{in} \times stride$。

    ---

    5. 代码示例

        下面的例子展示了如何将一个 $2 \times 2$ 的输入上采样到 $4 \times 4$。

        ```python
        import torch
        import torch.nn as nn

        # 定义转置卷积层
        # 输入通道1，输出通道1，卷积核3x3，步长2，填充1，输出填充1
        m = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 创建一个 1x1x2x2 的随机张量
        input_tensor = torch.randn(1, 1, 2, 2)

        # 执行前向传播
        output = m(input_tensor)

        print(f"输入尺寸: {input_tensor.shape}") # torch.Size([1, 1, 2, 2])
        print(f"输出尺寸: {output.shape}")       # torch.Size([1, 1, 4, 4])
        ```

    ---

    6. 注意事项：棋盘效应 (Checkerboard Artifacts)

        转置卷积的一个常见副作用是**棋盘效应**。当 `kernel_size` 不能被 `stride` 整除时，卷积核在重叠区域的覆盖不均匀，导致生成的图像出现类似棋盘的方格纹路。

    **解决方法：**

    1.  确保 `kernel_size` 可以被 `stride` 整除。
    2.  使用 `nn.Upsample`（插值）后接一个普通的 `nn.Conv2d` 来代替转置卷积。

* 详细讲讲 nn.BatchNorm2d() 的原型、作用和用法

    在深度学习中，`nn.BatchNorm2d`（二维批归一化）是训练深层卷积神经网络（CNN）的“标配”。它由 Google 在 2015 年提出，主要用于解决深层网络中的**梯度消失**和**内部协变量偏移**问题。

    ---

    1. 函数原型

        ```python
        torch.nn.BatchNorm2d(
            num_features, 
            eps=1e-05, 
            momentum=0.1, 
            affine=True, 
            track_running_stats=True
        )
        ```

        **参数详解：**
        *   **num_features**：最重要的参数，指输入张量的通道数 $C$（即 `(N, C, H, W)` 中的 $C$）。
        *   **eps**：分母中添加的一个微小值，防止除以零，默认 $10^{-5}$。
        *   **momentum**：用于计算动态均值和方差的动量值。
        *   **affine**：布尔值。设为 `True` 时，该层会学习两个参数 $\gamma$（缩放）和 $\beta$（偏移）。
        *   **track_running_stats**：设为 `True` 时，训练时会记录全局均值和方差，以便在测试时使用。

    ---

    2. 核心作用

        BatchNorm 的本质是对每一个特征通道进行**标准化处理**。

        (1) 加速收敛
        由于将每层的数据拉回到均值为 0、方差为 1 的分布，激活函数（如 Sigmoid 或 ReLU）不容易进入饱和区，从而允许使用更大的学习率。

        (2) 降低初始化敏感度
        传统的深层网络对权重初始化非常敏感，BN 减弱了这种依赖。

        (3) 正则化效果
        BN 在计算均值和方差时引入了当前 Batch 的随机性，起到了一定的正则化作用，有时甚至可以省掉 Dropout。

    ---

    3. 数学原理

        对于一个 Batch 中的特征图，BatchNorm 在 **(N, H, W)** 三个维度上计算均值和方差。

        1.  **计算均值：** $\mu = \frac{1}{N \times H \times W} \sum x_i$
        2.  **计算方差：** $\sigma^2 = \frac{1}{N \times H \times W} \sum (x_i - \mu)^2$
        3.  **标准化：** $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
        4.  **仿射变换（Scale and Shift）：** $y_i = \gamma \hat{x}_i + \beta$

        > **注意：** $\gamma$ 和 $\beta$ 是可学习的参数，目的是让网络自己决定是否要保留归一化后的分布，还是恢复回原始分布。

    ---

    4. 用法示例

        在 PyTorch 中，`nn.BatchNorm2d` 通常放置在卷积层（`Conv2d`）之后、激活函数（`ReLU`）之前。

        ```python
        import torch
        import torch.nn as nn

        # 假设输入是一个 Batch 大小为 4，通道为 3，尺寸为 32x32 的图像
        input_tensor = torch.randn(4, 3, 32, 32)

        # 定义层：num_features 必须等于输入通道数 3
        bn = nn.BatchNorm2d(3)

        # 训练模式：使用当前 batch 的统计量，并更新全局统计量
        bn.train()
        output = bn(input_tensor)

        # 评价模式：使用训练时保存的全局统计量
        bn.eval()
        output_eval = bn(input_tensor)

        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}") # 形状保持不变
        ```

    ---

    5. 常见坑点与注意事项

        * **训练与测试状态切换：** 必须使用 `model.train()` 和 `model.eval()`。在 `eval()` 模式下，BN 使用的是整个训练集累计的“运行均值（running_mean）”，而不是当前 Batch 的均值。

        * **Batch Size 的限制：** BN 的效果高度依赖于 Batch Size。如果 Batch Size 太小（如 1 或 2），计算出的均值和方差无法代表总体分布，会导致效果极差。此时建议改用 `GroupNorm` 或 `InstanceNorm`。
        
        * **偏置 (Bias) 的冗余：** 如果卷积层后面紧跟 BN 层，卷积层的 `bias` 参数应该设为 `False`。

            * 原因：BN 的减均值操作会把卷积的 `bias` 直接抵消掉。

            * `nn.Conv2d(..., bias=False)` 这样写更省内存。

    你目前是在处理一个小型数据集，还是在搭建复杂的深层架构？
* 讲讲 pytorch 中`nn.Unflatten()`的原型、作用和用法

    在 PyTorch 的神经网络构建中，`nn.Unflatten()` 是一个非常实用的层，它主要用于**将一个平坦的维度“展开”回高维形状**。它通常出现在 **卷积神经网络（CNN）的解码器（Decoder）** 或 **生成对抗网络（GAN）** 中，作为 `nn.Flatten()` 的逆操作。

    ---

    1. 函数原型

        ```python
        torch.nn.Unflatten(dim, unflattened_size)
        ```

        * **`dim` (int)**: 要被展开的维度索引。通常是 `1`，因为第 `0` 维通常是 Batch Size。
        * **`unflattened_size` (tuple/torch.Size)**: 展开后的新形状（不包括 Batch 维度）。

    ---

    2. 核心作用

        在全连接层（`nn.Linear`）之后，数据通常变成了形状为 `(Batch, Features)` 的 2D 张量。如果你想把这些特征喂给卷积层（`nn.Conv2d`），你需要把 `Features` 这一维重新变成 `(Channels, Height, Width)`。

        **`nn.Unflatten` 的本质就是 `tensor.view()` 或 `tensor.reshape()` 的封装，但它更适合放在 `nn.Sequential` 中。**

    ---

    3. 用法示例

        假设我们有一个全连接层的输出，维度是 `(64, 128)`（64 是 Batch Size），我们想把它转换成 `(64, 1, 8, 16)` 的特征图：

        **在 `nn.Sequential` 中使用（最常见用法）**

        ```python
        import torch
        import torch.nn as nn

        # 模拟输入：Batch=64, 特征数=128
        input_tensor = torch.randn(64, 128)

        model = nn.Sequential(
            # 将维度 1 (128) 展开为 (1, 8, 16)
            # 注意：1 * 8 * 16 必须等于 128
            nn.Unflatten(dim=1, unflattened_size=(1, 8, 16)),
            
            # 接着可以接卷积层了
            nn.Conv2d(1, 32, kernel_size=3, padding=1)
        )

        output = model(input_tensor)
        print(output.shape)  # 输出: torch.Size([64, 32, 8, 16])
        ```

    ---

    4. 关键点与注意事项

        * **元素总数匹配**：`unflattened_size` 中所有数字的乘积，必须等于原维度 `dim` 的大小。例如，上面例子中 $1 \times 8 \times 16 = 128$。
        * **Batch 维度自动保留**：你只需要定义“特征”部分的形状。PyTorch 会自动保留第 0 维的 Batch 数量。
        * **与 `view()` 的区别**：
            * `view()` 是张量的方法，需要写在 `forward` 函数里。
            * `nn.Unflatten` 是一个 **Layer（层）**，可以像 `nn.ReLU` 一样直接塞进 `nn.Sequential` 容器，让模型结构更清晰。

    5. 常见应用场景

        1.  **VAE（变分自编码器）**：将隐藏向量（Latent Vector）转回特征图。
        2.  **GAN（生成对抗网络）**：将随机噪声向量转为图像的初始特征图。
        3.  **从特征还原空间信息**：在一些多模态模型中，将一维的语义嵌入向量映射回空间维度进行后续处理。

    你是在处理图像生成类任务（比如自编码器），还是在尝试把全连接网络改造成卷积网络？

* accuracy 有 micro 和 macro 之分吗？

    这是一个很好的问题！**准确率（Accuracy）通常没有 micro 和 macro 之分**，原因让我详细解释一下。

    **核心原因：Accuracy 的本质**

    Accuracy 的定义是：
    \[ \text{Accuracy} = \frac{\text{所有正确预测的样本数}}{\text{总样本数}} \]

    这个定义本身就已经是 **"micro" 视角** 了 - 它统计所有样本的预测结果，然后计算总体比例。

    **为什么 Accuracy 没有 Macro 平均？**

    1. 计算层面的原因
        
        对于多分类问题，Accuracy 只能这样计算：
        ```python
        # 这本身就是 micro 方式
        accuracy = (TP₁ + TP₂ + ... + TPₙ) / N
        ```

        没有办法为每个类别单独计算 "类别准确率"，因为：
        - 每个样本只能属于一个真实类别
        - 每个样本只能被预测为一个类别
        - 正确预测一个样本就同时贡献了该类的 TP 和其他类的 TN

    2. 概念层面的原因
        
        尝试定义 "每个类别的准确率" 会很不直观：
        ```python
        # 不合理的"类别准确率"概念
        accuracy_class_i = (TP_i + TN_i) / N
        ```
        这里 TN_i（真负例）对于类别 i 来说意义模糊，因为它包含了所有其他类别的正确预测。

    **对比 Precision/Recall/F1 的情况**

    为了更清楚理解，我们对比一下：

    | 指标 | 是否有 Micro | 是否有 Macro | 原因 |
    |------|-------------|--------------|------|
    | **Accuracy** | ✅ (本身就是) | ❌ | 天然就是全局统计，无法合理定义"类别准确率" |
    | **Precision** | ✅ | ✅ | 可以为每个类别单独计算 Precision |
    | **Recall** | ✅ | ✅ | 可以为每个类别单独计算 Recall |
    | **F1-score** | ✅ | ✅ | 基于 Precision 和 Recall 计算 |

    **实际计算示例**

    假设一个3分类问题：
    ```python
    import torch
    from torchmetrics import Accuracy

    # 真实标签和预测
    target = torch.tensor([0, 1, 2, 0, 1, 2])  # 真实类别
    preds = torch.tensor([0, 1, 1, 0, 0, 2])   # 预测类别

    # Accuracy 只有一种计算方式
    accuracy = Accuracy(task="multiclass", num_classes=3)
    acc = accuracy(preds, target)
    print(f"Accuracy: {acc:.3f}")  # 输出: 0.667

    # 验证：手动计算
    # 正确预测：索引0,1,5 → 3个正确，总共6个样本 → 3/6=0.5
    ```

    **替代方案：当你想"公平"评估每个类别时**

    如果你想要一个类似 "macro accuracy" 的指标来公平对待所有类别，可以考虑：

    1. **平衡准确率（Balanced Accuracy）**
        
        ```python
        from torchmetrics import Accuracy

        # 平衡准确率：对每个类别的召回率求平均
        balanced_acc = Accuracy(task="multiclass", num_classes=3, average='macro')
        result = balanced_acc(preds, target)
        print(f"Balanced Accuracy: {result:.3f}")
        ```

        实际上，在 `torchmetrics` 中，当你在多分类任务中使用 `average='macro'` 时，它计算的就是平衡准确率。

    2. **每类准确率的概念替代**
        
        虽然不能直接计算"类别准确率"，但可以通过其他方式了解各类别表现：

        ```python
        from torchmetrics import Precision, Recall, ConfusionMatrix

        # 通过其他指标了解各类别表现
        precision = Precision(task="multiclass", num_classes=3, average='none')
        recall = Recall(task="multiclass", num_classes=3, average='none')
        conf_matrix = ConfusionMatrix(task="multiclass", num_classes=3)

        print("每类 Precision:", precision(preds, target))
        print("每类 Recall:", recall(preds, target))
        print("混淆矩阵:\n", conf_matrix(preds, target))
        ```

    **在 torchmetrics 中的实际行为**

    在 `torchmetrics` 的 Accuracy 中：
    - `average='micro'` 或 `average=None`：标准的准确率
    - `average='macro'`：平衡准确率（每类召回率的平均）
    - `average='weighted'`：按类别样本数加权的准确率

    ```python
    # 对比不同 average 参数
    acc_micro = Accuracy(task="multiclass", num_classes=3, average='micro')
    acc_macro = Accuracy(task="multiclass", num_classes=3, average='macro')
    acc_weighted = Accuracy(task="multiclass", num_classes=3, average='weighted')

    print(f"Micro Accuracy: {acc_micro(preds, target):.3f}")    # 标准准确率
    print(f"Macro Accuracy: {acc_macro(preds, target):.3f}")    # 平衡准确率
    print(f"Weighted Accuracy: {acc_weighted(preds, target):.3f}") # 加权准确率
    ```

    **总结**

    - **Accuracy 没有 micro/macro 之分**，因为它天然就是 micro 视角的全局指标
    - 所谓的 "macro accuracy" 实际上计算的是**平衡准确率**
    - 如果你需要公平评估每个类别的表现，应该使用：
      - **平衡准确率**（Balanced Accuracy）
      - **每类的 Precision 和 Recall**
      - **F1-score 的 macro 平均**

    所以当你看到有人讨论 "macro accuracy" 时，他们通常指的是平衡准确率，而不是传统意义上的准确率。

* 简述 pytorch 中 view() 的原型、作用和用法

    在 PyTorch 中，`view()` 是一个非常常用且高效的方法，用于改变张量（Tensor）的形状。它的核心思想是在**不改变底层数据**的情况下，重新解释数据的维度。

    ---

    1. 函数原型

        ```python
        tensor.view(*shape) -> Tensor
        ```
        * **参数 `*shape`**：期望输出的新形状（整数序列）。
        * **返回值**：返回一个新的张量，它与原张量共享内存空间。

    ---

    2. 主要作用

        `view()` 的作用是**维度重塑（Reshaping）**。
        它类似于 NumPy 中的 `reshape()`，但有一个关键约束：**原张量必须是连续的（Contiguous）**。如果张量在内存中不连续（例如经过了 `transpose()` 或 `permute()` 操作），直接调用 `view()` 会报错，此时通常需要先调用 `.contiguous()`。

    ---

    3. 基本用法与示例

        A. 基础变换

        最常见的用法是将多维张量展平或重组。
        ```python
        import torch

        x = torch.randn(4, 4)  # 形状为 (4, 4)
        y = x.view(16)         # 展平为 (16,)
        z = x.view(2, 8)       # 重塑为 (2, 8)
        ```

        B. 使用 `-1` 自动计算维度

        如果你不想手动计算某个维度的数值，可以用 `-1` 代替，PyTorch 会根据总元素数量自动推导该维度的值。
        ```python
        x = torch.randn(4, 4)
        # 只要确定第一维是 2，总元素 16 个，第二维自动推导为 8
        y = x.view(2, -1) 
        print(y.shape) # torch.Size([2, 8])
        ```

        C. 在深度学习模型中的典型场景

        在全连接层（Linear Layer）之前，通常需要将卷积层输出的多维特征图展平：
        ```python
        # 假设 batch_size=32, 图像特征为 64通道, 7x7大小
        features = torch.randn(32, 64, 7, 7)

        # 保留 batch 维度，将其余维度展平
        flattened = features.view(32, -1) 
        print(flattened.shape) # torch.Size([32, 3136])
        ```

    ---

    4. 注意事项（避坑指南）

        1.  **共享内存**：`view()` 产生的新张量与原张量**共用内存**。修改其中一个，另一个也会变。
        2.  **连续性限制**：如果遇到 `RuntimeError: input is not contiguous`，请改用 `x.contiguous().view(...)` 或者直接使用 `x.reshape(...)`（后者会自动处理不连续的情况，但可能会产生内存拷贝）。
        3.  **元素总数必须匹配**：重塑前后的元素总乘积必须一致。例如 $4 \times 4 = 16$，你不能把它 `view` 成 $(3, 5)$。

    ---

    **需要我为你对比一下 `view()`、`reshape()` 和 `transpose()` 之间的深层区别吗？**

* 详细讲讲 unsqueeze()

    简单来说，`unsqueeze()` 的作用是**“升维”**。它在指定位置插入一个长度为 **1** 的新维度。

    这在深度学习中极其重要，因为模型通常要求输入具有特定的维度（例如 Batch 维度），而你的原始数据可能少了一维。

    ---

    1. 函数原型
        
        ```python
        tensor.unsqueeze(dim) -> Tensor
        ```
        * **参数 `dim`**：新维度插入的位置索引。
        * **返回值**：返回一个新张量，与原张量共享内存。

    ---

    2. 核心逻辑：插入“1”
        
        假设你有一个形状为 `(3, 4)` 的张量 $A$：
        * `A.unsqueeze(0)` $\rightarrow$ 形状变为 `(1, 3, 4)`
        * `A.unsqueeze(1)` $\rightarrow$ 形状变为 `(3, 1, 4)`
        * `A.unsqueeze(2)` 或 `A.unsqueeze(-1)` $\rightarrow$ 形状变为 `(3, 4, 1)`

        > **直观理解**：它就像是在现有的维度列表里挤进一个数字 1，其他维度往后挪。

    ---

    3. 为什么要用它？（三大实战场景）

        **场景 A：添加 Batch 维度（最常用）**

        PyTorch 的模型（如 `nn.Conv2d` 或 `nn.Linear`）通常要求输入的第一维是 **Batch Size**。如果你只有一张图片或一个向量，就需要手动补齐这一维。

        ```python
        # 假设是一个单张图片的特征，形状为 [C, H, W]
        image = torch.randn(3, 224, 224) 

        # 模型要求输入为 [Batch, C, H, W]
        input_tensor = image.unsqueeze(0) 

        print(input_tensor.shape) # torch.Size([1, 3, 224, 224])
        ```

        **场景 B：为了进行广播（Broadcasting）**

        当你需要对两个形状不匹配的张量进行加减乘除时，可以用 `unsqueeze` 扩展维度，触发自动广播。

        ```python
        x = torch.tensor([1, 2, 3])      # 形状 (3,)
        y = torch.tensor([10, 20])       # 形状 (2,)

        # 想让 x 的每个元素都加到 y 的每一行上
        # 将 x 变为 (3, 1)，将 y 变为 (1, 2)
        result = x.unsqueeze(1) + y.unsqueeze(0) 
        # 结果形状为 (3, 2)
        ```

        **场景 C：配合特定的 Loss 函数**
        
        有些损失函数（如 `nn.CrossEntropyLoss`）对标签的维度有严格要求，有时需要将 `(N,)` 变成 `(N, 1)`。

        ---

    4. 逆操作：`squeeze()`

        有增就有减。`squeeze()` 用于**“降维”**，它会移除所有长度为 **1** 的维度。

        * `tensor.squeeze()`：移除所有长度为 1 的维度。
        * `tensor.squeeze(dim)`：只移除指定位置且长度为 1 的维度。

        ```python
        x = torch.randn(1, 3, 1, 5)

        y = x.squeeze()     # 形状变为 (3, 5)
        z = x.squeeze(0)    # 形状变为 (3, 1, 5)
        ```

    ---

    5. 进阶小技巧：None 索引
        
        在 Python 中，你可以使用 `None` 这种切片语法来实现和 `unsqueeze` 一样的效果，代码更简洁：

        ```python
        x = torch.randn(3, 4)

        y = x[None, :, :]   # 等同于 x.unsqueeze(0) -> (1, 3, 4)
        z = x[:, None, :]   # 等同于 x.unsqueeze(1) -> (3, 1, 4)
        w = x[..., None]    # 等同于 x.unsqueeze(-1) -> (3, 4, 1)
        ```

    ---

    **总结**

    * **`unsqueeze(dim)`**：在 `dim` 处塞入一个大小为 1 的维度。
    * **核心用途**：对齐维度、适配模型接口、准备广播运算。

    **想看看 `unsqueeze` 是如何帮助你在卷积神经网络中处理单张图片的完整代码吗？**

* 那么在 torchmetric 中，在多分类任务中，如果我想获得一个指定类别的 precision / recall / f1 该怎么办？

    在 `torchmetrics` 中，有几种方法可以获得多分类任务中指定类别的 precision、recall 和 f1。让我为你详细介绍每种方法。

    **方法1：使用 `average='none'` 然后选择特定类别**

    这是最直接的方法：

    ```python
    import torch
    from torchmetrics import Precision, Recall, F1Score

    # 设置任务
    num_classes = 3
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='none')
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='none')
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average='none')

    # 模拟数据
    preds = torch.tensor([0, 1, 2, 0, 1, 2])  # 预测的类别索引
    target = torch.tensor([0, 1, 1, 0, 0, 2]) # 真实的类别索引

    # 计算所有类别的指标
    all_precision = precision_metric(preds, target)
    all_recall = recall_metric(preds, target)
    all_f1 = f1_metric(preds, target)

    print("所有类别的 Precision:", all_precision)
    print("所有类别的 Recall:", all_recall)
    print("所有类别的 F1:", all_f1)

    # 获取指定类别（比如类别1）的指标
    class_idx = 1
    print(f"\n类别 {class_idx} 的指标:")
    print(f"Precision: {all_precision[class_idx]:.3f}")
    print(f"Recall: {all_recall[class_idx]:.3f}")
    print(f"F1: {all_f1[class_idx]:.3f}")
    ```

    输出类似：
    ```
    所有类别的 Precision: tensor([0.6667, 0.5000, 1.0000])
    所有类别的 Recall: tensor([1.0000, 0.5000, 0.5000])
    所有类别的 F1: tensor([0.8000, 0.5000, 0.6667])

    类别 1 的指标:
    Precision: 0.500
    Recall: 0.500
    F1: 0.500
    ```

    **方法2：使用 `class` 参数（某些版本支持）**

    在新版本的 torchmetrics 中，可以直接指定类别：

    ```python
    # 直接计算特定类别的指标
    precision_class1 = Precision(task="multiclass", num_classes=num_classes, average='none', class=1)
    recall_class1 = Recall(task="multiclass", num_classes=num_classes, average='none', class=1)
    f1_class1 = F1Score(task="multiclass", num_classes=num_classes, average='none', class=1)

    # 计算特定类别的指标
    prec_1 = precision_class1(preds, target)
    rec_1 = recall_class1(preds, target)
    f1_1 = f1_class1(preds, target)

    print(f"\n直接计算类别 1:")
    print(f"Precision: {prec_1:.3f}")
    print(f"Recall: {rec_1:.3f}")
    print(f"F1: {f1_1:.3f}")
    ```

    **方法3：使用 `MulticlassPrecision` 等具体类**

    ```python
    from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

    # 为特定类别创建指标
    precision_class_specific = MulticlassPrecision(num_classes=num_classes, average='none')
    recall_class_specific = MulticlassRecall(num_classes=num_classes, average='none')
    f1_class_specific = MulticlassF1Score(num_classes=num_classes, average='none')

    # 使用方法相同
    precisions = precision_class_specific(preds, target)
    recalls = recall_class_specific(preds, target)
    f1_scores = f1_class_specific(preds, target)

    class_of_interest = 2
    print(f"\n类别 {class_of_interest} 的指标:")
    print(f"Precision: {precisions[class_of_interest]:.3f}")
    print(f"Recall: {recalls[class_of_interest]:.3f}")
    print(f"F1: {f1_scores[class_of_interest]:.3f}")
    ```

    **方法4：批量获取所有类别的指标（推荐用于分析）**

    ```python
    def get_per_class_metrics(preds, target, num_classes, class_names=None):
        """获取每个类别的详细指标"""
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        precision = Precision(task="multiclass", num_classes=num_classes, average='none')
        recall = Recall(task="multiclass", num_classes=num_classes, average='none')
        f1 = F1Score(task="multiclass", num_classes=num_classes, average='none')
        
        precisions = precision(preds, target)
        recalls = recall(preds, target)
        f1_scores = f1(preds, target)
        
        print("每个类别的详细指标:")
        print("=" * 50)
        for i, name in enumerate(class_names):
            print(f"{name}: Precision={precisions[i]:.3f}, Recall={recalls[i]:.3f}, F1={f1_scores[i]:.3f}")
        
        return precisions, recalls, f1_scores

    # 使用示例
    class_names = ["Cat", "Dog", "Fish"]
    precisions, recalls, f1_scores = get_per_class_metrics(preds, target, num_classes, class_names)

    # 然后可以轻松访问任何类别
    interest_class = "Dog"
    class_idx = class_names.index(interest_class)
    print(f"\n我们关心的类别 '{interest_class}':")
    print(f"Precision: {precisions[class_idx]:.3f}")
    print(f"Recall: {recalls[class_idx]:.3f}")
    print(f"F1: {f1_scores[class_idx]:.3f}")
    ```

    **方法5：使用 ClassificationReport（最方便的分析工具）**

    ```python
    from torchmetrics import ClassificationReport

    # 使用分类报告一次性获取所有信息
    report = ClassificationReport(task="multiclass", num_classes=num_classes)
    result = report(preds, target)
    print(result)
    ```

    这会输出一个格式化的报告，包含每个类别的 precision、recall、f1-score。

    **实际应用建议**

    1. **对于单类别分析**：使用方法1（`average='none'`）最直接
    2. **对于完整分析**：使用方法4或5查看所有类别
    3. **在训练循环中**：可以同时计算宏观平均和单个类别指标

    ```python
    # 训练时的典型用法
    def calculate_metrics(preds, target, num_classes, class_of_interest):
        # 宏观指标用于整体监控
        macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
        # 特定类别指标用于重点关注
        per_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average='none')
        
        overall_f1 = macro_f1(preds, target)
        class_f1s = per_class_f1(preds, target)
        interest_f1 = class_f1s[class_of_interest]
        
        return overall_f1, interest_f1
    ```

    **总结**：在 `torchmetrics` 中获取指定类别的指标，主要就是使用 `average='none'` 参数来计算所有类别的指标，然后通过索引选择你关心的特定类别。

* torch.cuda 中都有哪些常用函数？

    在 PyTorch 的 GPU 编程中，`torch.cuda` 模块提供了管理显存、查询设备状态及优化计算的核心工具。

    以下是开发中最常用的函数分类整理：

    1. 设备查询与基础检查
        
        这些函数通常用于编写“设备无关”的代码，确保程序在有或没有 GPU 的环境下都能运行。
        * **`torch.cuda.is_available()`**: 返回 `True` 如果当前系统支持 CUDA。
        * **`torch.cuda.device_count()`**: 返回可用 GPU 的数量。
        * **`torch.cuda.current_device()`**: 返回当前正在使用的 GPU 索引。
        * **`torch.cuda.get_device_name(device=None)`**: 获取指定设备的型号名称（如 "NVIDIA GeForce RTX 4090"）。
        * **`torch.cuda.get_device_properties(device)`**: 返回包含显存总量、计算能力（major, minor）等详细信息的属性对象。

    ---

    2. 显存管理（调优必备）
        
        PyTorch 使用缓存分配器（Caching Allocator）来加速内存管理，这些函数能帮你监控和回收显存。
        * **`torch.cuda.empty_cache()`**: 释放 PyTorch 缓存分配器中闲置的显存。如果你在循环中频繁遇到 **OOM (Out of Memory)**，手动调用它可能有帮助（虽然会略微降低速度）。
        * **`torch.cuda.memory_allocated(device=None)`**: 查看当前已分配给 Tensor 的显存大小（单位：字节）。
        * **`torch.cuda.memory_reserved(device=None)`**: 查看缓存分配器目前占用的总显存。
        * **`torch.cuda.max_memory_allocated(device=None)`**: 查看自程序启动以来显存占用的**峰值**。

    ---

    3. 设备状态控制
        
        * **`torch.cuda.set_device(device)`**: 设置当前默认使用的 GPU。
            > **提示**：官方更推荐使用环境变量 `CUDA_VISIBLE_DEVICES` 或在创建 Tensor 时指定 `device` 参数，而不是频繁调用 `set_device`。
        * **`torch.cuda.synchronize(device=None)`**: 等待当前设备上所有流中的任务完成。因为 GPU 操作是异步的，做性能测试（计时）时必须先调用此函数。
        * **`torch.cuda.device(device_idx)`**: 上下文管理器，用于临时切换当前设备。
            ```python
            with torch.cuda.device(1):
                # 这里的操作都会默认在 GPU 1 上执行
                x = torch.tensor([1., 2.]).cuda()
            ```

    ---

    4. 流与事件（进阶优化）
        
        用于实现多个核函数（Kernel）的并行执行或精细的同步控制。
        * **`torch.cuda.Stream()`**: 创建一个新的 CUDA 流，用于实现计算与数据传输的重叠（Overlap）。
        * **`torch.cuda.current_stream()`**: 获取当前的流。
        * **`torch.cuda.Event(enable_timing=True)`**: 创建一个事件，常用于精确测量 GPU 算子的执行时间。

    ---

    5. 随机数种子（复现性）
        
        * **`torch.cuda.manual_seed(seed)`**: 为当前 GPU 设置随机种子。
        * **`torch.cuda.manual_seed_all(seed)`**: 为所有可用的 GPU 设置随机种子。

    **总结建议**

    如果你是刚入门，最需要掌握的是 `is_available()`、`device_count()` 以及如何监控 `memory_allocated()`。当你的模型开始变得庞大且复杂时，`empty_cache()` 和 `synchronize()` 会成为你调试性能和显存压力的重要工具。

    **您是遇到了具体的 OOM 报错，还是想针对多显卡环境进行优化呢？**我可以为您提供针对性的代码示例。




* PyTorch Tensor `.float()` 的作用

    基本作用

    `.float()` 用于将 PyTorch 张量（tensor）的数据类型转换为 **32位浮点型（float32）**。这是深度学习中默认和最常用的数据类型。

    ```python
    import torch

    # 整数张量
    x = torch.tensor([1, 2, 3])
    print(x.dtype)  # torch.int64

    x_float = x.float()
    print(x_float.dtype)  # torch.float32
    print(x_float)  # tensor([1., 2., 3.])
    ```

    应用场景

    1. **神经网络输入**：大多数 PyTorch 模型默认使用 float32

    2. **计算精度要求**：平衡计算精度和内存使用

    3. **与其他操作兼容**：某些操作要求特定数据类型

    类似的数据类型转换函数

    1. **整数类型转换**

        ```python
        .int()        # 转换为 int32
        .long()       # 转换为 int64（常用作标签索引）
        .short()      # 转换为 int16
        .byte()       # 转换为 uint8（常用于图像数据，0-255）
        ```

    2. **浮点类型转换**

        ```python
        .double()     # 转换为 float64（双精度）
        .half()       # 转换为 float16（半精度，节省显存）
        .bfloat16()   # 转换为 bfloat16（Google Brain 格式）
        ```

    3. **其他类型转换**

        ```python
        .bool()       # 转换为布尔型
        .cfloat()     # 转换为复数 float32
        .cdouble()    # 转换为复数 float64
        ```

    使用示例

    ```python
    import torch

    # 创建不同类型的张量
    x = torch.tensor([1, 2, 3])

    # 各种类型转换
    print(x.float())    # tensor([1., 2., 3.]) - float32
    print(x.double())   # tensor([1., 2., 3.], dtype=torch.float64)
    print(x.half())     # tensor([1., 2., 3.], dtype=torch.float16)
    print(x.long())     # tensor([1, 2, 3]) - int64
    print(x.byte())     # tensor([1, 2, 3], dtype=torch.uint8)
    print(x.bool())     # tensor([True, True, True])

    # 处理图像数据（0-255范围）
    image = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    print(image.dtype)  # torch.uint8

    # 神经网络输入需要归一化到 [0,1]
    image_float = image.float() / 255.0
    print(image_float.dtype)  # torch.float32
    ```

    注意事项

    1. **内存占用**：float32 (4字节) vs float64 (8字节) vs int64 (8字节)

    2. **精度需求**：大多数深度学习任务 float32 足够

    3. **GPU 支持**：某些 GPU 操作只支持特定数据类型

    4. **自动混合精度**：训练时可使用 `.half()` 加速

    5. cuda tensor 也可以使用，但是可能支持的类型有限。

    6. 设备保持不变：转换操作不会改变张量所在的设备

* pytorch 使用 datasets 下载数据集时，可以使用`export http_proxy=xxx`指定代理

* `from torchvision import datasets`, datasets 默认会把数据下载到当前文件夹下

* 使用 pip 安装 cuda 12.1 版本的 pytorch，至少要下载 2.8G 的数据

* 使用 pytorch 在 20 个 epoch 内尝试对 sin 曲线过拟合

    ```python
    import torch as t
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torch.optim.sgd as sgd

    def main():
        my_model: nn.Sequential = nn.Sequential(
            nn.Linear(1, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

        l1_loss = t.nn.L1Loss()

        params = my_model.parameters()
        params = my_model.named_parameters()
        for i, (name, param) in enumerate(params):
            name: str
            param: nn.Parameter
            print('i: %d, name: %s, shape:' % (i, name), param.shape)
            # nn.init.xavier_normal_(param.data)
            # nn.init.normal_(param.data, std=3)
            nn.init.uniform_(param.data, -4, 4)

        x_train = t.arange(0, t.pi * 2, 0.01)
        x_train = x_train.reshape(x_train.shape[0], 1)
        print(x_train.shape)
        y_gt = t.sin(x_train)
        y_gt = y_gt.reshape(y_gt.shape[0], 1)

        x_test = x_train

        optim_sgd = sgd.SGD(my_model.parameters(), lr=0.001, weight_decay=0.005)

        for epoch in range(20):
            epoch_loss = 0
            for batch_id in range(x_train.size()[0]):
                x_train_batch = x_train[batch_id]
                y_gt_batch = y_gt[batch_id]
                y_pre_batch: t.Tensor = my_model(x_train_batch)
                loss = l1_loss(y_pre_batch, y_gt_batch)
                loss.backward()
                optim_sgd.step()
                optim_sgd.zero_grad()
                epoch_loss += loss
            print('epoch %d, loss %.2f' % (epoch, epoch_loss))

        y_pre = my_model(x_test)

        plt.figure()
        plt.plot(x_train, y_gt, 'r')
        plt.plot(x_test, y_pre.detach().numpy(), 'b')
        plt.show()

        return

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    i: 0, name: 0.weight, shape: torch.Size([32, 1])
    i: 1, name: 0.bias, shape: torch.Size([32])
    i: 2, name: 2.weight, shape: torch.Size([1, 32])
    i: 3, name: 2.bias, shape: torch.Size([1])
    torch.Size([629, 1])
    epoch 0, loss 242.71
    epoch 1, loss 566.46
    epoch 2, loss 266.65
    epoch 3, loss 172.81
    epoch 4, loss 136.53
    epoch 5, loss 123.09
    epoch 6, loss 112.77
    epoch 7, loss 107.53
    epoch 8, loss 97.30
    epoch 9, loss 86.32
    epoch 10, loss 79.17
    epoch 11, loss 68.87
    epoch 12, loss 61.15
    epoch 13, loss 53.27
    epoch 14, loss 46.70
    epoch 15, loss 40.95
    epoch 16, loss 35.92
    epoch 17, loss 31.23
    epoch 18, loss 28.30
    epoch 19, loss 26.95
    ```

    经测试，在 init parameter 时，需要将范围设置得大一点，拟合效果才比较好。此时 lr 取值适中比较好，太大（0.01）或者太小（0.0001）效果都不太好。

    这个有点像遗传算法中变异率，变异率太小时无法搜索到全局最优点。

    weight decay 对结果影响较大，经测试，取值 0.005 时效果最好。

    Linear 的宽度变到 128 并不能显著影响结果。增加一层 32 节点 Linear 后效果会好些。

    l1 loss 变换 mse loss 并不能显著影响最终结果。

    最终的效果：

    <div style='text-align:center'>
    <img width=500 src='../../Reference_resources/ref_34/pic_1.png'>
    </div>

* `nn.init.xavier_normal_()`只能作用于 shape size 为 2 的 tensor

    如果一个 tensor 的 shape size 为 1，那么这个函数会报错。典型的场景就是 Linear 层的 bias 参数。

    `nn.init.normal_()`没有这个限制，可作用于任意 shape size 的 tensor。

* pytorch 在对 tensor 做计算时，会对齐低维度（假设越往右维度越低），对高维度做 broadcast

    ```py
    import torch as t
    import torch.nn as nn

    def main():
        my_model: nn.Sequential = nn.Sequential(
            nn.Linear(1, 32),  # 接收 (*, 1) 形式的输入
            nn.Sigmoid(),
            nn.Linear(32, 1)  # 输出也为 (*, 1) 的形式
        )

        x = t.rand(1)
        y: t.Tensor = my_model(x)
        print(y)  # (1, )
        print(y.shape)

        x = t.rand(5, 1)
        y: t.Tensor = my_model(x)
        print(y)  # (5, 1)
        print(y.shape)

        x = t.rand(3, 5, 1)  # (15, 1)
        y: t.Tensor = my_model(x)
        print(y)  # (3, 5, 1)
        print(y.shape)
        return

    if __name__ == '__main__':
        main()
    ```

* 使用 cuda 训练神经网络拟合 sine 函数

    ```py
    import torch as t
    from torch import nn
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    class MyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(1, 32, dtype=t.float32)
            self.act_1 = nn.ReLU()
            self.linear_2 = nn.Linear(32, 64, dtype=t.float32)
            self.act_2 = nn.ReLU()
            self.linear_3 = nn.Linear(64, 1, dtype=t.float32)
            
        def forward(self, x):
            x = self.linear_1(x)
            x = self.act_1(x)
            x = self.linear_2(x)
            x = self.act_2(x)
            x = self.linear_3(x)
            return x
            
    model = MyBlock()
    model.cuda()
    x = t.tensor([1], device='cuda', dtype=t.float32)
    y = model(x)
    print(y)

    optimizer = t.optim.SGD(model.parameters(), lr=0.01)

    x = t.arange(0, 2 * math.pi, 0.01, device='cuda')
    x = x.reshape(x.shape[0], 1)
    print(x.device)
    print(x.shape)
    y_gt = t.sin(x)
    print(y_gt.device)
    print(y_gt.shape)

    calc_loss = nn.MSELoss()

    model.train()
    for epoch in range(3000):
        output = model(x)
        loss = calc_loss(output, y_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 300 == 0:
            print('epoch: %d, epoch loss: %f' % (epoch, loss))

    model.eval()
    X = np.arange(0, 2 * np.pi, 0.01)
    print(X.shape)
    with t.no_grad():
        X_cuda = t.from_numpy(X.reshape((629, 1)).astype('float32')).cuda()
        pred = model(X_cuda)
    print(pred.shape)
    pred_cpu = pred.cpu().detach().numpy()
    print(pred.shape)

    Y = pred.flatten()
    print(Y.shape)
    Y_cpu = Y.cpu()

    Y_gt = np.sin(X)
    print(Y_gt.shape)

    fig = plt.figure()
    plt.plot(X, Y_cpu)
    plt.plot(X, Y_gt, 'r')
    fig.savefig('curve.png')
    plt.show()
    ```

    输出：

    ```
    tensor([-0.1299], device='cuda:0', grad_fn=<ViewBackward0>)
    cuda:0
    torch.Size([629, 1])
    cuda:0
    torch.Size([629, 1])
    epoch: 0, epoch loss: 0.549724
    epoch: 300, epoch loss: 0.130778
    epoch: 600, epoch loss: 0.108640
    epoch: 900, epoch loss: 0.088971
    epoch: 1200, epoch loss: 0.071567
    epoch: 1500, epoch loss: 0.056711
    epoch: 1800, epoch loss: 0.044086
    epoch: 2100, epoch loss: 0.033470
    epoch: 2400, epoch loss: 0.024818
    epoch: 2700, epoch loss: 0.018170
    (629,)
    torch.Size([629, 1])
    torch.Size([629, 1])
    torch.Size([629])
    (629,)
    ```

    图片输出：

    <div style='text-align:center'>
    <img width=700 src='../../Reference_resources/ref_26/pic_2.png'>
    </div>

    注：

    * 只需要调用`model.cuda()`，就可以把模型里的参数转移到 gpu 设备上

        不需要使用等号：`model = model.cuda()`

    * 在创建 tensor 时，可以使用`device='cuda'`参数将 tensor 创建到 gpu memory 上

    * 对于一个已经创建的 tensor，使用`.cuda()`也可以在 cuda 上创建一个副本

        但是它并不是 in-place 修改，所以经常需要用等号覆盖：`x = x.cuda()`

    * v100 gpu 的带宽大，但是延迟也高，所以尽量使用大 bach size，减少 i/o 通信凑数比较好

* 使用两层 linear + activation 层拟合一条 sin 曲线

    ```python
    import torch as t
    from torch import nn
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    class MyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(1, 32, dtype=t.float32)
            self.act_1 = nn.ReLU()
            self.linear_2 = nn.Linear(32, 64, dtype=t.float32)
            self.act_2 = nn.ReLU()
            self.linear_3 = nn.Linear(64, 1, dtype=t.float32)
            
        def forward(self, x):
            x = self.linear_1(x)
            x = self.act_1(x)
            x = self.linear_2(x)
            x = self.act_2(x)
            x = self.linear_3(x)
            return x
            
    my_block = MyBlock()
    x = t.tensor([1], dtype=t.float32)
    y = my_block(x)
    print(y)

    params = my_block.parameters()
    optimizer = t.optim.SGD(my_block.parameters(), lr=0.01)
    x = t.arange(0, 2 * math.pi, 0.01)
    x = x.reshape(x.shape[0], 1)
    print(x.shape)
    y_gt = t.sin(x)
    print(y_gt.shape)

    calc_loss = nn.MSELoss()

    my_block.train()
    for epoch in range(3000):
        output = my_block(x)
        loss = calc_loss(output, y_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch: %d, epoch loss: %f' % (epoch, loss))

    my_block.eval()
    X = np.arange(0, 2 * np.pi, 0.01)
    print(X.shape)
    with t.no_grad():
        pred = my_block(t.from_numpy(X.reshape((629, 1)).astype('float32')))
    print(pred.shape)
    pred = pred.detach().numpy()
    print(pred.shape)

    Y = pred.flatten()
    print(Y.shape)

    Y_gt = np.sin(X)
    print(Y_gt.shape)

    plt.plot(X, Y)
    plt.plot(X, Y_gt, 'r')
    plt.show()
    ```

    output:

    ```
    tensor([0.0518], grad_fn=<ViewBackward0>)
    torch.Size([629, 1])
    torch.Size([629, 1])
    epoch: 0, epoch loss: 0.390523
    epoch: 1, epoch loss: 0.367919
    epoch: 2, epoch loss: 0.359639
    epoch: 3, epoch loss: 0.351311
    epoch: 4, epoch loss: 0.343440
    epoch: 5, epoch loss: 0.336687
    epoch: 6, epoch loss: 0.330301
    epoch: 7, epoch loss: 0.324224
    epoch: 8, epoch loss: 0.318377
    epoch: 9, epoch loss: 0.312717
    epoch: 10, epoch loss: 0.307221
    epoch: 11, epoch loss: 0.301877
    epoch: 12, epoch loss: 0.296678
    epoch: 13, epoch loss: 0.291617
    epoch: 14, epoch loss: 0.286691
    epoch: 15, epoch loss: 0.281895
    epoch: 16, epoch loss: 0.277223
    epoch: 17, epoch loss: 0.272677
    epoch: 18, epoch loss: 0.268257
    epoch: 19, epoch loss: 0.263966
    epoch: 20, epoch loss: 0.259807
    epoch: 21, epoch loss: 0.255789
    epoch: 22, epoch loss: 0.251917
    epoch: 23, epoch loss: 0.248195
    epoch: 24, epoch loss: 0.244619
    ...
    epoch: 2990, epoch loss: 0.005504
    epoch: 2991, epoch loss: 0.005398
    epoch: 2992, epoch loss: 0.005502
    epoch: 2993, epoch loss: 0.005396
    epoch: 2994, epoch loss: 0.005500
    epoch: 2995, epoch loss: 0.005395
    epoch: 2996, epoch loss: 0.005501
    epoch: 2997, epoch loss: 0.005396
    epoch: 2998, epoch loss: 0.005502
    epoch: 2999, epoch loss: 0.005396
    (629,)
    torch.Size([629, 1])
    (629, 1)
    (629,)
    (629,)
    ```

    图片输出：

    <div style='text-align:center'>
    <img width=700 src='../../Reference_resources/ref_26/pic_1.png'>
    </div>

    注：

    * 三层 linear 层即可拟合一条 sine 曲线，前两层 linear 层后需跟一个 activation layer，最后一层 linear 层不要跟 activation layer

    * activation 选 ReLU 的效果要比 Sigmoid 好很多

    * 在设置 layer 的参数时，不需要考虑 batch 维度，后面在 input data 和 ground truth 在第一个维度处加上 batch 后，会自动广播

    * num epoch 至少得有 3000 才能有拟合的效果，不清楚怎么把 epoch 减少一点

        这点可能和 input data 的 sample 有关。如果每次都只在训练集上均匀采样几个训练数据，效果可能会好一些

    * 在开始训练前要先运行下`my_model.train()`，在预测的时候运行下`my_model.eval()`

        目前不清楚这个有没有用，也不清楚原理。看教程上这么写的。

    * `loss`只有为 scalar 时才可以直接`backward()`

    * optimizer 需要手动清零梯度：`optimizer.zero_grad()`

    * 各种各样的优化器放在`torch.optim`模块下

    * 不需要记录梯度时，可以使用`with torch.no_grad():`

* define a customized module

    define a customized module:

    ```py
    import torch as t
    from torch import nn

    class MyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(1, 32, dtype=t.float32)
            self.linear_2 = nn.Linear(32, 1, dtype=t.float32)
            
        def forward(self, x):
            x = self.linear_1(x)
            x = self.linear_2(x)
            return x

    my_block = MyBlock()
    x = t.tensor([1], dtype=t.float32)
    y = my_block(x)
    print(y)
    ```

    output:

    ```
    tensor([0.8138], grad_fn=<ViewBackward0>)
    ```

    note:

    * 如果要写自定义的 Module，需要自己写一个 class 然后继承`nn.Module`

    * class 中至少需要实现 2 个函数：

        `__init__(self)`和`forward(self, x)`

        其中`__init__(self)`需要调用`super().__init__()`初始化基类。

    * `nn`模块预先定义了很多带 grad 的参数，比如`nn.Linear()`就是一个定义了矩阵乘法参数的 module

        `nn.Linear()`执行的数学操作是`y = Ax + b`，其中`A`是带 grad 的矩阵，`b`是带 grad 的向量。

    * 使用`MyBlock`定义出来的变量可以直接作为函数使用，

        `y = my_block(x)`

        其实调用的就是`forward()`函数。

    * torch 的 grad 记录默认是打开的，所以可以直接 backward 拿到各个参数的梯度

        ```py
        y.backword()
        params = list(my_block.parameters())
        param = params[0]
        print(param.grad)
        ```

        output:

        ```
        tensor([[-0.0996],
                [-0.0583],
                [ 0.1459],
                [ 0.0239],
                [ 0.0167],
                [ 0.1735],
                [-0.1674],
                [ 0.0899],
                [ 0.0500],
                [-0.1269],
                [ 0.0909],
                [-0.0420],
                [-0.1046],
                [-0.0768],
                [-0.0824],
                [-0.1338],
                [ 0.1352],
                [-0.0911],
                [ 0.0707],
                [ 0.1464],
                [ 0.1145],
                [ 0.0605],
                [-0.0283],
                [ 0.0986],
                [ 0.0872],
                [-0.1290],
                [ 0.1707],
                [-0.0363],
                [ 0.0244],
                [-0.0511],
                [-0.1112],
                [ 0.0171]])
        ```

