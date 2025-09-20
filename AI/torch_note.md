# Torch Note

## cache

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

* python class 中定义成员变量

    1. 在`__init__()`或其他成员函数中，使用`self.xxx = yyy`定义成员变量

        ```py
        class DynamicClass:
            def __init__(self):
                self.defined_in_init = "I'm from init" 

            def add_attribute_later(self):
                self.defined_later = "I was created later!"

        # 使用
        obj = DynamicClass()
        print(obj.defined_in_init) # 正常工作

        # print(obj.defined_later) # 这里会报错，因为还没有执行定义它的方法

        obj.add_attribute_later() # 调用方法，动态创建了成员
        print(obj.defined_later)  # 现在可以正常工作了
        ```

    2. 使用类属性

        ```py
        class MyClass:
            # 这是类属性
            class_attr = "I'm a class attribute"

            def __init__(self, instance_attr):
                # 这是实例属性
                self.instance_attr = instance_attr

        # 使用
        obj1 = MyClass("Obj1 value")
        obj2 = MyClass("Obj2 value")

        # 访问实例属性：每个对象独有
        print(obj1.instance_attr) # Obj1 value
        print(obj2.instance_attr) # Obj2 value

        # 访问类属性：所有对象共享，也可以通过类本身访问
        print(obj1.class_attr)    # I'm a class attribute
        print(obj2.class_attr)    # I'm a class attribute
        print(MyClass.class_attr) # I'm a class attribute
        ```

        共享性：所有实例对象共享同一个类属性。如果通过类名修改它（如 MyClass.class_attr = "new"），所有实例看到的都会改变。

        实例访问的陷阱：如果你通过实例对类属性进行赋值（如 obj1.class_attr = "new for obj1"），你实际上是在该实例的命名空间内创建了一个新的同名实例属性，它会遮蔽（shadow）掉类属性。此时，obj1.class_attr 是实例属性，而 obj2.class_attr 和 MyClass.class_attr 仍然是原来的类属性。

    3. 使用`@property`装饰器

        ```py
        class Circle:
            def __init__(self, radius):
                self.radius = radius # 这里只存储了半径

            @property
            def area(self):
                # 面积不需要存储，每次访问时根据半径计算
                return 3.14159 * self.radius ** 2

            @property
            def diameter(self):
                return self.radius * 2

        # 使用
        c = Circle(5)
        print(c.radius)   # 5 (实例属性)
        print(c.diameter) # 10 (看起来是属性，实则是方法计算的结果)
        print(c.area)     # 78.53975 (看起来是属性，实则是方法计算的结果)

        # c.area = 100 # 这会报错，因为@property默认是只读的
        ```

    在使用类成员时，如果不知道初始值，可以使用`Nonde`:

    ```py
    class User:
        # 使用 None 作为占位符，表示这些属性需要后续初始化
        name = None
        email = None
        age = None
    ```

    但是只有`None`无法提供类型信息，可以使用类型注解（Type Hints）配合 None:

    ```py
    class User:
        name: str | None = None
        email: str | None = None
        age: int | None = None
    ```

    不可以只写类型注解，不写初始化值：

    ```py
    class User:
        name: str          # 这只是类型注解
        age: int = 0       # 这是真正的属性定义 + 类型注解

    # 测试
    user = User()
    print(user.age)        # 正常工作，输出: 0
    print(user.name)       # 报错！AttributeError: 'User' object has no attribute 'name'
    ```

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

* 获取 hugging face 的 imdb 数据集

    ```py
    from datasets import load_dataset
    dataset = load_dataset('imdb')
    print(dataset['train'][0])
    ```

    数据会被下载到`~/.cache/huggingface/datasets`中。imdb 数据集大小为 128 M。

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

* Pytorch - Index-based Operation

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

## topics