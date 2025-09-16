# Torch Note

## cache

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