# PyTorch Note

## cache

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

