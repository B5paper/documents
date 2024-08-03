# PyTorch Note

## cache

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

