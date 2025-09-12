# Torch Note

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