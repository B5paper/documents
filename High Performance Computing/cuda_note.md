# CUDA Note

## cache

* cuda stream 在创建时，靠`cudaSetDevice()`来指定具体的 device

    这个操作看起来比较像 opengl，提前指定好上下文环境。

* 可以直接用 apt 安装 nvidia-cuda-tookit，这样可以安装上`nvcc`等开发环境

    cuda 的版本会落后一些，但是提供了提示和编译环境，可以用来跳转和写代码。

* nvcc 需要安装 g++ 编译器