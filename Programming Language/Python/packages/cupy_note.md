## cache

* cupy 在安装时，需要提前选择 cuda 版本，否则会现场编译，容易出错且非常耗时

    **如果是 CUDA 12.8 环境，那么执行：**

    ```bash
    pip install cupy-cuda12x
    ```

    Cupy 的 `cuda12x` 系列包支持 CUDA 12.0 到 12.x 的所有版本，通常不需要从源码构建。

