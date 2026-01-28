# jax note

## cache

* 简述 python 中 jax 的用法

    JAX 是一个用于高性能数值计算的 Python 库，结合了 NumPy 的易用性、自动微分 和 硬件加速（CPU/GPU/TPU）。以下是其核心用法的简述：

    1. 类 NumPy 语法

        JAX 提供了类似 NumPy 的 API（jax.numpy），可无缝替换：

        ```python
        import jax.numpy as jnp
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sin(x)  # 支持向量化操作
        ```

    2. 自动微分（Grad）

        通过 grad 可自动计算梯度：

        ```python
        from jax import grad
        def loss(params, x):
            return jnp.sum(params * x ** 2)
        grad_loss = grad(loss)  # 对第一个参数（params）求导
        gradients = grad_loss(2.0, 3.0)  # 返回梯度值
        ```

        * 支持高阶导数（grad(grad(...))）。

        * value_and_grad 可同时返回函数值和梯度。

    3. 即时编译（JIT）

        使用 jit 将函数编译为 XLA 优化代码，加速计算：

        ```python
        from jax import jit
        @jit
        def slow_function(x):
            return x ** 2 + jnp.sin(x).mean()
        fast_result = slow_function(jnp.ones(1000))  # 首次编译，后续加速
        ```

        * 适用于循环或复杂运算。

        * 注意：编译要求输入形状固定（动态形状需额外处理）。

    4. 自动向量化（vmap）

        将函数自动向量化，避免显式循环：

        ```python
        from jax import vmap
        def apply_matrix(x):
            return jnp.dot(jnp.ones((3, 3)), x)
        batched_apply = vmap(apply_matrix)  # 自动批处理
        result = batched_apply(jnp.ones((10, 3)))  # 输入 shape=(10,3)
        ```

        * 提升批处理效率，简化代码。

    5. 并行化（pmap）

        在多设备（如多个 GPU）上并行执行：

        ```python
        from jax import pmap
        @pmap
        def add_one(x):
            return x + 1
        multi_gpu_result = add_one(jnp.ones(4, device_count()))  # 分布在多个设备上
        ```

        * 需注意设备间通信开销。

    6. 随机数生成

        JAX 采用显式的随机状态（PRNGKey）：

        ```python
        from jax import random
        key = random.PRNGKey(0)
        subkey, key = random.split(key)  # 分割密钥
        x = random.normal(subkey, shape=(10,))
        ```

        * 确保随机操作的确定性和可重现性。

    7. 与机器学习库集成

        * Flax：基于 JAX 的神经网络库。

        * Optax：优化器库。

        * Haiku：模块化深度学习库。

    注意事项

    * 函数纯度：JAX 要求函数为纯函数（无副作用），否则行为可能不符合预期。

    * 数组不可变：JAX 数组不可直接修改，需使用 x.at[index].set(value) 等操作。

    * 设备内存管理：注意 GPU/TPU 内存限制，可结合 jit 控制内存使用。

    简单示例

    ```python
    import jax
    import jax.numpy as jnp

    # 自动微分 + JIT
    @jax.jit
    @jax.grad
    def f(x):
        return jnp.sum(jnp.sin(x) ** 2)

    print(f(jnp.array([1.0, 2.0])))  # 输出梯度
    ```

    JAX 的核心优势在于将 自动微分、编译优化 和 并行化 统一在 NumPy 接口下，适合科学计算、机器学习研究和高性能数值任务。


## topics
