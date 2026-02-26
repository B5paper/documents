# Unclassified Packages Note

笔记量比较小的 python package 的笔记。如果某个 package 的笔记积累足够，则独立到文件夹下。

## cache

* 简述msgpack-numpy的用法

    `msgpack-numpy` 是一个用于扩展 MessagePack 序列化协议以支持 NumPy 数据类型的库。它允许你将包含 NumPy 数组的对象序列化为高效的二进制格式，并在反序列化时恢复为 NumPy 数组。

    基本用法

    1. 安装

        ```bash
        pip install msgpack-numpy
        ```

    2. 序列化和反序列化

        ```python
        import msgpack
        import msgpack_numpy as m
        import numpy as np

        # 必须调用此函数来注册解包钩子
        m.patch()

        # 创建一个包含 NumPy 数组的对象
        data = {
            'array': np.array([1, 2, 3, 4, 5]),
            'matrix': np.random.rand(3, 3),
            'scalar': np.float64(3.14),
            'integer': np.int32(42)
        }

        # 序列化
        packed = msgpack.packb(data)

        # 反序列化
        unpacked = msgpack.unpackb(packed)

        print(unpacked['array'])  # 恢复为 NumPy 数组
        print(type(unpacked['array']))  # <class 'numpy.ndarray'>
        ```

    3. 处理不同数据类型

        ```python
        import msgpack
        import msgpack_numpy as m
        import numpy as np

        m.patch()

        # 支持多种 NumPy 数据类型
        data = {
            'int_array': np.array([1, 2, 3], dtype=np.int32),
            'float_array': np.array([1.1, 2.2, 3.3], dtype=np.float64),
            'bool_array': np.array([True, False, True]),
            'complex_array': np.array([1+2j, 3+4j])
        }

        packed = msgpack.packb(data)
        unpacked = msgpack.unpackb(packed)
        ```

    4. 使用不同的序列化设置

        ```python
        import msgpack
        import msgpack_numpy as m
        import numpy as np

        # 自定义设置
        def custom_pack(obj):
            # 自定义序列化逻辑
            pass

        # 默认使用高效的二进制序列化
        packed = msgpack.packb(data, default=m.encode)
        unpacked = msgpack.unpackb(packed, object_hook=m.decode)
        ```

    高级用法

    处理大型数组

    ```python
    import msgpack
    import msgpack_numpy as m
    import numpy as np

    m.patch()

    # 大型数组会被高效压缩
    large_array = np.random.rand(1000, 1000)
    data = {'large': large_array}

    # 使用压缩选项减少大小
    packed = msgpack.packb(data, default=m.encode, use_bin_type=True)
    unpacked = msgpack.unpackb(packed, object_hook=m.decode)
    ```

    嵌套数据结构

    ```python
    import msgpack
    import msgpack_numpy as m
    import numpy as np

    m.patch()

    # 复杂的嵌套结构
    data = {
        'metadata': {'name': 'test', 'version': 1.0},
        'arrays': {
            'train': np.random.rand(100, 10),
            'test': np.random.rand(20, 10),
            'labels': np.array([0, 1, 2, 3])
        },
        'list_of_arrays': [np.array([1,2]), np.array([3,4])]
    }

    packed = msgpack.packb(data)
    unpacked = msgpack.unpackb(packed)
    ```

    注意事项

    1. **必须调用 `m.patch()`** 来注册解码钩子，否则无法正确恢复 NumPy 数组
    2. **性能考虑**：对于非常大的数组，序列化可能需要较多内存和时间
    3. **版本兼容性**：确保序列化和反序列化使用相同版本的 msgpack-numpy
    4. **数据类型**：某些特殊的 NumPy 数据类型可能需要额外处理

    完整示例

    ```python
    import msgpack
    import msgpack_numpy as m
    import numpy as np

    # 初始化
    m.patch()

    # 创建数据
    data = {
        'name': 'dataset',
        'features': np.random.rand(5, 3),
        'targets': np.array([0, 1, 0, 1, 0]),
        'weights': np.ones(5) * 2.5
    }

    # 保存到文件
    with open('data.msgpack', 'wb') as f:
        packed = msgpack.packb(data)
        f.write(packed)

    # 从文件读取
    with open('data.msgpack', 'rb') as f:
        unpacked = msgpack.unpackb(f.read())

    print(unpacked['features'])  # 恢复为 NumPy 数组
    ```

    `msgpack-numpy` 特别适合需要在不同系统间传输 NumPy 数据，或者需要高效存储大量数值数据的场景。

* 简述 memoryview 的用法

    memoryview 简述

    **memoryview** 是 Python 的内置类型，它允许你在**不复制数据**的情况下访问支持缓冲区协议的对象（如 bytes、bytearray、array 等）的内部数据。

    主要特点

    - **零拷贝**：直接操作原对象的内存，避免数据复制
    - **内存效率高**：适合处理大文件或大数据流
    - **支持切片**：切片返回新的 memoryview，仍不复制数据
    - **可修改**：如果原对象可变，可通过 memoryview 修改数据

    基本用法

    ```python
    # 创建 memoryview
    data = bytearray(b'hello world')
    view = memoryview(data)

    # 访问数据
    print(view[0])        # 104 (ASCII 'h')
    print(view[1:5])      # <memory at 0x...> 切片不复制数据

    # 修改数据（原对象可变时）
    view[0] = 74          # 修改为 'J'
    print(data)           # bytearray(b'Jello world')

    # 转换为其他类型
    bytes_view = view.cast('B')  # 转为无符号字节视图
    ```

    常见应用场景

    1. **处理大文件**

        ```python
        with open('large_file.bin', 'rb') as f:
            data = f.read()
            view = memoryview(data)
            # 无需复制即可处理部分数据
            header = view[:100]
        ```

    2. **网络编程**

        ```python
        data = bytearray(1024)
        view = memoryview(data)
        nbytes = sock.recv_into(view)  # 直接接收数据到内存
        ```

    3. **高效的数据切片**

        ```python
        large_bytes = b'a' * 1000000
        view = memoryview(large_bytes)

        # 不会复制 500KB 数据
        slice_view = view[:500000]
        ```

    注意事项

    - 原对象被修改时，memoryview 会反映变化
    - 原对象被删除后，memoryview 可能失效
    - 使用 `.release()` 可以手动释放内存视图
    - 某些操作（如排序）需要先转换为 list


## topics
