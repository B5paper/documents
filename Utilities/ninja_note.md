# Ninja Note

## cache

* 简述 ninja 的用法

    Ninja 是一个高效的构建系统，专注于速度。它通常作为更高级构建系统（如 CMake、GNN）的后端使用，而非直接编写。

    核心特点

    * 极速启动和运行：增量构建极快

    * 基于文件的依赖关系：只重新编译需要的内容

    * 并行构建：充分利用多核 CPU

    基本用法

    1. 生成 build.ninja 文件

        ```bash
        # 通过 CMake 生成
        cmake -G Ninja -B build

        # 通过 meson 生成
        meson setup builddir
        ```

    2. 常用命令

        ```bash
        # 构建项目
        ninja

        # 构建特定目标
        ninja target_name

        # 并行构建（默认自动检测核心数）
        ninja -j8

        # 清理构建产物
        ninja -t clean

        # 查看依赖关系
        ninja -t deps

        # 显示所有目标
        ninja -t targets
        ```

    3. 直接使用 ninja 文件

        ```bash
        # 指定构建文件
        ninja -f build.ninja

        # 详细输出
        ninja -v
        ```

    ninja.build 文件示例

    ```ninja
    # 变量
    cflags = -Wall -O2

    # 规则
    rule cc
      command = gcc $cflags -c $in -o $out
      description = CC $out

    rule link
      command = gcc $in -o $out
      description = LINK $out

    # 构建语句
    build main.o: cc main.c
    build utils.o: cc utils.c
    build app: link main.o utils.o

    # 默认目标
    default app
    ```

    与 CMake 配合使用

    ```bash
    # 配置阶段
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

    # 构建阶段
    cd build && ninja

    # 或直接
    cmake --build build --parallel
    ```

    实用技巧

    * 性能调优：

        ```bash
        # 限制并行任务数
        ninja -j4

        # 负载控制
        ninja -l 2
        ```

    * 调试构建：

        ```bash
        # 解释模式（不实际执行）
        ninja -n

        # 查看图表依赖
        ninja -t graph > graph.dot
        ```

    * 重编译检查：

        ```bash
        # 为什么需要重构建？
        ninja -t commands target
        ```

    Ninja 的设计哲学是"只做一件事并做到极致"，因此它没有条件语句、函数等复杂功能，所有逻辑都在生成阶段由前端工具处理。

## topics