## cache

* 详细讲讲 asan 的作用和用法

    **ASan (AddressSanitizer)** 是 Google 开发的一个快速内存错误检测工具，集成在 GCC 和 Clang 编译器中。它通过在编译时插桩（Instrumentation）和运行时替换内存分配函数，能够以极高的效率捕捉 C/C++ 程序中的各种内存越界和泄漏问题。

    相比于传统的内存检测工具（如 Valgrind 带来 10~50 倍的降速），ASan 的性能非常优秀，通常只会让程序变慢 **2倍** 左右，内存开销增加 2~4 倍。因此它非常适合用于日常开发、测试和 CI/CD 自动化流水线中。

    ---

    ## 一、 ASan 的核心作用（它能检测什么？）

    ASan 能够精准识别以下几种最致命的 C/C++ 内存错误：

    * **堆内存越界 (Heap buffer overflow)**：访问了动态分配（`malloc`/`new`）内存块之外的区域。
    * **栈内存越界 (Stack buffer overflow)**：局部变量或数组越界访问。
    * **全局变量越界 (Global buffer overflow)**：全局或静态变量越界访问。
    * **释放后使用 (Use after free / UAF)**：使用了已经被 `free` 或 `delete` 的指针（僵尸指针）。
    * **返回后使用 (Use after return)**：函数返回后，访问了该函数内部的局部变量指针（需通过环境变量额外开启）。
    * **离开作用域后使用 (Use after scope)**：在代码块（如 `if` 或 `for` 循环）之外，使用了块内定义的局部变量。
    * **初始顺序初始化错误 (Initialization order bugs)**：不同编译单元（`.cpp` 文件）间全局变量初始化的顺序依赖问题。
    * **内存泄漏 (Memory leaks)**：程序结束时未释放的内存（ASan 集成了 LSan 功能）。

    ---

    ## 二、 ASan 的基本用法

    使用 ASan 非常简单，不需要修改任何代码，只需要在编译和链接时加上特定的参数。

    ### 1. 编译和链接参数

    最核心的参数是 `-fsanitize=address`。为了能看到具体的报错行号，还需要加上 `-g`（生成调试信息）。

    ```bash
    # 编译并链接
    g++ -fsanitize=address -g main.cpp -o main

    ```

    > **优化建议：** 建议搭配 `-O1` 或 `-O2` 优化级别，既能保证代码运行不会太慢，ASan 也能更好地进行内联代码插桩，同时保留清晰的调用栈。

    ### 2. 运行程序

    直接像平常一样运行编译出来的可执行文件即可：

    ```bash
    ./main

    ```

    如果程序存在内存错误，ASan 会在触发错误的那一行代码处**立即拦截程序**，并向终端输出一份详细的错误报告，然后退出。

    ---

    ## 三、 代码示例与报错分析

    ### 示例 1：堆内存越界 (Heap Buffer Overflow)

    编写一个简单的越界访问代码 `test.cpp`：

    ```cpp
    #include <stdlib.h>

    int main() {
        int *array = (int*)malloc(10 * sizeof(int));
        // 故意越界访问第 11 个元素
        int res = array[10]; 
        free(array);
        return res;
    }

    ```

    ### 编译并运行：

    ```bash
    gcc -fsanitize=address -g test.cpp -o test
    ./test

    ```

    ### ASan 报错报告解读：

    ASan 的报告通常非常长，但关键信息在最前面的几行：

    ```text
    =================================================================
    ==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x604000000038 at pc 0x5555555551a4 bp 0x7fffffffdc40 sp 0x7fffffffdc38
    READ of size 4 at 0x604000000038 thread T0
        #0 0x5555555551a3 in main /path/to/test.cpp:6  <-- 精确指出了触发错误的代码行号
    ...
    allocated by thread T0 here:
        #0 0x7ffff7ec8bc8 in malloc
        #1 0x55555555516e in main /path/to/test.cpp:4  <-- 指出了这块内存是在哪里分配的
    ...
    SUMMARY: AddressSanitizer: heap-buffer-overflow /path/to/test.cpp:6 in main

    ```

    * **第一行**：明确指出了错误类型是 `heap-buffer-overflow`（堆越界）。
    * **READ of size 4**：表示在尝试读取一个 4 字节的大小（即 `int`）。
    * **第一个调用栈 (`#0`)**：告诉你崩溃发生在 `test.cpp` 的第 6 行。
    * **第二个调用栈 (`allocated by...`)**：告诉你被越界的这块内存在 `test.cpp` 的第 4 行被分配。

    ---

    ## 四、 常用高级配置（环境变量）

    ASan 提供了丰富的运行时环境变量 `ASAN_OPTIONS`，可以用来调整它的行为。

    使用语法：

    ```bash
    ASAN_OPTIONS=key1=value1:key2=value2 ./main

    ```

    ### 常用配置项：

    | 配置项 | 默认值 | 作用 |
    | --- | --- | --- |
    | `halt_on_error` | `true` | 检测到第一个错误时是否立即崩溃退出。设为 `false` 可以让程序继续运行以捕获更多错误。 |
    | `detect_leaks` | `true` | 是否在程序退出时检测内存泄漏（LSan）。Linux 下默认开启。 |
    | `detect_stack_use_after_return` | `false` | 是否检测“返回后使用局部变量”的错误。开启此项会带来额外的性能开销。 |
    | `log_path` | `stderr` | 将 ASan 的报告重定向输出到指定文件（如 `log_path=/tmp/asan.log`），适合后台服务。 |

    **示例：**

    ```bash
    # 允许程序发生错误不退出，并检测返回后使用的错误
    export ASAN_OPTIONS=halt_on_error=false:detect_stack_use_after_return=1
    ./main

    ```

    ---

    ## 五、 使用注意事项与最佳实践

    1. **不要在生产环境/发布版中开启**：ASan 会增加额外的内存开销和 CPU 损耗，且如果遇到错误会直接中断程序，不适合高并发、低延迟的生产环境。
    2. **全量编译**：尽可能将程序中的所有源文件以及静态库都加上 `-fsanitize=address` 编译。如果链接了没有开启 ASan 的第三方动态库，ASan 可能无法拦截该库内部的内存错误。
    3. **与 C++ 容器结合**：现代 GCC/Clang 的 C++ 标准库（libstdc++/libc++）已经对 `std::vector` 等容器进行了 ASan 适配。即使是 `vector` 内部未使用的预留空间（Capacity 范围内但超过 Size 的部分），越界访问也能被 ASan 捕捉。

