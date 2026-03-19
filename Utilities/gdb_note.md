# Gdb Note

## cache

* 简述 gdb 如何 attach 调试

    **gdb attach 调试简述**

    **gdb attach** 用于调试已经运行的进程，而不需要从程序开始处启动。

    **基本步骤**

    1. 查找进程 PID
        
        ```bash
        ps aux | grep 程序名          # 查看进程PID
        pidof 程序名                  # 直接获取PID
        pgrep 程序名                  # 另一种获取PID的方式
        ```

    2. 启动 gdb 并 attach

        ```bash
        # 方法一：先启动gdb，再attach
        gdb
        (gdb) attach <PID>

        # 方法二：直接attach
        gdb -p <PID>
        gdb --pid=<PID>

        # 方法三：同时指定程序文件
        gdb 程序名 <PID>
        gdb 程序名 -p <PID>
        ```

    **常用操作**

    **调试控制**

    ```bash
    (gdb) continue 或 c            # 继续执行程序
    (gdb) break 或 b               # 设置断点
    (gdb) next 或 n                # 下一步（跳过函数）
    (gdb) step 或 s                # 步入函数
    (gdb) print 或 p               # 打印变量值
    (gdb) backtrace 或 bt          # 查看调用栈
    (gdb) info threads             # 查看所有线程
    (gdb) thread <ID>              # 切换线程
    ```

    **分离进程**

    ```bash
    (gdb) detach                    # 分离进程，让程序继续运行
    (gdb) quit                      # 退出gdb
    ```

    **注意事项**

    1. **权限要求**

       - 需要与被调试进程相同的用户权限或root权限
       - 某些系统可能需要设置ptrace权限

    2. **编译要求**

       - 程序需包含调试信息（编译时加 `-g` 选项）
       - 最好使用 `-O0` 避免优化导致调试困难

    3. **进程状态**

       - attach会暂停目标进程
       - 生产环境attach可能导致服务短暂中断
       - detach后程序继续正常执行

    4. **常见问题**

       ```bash
       # 权限不足时尝试
       sudo gdb -p <PID>
       
       # 或临时修改ptrace限制
       echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
       ```

    **实用场景**

    - 调试已运行的服务进程
    - 分析程序卡死或性能问题
    - 排查生产环境问题（谨慎使用）

* 调试 c++ 代码时，在宏前打断点停不下来，为什么？

    **宏前打断点停不下来的原因**

    **主要原因**

    1. **宏不是函数/代码行**
       - 宏在**预处理阶段**被展开，编译器生成的调试信息中没有宏的符号
       - gdb实际看到的是宏展开后的代码，而非宏定义本身

    2. **调试信息不包含宏**
       - 默认编译选项下，调试信息（DWARF格式）不包含宏定义
       - 即使有 `-g` 选项，宏信息通常也不被保留

    **示例说明**

    ```cpp
    // 源代码 test.cpp
    #define DEBUG_PRINT(x) do { \
        std::cout << "Debug: " << x << std::endl; \
    } while(0)

    void func() {
        DEBUG_PRINT("hello");  // 想在这行打断点
    }
    ```

    ```bash
    # 预处理后实际代码
    void func() {
        do { 
            std::cout << "Debug: " << "hello" << std::endl; 
        } while(0);
    }
    ```

    **解决方案**

    1. **在宏展开后的位置打断点**

        ```bash
        # 查看宏展开后的代码行
        gdb test
        (gdb) info line func         # 查看func函数对应的代码行
        (gdb) break test.cpp:5       # 在宏调用的代码行打断点
        (gdb) break test.cpp:6       # 或在宏展开后的具体行
        ```

    2. **使用保存宏信息的编译选项**
        
        ```bash
        # GCC编译时添加 -g3 选项（保存宏信息）
        g++ -g3 test.cpp -o test

        # 然后可以直接在宏名打断点
        gdb test
        (gdb) break DEBUG_PRINT       # 现在可以直接对宏打断点
        (gdb) info macro DEBUG_PRINT  # 查看宏定义
        (gdb) macro expand DEBUG_PRINT("hello")  # 展开宏
        ```

    3. **在宏展开的代码处打断点**

        ```bash
        # 查看预处理后的代码
        g++ -E test.cpp > test.i      # 生成预处理文件
        # 查看test.i找到实际代码行，然后打断点
        ```

    4. **在宏调用的上下文打断点**

        ```cpp
        void func() {
            DEBUG_PRINT("hello");
            // 可以在调用宏的下一行打断点
            normal_function();  // 在这里打断点
        }
        ```

    5. **临时将宏改为函数（调试用）**

        ```cpp
        // 调试时临时改为内联函数
        inline void debug_print(const char* x) {
            std::cout << "Debug: " << x << std::endl;
        }
        #define DEBUG_PRINT(x) debug_print(x)  // 或直接使用函数
        ```

    **调试宏的技巧**

    ```bash
    # 查看宏信息（需要 -g3）
    (gdb) info macro DEBUG_PRINT  # 显示宏定义
    (gdb) macro list              # 列出所有宏
    (gdb) macro expand DEBUG_PRINT("test")  # 展开宏

    # 在宏调用处设置条件断点
    (gdb) break test.cpp:5 if strcmp(x, "error") == 0
    ```

    **总结**

    - **根本原因**：宏在预处理阶段被展开，没有对应的调试符号
    - **最佳方案**：使用 `-g3` 编译选项保留宏信息
    - **备选方案**：在宏调用的行号或宏展开后的代码位置打断点
    - **调试技巧**：善用 `info macro`、`macro expand` 等命令辅助调试

* `my_print()`在调试时无法一次性输出的问题

    我写的代码中 

    ```cpp
    void my_print() {
        printf("xxx 1\n");
        printf("xxx 2\n");
        // xxxx
    }
    ```

    在给 my_print() 打上断点时，由于是多线程环境，通常 printf("xxx 2\n"); 还没来得及输出，就被其他地方的断点断下了。我想让 my_print() 作为一个“原子操作”，完整打印出函数内的所有内容，怎么办？

    这是一个非常经典的调试痛点。在多线程环境下，调试器（如 GDB 或 LLDB）默认的 **“All-stop 模式”** 会在任何一个线程触发断点时，立即挂起进程中的**所有**线程。

    要让你的 `my_print()` 看起来像“原子”一样执行完再断下，可以参考以下几种方案：

    1. 修改调试器的线程行为（最推荐）

        你可以告诉调试器，在当前线程运行完特定逻辑之前，不要去管其他线程。

        * **GDB 用户：**
        可以使用 `scheduler-locking` 命令。当你停在 `my_print` 的第一行时：
        ```bash
        set scheduler-locking on  # 锁定调度器，只运行当前线程
        next                      # 或者 continue 到函数末尾的断点

        ```


        这样，只有当前线程会运行，其他线程保持挂起，直到你执行 `set scheduler-locking off`。
        * **LLDB 用户：**
        使用 `--run-mode` 参数：
        ```bash
        thread step-over --run-mode one-thread

        ```

    2. 巧用“条件断点”或“断点命令”

        如果你不想手动输入命令，可以利用调试器的自动化功能。

        在 `printf("xxx 1\n");` 设置断点，并配置如下动作（以 GDB 为例）：

        ```gdb
        break my_print
        commands
          silent
          set scheduler-locking on
          continue
        end

        ```

        然后在 `my_print` 的**最后一行**设置另一个断点，并在那里恢复 `set scheduler-locking off`。

    3. 代码层面的“伪原子”打印

        如果调试器设置太麻烦，最稳妥的方法是改变打印方式，**减少 I/O 次数**。

        `printf` 是行缓冲的，多次调用会给其他线程留出大量的插入机会。你可以将内容合并到一个缓冲区一次性输出：

        ```c
        void my_print() {
            char buf[256];
            // 将所有内容格式化到内存字符串中
            snprintf(buf, sizeof(buf), "xxx 1\nxxx 2\n");
            // 一次性系统调用，大大降低被中断的概率
            fputs(buf, stdout); 
        }

        ```

    4. 使用互斥锁（Mutex）

        如果你是想防止**输出乱码**（即其他线程的 printf 混进了这两行之间），最标准的方法是加锁：

        ```c
        void my_print() {
            static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
            pthread_mutex_lock(&lock);
            printf("xxx 1\n");
            printf("xxx 2\n");
            pthread_mutex_unlock(&lock);
        }

        ```

        **注意：** 这只能解决输出不被干扰，如果你在函数内部打了断点，调试器依然会挂起整个进程。

    **建议总结**

    | 需求场景 | 推荐方案 |
    | --- | --- |
    | **调试时想一口气跑完函数** | 使用 GDB 的 `set scheduler-locking on` |
    | **防止多线程日志交织乱序** | 将多个 `printf` 合并为一个，或加 `mutex` |
    | **自动化调试** | 编写调试器脚本 (Breakpoint Commands) |

    你想尝试配置一下 GDB 的自动化脚本来一键实现这个功能吗？

    注：

    1. 起起来 ai 也没有什么好的解决办法

* 在GDB中，\b 和 \w 是用于寄存器值显示的特殊格式说明符：

    * `\b`: 按字节显示

        以十六进制格式显示寄存器的低 8 位（一个字节）

        只显示寄存器的最低字节内容

    * `\w`: 按字显示

        以十六进制格式显示寄存器的低 16 位（一个字）

        只显示寄存器的低两个字节

    example:

    ```
    # 显示EAX寄存器的完整值
    (gdb) p $eax
    $1 = 0x12345678

    # 只显示EAX的低字节（0x78）
    (gdb) p/x $eax \b
    $2 = 0x78

    # 只显示EAX的低16位（0x5678）
    (gdb) p/x $eax \w
    $3 = 0x5678
    ```

    `p/x`表示以十六进制显示，`p`表示以十进制显示。如果之前使用过 p/x，后续 p 可能保持十六进制格式。

    `p/t` → 二进制, `p/d` → 十进制

* gdb 锁定线程

    `set scheduler-locking on`

* gdb 中的`call`与`print`（`p`）都支持调用函数，唯一的不同是，`call func()`调用函数时，如果`func()`有返回值，那么返回值被记录到`$N`临时变量中，如果没有返回值（`void`类型），那么不记录`$N`。使用`p func()`时，即使`func()`返回值为 void，也会出现类似`$1 = void`的提示信息。

* gdb 中`p $rax`可以查看函数的返回值。

    `p $`返回上一个有返回值函数的返回值，如果上一个有返回值的语句是个表达式，那么返回表达式的结果。

* void 在 gdb 中的作用

    在 gdb 中每使用`call`调用一次函数，函数的返回值都会被存放在`$1`, `$2`等临时变量里，如果希望 gdb 忽略函数的返回值，不放到临时变量里，那么可以使用`call (void)my_func()`进行调用。`(void)`和函数名之间可以有空格。

    如果源代码中本身就有`(void)`，那么 gdb 也会忽略函数被调用时的返回值。

* gdb 可以使用`call`命令调用函数，`(gdb) call func(param_1, param_2)`

    被 call 的函数定义所在的文件必须使用`-g`编译才行。

    可以使用`call <var_name> = <val>`给变量赋值。如果不使用`call`，直接写`<var_name> = <val>`会报错。

* gdb 远程调试

    1. 安装 gdbserver

        `sudo apt install gdbserver`

    2. 在 host A 上创建新目录，写入下面的文件

        `main.c`:

        ```c
        #include <stdio.h>

        int main()
        {
            printf("hello from gdb server\n");
            return 0;
        }
        ```

        编译：`gcc -g main.c -o main`

    3. 在 host A 上使用 gdbserver 运行`main`

        `gdbserver :5432 ./main`

    4. 在 host B 上启动 gdb，加载符号表

        `gdb`

        ```
        (gdb) symbol-file ./main
        Reading symbols from ./main...
        ```

        这里的`./main`必须和 host A 上编译出来的相同才行。

    5. 在 host B 上执行

        `(gdb) target remote  <host_A_ipv4>:4321`

        根据提示操作，即可开始调试。

        ```
        (gdb) c
        Continuing.
        Reading /lib/x86_64-linux-gnu/libc.so.6 from remote target...

        Breakpoint 1, main () at main.c:6
        6	    return 0;
        (gdb) 
        ```

        这中间可能需要联网下载一些符号表。（如果是断网条件下，这些符号表该如何获得？）

* 对于多层依赖的库和 app 文件，编译时在哪一个文件上加`-g`，调试时就只能 hit 到哪个文件的断点。

    假如这个文件为`debug_valid.c`，如果这个文件的上一层和下一层库/app在编译时没有加上`-g`参数，那么就无法 hit 断点。

    即使 hit 了`debug_valid.c`文件的断点，程序暂停时上一层和下一层暂停的代码行上下文。

* <https://interrupt.memfault.com/blog/advanced-gdb>

* `file <exe_path>`

    加载某个程序

* `run`

    从头开始执行程序

* `break <func_name>`
* `break <src_path>:<line>`

    下断点

* `continue`

    从断点处继续执行

* `step`

    单步调试，相当于 step into

* `next`

    相当于 step over

* `Enter`键

    重复上一个命令

* `print <var_name>`

    打印变量值

    * `print/x <var_name>`

        以十六进制形式打印变量值

* `watch <var_name>`

    监视一个变量，当变量的值被修改时，程序中断

* `backtrace`

    显示调用函数栈

* `where`

    和`backtrace`功能差不多。教程上说`backtrace`主要用于程序崩溃时，而`where`用于任何正常情况，但是这个说法还没得到证实。

* `finish`

    runs until the current function is finished

* `delete`

    deletes a specified breakpoint

* `info breakpoints`

    显示所有的断点的信息

* `help`

    显示帮助信息

* `break <file>:<line> if i >= ARRAYSIZE`

    条件断点，conditional breakpoints

* `point <pointer>`

    显示指针所指向的地址

* `print <pointer>-><var_name>`

    显示成员的值

* `print (*<pointer>).<var_name>`

    对指针解引用后，打印成员值

* `print *<pointer>`

    显示整个结构体的值

* `print ptr->ptr2->data`

    链式显示指针值

* `list`

    显示上下文代码

* `frame`

    显示当前行

* 显示所有断点

    `info breakpoints`

* `set print elements 1000`

* `set print pretty on`

* `x/s pointer`

使用`root`权限调试：

1. 将`/usr/bin/gdb`改个名字，比如改成`hgdb`

    然后在`/usr/bin`下创建一个新文件`gdb`：

    ```bash
    #!/bin/bash
    pkexec /usr/bin/hgdb "$@"
    ```

    再加上可执行属性：`chmod +x /usr/bin/gdb`

    这样就可以了。

1. 另外一些或许有用的方案：<https://stackoverflow.com/questions/40033311/how-to-debug-programs-with-sudo-in-vscode>

1. 使用 gdb 调 rust 的 test 代码：<https://whamcloud.github.io/Online-Help/docs/Contributor_Docs/cd_Debugging_Rust_Tests.html>

1. 另外一个参考资料：<https://itecnote.com/tecnote/c-how-to-debug-programs-with-sudo-in-vscode/>

显示很长的字符串：<https://stackoverflow.com/questions/233328/how-do-i-print-the-full-value-of-a-long-string-in-gdb>

1. pretty print: <https://sourceware.org/gdb/onlinedocs/gdb/Pretty_002dPrinter-Commands.html>

## Problem shooting

1. How can I get address info on a structure member?

    <https://stackoverflow.com/questions/33673343/how-can-i-get-address-info-on-a-structure-member>

1. print struct in gdb

    <https://gnu.gpp.help.narkive.com/pJBTiPUJ/hash-map-print-in-gdb>

1. 有关 cout

    `std::cout`在 gdb 中是不可见的。可以自己写个函数 wrap 一层。

1. Error: `Cannot resolve function printf to any overloaded instance`

1. 在 gdb 中 print `wchar_t` 字符串

    <https://stackoverflow.com/questions/1721545/how-can-i-display-unicode-strings-while-debugging-on-linux>

1. gdb 设置字符集：<https://sourceware.org/gdb/onlinedocs/gdb/Character-Sets.html>

    <https://www-zeuthen.desy.de/unix/unixguide/infohtml/gdb/Character-Sets.html>

1. 在调试 c 代码时看到对应的汇编代码：<https://juejin.cn/post/6844903625077063693>

1. 用 gdb 调试汇编：<https://zhuanlan.zhihu.com/p/259625135>

    <https://resources.infosecinstitute.com/topic/debugging-your-first-x86-program/>

1. a tutorial about gdb: <https://www-users.cse.umn.edu/~kauffman/tutorials/gdb.html>

