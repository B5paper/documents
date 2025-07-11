# Gdb Note

## cache

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

