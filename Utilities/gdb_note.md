# Gdb Note

<https://interrupt.memfault.com/blog/advanced-gdb>

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

