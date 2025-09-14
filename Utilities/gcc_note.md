# Gcc Note

## cache

* 不能从`.so`中拿到库中所有的函数，因为有的`.c`中的函数可能是`static`的。也不能从`*.o`中拿到所有的函数，因为不一定所有的`.c`都生成`.o`，还有可能直接生成`.so`。

* `--enable-new-dtags`

    --enable-new-dtags 是 GNU 链接器 (ld) 的一个选项。它的主要作用是：在创建可执行文件或动态库时，使用 RUNPATH 而不是 RPATH。

    example:

    ```bash
    # 启用 new dtags，生成 RUNPATH
    gcc -Wl,--enable-new-dtags,-rpath=/opt/mylib -o my_program main.c

    # 禁用 new dtags（或不指定），生成 RPATH
    gcc -Wl,--disable-new-dtags,-rpath=/opt/mylib -o my_program main.c
    # 或者直接
    gcc -Wl,-rpath=/opt/mylib -o my_program main.c
    ```

    checkout:

    ```bash
    readelf -d my_program | grep -E '(RUNPATH|RPATH)'
    ```

    * 启用`--enable-new-dtags`后，输出会显示`0x000000000000001d (RUNPATH) Library runpath: [/opt/mylib]`

    * 禁用时，输出会显示`0x000000000000000f (RPATH) Library rpath: [/opt/mylib]`

* `RUNPATH`

    存储在可执行文件或动态库（.elf 文件）中的一个参数，它的主要作用是指定程序在运行时搜索动态链接库（.so 文件）的额外路径列表。

    动态链接器的典型搜索顺序如下（简化版，体现了 RUNPATH 的关键位置）：

    1. LD_LIBRARY_PATH 环境变量指定的目录。

    2. 可执行文件中嵌入的 RPATH 目录（如果存在，且没有 RUNPATH）。

    3. 系统缓存文件 /etc/ld.so.cache 中列出的目录。

    4. 默认的系统库目录，如 /lib 和 /usr/lib。

    5. 可执行文件中嵌入的 RUNPATH 目录（如果存在）。

    设置`RUNPATH`的方法：
    
    `gcc -Wl,-rpath=/path/to/your/libs -o my_program my_program.c`

* `-rpath-link`

    -rpath-link 是 GCC/ld 链接器选项，用于在 动态链接（shared library） 时指定额外的 库搜索路径

    example:

    ```bash
    gcc main.o -o myprog -L/usr/lib/mylibs -lmylib -Wl,-rpath-link,/usr/lib/mylibs
    ```

    解释：

    * `-L/usr/lib/mylibs`：告诉链接器查找`-lmylib`的路径。

    * `-Wl,-rpath-link,/usr/lib/mylibs`：告诉链接器，如果`libmylib.so`还依赖其他库，去`/usr/lib/mylibs`查找它们。

    * 不会在最终可执行文件中嵌入`/usr/lib/mylibs`。

    `-L`只能用来查找`-l`指定的库，如果要找`-l`指定的库的依赖库 (transitive shared libraries)，只能使用`-rpath-link`。

* `LIBRARY_PATH`与 `LD_LIBRARY_PATH`的区别

    `LIBRARY_PATH`用于编译时指示链接器（ld）在链接阶段查找需要链接的库文件（如 libxxx.a 或 libxxx.so）的目录列表。它用于解决 -l 选项指定的库的路径。功能与`-L`比较像。

    搜索顺序通常是：

    1. 首先搜索 -L 指定的目录。

    2. 然后搜索 LIBRARY_PATH 指定的目录。

    3. 最后搜索标准系统库目录（如 /usr/lib, /lib）。

    `LD_LIBRARY_PATH`用于运行时指使`ld.so`或`ld-linux.so`到对应的目录下查找动态共享库（.so 文件）。

* `ld.so`, `ld-linux.so`

    ld.so 和 ld-linux.so 都是 动态链接器/加载器（Dynamic Linker/Loader）。它们的主要作用是在程序运行时（run-time） 完成以下工作：

    * 加载共享库： 将程序所依赖的共享库（.so 文件）从文件系统加载到进程的地址空间。

    * 符号重定位： 解析程序与共享库之间、以及共享库相互之间的函数和变量引用（即符号），将其替换为实际的内存地址。

    * 处理依赖关系： 递归地处理所有传递性依赖（即库所依赖的其他库）。

    ld.so 批的是 /lib/ld.so

    ld-linux.so 指的是 /lib/ld-linux.so.{1,2,3} 或 /lib64/ld-linux-x86-64.so.2

    `ld.so`主要服务于a.out 格式的二进制文件（古老，已淘汰）。在现代系统中，ld.so 通常只是一个指向 ld-linux.so 的符号链接。

    `ld-linux.so`主要服务于 ELF 格式的二进制文件（现代标准）。

     ELF（Executable and Linkable Format）

* `-rpath`

    `-rpath <dir>`在生成可执行文件时，把指定的目录`<dir>`记录到可执行文件的 运行时库搜索路径（runtime library search path） 中。当程序运行时，动态链接器会优先在这些路径下查找共享库（`.so`），而不用依赖用户再设置`LD_LIBRARY_PATH`。

    example:

    ```bash
    gcc main.c -o main -L/opt/mylib -lmylib -Wl,-rpath,/opt/mylib
    ```

    其中，`-Wl`表示将接下来的参数传递给`ld`，多个传递参数用逗号`,`分隔。

    `ld`中的写法：

    ```bash
    ld -rpath /opt/mylib
    ```

    与其他选项的区别:

    * `-L<dir>`：告诉编译/链接阶段去哪里找库；

    * `-rpath <dir>`：告诉运行时去哪里找库；

    * `LD_LIBRARY_PATH`：环境变量，运行时指定库路径，作用类似 -rpath，但依赖用户环境。

    优先级：

    1. `LD_LIBRARY_PATH`

    2. `-rpath`

    3. 系统默认目录：`/lib`, `/usr/lib`, `/lib64`, `/usr/lib64` …

    4. /etc/ld.so.cache 里缓存的目录

    指定多个`rpath`：

    ```bash
    # method 1
    gcc main.c -Wl,-rpath,/opt/mylib1:/opt/mylib2 -lmylib

    # method 2
    gcc main.c -Wl,-rpath,/opt/mylib1 -Wl,-rpath,/opt/mylib2
    ```

    `rpath`可以用相对路径，但不推荐。动态链接器解析这个相对路径时，不会用你运行程序时的当前目录，而是相对于 可执行文件被加载时的工作目录（current working directory，CWD）。（未验证）

* 如果使用`gcc main.c /path/to/libxxx.so -o main`编译，那么`/path/to/libxxx.so`会被硬编码到`main`中。这个路径可以是软链接。

    这种情况下，如果`libxxx.so`換了位置，那么使用`LD_LIBRARY_PATH`也是无效的。

* gcc 编译时，不会记录`-L`的目录，只会指定`-l`指定的 so 文件。

* 在使用 gcc 编译时，如果有这样的编译命令：

    ```bash
    main: main.c A.o B.o
        gcc main.c A.o B.o -o main

    A.o:
        gcc -c A.c -o A.o

    B.o:
        gcc -c B.c -o B.o
    ```

    其中`B.o`中的函数依赖`A.o`中的函数，那么交换`gcc main.c A.o B.o -o main`中`A.o`和`B.o`的顺序，不影响`main`的编译。

* 有关 gcc 编译顺序的猜想

    * 猜想：结尾是`.o`，`.so`以及`-lxxx`的顺序，假如 A 依赖 B，那么`B`应该在`A`后面，即`gcc A B -o a.out`

    * 猜想：所有`.o`文件必须写在`.so`的前面，`-lxxx`也属于`.so`文件

* gcc 与 g++

    `gcc -g client.c ../rdma/tests/ibv_tests/utils/sock.o -o client`这个命令可以通过编译，但是把 gcc 換成 g++ 就不行。

## 编译 dll

`g++ -shared mylib.cpp -o mylib.dll`

`g++ main.cpp mylib.dll -o a.exe`或者`g++ main.cpp -L. -lmylib`

## 其他

编译程序时，将程序中的字符串存储成指定的编码：`-fexec-charset=gbk`
