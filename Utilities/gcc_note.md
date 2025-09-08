# Gcc Note

## cache

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
