# Gcc Note

## cache

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
