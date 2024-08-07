# Linux C Programming

Linux 下的 C 语言编程。

## cache

* `getopt()`

    linux 中处理 argc, argv 的函数。

    example:

    ```c
    #include <stdio.h>
    #include <unistd.h>

    int main(int argc, char **argv)
    {
        int opt;
        do {
            opt = getopt(argc, argv, "csp:");
            if (opt == -1)
                break;
            printf("opt: %c, optarg: %s\n", opt, optarg);
        } while (opt != -1);
        return 0;
    }
    ```

    运行：

    ```
    hlc@hlc-VirtualBox:~/Documents/Projects/cpp_test$ ./main -c
    opt: c, optarg: (null)
    hlc@hlc-VirtualBox:~/Documents/Projects/cpp_test$ ./main -s
    opt: s, optarg: (null)
    hlc@hlc-VirtualBox:~/Documents/Projects/cpp_test$ ./main -p
    ./main: option requires an argument -- 'p'
    opt: ?, optarg: (null)
    hlc@hlc-VirtualBox:~/Documents/Projects/cpp_test$ ./main -p 1234
    opt: p, optarg: 1234
    ```

    可以看到，对于格式`csp:`，当程序后跟的参数是`c`或`s`时，不需要再加额外的值。`p`的后面加了个冒号，表示`p`后面必须要跟一个参数值，否则就报错。

    具体的参数值可以使用`optarg`全局变量得到。

    如果解析结束，`getopt()`会返回`-1`.


    ref:

    1. Using getopt in C with non-option arguments
    
        <https://stackoverflow.com/questions/18079340/using-getopt-in-c-with-non-option-arguments>

    2. getopt(3) — Linux manual page

        <https://man7.org/linux/man-pages/man3/getopt.3.html>

    3. 12.17 getopt.h

        <https://www.gnu.org/software/gnulib/manual/html_node/getopt_002eh.html>

