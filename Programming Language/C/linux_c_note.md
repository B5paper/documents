# Linux C Programming

Linux 下的 C 语言编程。

## cache

* `asprintf()`

    `asprintf()`是 GNU 的 C 标准库的一个 extension，同时做了`malloc()`和`sprintf()`的工作。

    syntax:

    ```c
    #include <stdio.h>

    int asprintf(char **restrict strp, const char *restrict fmt, ...);
    int vasprintf(char **restrict strp, const char *restrict fmt,
                    va_list ap);
    ```

    return value:

    return the string length if succeed, return `-1` if fail.

    note:

    This pointer should be passed to free(3) to release the allocated storage when it is no longer needed.

    example:

    ```c
    #include <stdio.h>
    #include <string.h>
    #include <stdlib.h>

    int main()
    {
        char *buf;
        int ret = asprintf(&buf, "hello, %s", "world");
        if (ret < 0)
        {
            printf("fail to asprintf, ret: %d\n", ret);
            return -1;
        }
        int len = strlen(buf);
        printf("ret: %d, len: %d\n", ret, len);
        free(buf);
        return 0;
    }
    ```

    output:

    ```
    ret: 12, len: 12
    ```

    compile output:

    ```
    gcc -g main.c -o main
    main.c: In function ‘main’:
    main.c:8:15: warning: implicit declaration of function ‘asprintf’; did you mean ‘vsprintf’? [-Wimplicit-function-declaration]
        8 |     int ret = asprintf(&buf, "hello, %s", "world");
        |               ^~~~~~~~
        |               vsprintf
    ```

    说明：

    1. 似乎头文件不是 gnu c 的头文件，是标准 c 的头文件，导致写代码时没有函数参数提示，但是能正常编译和执行。

        正常写 c 语言程序不建议使用这个函数。

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

