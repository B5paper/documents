# C Note

C 语言标准库 tutorial：<https://www.tutorialspoint.com/c_standard_library/index.htm>

有时间了看看。

## cache

* c 定长数据类型 in `<cstdint>`, reference

    <https://en.cppreference.com/w/cpp/types/integer.html>

* c 中的变长参数函数

    example:

    `main.c`:

    ```cpp
    #include <cstdarg>
    #include <cstdio>

    int indented_print(int stage, ...) {
        for (int i = 0; i < stage; ++i) {
            for (int j = 0; j < 4; ++j) {
                putchar(' ');
            }
        }

        va_list args;
        va_start(args, stage);
        const char* format_str = va_arg(args, const char*);
        vprintf(format_str, args);
        va_end(args);
        return 0;
    }

    int indented_print_2(int stage, const char *format_str, ...) {
        for (int i = 0; i < stage; ++i) {
            for (int j = 0; j < 4; ++j) {
                putchar(' ');
            }
        }

        va_list args;
        va_start(args, format_str);
        vprintf(format_str, args);
        va_end(args);
        return 0;
    }

    int parse_args(int int_val, ...) {
        printf("int val: %d\n", int_val);

        va_list args;
        va_start(args, int_val);

        float float_val = va_arg(args, double);
        printf("float val: %.2f\n", float_val);

        const char *msg = va_arg(args, const char*);
        printf("msg: %s\n", msg);

        va_end(args);
        return 0;
    }

    int main() {
        indented_print(0, "this is a title:\n");
        indented_print(1, "name: %s\n", "hlc");
        indented_print(1, "age: %d\n", 29);
        indented_print(1, "friends:\n");
        indented_print(2, "1. Alice\n");
        indented_print(2, "2. Bob\n");

        indented_print_2(0, "title\n");
        indented_print_2(1, "child_1\n");
        indented_print_2(2, "child_2\n");

        parse_args(123, 456.7, "hello, world");

        return 0;
    }
    ```

    compile:

    `gcc main.c -o main`

    run:

    `./main`

    output:

    ```
    this is a title:
        name: hlc
        age: 29
        friends:
            1. Alice
            2. Bob
    title
        child_1
            child_2
    int val: 123
    float val: 456.70
    msg: hello, world
    ```

    C 变长参数函数使用三个点`...`来标记，使用几个宏函数来提取变长列表中的各个参数。以`parse_args()`为例，`va_list args;`定义`args`来表示变长参数，`va_start(args, int_val);`表示从`int_val`的下一个参数开始提取，接下来使用`float float_val = va_arg(args, double);`，`const char *msg = va_arg(args, const char*);`根据指定类型去从`args`中提取参数，最后调用`va_end(args);`释放资源，结束提取。使用这几个宏时，必须包含头文件`#include <cstdarg>`。

    `indented_print()`和`indented_print_2()`则是使用嵌套变长参数函数实现了带缩进的 printf 函数。

    说明：

    1. 变长参数列表至少需要填一个参数，再写`...`，不能只写`...`，比如`my_print(...);`，否则`va_start()`的第二个参数没法填。

    1. `va_arg(args, double);`在提取小数类型时，`va_arg()`只能填`double`，不能填`float`，否则会报错。

    1. 嵌套调用变长参数函数时，`args`的含义为从已经解析的参数的下一个开始，传入内层函数中。另外内层 print 函数必须使用`vprintf()`，否则会报 segmentation fault 错误。这个函数是专门为内层变长参数函数准备的。

    1. 如果无法一开始就确定所有参数的类型，通常做法是仿照`printf()`，第一个参数传字符串，字符串里包含后面参数类型的信息，比如`%d`,`%f`之类，在函数里只需要解析字符串就可以了。

* 因为有补码，所以`int a = 0xffffffff;`并不是`INT32_MIN`，而是`-1`。而`INT32_MIN`的 16 进制是`0x80000000`。

* 当`int32_t`类型的最高位为`1`时，C 语言中，若将其转换为`uint64_t`，那么高 32 位，以第 31 位（从 0 开始索引）都是`1`，`[30:0]`位仍保持原来的数据。

    `int16_t`, `int8_t`, `char`都是同理。

    以`int8_t`为例，当其为`int8_t a = 0b10000010;`时，`uint16_t b = a;`为`1111 1111 1000 0010`。

* 移位操作并不是循环移位，当移位的位数超出整数的二进制位数时，可能会出现问题

    ```cpp
    #include <stdio.h>
    #include <stdint.h>

    int main()
    {
        for (int i = 1; i <= 256; ++i)
        {
            int aaa = i;
            uint64_t bbb = aaa << 56;
            bbb = bbb >> 56;
            printf("i: %d, %lu; ", i, bbb);
        }
        putchar('\n');
        return 0;
    }
    ```

    > If the number is shifted more than the size of the integer, the behavior is undefined. For example, 1 << 33 is undefined if integers are stored using 32 bits. For bit shift of larger values 1ULL<<62  ULL is used for Unsigned Long Long which is defined using 64 bits that can store large values.

* `fgets()`用法及注意事项

    syntax:

    ```c
    char* fgets(char *s, int n, FILE *stream);
    ```

    读取文件中的内容，遇到`\0`或者`\n`时返回，数据存到`s`中，并返回`s`。
    
    `n`表示`s`的长度。由于`s`的最后一个字节要存储`\0`，所以`fgets()`一共要从文件读取`n - 1`个字节。

    example:

    ```c
    #include <string.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        FILE *f = fopen("test.txt", "wb");
        fwrite("hello\0world\n", 12, 1, f);
        fclose(f);

        f = fopen("test.txt", "r");
        char buf[128] = {0};
        char *ret = fgets(buf, 13, f);
        fclose(f);

        printf("%p, %s\n", buf, buf);
        printf("%p, %s\n", ret, ret);
        return 0;
    }
    ```

    output:

    ```
    0x7ffcf977aa70, hello
    0x7ffcf977aa70, hello
    ```

    可以看到，要求`fgets()`读 13 个字节，但是只读了 5 个有效字符，再加一个`\0`，相当于读了 6 个字节。

    `fwrite()`指定的 size `12`是字符串的长度，不带末尾的`\0`。如果强行写 13 个字节带上`\0`，那么文件会出错，vscode 中显示为一个红色出错位。

    example 2:

    ```c
    #include <string.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        FILE *f = fopen("test.txt", "wb");
        fwrite("hello\nworld\n", 12, 1, f);
        fclose(f);

        f = fopen("test.txt", "r");
        char buf[128] = {0};
        char *ret = fgets(buf, 12, f);
        fclose(f);

        printf("%p, %s\n", buf, buf);
        printf("%p, %s\n", ret, ret);
        return 0;
    }
    ```

    output:

    ```
    0x7fff78ddd800, hello

    0x7fff78ddd800, hello

    ```

    可以看到，要求`fgets()`读 12 - 1 = 11 个字符，但是实际只读了 6 个字符，这 6 个字符中包含`\n`。

* `fgetc()`读取文件，遇到`\0`退出

    ```cpp
    f = fopen("test.txt", "r");
    string str;
    char ch;
    while (true)
    {
        ch = fgetc(f);
        if (feof(f) || ch == '\0')
            break;
        str.push_back(ch); 
    }
    putchar('\n');
    fclose(f);
    ```

    说明：

    * `feof()`返回一个 bool 值，如果为`true`，则说明“当前位置”是文件的结尾。

        而当前位置是`fgetc()`调用过后的位置，所以必须先调用`fgetc()`，再调用`feof()`判断是否到结尾，如果未到结尾，再做后处理。

    * 由于不知道在何处会遇到`\0`，所以无法提前知道`str`的长度，因此只能使用`push_back()`的方式动态添加字符。

* c 使用`a+`向文件中追加内容

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    int main()
    {
        FILE *f = fopen("aaa.txt", "a+");
        char *str = "hello, world\n";
        size_t str_len = strlen(str);  // 13
        printf("str len: %lu\n", str_len);
        fwrite(str, str_len, 1, f);
        fclose(f);
    }
    ```

* printf 中的数据解析

    如果一个 long (64位) 数据类型，在填 format 时写成了`%d`，那么有可能导致后续的数据解析错误。

    ```c
    long a = 1, b = 2;
    int c = 3;
    printf("%d, %d, %d\n", a, b, c);  // Error output
    ```

* 长度为 0 的数组

    ```c
    #include <stdio.h>

    int main()
    {
        int acc[0];
        printf("%lu\n", sizeof(acc));
        printf("%p\n", acc);
        return 0;
    }
    ```

    output:

    ```
    0
    0x7fff7d10a164
    ```

    可以看到，并不会分配内存，只会创建一个指针。自定义类也是一样的情况，不会调用构造和析构。

* `uint64_t`在`stdint.h`里，不在`stddef.h`里。

* `printf()`使用`"%[-][N]s"`可以指定补全空格

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        printf("%8s\n", "hel");
        printf("%-8s\n", "hel");
        printf("%8s\n", "hello, world");
        printf("%-8s\n", "hello, world");
        return 0;
    }
    ```

    output:

    ```
         hel
    hel     
    hello, world
    hello, world
    ```

    `%8s`会在字符串的左侧补全空格，使得空格 + 字符串的总长度为 8 个字符。`%-8s`则会在字符串的右侧补全空格。`-`表示左对齐。

    如果字符串的长度超过 8 个字符，则会忽视对齐要求，按照字符串的实际长度输出。

* `printf()`格式化打印小数

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        double val = 12.345;
        printf("%f\n", val);
        printf("%10f\n", val);
        printf("%-10f\n", val);
        printf("%.2f\n", val);
        printf("%10.2f\n", val);
        return 0;
    }
    ```
    
    output:

    ```
    12.345000
     12.345000
    12.345000 
    12.35
         12.35
    ```

    * `%f`: 整数部分不删减，小数部分保留 6 位，如果小数位数不够则补 0，如果多于 6 位则按下面的方法舍入：

        从小数部分的第 7 位开始往后截取，设截取的数字为`x`，若`x <= 500...`，则舍去；若`x > 5000...`，则进位。

        比如`0.12345645`，会被转换为`0.123456`;`0.1234565`，会被转换为`0.123456`；`0.12345651`，会被转换为`0.123457`。

        这个也有可能是用二进制数做截断的，目前还不清楚原理。

    * `%10f`: 首先将小数部分保留 6 位，然后判断整数部分 + 小数点 + 小数部分如果小于 10 个字符，则在最左侧补 0，补够 10 个字符。如果大于等于 10 个字符，则按实际的小数输出。

    * `%-10f`: 行为同`%10f`，但是往右侧补空格。

    * `%.2f`: 把小数部分保留两位，整数部分不限制。

    * `%10.2f`: 把小数部分保留 2 位，整体（整数部分 + 小数点 + 小数部分）一共凑够 10 位，如果小数不够 10 位，则在左侧添 0。如果大于等于 10 位，则按实际小数输出。

* `printf()`格式化打印整数

    `%-8d`: 在整数的右侧补空格，直到凑够 8 个字符。如果整数的字符数大于等于 8，则如实输出。

* `printf()`使用`*`替代格式中的常数

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main()
    {
        int val = 123;
        for (int i = 0; i < 8; ++i)
            printf("%*d\n", i, val);

        float fval = 12.3;
        for (int i = 4; i < 7; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                printf("i = %d, j = %d: %*.*f\n", i, j, i, j, fval);
            }
        }
        return 0;
    }
    ```

    output:

    ```
    123
    123
    123
    123
     123
      123
       123
        123
    i = 4, j = 0:   12
    i = 4, j = 1: 12.3
    i = 4, j = 2: 12.30
    i = 4, j = 3: 12.300
    i = 5, j = 0:    12
    i = 5, j = 1:  12.3
    i = 5, j = 2: 12.30
    i = 5, j = 3: 12.300
    i = 6, j = 0:     12
    i = 6, j = 1:   12.3
    i = 6, j = 2:  12.30
    i = 6, j = 3: 12.300
    ```

* c 语言中 static global variable 的含义

    假如现在想在一个`.c`文件中使用另一个`.c`文件中定义的变量，通常可以用下面的方式：

    `main.c`:

    ```c
    #include <stdio.h>

    extern int val;

    int main()
    {
        printf("val: %d\n", val);
        return 0;
    }
    ```

    `aaa.c`:

    ```c
    int val = 34;
    ```

    compile: `gcc -g main.c aaa.c -o main`

    run: `./main`

    output:

    ```
    val: 34
    ```

    上面的例子展示了，`main.c`中并没有定义`val`的值，但是从`aaa.c`中拿到了`val`的值。

    如果我们删掉`main.c`中的`extern`：

    `main.c`:

    ```
    #include <stdio.h>

    int val;

    int main()
    {
        printf("val: %d\n", val);
        return 0;
    }
    ```

    则会编译时报错：

    ```
    gcc -g main.c aaa.c -o main
    /usr/bin/ld: /tmp/ccDKlEoA.o:/home/hlc/Documents/Projects/c_test/aaa.c:1: multiple definition of `val'; /tmp/cc6LTykz.o:/home/hlc/Documents/Projects/c_test/main.c:3: first defined here
    collect2: error: ld returned 1 exit status
    make: *** [Makefile:2: main] Error 1
    ```

    如果此时我们不想让`main.c`拿到`val`的值，可以在`aaa.c`中给`val`加上`static`：

    `aaa.c`:

    ```c
    static int val = 34;
    ```

    此时会编译时报错：

    ```
    gcc -g main.c aaa.c -o main
    /usr/bin/ld: /tmp/ccpwcbKt.o: warning: relocation against `val' in read-only section `.text'
    /usr/bin/ld: /tmp/ccpwcbKt.o: in function `main':
    /home/hlc/Documents/Projects/c_test/main.c:7: undefined reference to `val'
    /usr/bin/ld: warning: creating DT_TEXTREL in a PIE
    collect2: error: ld returned 1 exit status
    make: *** [Makefile:2: main] Error 1
    ```

    总结：可以用`extern`将别的`.c`文件中的变量引入到当前`.c`文件中，如果不想让别人引用自己的全局变量，可以在全局变量／函数前加`static`。`.h`文件可以看作直接写入到`.c`文件的代码，没有额外的核心作用。

* `stdio.h`中的`puts(char *msg)`可以打印一个字符串并自动换行。

* C 语言`realloc()`

    `realloc()`会释放一段内存，并申请一段新内存，并将旧内存中的数据尽可能多地复制到新内存中。

    example:

    `main.c`:

    ```c
    #include <stdlib.h>
    #include <stdio.h>

    int main()
    {
        char *a = malloc(128);
        printf("ptr a: %p\n", a);

        for (int i = 0; i < 128; ++i)
            a[i] = 123;

        char *b = realloc(a, 256);
        printf("after realloc:\n");
        printf("ptr a: %p\n", a);
        printf("ptr b: %p\n", b);

        for (int i = 0; i < 256; ++i)
            b[i] = 234;

        free(b);
        free(a);  // Error
        return 0;
    }
    ```

    compile: `gcc -g main.c -o main`

    run: `./main`

    output:

    ```
    ptr a: 0x6517528f72a0
    after realloc:
    ptr a: 0x6517528f72a0
    ptr b: 0x6517528f7740
    free(): double free detected in tcache 2
    Aborted (core dumped)
    ```

    由于`realloc()`已经释放了指针`a`，所以调用`free(a)`会报错。

    `realloc()`并不是保留内存地址，仅扩大 size，而是既改变 addr，又扩大 size。

    `realloc()`有可能 in-place 地扩大 size，但是并不保证总是这样。

* int32 max 差不多是 2 * 10^9 多一点

* C 语言中使用指针获取数组中的成员

    我们考虑这样一个 example

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    typedef struct Buf
    {
        float *addr;
        int num_elm;
        char name[20];
    } Buf;

    int main()
    {
        int num_bufs = 4;
        Buf *bufs = malloc(num_bufs * sizeof(Buf));
        for (int i = 0; i < num_bufs; ++i)
        {
            Buf buf = bufs[i];
            buf.num_elm = 3;
            buf.addr = malloc(buf.num_elm * sizeof(float));
            for (int j = 0; j < buf.num_elm; ++j)
                buf.addr[j] = i + j;
            sprintf(buf.name, "buffer %d", i);
        }

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf buf = bufs[i];
            printf("buf name: %s\n", buf.name);
            for (int j = 0; j < buf.num_elm; ++j)
                printf("%.1f, ", buf.addr[j]);
            putchar('\n');
        }

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf buf = bufs[i];
            free(buf.addr);
        }
        free(bufs);
        return 0;
    }
    ```

    其输出为

    ```
    buf name: 

    buf name: 

    buf name: 

    buf name: 

    ```

    明显是有问题的。

    为了避免在 for 中频繁地使用`bufs[i]`来访问成员，我们自作聪明地使用`Buf buf = bufs[i];`来拿到一个元素。观察`struct Buf`中的成员，要么是指针，要么是值，浅复制完全满足我们的需求，所以以为按值拷贝是没问题的。

    但是在第一次对`bufs`中的成员的成员赋值时，我们实际上赋值的是一个副本。这样就导致了输出错误。

    在 C 中可以使用指针来完成这个功能：

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    typedef struct Buf
    {
        float *addr;
        int num_elm;
        char name[20];
    } Buf;

    int main()
    {
        int num_bufs = 4;
        Buf *bufs = malloc(num_bufs * sizeof(Buf));

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf *buf = &bufs[i];
            buf->num_elm = 3;
            buf->addr = malloc(buf->num_elm * sizeof(float));
            for (int j = 0; j < buf->num_elm; ++j)
                buf->addr[j] = i + j;
            sprintf(buf->name, "buffer %d", i);
        }

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf buf = bufs[i];
            printf("buf name: %s\n", buf.name);
            for (int j = 0; j < buf.num_elm; ++j)
                printf("%.1f, ", buf.addr[j]);
            putchar('\n');
        }

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf buf = bufs[i];
            free(buf.addr);
        }
        free(bufs);
        return 0;
    }
    ```

    output:

    ```
    buf name: buffer 0
    0.0, 1.0, 2.0, 
    buf name: buffer 1
    1.0, 2.0, 3.0, 
    buf name: buffer 2
    2.0, 3.0, 4.0, 
    buf name: buffer 3
    3.0, 4.0, 5.0,
    ```

    这次的结果就正确了。

    在 c++ 中，通常用引用拿到成员，因此不会遇到这个问题：

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>

    typedef struct Buf
    {
        float *addr;
        int num_elm;
        char name[20];
    } Buf;

    int main()
    {
        int num_bufs = 4;
        Buf *bufs = (Buf*) malloc(num_bufs * sizeof(Buf));
        for (int i = 0; i < num_bufs; ++i)
        {
            Buf &buf = bufs[i];
            buf.num_elm = 3;
            buf.addr = (float*) malloc(buf.num_elm * sizeof(float));
            for (int j = 0; j < buf.num_elm; ++j)
                buf.addr[j] = i + j;
            sprintf(buf.name, "buffer %d", i);
        }

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf &buf = bufs[i];
            printf("buf name: %s\n", buf.name);
            for (int j = 0; j < buf.num_elm; ++j)
                printf("%.1f, ", buf.addr[j]);
            putchar('\n');
        }

        for (int i = 0; i < num_bufs; ++i)
        {
            Buf &buf = bufs[i];
            free(buf.addr);
        }
        free(bufs);
        return 0;
    }
    ```

    output:

    ```
    buf name: buffer 0
    0.0, 1.0, 2.0, 
    buf name: buffer 1
    1.0, 2.0, 3.0, 
    buf name: buffer 2
    2.0, 3.0, 4.0, 
    buf name: buffer 3
    3.0, 4.0, 5.0,
    ```

    总结：C 语言的 for 循环中，如果不想使用索引，尽量使用指针拿到数组的元素，不要使用值复制。

* 如果一个函数返回一个指针，或者要求参数传入二级指针，那么说明是由这个函数负责相关结构体的内存申请与释放

    如果一个函数要求参数传入一个指针，那么说明函数只负责填充 struct 的字段，由用户负责 struct 的内存管理

* 大端与小商

    对于一个 int 类型的数字：`int a = 0x12345678;`，假如现在有一段内存，内存的地址及存放 a 的方式为

    ```
    0x0001 -> 0x78
    0x0002 -> 0x56
    0x0003 -> 0x34
    0x0004 -> 0x12
    ```

    那么这个存储方式就是小端。

    如果 a 的存储方式为

    ```
    0x0001 -> 0x12
    0x0002 -> 0x34
    0x0003 -> 0x56
    0x0004 -> 0x78
    ```

    那么这个存储方式就是大端。

* 十六进制数字打印

    ```c
    #include <stdio.h>

    int main()
    {
        // case 1
        int a = 0x1234;
        printf("%08x\n", a);

        // case 2
        a = 0x12345678;
        for (int i = 0; i < 4; ++i)
        {
            printf("%02x ", *((char*)&a + i));
        }
        putchar('\n');

        // case 3
        a = 0x12345678;
        printf("%p\n", a);
        return 0;
    }
    ```

    output:

    ```
    00001234
    78 56 34 12 
    0x12345678
    ```

    说明：

    1. 在 case 1 中，使用`%08x`输出十六进制数字

        其含义为，至少输出 8 位十六进制数，如果不够 8 位，那么在前面补 0.

        如果写成`%8x`的形式，那么如果不够 8 位，则在之前填充空格。

        `%x`要求对应的数值是 int / unsigned int 类型。如果填 short, long 类型，编译器会报 warning。

        `%x`输出的十六进制，abcdef 为小写，且前面不添加`0x`。

    2. 在 case 2 中，循环输出 2 位十六进制
    
        `(char*)&a`，先取`a`的地址，然后将其解释为`char*`，为的是后面`+ i`时每次把地址增加 1。

        类似地，如果是`(int16_t*)&a + i`，则当 i = 1 时，地址增加 2.

        最后使用`*((char*)&a + i)`解引用，拿到单个字节的值。

        由于 x86 是小端存储，所以 output 和阅读顺序正好相反。

    3. case 3 使用`%p`将其视作一个指针并输出，会自动在开头加上`0x`。

        缺点是只能处理最大 8 个字节的数字。

    4. 在十六进制表示中，每 2 个十六进制数字正好可以表示 1 个字节。

* 对于每个程序来说，so 文件中的全局变量都是独立的, 函数中的 static 变量也是独立的

* 指针数组

    指针是`int *a`，指针的数组就变成了`int **aa;`，可以使用`aa[i]`或`*aa + i`访问到每一个指针。

    如果需要在函数里创建一个指针的数组，那么函数的参数就变成了`int ***aaa`，常见的 code 如下：

    ```c
    void alloc_pointer_arr(int ***aaa, int num)
    {
        *aaa = malloc(sizeof(int *) * num);
    }

    void free_pointer_arr(int **aa)
    {
        free(aa);
    }

    int main()
    {
        int **aa;
        alloc_pointer_arr(&aa, 3);
        return 0;
    }
    ```

* 关于颜色写两个常用的函数

    ```c
    void print_ok_msg(const char *msg)
    {
        printf("[" "\x1b[32m" "\x1b[1m" "OK" "\x1b[0m" "]");
        printf(" ");
        puts(msg);
    }

    void print_err_msg(const char *msg)
    {
        printf("[" "\x1b[31m" "\x1b[1m" "Error" "\x1b[0m" "]");
        printf(" ");
        puts(msg);
    }
    ```

    这样就可以很方便地输出

    ```
    [OK] hello world
    [Error] asdfsdfasdf
    ```

    这样的格式了。其中`OK`是绿色的，`Error`是红色的。

* c 语言输出多种颜色的字符

    ```c
    #include <stdio.h>

    #define ANSI_RESET_ALL          "\x1b[0m"

    #define ANSI_COLOR_BLACK        "\x1b[30m"
    #define ANSI_COLOR_RED          "\x1b[31m"
    #define ANSI_COLOR_GREEN        "\x1b[32m"
    #define ANSI_COLOR_YELLOW       "\x1b[33m"
    #define ANSI_COLOR_BLUE         "\x1b[34m"
    #define ANSI_COLOR_MAGENTA      "\x1b[35m"
    #define ANSI_COLOR_CYAN         "\x1b[36m"
    #define ANSI_COLOR_WHITE        "\x1b[37m"

    #define ANSI_BACKGROUND_BLACK   "\x1b[40m"
    #define ANSI_BACKGROUND_RED     "\x1b[41m"
    #define ANSI_BACKGROUND_GREEN   "\x1b[42m"
    #define ANSI_BACKGROUND_YELLOW  "\x1b[43m"
    #define ANSI_BACKGROUND_BLUE    "\x1b[44m"
    #define ANSI_BACKGROUND_MAGENTA "\x1b[45m"
    #define ANSI_BACKGROUND_CYAN    "\x1b[46m"
    #define ANSI_BACKGROUND_WHITE   "\x1b[47m"

    #define ANSI_STYLE_BOLD         "\x1b[1m"
    #define ANSI_STYLE_ITALIC       "\x1b[3m"
    #define ANSI_STYLE_UNDERLINE    "\x1b[4m"

    int main(int argc, const char **argv)
    {
        puts("## Print color ##");
        printf("=> " ANSI_COLOR_BLACK   "This text is BLACK!"   ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_RED     "This text is RED!"     ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_GREEN   "This text is GREEN!"   ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_YELLOW  "This text is YELLOW!"  ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_BLUE    "This text is BLUE!"    ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_MAGENTA "This text is MAGENTA!" ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_CYAN    "This text is CYAN!"    ANSI_RESET_ALL "\n");
        printf("=> " ANSI_COLOR_WHITE   "This text is WHITE!"   ANSI_RESET_ALL "\n");

        puts("\n## Print style ##");
        printf("=> " ANSI_STYLE_BOLD        "This text is BOLD!"      ANSI_RESET_ALL "\n");
        printf("=> " ANSI_STYLE_ITALIC      "This text is ITALIC!"    ANSI_RESET_ALL "\n");
        printf("=> " ANSI_STYLE_UNDERLINE   "This text is UNDERLINE!" ANSI_RESET_ALL "\n");

        puts("\n## Print background ##");
        printf("=> " ANSI_BACKGROUND_BLACK   "This BG is BLACK!"   ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_RED     "This BG is RED!"     ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_GREEN   "This BG is GREEN!"   ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_YELLOW  "This BG is YELLOW!"  ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_BLUE    "This BG is BLUE!"    ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_MAGENTA "This BG is MAGENTA!" ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_CYAN    "This BG is CYAN!"    ANSI_RESET_ALL "\n");
        printf("=> " ANSI_BACKGROUND_WHITE   "This BG is WHITE!"   ANSI_RESET_ALL "\n");

        return 0;
    }
    ```

    说明；

    * 不同选项间可以互相组合

    * 这里的 black 其实不是黑色，而是深灰色，white 也不是纯白，而是浅灰

* C 语言的接口基本都是 handle pointer, 全是数据是 struct

    如果 C 对接 C++ 接口时还要写 linked list，那就太麻烦了

    C 接口的函数设计也基本只能设计成`my_func(type_pointer *type, args)`的形式。

    这里的 type pointer 如果想偷情的话可以直接设置成`void*`，如果需要有一个类型提示的话可以设置成`MyStruct *`这样的，但是`MyStruct`如果同时也是 c++ 代码里的类型，那么就会冲突。

    这里想到的几个解决方案：

    * c++ 代码里使用`class _MyStruct`避免和 C 接口冲突

    * c++ 代码使用 namespace

    * c 语言接口再另起一套名字，比如`MyStruct`变成`MyCStruct`之类的

        由于名字是稀缺资源，所以这个方案其实并不是太好。

* `uint64_t`这些类型定义在`stdint.h`头文件里。

* 在 c 语言中计算浮点数级的 duration

    ```c
    #include <time.h>
    #include <stdio.h>

    void spend_time()
    {
        int a = 1, b = 2, c;
        for (int i = 0; i < 1000; ++i)
        {
            for (int j = 0; j < 1000; ++j)
            {
                for (int k = 0; k < 100; ++k)
                {
                    c = a + b;
                }
            }
        }
    }

    int main()
    {
        timespec tmspec_1, tmspec_2;
        timespec_get(&tmspec_1, TIME_UTC);
        spend_time();
        timespec_get(&tmspec_2, TIME_UTC);
        time_t secs = tmspec_2.tv_sec - tmspec_1.tv_sec;
        long nano_secs = tmspec_2.tv_nsec - tmspec_1.tv_nsec;
        printf("duration: %ld secs + %ld nano secs\n", secs, nano_secs);
        float fsecs = (float) secs + ((float) nano_secs / 1000 / 1000 / 1000);
        printf("duration: %f secs\n", fsecs);
        return 0;
    }
    ```

    主要是这个`timespec_get()`函数的使用。注意不要忘了`TIME_UTC`，否则得到的结果不正确。

    ref: <https://en.cppreference.com/w/c/chrono/timespec_get>

* c/c++ 中都不允许两个指针直接相加

    但是 c 将指针显式转换为整数，可以使用指针 + 整数，或整数 + 整数。

* `printf()`的格式化

    使用`%x`打印十六进制数时，不会在前面加上`0x`。如果需要加上前缀，可以使用`%#010x`。不清楚这个 spec 是在哪写的。ref: <https://stackoverflow.com/questions/14733761/printf-formatting-for-hexadecimal>

    `%d`打印的是 integer，`%ld`打印的是 long integer，注意这些都是 signed 值。

    `%u`打印的是 unsigned integer，`%lu`打印的是 unsigned long integer。

    单独一个`%l`没有什么意义。

    long 类型是 64 位，int 类型是 32 位。如果用打印 int 类型的命令打印 long 类型，那么会截取低 32 位打印。

    如果使用打印 signed 的命令打印 unsigned 类型，那么由于对最高位的处理不同，可能会出现负数。

    `printf()`打印不同颜色的字体：<https://blog.csdn.net/qq_41673920/article/details/80334557/>


* 没有办法知道 enum 中有几个元素，C 有一些比较特别的写法可以提供便捷：

    ```cpp
    #include <iostream>

    enum MyEnum {
        mem_1 = 0,
        mem_2,
        mem_3,
        NUM_MYENUM_MEMBERS
    };

    int main()
    {
        printf("the number of MyEnum is %d\n", NUM_MYENUM_MEMBERS);
        return 0;
    }
    ```

    output:

    ```
    the number of MyEnum is 3
    ```

    另外，`sizeof(MyEnum)`无法得到整个`MyEnum`的大小，只会得到单个元素（即`int`类型）所占的大小。通常为 4 字节。

* `memcpy()`在`<memory.h>`中，不在`<memory>`中，也不在`<stdlib.h>`中

## Hello world

file `hello.c`:

```c
#include <stdio.h>

main()
{
    printf("hello, world\n");
}
```

编译：`cc hello.c`，此时会生成`a.out`文件。

运行：`./a.out`

转义字符：`\t`, `\b`, `\"`, `\\`

注释：

```c
/* xxx
xxx
xxxxx
*/

// xxxx
```

## Basic types

* `char`

    字符类型，占 4 个字节。

* `int`

    整数，占 4 个字节。

* `short`

    整数，占 2 个字节。

* `long`

    整数，占 8 个字节。

* `float`

    浮点数，占 4 个字节。至少有 6 位有效数字，取值范围在`10^-38 ~ 10^38`之间。

* `double`

    浮点数，占 8 个字节。

C 语言中，变量会自动类型转换。在一个运算符（包括逻辑运算符与算数运算符）左右，如果两个都是整数，那么执行整数运算；如果其中一个是小数，那么执行浮点运算。

## flow control

* `while`

    ```c
    while (a < b) {
        // ...
    }

    while (a < b)
        a = 2 * a;
    ```

* `for`

    ```c
    main()
    {
        int fahr;
        for (fahr = 0; fahr <= 300; fahr = fahr + 20)
            printf("%3d %6.1f\n", fahr, (5.0 / 9.0) * (fahr - 32));
    }
    ```

说明：

1. 在写`while`和`for`时，要检查循环体一次也不执行的情况，看其结果是否符合要求。

## define

`#define 名字 替换文本`

```c
#define LOWER 0
#define UPPER 300
#define STEP 20

main()
{
    int fahr;
    for (fahr = LOWER; fahr <= UPPER; fahr = fahr + STEP)
        printf("%3d %6.1f\n", fahr, (5.0 / 9.0) * (fahr - 32));
}
```

## standard library

### String

* 初始化字符串

    ```c
    #include <string>

    void *memset(void *s, int c, size_t n);
    ```

    return value: `s`指向哪，返回的指针就指向哪。

    通常全局变量和静态变量会被自动初始化为 0，而局部变量和`malloc`申请的内存则不会，需要手动初始化。

    Example:

    ```c
    char buf[10];
    memset(buf, 0, 10);
    ```

1. 

### IO

`<stdio.h>`

* `printf`

    `%3d`可以右对齐输出整数，这个整数占 3 位。

    `%f`表示浮点数。

    `%3.0f`表示浮点数至少共占 3 位，没有小数点和小数部分。也可以写成`%3f`。

    `%6.1f`表示浮点数至少共占 6 位，且小数点后有 1 位数字。

    `%.2f`表示浮点数的宽度没有限制，但是小数点后有 2 位小数。

    `%.0f`表示不打印小数点和小数部分。

    `%o`表示八进制数，`%x`表示十六进制数，`%c`表示字符，`%s`表示字符串，`%%`表示百分号。
    
    `%ld`表示对应的参数是`long`类型。

* `getchar()`

    从标准输入中读取一个字符，并返回。

    ```c
    char c;
    c = getchar();
    ```

* `putchar(char c)`

    输出一个字符

* `EOF`

    输入的结束标志。被定义为 -1。

#### FILE

turorial: <https://www.programiz.com/c-programming/c-file-input-output>

* `fread()`, `fwrite()`的返回值指的是读取了多少个 block。这个值和设置的`n`参数有关。

    如果返回值等于`n`，那么说明成功读取。

    还有一种情况，如果返回值等于`n - 1`，但是`feof(f_ptr)`返回 true，那么说明读到了文件结尾，也相当于成功读取。

### random number

The function of generating random numbers is integrated in the `<stdlib.h>` file.

A normal usage:

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
int main(void)
{
    srand(time(NULL)); // use current time as seed for random generator
    int random_variable = rand();
    printf("Random value on [0,%d]: %d\n", RAND_MAX, random_variable);
 
    // roll a 6-sided die 20 times
    for (int n=0; n != 20; ++n) {
        int x = 7;
        while(x > 6) 
            x = 1 + rand()/((RAND_MAX + 1u)/6); // Note: 1+rand()%6 is biased
        printf("%d ",  x); 
    }
}
```

Examples:

1. generate 10 random integers

    ```c
    #include <stdlib.h>
    #include <stdio.h>
    #include <time.h>

    int main()
    {
        int sec, clk, seed;
        sec = time(NULL);
        clk = clock();
        seed = sec + clk;
        srand(seed);
        int rnd_num;
        printf("The maximum random number is %d\n", RAND_MAX);
        for (int i = 0; i < 10; ++i)
        {
            rnd_num = rand();
            printf("epoch %d: , random num: %d\n", i, rnd_num);
        }
        return 0;
    }
    ```

Something unusual:

If the `seed` is set with a little variant, the first number generated by `rand()` will vary a little, too.

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

int main()
{
    unsigned int seed = time(NULL);
    unsigned int rnd_num = rand();
    cout << (unsigned int)RAND_MAX << endl;
    int sec, clk;
    for (int i = 0; i < 10; ++i)
    {
        sec = time(NULL);
        clk = clock();
        seed = sec + clk;
        srand(seed);
        rnd_num = rand();
        cout << "epoch " << i << ": ";
        cout << "sec: " << sec << ", ";
        cout << "clk: " << clk << ", ";
        cout << "rnd_num: " << rnd_num << endl; 
    }
    return 0;
}
```

输出：

```
32767
epoch 0: sec: 1677465618, clk: 93, rnd_num: 5116
epoch 1: sec: 1677465618, clk: 94, rnd_num: 5120
epoch 2: sec: 1677465618, clk: 95, rnd_num: 5123
epoch 3: sec: 1677465618, clk: 96, rnd_num: 5126
epoch 4: sec: 1677465618, clk: 96, rnd_num: 5126
epoch 5: sec: 1677465618, clk: 98, rnd_num: 5133
epoch 6: sec: 1677465618, clk: 99, rnd_num: 5136
epoch 7: sec: 1677465618, clk: 100, rnd_num: 5139
epoch 8: sec: 1677465618, clk: 100, rnd_num: 5139
epoch 9: sec: 1677465618, clk: 101, rnd_num: 5142
```

I think the first random number generated after setting seed should be abandoned. Because they are located in almost the same range.

## 预处理命令

`#define`中，`##`的作用是拼接两个字符串。如果`##`的左右有变量，那么将变量替换为字符串，如果没有变量，那么将其直接看作字符串。

Ref: <https://stackoverflow.com/questions/29577775/what-does-mean-in-the-define-directive-in-the-code-here>

## Examples

* 读一个字符，并将其复制到输出

    版本一：

    ```c
    #include <stdio.h>
    
    main()
    {
        int c;

        c = getchar();
        while (c != EOF) {
            putchar(c);
            c = getchar();
        }
    }
    ```

    版本二：

    ```c
    #include <stdio.h>

    main()
    {
        int c;
        while ((c = getchar()) != EOF)
            putchar(c);
    }
    ```

    说明：

    1. 如果`while`语句上面一行和`while`语句中最后一行完全相同，那么似乎就可以写成版本二这种形式。

    1. `!=`运算符的优先级比`=`要高，因此在`=`周围需要加圆括号`()`。

    1. 为什么`putchar()`没有同步显示呢。

* 统计字符数量

    版本一：

    ```c
    #include <stdio.h>

    main()
    {
        long nc;

        nc = 0;
        while (getchar() != EOF)
            ++nc;
        printf("%ld\n", nc);
    }
    ```

    版本二：

    ```c
    #include <stdio.h>

    main()
    {
        double nc;

        for (nc = 0; getchar() != EOF; ++nc)
            ;
        printf("%.0f\n", nc);
    }
    ```

目前已经看到：1.5.3 行计数

## 其他

* 读 utf-8 文件：<https://stackoverflow.com/questions/21737906/how-to-read-write-utf8-text-files-in-c>