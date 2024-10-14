# C Note

C 语言标准库 tutorial：<https://www.tutorialspoint.com/c_standard_library/index.htm>

有时间了看看。

## cache

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