# C Note

C 语言标准库 tutorial：<https://www.tutorialspoint.com/c_standard_library/index.htm>

有时间了看看。

## cache

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