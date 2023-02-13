# C Note

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