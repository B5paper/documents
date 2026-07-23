# C Note

C 语言标准库 tutorial：<https://www.tutorialspoint.com/c_standard_library/index.htm>

有时间了看看。

## cache

* 详细讲讲 c 中的 printf 函数

    好的，我们详细地来剖析 C 语言中的 `printf` 函数。它不仅仅是“打印输出”那么简单，其背后涉及可变参数、格式化解析、缓冲区管理，甚至是系统调用。

    ---

    ## 1. 函数原型与头文件

    ```c
    #include <stdio.h>

    int printf(const char *format, ...);
    ```

    - **返回值**：成功时返回**输出的字符总数**（不包括结尾的 `'\0'`），失败时返回负数。
    - **第一个参数**：`format` 是**格式化字符串**，包含普通字符和**格式占位符**（以 `%` 开头）。
    - **`...`**：表示**可变参数**，数量与类型由 `format` 中的占位符决定。

    ---

    ## 2. 格式化占位符详解

    基本结构：`%[flags][width][.precision][length]specifier`

    ### 2.1 必选部分：转换说明符（specifier）

    | 说明符 | 输出类型                     |
    |--------|------------------------------|
    | `d` / `i` | 有符号十进制整数           |
    | `u`      | 无符号十进制整数           |
    | `o`      | 无符号八进制               |
    | `x` / `X`| 无符号十六进制（小写/大写）|
    | `f` / `F`| 浮点数（小数形式）         |
    | `e` / `E`| 浮点数（科学计数法）       |
    | `g` / `G`| 根据数值自动选 `f` 或 `e`  |
    | `c`      | 单个字符                   |
    | `s`      | 字符串（以 `\0` 结尾）    |
    | `p`      | 指针地址（void*）          |
    | `%`      | 输出一个百分号 `%`         |

    ### 2.2 可选修饰符

    #### （1）标志（flags）

    | 标志 | 含义 |
    |------|------|
    | `-`  | 左对齐（默认右对齐） |
    | `+`  | 强制显示正负号 |
    | `空格` | 正数显示空格，负数显示 `-` |
    | `#`  | 对于 `o`/`x`/`X` 显示前缀（0, 0x, 0X）；对浮点强制显示小数点 |
    | `0`  | 用 `0` 填充宽度（而非空格） |

    #### （2）宽度（width）

    - 数字：最小输出宽度（若不足则填充空格或 `0`）。
    - `*`：宽度由**下一个参数**提供（int 类型）。

    #### （3）精度（.precision）

    - 对整数：最少数字位数（不足补 `0`）。
    - 对浮点数：小数点后的位数。
    - 对字符串：最大输出字符数。
    - `.*`：精度由下一个参数提供。

    #### （4）长度修饰符（length）

    | 修饰符 | 影响的数据类型 |
    |--------|---------------|
    | `hh`   | signed char / unsigned char |
    | `h`    | short / unsigned short |
    | `l`    | long / unsigned long |
    | `ll`   | long long / unsigned long long |
    | `L`    | long double |
    | `z`    | size_t |
    | `t`    | ptrdiff_t |

    ---

    ## 3. 工作原理（底层机制）

    ### 3.1 可变参数处理

    `printf` 使用 `<stdarg.h>` 中的宏来遍历参数列表：

    ```c
    va_list args;
    va_start(args, format);
    // 逐个解析 format 字符串，遇到 % 则根据类型取参数
    int ret = vprintf(format, args);
    va_end(args);
    ```

    内部通常会调用 `vprintf` 或 `vfprintf`，真正实现格式化与输出。

    ### 3.2 格式化解析过程（简化）

    1. 逐字符扫描 `format`。
    2. 遇到普通字符直接输出。
    3. 遇到 `%`：
       - 解析 flags、width、precision、length。
       - 根据 specifier 从 `va_list` 中取出对应类型的参数。
       - 按照规则转换成字符串（如整数转十进制/十六进制，浮点数转小数形式等）。
    4. 将结果字符串放入**缓冲区**。

    ### 3.3 缓冲区与系统调用

    - `printf` 默认使用 **行缓冲**（stdout 是行缓冲模式，除非重定向）。
    - 数据先写入**用户态缓冲区**，遇到 `\n` 或缓冲区满或程序结束时，调用 `write` 系统调用将数据送入内核，再写到终端或文件。
    - 可通过 `setbuf` / `setvbuf` 更改缓冲模式。

    ---

    ## 4. 常见使用示例

    ```c
    printf("Hello\n");                          // 普通字符串
    printf("%d %x %o", 255, 255, 255);          // 255 ff 377
    printf("%10d", 42);                         // "        42"
    printf("%-10d", 42);                        // "42        "
    printf("%.2f", 3.14159);                    // "3.14"
    printf("%*.*f", 6, 2, 3.14);                // "  3.14"
    printf("%s", "abc");                        // "abc"
    printf("%.5s", "abcdef");                   // "abcde"
    printf("%p", (void*)&a);                    // 地址如 0x7ffc...
    printf("%%");                               // "%"
    ```

    ---

    ## 5. 安全性与常见陷阱

    ### ⚠️ 类型不匹配
    - `%d` 传 `long` → 可能截断或乱码。
    - `%f` 传 `int` → 未定义行为。
    - 必须使用长度修饰符匹配实际类型。

    ### ⚠️ 格式化字符串漏洞
    - 如果 `format` 来自用户输入（如 `printf(user_input)`），攻击者可利用 `%x`、`%n` 泄露或篡改内存。
    - **修复**：始终使用 `printf("%s", user_input)`。

    ### ⚠️ 缓冲区溢出风险（极少）
    - `printf` 自身不会溢出，因为它内部动态分配或使用静态缓冲，但过长的输出可能耗光内存（但很少见）。

    ---

    ## 6. 性能考量

    - 频繁调用 `printf`（尤其是大量小输出）会因系统调用和解析开销变慢。
    - 建议：大批量输出时先 `sprintf` 到字符串再一次性输出。
    - 嵌入式系统常使用自定义的微型 `printf` 以节省代码空间。

    ---

    ## 7. 扩展：相关函数族

    | 函数 | 作用 |
    |------|------|
    | `fprintf(FILE*, ...)` | 输出到指定文件流 |
    | `sprintf(char*, ...)` | 输出到字符串（不安全，可能溢出） |
    | `snprintf(char*, size, ...)` | 安全版，限制输出长度 |
    | `vprintf` / `vfprintf` / `vsnprintf` | 接受 `va_list`，用于自定义包装函数 |

    ---

    ## 8. 实现简易 printf（思路）

    如果你自己实现一个极简版：

    ```c
    int my_printf(const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        int ret = my_vprintf(fmt, args);
        va_end(args);
        return ret;
    }
    ```

    其中 `my_vprintf` 需要：
    - 手动解析 `%` 和修饰符。
    - 用 `va_arg` 取参数。
    - 使用 `itoa` / `ftoa` 转换数字。
    - 用 `putchar` 或 `write` 输出。

    ---

    ## 9. 标准符合性

    - `printf` 由 **C89 / C99 / C11 / C17** 标准定义。
    - 各平台（glibc、MSVC、musl）实现细节略有差异，但核心行为一致。
    - 浮点数转换可能受舍入模式、环境变量（如 `printf` 在 glibc 中受 `LOCALE` 影响）影响。

    ---

    ## 总结

    `printf` 是一个**功能强大但需要谨慎使用**的函数。它通过解析格式字符串和可变参数，将多种类型的数据转化为字符流输出，背后涉及格式化、缓冲、系统调用等机制。正确使用时，注意类型匹配和安全性；在性能敏感或资源受限场景下，可以考虑替代方案或自定义实现。

    如果你对某个具体部分（如浮点数转换细节、自定义 `printf` 实现、漏洞利用）有更深兴趣，我可以继续展开。

* printf("%2x ", prop.uuid.bytes[i]); 为什么打印出来是 uuid: 33 63 70 23 ffffffff 22 4e ffffff84 ffffffa9 ffffffb0 ffffff8a ffffffc0  0  2 20 10，我希望只显示低2 位

    这个问题非常经典！你的代码 `printf("%2x ", prop.uuid.bytes[i])` 打印出 `ffffffff` 而不是期望的 `ff`，根本原因是 **`bytes` 数组的元素类型是有符号的 `char`**（`signed char`），而 `%x` 期望的是 `unsigned int`。

    ## 问题根源

    ### 1. 类型提升（Integer Promotion）
    在 C 语言中，当传递 `char` 类型给可变参数函数（如 `printf`）时，会发生**整数提升**（Integer Promotion）：
    - 如果 `char` 是有符号的（默认在很多系统上），且该字节的最高位是 `1`（即值 ≥ 0x80），它会被**符号扩展**（Sign-extend）为 `int`。
    - 例如，`0x84`（二进制 `1000 0100`）作为 `signed char` 是负数（-124），提升为 `int` 后变成 `0xFFFFFF84`。
    - 然后 `%x` 将这个 `int` 按十六进制打印，就得到了 `ffffff84`。

    ### 2. 宽度修饰符 `%2x` 的含义
    - `%2x` 表示**最小宽度为 2**，不是"只取低 2 位"。
    - 当数值是 `0xFFFFFF84` 时，宽度早已超过 2，所以会原样输出全部 8 位十六进制数字。

    ---

    ## 解决方案

    ### 方案一：强制转换为 `unsigned char`（推荐）
    ```c
    printf("%02x ", (unsigned char)prop.uuid.bytes[i]);
    ```
    - `(unsigned char)` 强制将值转为无符号字符，范围 0~255。
    - 整数提升时变成 `unsigned int`（0x00000084），不会符号扩展。
    - `%02x` 确保至少输出 2 位，不足补零。

    ---

    ### 方案二：使用 `%hhx` 长度修饰符（C99 起）
    ```c
    printf("%02hhx ", prop.uuid.bytes[i]);
    ```
    - `%hhx` 表示参数是 `unsigned char`（或 `signed char` 也会被正确转换）。
    - 但某些老旧编译器可能不完全支持，且仍需注意符号扩展问题（建议配合 `unsigned char` 使用）。

    ---

    ### 方案三：手动掩码（最保险）
    ```c
    printf("%02x ", prop.uuid.bytes[i] & 0xFF);
    ```
    - `& 0xFF` 强制只保留低 8 位，无论原类型是什么。

    ---

    ## 完整对比示例

    ```c
    #include <stdio.h>

    int main() {
        signed char bytes[] = {0x84, 0x23, 0x70, 0x33};
        
        // 错误写法
        printf("错误: ");
        for (int i = 0; i < 4; i++) {
            printf("%2x ", bytes[i]);  // 输出: ffffff84 23 70 33
        }
        printf("\n");
        
        // 正确写法1：强制转换
        printf("正确1: ");
        for (int i = 0; i < 4; i++) {
            printf("%02x ", (unsigned char)bytes[i]);  // 输出: 84 23 70 33
        }
        printf("\n");
        
        // 正确写法2：掩码
        printf("正确2: ");
        for (int i = 0; i < 4; i++) {
            printf("%02x ", bytes[i] & 0xFF);  // 输出: 84 23 70 33
        }
        printf("\n");
        
        return 0;
    }
    ```

    **输出：**
    ```
    错误: ffffff84 23 70 33 
    正确1: 84 23 70 33 
    正确2: 84 23 70 33 
    ```

    ---

    ## 额外建议

    ### 1. 用 `%02x` 代替 `%2x`
    - `%2x`：最小宽度 2，不足补空格 → `" 84"`
    - `%02x`：最小宽度 2，不足补零 → `"84"`（更符合 UUID 显示习惯）

    ### 2. 声明时就用 `unsigned char`
    如果 `prop.uuid.bytes` 是数组，考虑定义为：
    ```c
    unsigned char bytes[16];  // 更明确表示字节数据
    ```
    这样就不需要每次强制转换了。

    ### 3. 检查 UUID 结构定义
    很多 UUID 结构使用 `uint8_t`（来自 `<stdint.h>`）：
    ```c
    #include <stdint.h>
    uint8_t bytes[16];  // 明确为无符号 8 位
    ```

    ---

    ## 总结

    | 代码 | 问题 | 解决方案 |
    |------|------|----------|
    | `printf("%2x", bytes[i])` | 符号扩展导致高位变 `ff` | 用 `(unsigned char)` 或 `& 0xFF` |
    | `printf("%2x", ...)` | 宽度不足补空格 | 用 `%02x` 补零 |
    | 数据类型不明确 | 依赖默认 `char` 符号性 | 使用 `uint8_t` 或 `unsigned char` |

    你的问题核心就是**符号扩展**。加上强制转换后，就能得到你期望的 `33 63 70 23 ff 22 4e 84 a9 b0 8a c0 00 02 20 10` 这样的格式了。

    * 那为什么 (unsigned int)prop.uuid.bytes[i] 不行

        非常好的问题！这涉及到 C 语言中类型转换和整数提升的微妙之处。

        ## 为什么 `(unsigned int)bytes[i]` 不行？

        ```c
        printf("%02x ", (unsigned int)prop.uuid.bytes[i]);
        ```

        **结论：这样写依然会打印 `ffffff84`，问题没有解决！**

        ---

        ## 根本原因：转换顺序问题

        关键在于 **类型转换发生在整数提升之前还是之后**：

        ### 错误的执行顺序（你写的代码）：
        ```
        1. bytes[i] 是 signed char，值为 0x84（-124）
        2. (unsigned int) 强制转换 → 将 -124 转为 unsigned int
           - 转换规则：-124 + UINT_MAX + 1 = 0xFFFFFF84
           - 结果已经是 0xFFFFFF84（unsigned int）
        3. 传递给 printf，%x 打印 → ffffff84
        ```

        ### 正确的执行顺序（我们需要的）：
        ```
        1. 先将 signed char 转为 unsigned char → 0x84（132）
        2. 整数提升为 int → 0x00000084（正数）
        3. 再转为 unsigned int → 0x00000084
        4. 打印 → 84
        ```

        ---

        ## 详细拆解

        ### 情况1：直接转换 `(unsigned int)`

        ```c
        signed char c = 0x84;  // 二进制 10000100，值为 -124
        unsigned int u = (unsigned int)c;
        // 转换过程：-124 → 0xFFFFFF84（UINT_MAX - 123）
        printf("%x", u);  // 输出: ffffff84
        ```

        C 标准规定：将有符号整数转换为无符号整数时，如果原值为负，结果 = 原值 + (最大值 + 1)。所以：
        - -124 + 4294967296 = 4294967172 = 0xFFFFFF84

        ### 情况2：先转 `unsigned char`

        ```c
        signed char c = 0x84;  // -124
        unsigned char uc = (unsigned char)c;  // 0x84（132），保留位模式
        unsigned int u = (unsigned int)uc;    // 132 → 0x00000084
        printf("%x", u);  // 输出: 84
        ```

        `unsigned char` 的转换规则是**保持位模式不变**，范围 0~255，所以 0x84 就是 132（正数）。

        ---

        ## 为什么 `unsigned char` 能解决问题？

        因为当 `unsigned char` 参与整数提升时：

        ```c
        unsigned char uc = 0x84;
        // uc 在表达式中会被提升为 int
        // 规则：unsigned char → int（因为 int 能表示所有 unsigned char 的值）
        // 结果：0x00000084（正数，32位）
        ```

        而 `signed char` 提升时：
        ```c
        signed char sc = 0x84;  // 最高位为1
        // sc 在表达式中被提升为 int
        // 规则：保持符号，符号扩展
        // 结果：0xFFFFFF84（负数）
        ```

        ---

        ## 完整对比示例

        ```c
        #include <stdio.h>

        int main() {
            signed char c = 0x84;
            
            // 错误方式1：直接转 unsigned int
            printf("(unsigned int): %02x\n", (unsigned int)c);
            // 输出: ffffff84
            
            // 错误方式2：转 unsigned int（没有效果）
            printf("(unsigned int): %02x\n", (unsigned int)(c));
            // 输出: ffffff84（一样的问题）
            
            // 正确方式：先转 unsigned char
            printf("(unsigned char): %02x\n", (unsigned char)c);
            // 输出: 84
            
            // 正确方式：使用掩码
            printf("& 0xFF: %02x\n", c & 0xFF);
            // 输出: 84
            
            // 正确方式：使用 uint8_t
            uint8_t uc = (uint8_t)c;
            printf("uint8_t: %02x\n", uc);
            // 输出: 84
            
            return 0;
        }
        ```

        ---

        ## 为什么 `%hhx` 也有同样的问题？

        ```c
        printf("%02hhx ", prop.uuid.bytes[i]);  // 仍可能出错！
        ```

        `%hhx` 只是告诉 `printf` 参数是 `unsigned char` 类型，但**参数传递时已经被提升了**：
        1. `bytes[i]` 被提升为 `int`（0xFFFFFF84）
        2. `printf` 收到 `int` 类型的参数
        3. `%hhx` 会将这个 `int` 截断为 `unsigned char`
        4. 结果：0xFFFFFF84 截断为 0x84 ✅

        **但这依赖于 `printf` 的实现，不安全！** 因为在某些架构上，`int` 和 `char` 的传递方式可能不同。最佳实践是显式转换。

        ---

        ## 总结

        | 写法 | 结果 | 原因 |
        |------|------|------|
        | `(unsigned int)c` | ❌ `ffffff84` | 负值转无符号，值变了 |
        | `(unsigned char)c` | ✅ `84` | 保留位模式，变为正数 |
        | `c & 0xFF` | ✅ `84` | 位掩码，强制只取低8位 |
        | `%02hhx` | ⚠️ 可能对，但不安全 | 依赖 printf 实现 |

        **关键记忆**：
        - `(unsigned int)` 转换的是**值**，负值会变成很大的正数
        - `(unsigned char)` 转换的是**位模式**，保留低 8 位
        - 要解决符号扩展，必须**在整数提升之前**将数据转为无符号小类型

        所以正确的做法是：
        ```c
        printf("%02x ", (unsigned char)prop.uuid.bytes[i]);
        // 或
        printf("%02x ", prop.uuid.bytes[i] & 0xFF);
        ```

* 简述内联汇编，并给出 example

    好的，我们来简述内联汇编并给出示例。

    ### 简述内联汇编

    内联汇编（Inline Assembly）是高级编程语言（如 C 和 C++）中的一个特性，它允许开发者直接在代码中嵌入特定处理器架构的低级汇编指令。

    **主要特点和用途：**

    1.  **性能关键代码：** 用于手动优化对性能极其敏感的代码段，充分发挥硬件优势，实现高级语言难以生成或无法达到的高效指令序列。
    2.  **访问特殊硬件特性：** 用于访问和操作那些没有对应高级语言语法的特殊处理器寄存器（如状态寄存器、控制寄存器）或执行特权指令。
    3.  **系统级编程：** 在操作系统内核、设备驱动或嵌入式系统开发中非常常见，用于实现上下文切换、中断处理、直接硬件访问等底层操作。

    **实现方式：**
    不同编译器有自己独特的内联汇编语法，这导致了代码可移植性很差。

    *   **GCC/Clang（GNU 风格）:** 在 Unix-like 系统和许多其他平台上主流的编译器。其语法为 `asm [volatile] ( “汇编模板” : “输出操作数” : “输入操作数” : “Clobbered 寄存器” );`。功能强大但语法稍复杂。
    *   **MSVC (Microsoft 风格):** Windows 平台上的主流编译器。其语法为 `__asm { 汇编指令 }`。语法更接近传统的汇编语言，但灵活性不如 GNU 风格。

    ---

    ### 示例

    下面我们分别使用 GNU (GCC/Clang) 风格和 MSVC 风格实现一个简单的加法函数。

    #### 示例 1：GNU (GCC/Clang) 风格内联汇编

    这个例子中，`add_numbers` 函数使用内联汇编将两个整数相加。

    ```c
    #include <stdio.h>

    int add_numbers(int a, int b) {
        int sum;
        // 内联汇编开始
        asm volatile (
            "addl %%ebx, %%eax;"   // 汇编模板：将 ebx 加到 eax
            : "=a" (sum)           // 输出操作数：将 eax 的值输出到变量 sum
            : "a" (a), "b" (b)     // 输入操作数：将 a 放入 eax, 将 b 放入 ebx
            // Clobbered 列表：这里没有显式修改其他寄存器，所以省略
        );
        return sum;
    }

    int main() {
        int result = add_numbers(10, 25);
        printf("The result is: %d\n", result); // 输出： The result is: 35
        return 0;
    }
    ```

    **代码解释：**

    1.  `asm volatile (...)`: `volatile` 关键字告诉编译器不要优化这段汇编代码，确保它按书写顺序执行。
    2.  **汇编模板 `"addl %%ebx, %%eax;"`**:
        *   这是实际的汇编指令。`addl` 表示 “add long”（32位加法）。
        *   在 GNU 语法中，为了区分操作数和寄存器，寄存器名前需要**两个** `%` 符号（`%%eax`）。单个 `%` 用于操作数。
    3.  **输出操作数 `: “=a” (sum)`**:
        *   格式为 `"约束" (变量)`。
        *   `=a` 表示：
            *   `=` 代表这是一个输出操作数（只写）。
            *   `a` 是一个约束，要求将变量放入 `eax` 寄存器。
        *   `(sum)` 是 C 语言中的变量，结果将从 `eax` 寄存器写回到这个变量。
    4.  **输入操作数 `: “a” (a), “b” (b)`**:
        *   格式同上。
        *   `“a” (a)`：将变量 `a` 的值放入 `eax` 寄存器。
        *   `“b” (b)`：将变量 `b` 的值放入 `ebx` 寄存器。
        *   编译器负责在汇编代码执行前，将 C 变量 `a` 和 `b` 的值加载到指定的寄存器中。

    **执行流程：**
    编译器将 `a` (10) -> `eax`, `b` (25) -> `ebx` -> 执行 `addl %ebx, %eax` (`eax` = 10 + 25) -> 将 `eax` 的值写回 `sum` -> 函数返回 `sum` (35)。

    ---

    #### 示例 2：MSVC 风格内联汇编

    同样的功能，使用 MSVC 编译器在 Windows 上的实现。

    ```c
    #include <stdio.h>

    int add_numbers_msvc(int a, int b) {
        __asm {           // MSVC 使用 __asm 关键字
            mov eax, a    // 将变量 a 的值移动到 eax 寄存器
            mov ebx, b    // 将变量 b 的值移动到 ebx 寄存器
            add eax, ebx  // 将 ebx 加到 eax
            // 结果已经在 eax 中，直接返回即可
            // 不需要显式移动到 ‘sum’ 变量
        }
        // MSVC 内联汇编默认使用 eax 作为返回值（对于整数类型）
        // 编译器会自动在这里生成 return 指令
    }

    int main() {
        int result = add_numbers_msvc(10, 25);
        printf("The result is: %d\n", result);
        return 0;
    }
    ```

    **代码解释：**

    1.  `__asm { ... }`: 使用 `__asm` 块来包含汇编指令。
    2.  语法更简单直观：
        *   可以直接使用 C 变量的名字（如 `a`, `b`）。
        *   寄存器名只需一个 `%`（但 MSVC 汇编器通常直接使用名称如 `eax`）。
        *   不需要显式地指定输入、输出操作数或约束，编译器会自动处理变量到寄存器的移动。
    3.  **返回值：** 对于返回整数值的函数，MSVC 约定返回值放在 `eax` 寄存器中。因为我们的结果已经在 `eax` 里，所以函数会直接返回它。

    ---

    ### 重要注意事项

    *   **可移植性极差：** 为 GCC 写的內联汇编在 MSVC 上完全无法编译，反之亦然。甚至不同 CPU 架构（如 x86 vs ARM）的汇编指令完全不同。
    *   **容易出错：** 你必须非常清楚指令的副作用、寄存器的使用约定以及编译器如何分配寄存器。错误使用会导致难以调试的内存损坏或程序崩溃。
    *   **阻碍编译器优化：** 内联汇编对编译器来说是一个“黑盒”，可能会阻止编译器进行跨指令的优化。应尽量避免在性能无关的代码中使用。
    *   **现代 C++ 的替代方案：** 对于需要高性能或硬件特定操作的任务，现代 C++ 提供了更安全、可移植性更好的替代方案，如 `std::atomic` 用于原子操作，或者编译器内置函数（Intrinsics，如 SSE/AVX 指令）。

    **总结：** 内联汇编是一个强大但危险的工具，应仅作为最后的手段，由对其有深刻理解的开发者在绝对必要时使用。

    * 所以对于上面的 gpu 风格的内联汇编 example，a 表示 eax 寄存器，那是否 b 代表 ebx 寄存器，c 代表 ecx 寄存器？如果我想访问 ea 寄存器，又该怎么访问？

        非常好的问题！这触及了 GNU 风格内联汇编语法的核心细节。

        您的观察是**部分正确**的。确实存在一些约定俗成的**单字母约束**，但它们代表的不是“b 代表 ebx”，而是代表一类**约束类型**。

        ### 单字母约束的含义

        在 GNU 内联汇编中，`a`, `b`, `c`, `d` 等确实是特定寄存器的约束，但它们对应的是 x86 架构的**通用寄存器**：

        *   `"a"`： 代表 `eax` / `rax` / `ax` / `al` 寄存器（具体取决于操作数大小）
        *   `"b"`： 代表 `ebx` / `rbx` / `bx` / `bl` 寄存器
        *   `"c"`： 代表 `ecx` / `rcx` / `cx` / `cl` 寄存器
        *   `"d"`： 代表 `edx` / `rdx` / `dx` / `dl` 寄存器
        *   `"S"`： 代表 `esi` / `rsi` / `si`
        *   `"D"`： 代表 `edi` / `rdi` / `di`

        所以你的猜测 `a`->`eax`, `b`->`ebx`, `c`->`ecx` 是完全正确的。

        ---

        ### 关于 “ea 寄存器” 的问题

        这是一个非常关键的点：**不存在名为 `ea` 的独立寄存器**。

        我猜您可能是看到了 `eax` 这个名称，并把它拆分理解成了 `e` 和 `ax`。在 x86 架构中：

        *   `eax` 是一个完整的 32 位寄存器。
        *   它的低 16 位可以被单独访问，称为 `ax`。
        *   `ax` 的高 8 位是 `ah`，低 8 位是 `al`。

        **你不能直接访问一个不存在的 `ea` 部分。** 你只能操作整个 `eax` 寄存器，或者它的子部分 (`ax`, `ah`, `al`)。

        **那么，如何访问 `eax` 寄存器呢？**
        答案就是使用我们上面提到的约束 `"a"`。当你写 `: “=a” (sum)` 时，你就是在告诉编译器：“请把 C 变量 `sum` 和 `eax` 寄存器绑定在一起”。

        ---

        ### 如何访问其他寄存器（如 ebp, esp）？

        对于那些没有单字母约束的寄存器（如 `ebp`, `esp`），你需要使用更通用的 `"r"` 约束，或者直接**在汇编模板中硬编码**它们。

        #### 方法 1：使用 `"r"` 约束（让编译器分配）
        `"r"` 约束告诉编译器：“请帮我把这个变量放在任何一个可用的通用寄存器里”。你不需要关心具体是哪个寄存器。

        ```c
        int foo(int x) {
            int result;
            asm volatile (
                "movl $0xFFFF, %%ebp \n\t" // 直接写死 ebp 寄存器
                "incl %1 \n\t"             // 操作编译器分配的寄存器（可能是eax, ebx等，不一定是ebp）
                "movl %1, %0"
                : "=r" (result)  // 输出： result 对应某个寄存器
                : "r" (x)        // 输入： x 对应某个寄存器
                : "ebp"          // 非常重要！告诉编译器我们修改了 ebp，让它做好准备
            );
            return result;
        }
        ```
        在上面的例子中，`%1` 代表编译器为输入变量 `x` 选择的寄存器，它**不一定**是 `ebp`。

        #### 方法 2：在汇编模板中硬编码寄存器（你必须自己管理）
        如果你想**强制**使用某个特定的寄存器（比如 `ebp`），你必须在汇编指令字符串中直接写出它的名字，并**明确告知编译器你破坏了它**。

        ```c
        #include <stdio.h>

        void example_with_ebp() {
            int stack_value;

            // 使用内联汇编读取栈指针 (esp) 附近的一个值
            asm volatile (
                "movl %%ebp, %0"  // 汇编模板：将 ebp 的值移动到输出变量
                : "=r" (stack_value) // 输出操作数：由编译器决定用哪个寄存器来暂存这个值
                : // 无输入操作数
                : // 无 clobber，因为我们只是读取了 ebp，没有修改它
            );

            printf("The value of EBP is: 0x%x\n", stack_value);
        }

        // 一个修改 ebp 的例子
        void modify_ebp() {
            asm volatile (
                "push %%ebp \n\t"   // 保存旧的 ebp
                "movl %%esp, %%ebp \n\t" // 建立一个新栈帧（像函数开头一样）
                "movl $0x12345678, %%ebp \n\t" // 强制修改 ebp 的值
                "pop %%ebp"         // 恢复旧的 ebp
                : // 无输出
                : // 无输入
                : "memory"          // 告诉编译器内存可能被修改了（因为操作了栈）
            );
            // 注意：随意修改 ebp 极其危险，会导致程序崩溃！
        }

        int main() {
            example_with_ebp();
            // modify_ebp(); // 非常危险的函数，谨慎运行
            return 0;
        }
        ```

        ### 关键总结

        1.  **`a`, `b`, `c`, `d`** 是**特定寄存器约束**，分别对应 `eax`, `ebx`, `ecx`, `edx`。
        2.  **`r`** 是**通用寄存器约束**，编译器会为你自动选择一个可用的寄存器。
        3.  **`ebp`, `esp` 等寄存器**没有单字母约束。要使用它们，你必须：
            *   在**汇编模板字符串中直接写死**（如 `movl %%ebp, %0`）。
            *   在 **Clobber 列表**（最后一个冒号后的部分）中**声明**你修改了它们（如 `: “memory”, “ebp”`），这样编译器才能做好保护现场的准备。
        4.  **`ea` 不是寄存器**，不能单独访问。`eax` 是一个完整的寄存器。

        **最重要的建议：** 操作像 `ebp`, `esp` 这样的关键寄存器是**高度危险**的行为，通常只有在你编写极其底层的代码（如上下文切换、任务调度器）时才需要这样做，并且你必须完全清楚自己在做什么。绝大多数情况下，你应该使用输入/输出操作数和约束，让编译器来管理寄存器分配。

    * 这些内联汇编的语法在哪里查？

        非常好的问题！查找内联汇编语法是掌握它的关键。由于其复杂性且高度依赖于编译器和处理器架构，没有一个“万能手册”，但有一系列权威的参考来源。

        以下是查找内联汇编语法最有效的方法和途径，我将其分为几个层次：

        ---

        ### 层次一：官方编译器文档（最权威）

        这是最准确、最权威的来源。所有其他教程和资料都基于这些文档。

        1.  **GCC (GNU Compiler Collection) 官方手册**
            *   **主题**: “How to Use Inline Assembly Language in C Code”
            *   **查找方式**:
                *   **在线**: 直接搜索 **“GCC inline assembly”**，通常第一个结果就是 `gcc.gnu.org` 的官方文档。
                *   **本地**: 如果你安装了 GCC，可以在命令行使用 `info gcc`（如果安装了 info）或 `man gcc` 来查看手册，然后搜索 “Extended Asm” 或 “Inline Assembly”。
            *   **内容**: 这里详细解释了 `asm` 关键字、扩展汇编语法（带操作数列表的那个）、约束字母（`r`, `m`, `a`, `=`, `+` 等的完整列表）、Clobber 列表等。这是**必读的终极参考**。

        2.  **Microsoft MSVC 官方文档**
            *   **主题**: “Inline Assembly in C/C++” (注意：x64 MSVC 编译器**不再支持**内联汇编，只能在 x86 模式下使用)
            *   **查找方式**:
                *   **在线**: 搜索 **“MSVC inline assembly”**，结果会指向 `learn.microsoft.com` 的官方文档。
            *   **内容**: 解释 `__asm` 块的语法、如何在其中使用 C/C++ 符号和宏、以及与 MASM 的差异。

        ---

        ### 层次二：处理器架构的指令集参考（理解指令本身）

        内联汇编只是“包装”，核心是里面的汇编指令。你必须查阅指令手册。

        1.  **Intel 和 AMD 官方手册 (对于 x86/x86-64)**
            *   **Intel® 64 and IA-32 Architectures Software Developer’s Manuals**: 这是圣经。你需要重点关注 **Volume 2 (Instruction Set Reference)**。它详细说明了每一条指令的功能、操作码、受影响的状态标志以及异常行为。
            *   **查找方式**: 搜索 **“Intel SDM”** 即可找到官网下载页面。通常是 PDF 合集。

        2.  **ARM 架构参考手册 (对于 ARM)**
            *   同样，ARM 也提供详尽的架构参考手册，如果你在为 ARM 平台（如手机、树莓派）编写内联汇编，这是必备的。

        ---

        ### 层次三：高质量的教程和指南（用于学习和理解概念）

        官方文档有时过于晦涩，这时优秀的教程就非常重要。

        1.  **IBM 开发者网站 (IBM Developer)**
            *   **主题**: 一篇非常著名的文章，名为 **“GCC inline assembly HOWTO”**。
            *   **内容**: 这是学习 GNU 内联汇编的**经典入门教程**。它用清晰的例子解释了基本概念和语法，比直接啃 GCC 手册要友好得多。强烈建议初学者从这里开始。

        2.  **Stack Overflow 和 Stack Exchange**
            *   **主题**: 搜索 `[inline-assembly]` 标签下的问题。
            *   **内容**: 这里有无数个真实世界的例子和常见错误。当你遇到具体问题时（例如，“为什么我的约束不起作用？”或“这个错误是什么意思？”），在这里搜索很可能已经有人问过并得到了解答。这是一个巨大的知识库。

        3.  **专业的编程 Wiki 和网站**
            *   **OS Dev Wiki**: 如果你做操作系统开发，这个 Wiki 是宝藏，里面有大量与内联汇编相关的底层知识。
            *   **Felix Cloutier’s Website**: 这是一个将 Intel 指令集手册重新排版成更易于网页搜索的网站。查找某条指令的详细用法时非常方便。

        ---

        ### 层次四：实践与查看编译器输出（最实用的技巧）

        这是最有效的学习方法之一。

        1.  **让编译器教你：生成汇编输出**
            *   你可以让 GCC 生成汇编代码而不是目标文件，这样就能看到它如何处理你的内联汇编和周围的 C 代码。
            *   **命令**: `gcc -S myfile.c -o myfile.s`
            *   打开生成的 `.s` 文件，你就可以清晰地看到你的内联汇编代码被嵌入到了编译器生成的汇编代码中，这有助于你理解操作数是如何被替换的（`%0`, `%1` 变成了什么）。

        2.  **从一个简单的 C 代码开始**
            *   先写一个简单的 C 函数，比如 `c = a + b;`。
            *   用 `gcc -S` 生成它的汇编代码。
            *   看看编译器自己是怎么实现这个加法的（用了哪些指令和寄存器）。
            *   然后尝试用内联汇编去实现完全相同的功能。这能让你很好地理解高级代码和低级指令之间的映射关系。

        ### 总结：一个高效的查询路径

        1.  **遇到概念问题** (如约束是什么意思？Clobber 是干嘛的？) -> 去读 **IBM GCC HOWTO** 或 **GCC 官方手册**。
        2.  **遇到指令问题** (如 `addl` 和 `addw` 的区别？`cmpxchg` 怎么用？) -> 去查 **Intel 指令集手册** 或 **Felix Cloutier’s 网站**。
        3.  **遇到具体错误/诡异行为** -> 去 **Stack Overflow** 搜索错误信息或类似代码。
        4.  **想看看编译器背后做了什么** -> 使用 `gcc -S` **生成汇编输出**自己分析。

        记住，内联汇编是“最后的武器”。始终优先考虑使用编译器内置函数（Intrinsics，如 `_mm_add_epi32` for SSE）或更安全的纯 C/C++ 写法，因为它们更具可移植性和可维护性。

* 如何快速判断一个字符串是否符合指定的格式，比如 0000:00:09.0，是否可以 scanf("%x%x%x%x:%x%x:%x%x:%x", &domain[0], ＆domain[1], ...) 如果 scanf 返回成功值，那么说明符合格式，否则就是不符合格式？

    这是一个非常经典的问题。简单来说：**可以使用 `sscanf` 来初步判断，但直接用你提供的写法会有逻辑漏洞。**

    你的思路（通过返回值判断）是正确的，但 `scanf` 的匹配逻辑比想象中要“宽容”得多。

    1. 为什么你提供的 `scanf` 写法不安全？

        如果你使用 `%x%x%x%x:%x%x...`，会遇到以下几个问题：

        * **`%x` 会尽可能多地读取：** `%x` 会读取一串十六进制数字。对于 `0000:00:09.0`，第一个 `%x` 就会直接把 `0000` 全部读走。你不需要为每一个数字位写一个 `%x`。
        * **无法强制长度：** 默认的 `%x` 会匹配 1 到 8 位（取决于 `int` 大小）的十六进制。如果输入是 `0:0:9.0`，它也会返回成功，而这可能不符合你要求的 `0000:00:09.0` 固定宽度格式。
        * **无法检测尾部多余字符：** 如果输入是 `0000:00:09.0-extra-junk`，`sscanf` 依然会返回成功，因为它成功匹配了前面的部分。

    2. 推荐的快速判断方案：`sscanf` 进阶版

        要更严谨地判断格式，建议结合 **宽度限制** 和 **`%n` 说明符**。

        **正确的 C 代码示例：**

        ```c
        #include <stdio.h>

        int is_valid_format(const char *str) {
            int domain, bus, slot, func;
            int pos = 0;

            // %x: 匹配十六进制
            // %4x: 最多匹配4位
            // %n: 记录截止到目前处理了多少个字符
            int items = sscanf(str, "%4x:%2x:%2x.%1x%n", &domain, &bus, &slot, &func, &pos);

            // 判断标准：
            // 1. 必须匹配成功 4 个字段
            // 2. pos 必须等于字符串的总长度（确保后面没有垃圾字符）
            // 3. 这里的例子假设总长度是 12 (4+1+2+1+2+1+1)
            if (items == 4 && str[pos] == '\0') {
                return 1; // 格式正确
            }
            return 0; // 格式错误
        }
        ```

    **总结建议**

    1. **如果你只是想快速抓取数据：** 使用 `sscanf(str, "%x:%x:%x.%x", ...)` 并检查返回值是否为 4。
    2. **如果你要确保字符串完整且合法：** 使用 `sscanf` 配合 `%n` 检查处理长度，并判断是否到达字符串末尾 `\0`。
    3. **如果这是在处理 PCI 设备地址：** 建议直接参考内核或 `libpci` 的解析逻辑，它们通常会处理更多边界情况。

    你想让我帮你写一个完整的、带正则表达式校验的 C 语言函数示例吗？
* __atomic_exchange_n() 简述这个函数的原型、作用、用法

    **函数原型**
    
    ```c
    #include <stdatomic.h>  // 或编译器内置头文件

    type __atomic_exchange_n(type *ptr, type val, int memorder);
    ```

    **作用**

    原子地将 `ptr` 指向的值替换为 `val`，并返回原来的值。这是一个**原子交换操作**（read-modify-write）。

    **参数说明**

    - **ptr**：指向要操作的变量的指针
    - **val**：要设置的新值
    - **memorder**：内存顺序约束

    **内存顺序参数（memorder）**

    | 内存顺序 | 描述 |
    |----------|------|
    | `__ATOMIC_RELAXED` | 宽松内存顺序，仅保证原子性 |
    | `__ATOMIC_CONSUME` | 消费操作（很少使用） |
    | `__ATOMIC_ACQUIRE` | 获取操作，防止之后的内存操作重排到此之前 |
    | `__ATOMIC_RELEASE` | 释放操作，防止之前的内存操作重排到此之后 |
    | `__ATOMIC_ACQ_REL` | 获取+释放操作（用于读写操作） |
    | `__ATOMIC_SEQ_CST` | 顺序一致性模型（最严格） |

    **使用示例**

    1. 基本用法

        ```c
        #include <stdio.h>

        int main() {
            int value = 10;
            int old_value;
            
            // 原子交换：将 value 改为 20，返回原值 10
            old_value = __atomic_exchange_n(&value, 20, __ATOMIC_SEQ_CST);
            
            printf("原值: %d, 新值: %d\n", old_value, value);  // 输出: 原值: 10, 新值: 20
            
            return 0;
        }
        ```

    2. 多线程环境实现锁

        ```c
        #include <stdio.h>
        #include <pthread.h>
        #include <unistd.h>

        int lock = 0;  // 0: 未锁定, 1: 已锁定

        void acquire_lock() {
            // 原子交换尝试获取锁
            while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE) == 1) {
                // 锁已被占用，忙等待
                usleep(100);  // 避免CPU空转
            }
        }

        void release_lock() {
            // 释放锁
            __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);
        }

        int shared_counter = 0;

        void* thread_func(void* arg) {
            for (int i = 0; i < 1000; i++) {
                acquire_lock();
                shared_counter++;  // 临界区
                release_lock();
            }
            return NULL;
        }

        int main() {
            pthread_t t1, t2;
            
            pthread_create(&t1, NULL, thread_func, NULL);
            pthread_create(&t2, NULL, thread_func, NULL);
            
            pthread_join(t1, NULL);
            pthread_join(t2, NULL);
            
            printf("最终计数: %d\n", shared_counter);  // 输出: 2000
            
            return 0;
        }
        ```

    3. 实现无锁数据结构（栈）
        
        ```c
        #include <stdio.h>
        #include <stdatomic.h>

        typedef struct node {
            int data;
            struct node* next;
        } node_t;

        node_t* top = NULL;  // 栈顶指针

        // 无锁入栈
        void push(int value) {
            node_t* new_node = malloc(sizeof(node_t));
            new_node->data = value;
            
            node_t* old_top;
            do {
                old_top = top;
                new_node->next = old_top;
                // 原子更新栈顶指针
            } while (!__atomic_compare_exchange_n(&top, &old_top, new_node, 
                                                 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
        }

        // 无锁出栈
        int pop() {
            node_t* old_top;
            node_t* new_top;
            
            do {
                old_top = top;
                if (old_top == NULL) return -1;  // 空栈
                new_top = old_top->next;
                // 原子更新栈顶指针
            } while (!__atomic_compare_exchange_n(&top, &old_top, new_top,
                                                 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
            
            int value = old_top->data;
            free(old_top);
            return value;
        }
        ```

    4. 不同内存顺序的比较

        ```c
        #include <stdio.h>
        #include <pthread.h>

        int data = 0;
        int flag = 0;

        void* writer(void* arg) {
            data = 42;  // 写入数据
            
            // 释放语义：确保 data 写入在 flag 设置之前完成
            __atomic_exchange_n(&flag, 1, __ATOMIC_RELEASE);
            return NULL;
        }

        void* reader(void* arg) {
            int f;
            
            // 获取语义：确保 flag 读取后，能正确读取 data
            while ((f = __atomic_exchange_n(&flag, 0, __ATOMIC_ACQUIRE)) == 0) {
                // 等待 flag 被设置
            }
            
            printf("data = %d\n", data);  // 保证看到 writer 设置的 42
            return NULL;
        }

        int main() {
            pthread_t t1, t2;
            pthread_create(&t1, NULL, writer, NULL);
            pthread_create(&t2, NULL, reader, NULL);
            pthread_join(t1, NULL);
            pthread_join(t2, NULL);
            return 0;
        }
        ```

    5. 实现自旋锁
        
        ```c
        #include <stdio.h>
        #include <pthread.h>

        typedef struct {
            int locked;  // 0: 未锁定, 1: 锁定
        } spinlock_t;

        void spinlock_init(spinlock_t* lock) {
            lock->locked = 0;
        }

        void spinlock_lock(spinlock_t* lock) {
            // 原子交换尝试获取锁
            while (__atomic_exchange_n(&lock->locked, 1, __ATOMIC_ACQUIRE)) {
                // 锁已被占用，继续自旋
                // 可以添加 CPU 暂停指令优化
                __asm__ volatile("pause" ::: "memory");
            }
        }

        void spinlock_unlock(spinlock_t* lock) {
            // 释放锁
            __atomic_store_n(&lock->locked, 0, __ATOMIC_RELEASE);
        }

        // 使用示例
        spinlock_t lock;
        int counter = 0;

        void* worker(void* arg) {
            for (int i = 0; i < 100000; i++) {
                spinlock_lock(&lock);
                counter++;
                spinlock_unlock(&lock);
            }
            return NULL;
        }
        ```

    **与相关函数对比**

    | 函数 | 描述 | 返回值 |
    |------|------|--------|
    | `__atomic_exchange_n()` | 原子交换 | 原值 |
    | `__atomic_load_n()` | 原子读取 | 当前值 |
    | `__atomic_store_n()` | 原子写入 | 无 |
    | `__atomic_compare_exchange_n()` | 条件交换 | 是否成功 |
    | `__atomic_fetch_add_n()` | 原子加法 | 原值 |

    ```c
    // 比较示例
    int val = 10;
    int old;

    // 原子交换
    old = __atomic_exchange_n(&val, 20, __ATOMIC_SEQ_CST);  // old=10, val=20

    // 原子读取
    int current = __atomic_load_n(&val, __ATOMIC_ACQUIRE);  // current=20

    // 原子写入
    __atomic_store_n(&val, 30, __ATOMIC_RELEASE);           // val=30
    ```

    **注意事项**

    1. **类型支持**：支持整型、指针等基本类型
    2. **对齐要求**：变量需要正确对齐
    3. **内存顺序**：选择合适的内存顺序影响性能和正确性
    4. **编译器支持**：GCC 4.7+、Clang 等支持
    5. **硬件支持**：需要底层硬件支持原子操作

    **应用场景**

    - 实现无锁数据结构
    - 实现自旋锁、读写锁
    - 多线程计数器
    - 状态标志管理
    - 引用计数

* strcasecmp() 简述这个函数的原型、作用及用法

    **函数原型**

    ```c
    #include <strings.h>

    int strcasecmp(const char *s1, const char *s2);
    ```

    **作用**

    比较两个字符串，**忽略大小写差异**，进行不区分大小写的字典序比较。

    **返回值**

    - **0**：两个字符串相等（忽略大小写）
    - **负数**：s1 < s2（忽略大小写）
    - **正数**：s1 > s2（忽略大小写）

    **使用示例**

    1. 基本用法

        ```c
        #include <stdio.h>
        #include <strings.h>

        int main() {
            char *str1 = "Hello";
            char *str2 = "hello";
            char *str3 = "HELLO";
            char *str4 = "world";
            
            if (strcasecmp(str1, str2) == 0) {
                printf("'Hello' 和 'hello' 相等（忽略大小写）\n");
            }
            
            if (strcasecmp(str1, str3) == 0) {
                printf("'Hello' 和 'HELLO' 相等（忽略大小写）\n");
            }
            
            if (strcasecmp(str1, str4) != 0) {
                printf("'Hello' 和 'world' 不相等\n");
            }
            
            return 0;
        }
        ```

    2. 用于比较判断

        ```c
        #include <stdio.h>
        #include <strings.h>

        int main() {
            char input[100];
            
            printf("请输入 'yes' 或 'no'：");
            scanf("%s", input);
            
            // 忽略大小写判断用户输入
            if (strcasecmp(input, "yes") == 0) {
                printf("用户同意\n");
            } else if (strcasecmp(input, "no") == 0) {
                printf("用户拒绝\n");
            } else {
                printf("无效输入\n");
            }
            
            return 0;
        }
        ```

    3. 在搜索/比较中的应用

        ```c
        #include <stdio.h>
        #include <strings.h>

        // 不区分大小写的字符串搜索
        int find_string(const char *target, const char *array[], int size) {
            for (int i = 0; i < size; i++) {
                if (strcasecmp(target, array[i]) == 0) {
                    return i;  // 返回匹配的索引
                }
            }
            return -1;  // 未找到
        }

        int main() {
            const char *colors[] = {"Red", "Green", "Blue", "Yellow"};
            int size = sizeof(colors) / sizeof(colors[0]);
            
            // 不区分大小写查找
            printf("查找 'GREEN'：%s\n", 
                   find_string("GREEN", colors, size) >= 0 ? "找到" : "未找到");
            printf("查找 'blue'：%s\n", 
                   find_string("blue", colors, size) >= 0 ? "找到" : "未找到");
            
            return 0;
        }
        ```

    **相关函数**

    | 函数 | 描述 |
    |------|------|
    | `strncasecmp()` | 比较前n个字符，忽略大小写 |
    | `strcmp()` | 区分大小写的字符串比较 |
    | `strncmp()` | 比较前n个字符，区分大小写 |

    ```c
    // strncasecmp 示例：只比较前n个字符
    char *s1 = "HelloWorld";
    char *s2 = "HELLOworld";
    if (strncasecmp(s1, s2, 5) == 0) {
        printf("前5个字符相等（忽略大小写）\n");
    }
    ```

    **注意事项**

    1. **头文件**：`<strings.h>`（注意不是 `<string.h>`）
    2. **平台差异**：
       - Linux/Unix：`strcasecmp()` 在 `<strings.h>`
       - Windows：使用 `_stricmp()` 在 `<string.h>`
       - POSIX标准函数

    3. **本地化影响**：
       - 默认基于ASCII字符集
       - 某些系统可能受locale设置影响

    4. **性能考虑**：
       - 比 `strcmp()` 稍慢，因为需要处理大小写转换
       - 适合配置解析、命令处理等需要忽略大小写的场景

* `sscanf()`

    从给定的字符串（str） 中按照指定的格式读取数据，并将数据存储到给定的变量地址中。可以看作是 scanf() 的“字符串版本”。

    syntax:

    ```c
    int sscanf(const char *str, const char *format, ...);
    ```

    参数：

    * `str`：源字符串，数据将从这里被读取。

    * `format`：格式控制字符串，与 scanf() 的 format 参数完全相同。

    * `...`：可变参数列表，同样是需要存储数据的变量的地址。

    返回值：

    * 成功：返回成功匹配并赋值的输入项的数量。

    * 失败：如果没有任何项被成功匹配，则返回 EOF。

    example:

    ```c
    #include <stdio.h>

    int main() {
        char info[] = "张三,25,80.5";
        char name[50];
        int age;
        float weight;

        // 从字符串 info 中解析数据
        int items_matched = sscanf(info, "%[^,],%d,%f", name, &age, &weight);

        printf("成功从字符串中解析了 %d 项数据。\n", items_matched);
        printf("姓名：%s，年龄：%d，体重：%.1f\n", name, age, weight);

        return 0;
    }
    ```

    示例解析：

    * 源字符串是`"张三,25,80.5"`。

    * 格式字符串`"%[^,],%d,%f"`的含义是：

        * `%[^,]`：读取一个字符串，直到遇到逗号`,`为止（`[^,]`是一个扫描集，表示匹配所有非逗号的字符）。

            这样可以匹配含有空格的字符串。

        * `,`：匹配一个逗号字符。

        * `%d`：读取一个整数。

        * `,`：再匹配一个逗号字符。

        * `%f`：读取一个浮点数。

* `scanf()`

    从标准输入流（stdin），通常是键盘，按照指定的格式读取数据，并将数据存储到给定的变量地址中。

    syntax:

    ```c
    int scanf(const char *format, ...);
    ```

    参数：

    * `format`：一个格式控制字符串，指定了期望输入的序列格式。它包含以下两种类型的字符：

        * 格式说明符：以 % 开头，如 %d（整数）、%f（浮点数）、%s（字符串，遇到空白字符停止）、%c（字符）等。

        * 空白字符：如空格、制表符、换行符。在格式字符串中，一个空白字符可以匹配输入中任意数量的空白字符。

        * 非空白字符：除了 % 之外的普通字符。输入时必须原样输入这些字符。

    * `...`：一个可变参数列表，是多个变量的地址。参数的数量和类型必须与 format 中的格式说明符一一对应。

    返回值：

    * 成功：返回成功匹配并赋值的输入项的数量。

    * 失败/错误：如果在读取任何数据前发生输入失败，则返回 EOF。

    example:

    ```c
    #include <stdio.h>

    int main() {
        int age;
        float height;
        char name[50];

        printf("请输入您的姓名、年龄和身高（例如：Alice 20 1.65）: ");
        
        // 从标准输入读取数据
        // 注意：变量前需要加 & 取地址操作符（数组名本身是地址，所以 name 不需要 &）
        int items_matched = scanf("%s %d %f", name, &age, &height);

        printf("成功读取了 %d 项数据。\n", items_matched);
        printf("姓名：%s，年龄：%d，身高：%.2f\n", name, age, height);

        return 0;
    }
    ```

    注意事项：

    * `scanf("%s", str)`读取字符串时，遇到空格、制表符或换行符就会停止，因此它不能读取包含空格的字符串。

    * 一定要在变量前加上取地址运算符 &，除非变量本身就是一个地址（如数组名、指针）。

* `__attribute__((packed))`

    这个主要用于取消结构体对齐。

    ```c
    // 正常情况（假设 4 字节对齐）
    struct normal {
        char a;      // 1 字节
        int b;       // 4 字节
        char c;      // 1 字节
    };
    // sizeof(struct normal) = 12 字节（有填充）

    // 使用 packed
    struct __attribute__((packed)) packed_struct {
        char a;      // 1 字节
        int b;       // 4 字节
        char c;      // 1 字节
    };
    // sizeof(struct packed_struct) = 6 字节（无填充）
    ```

    主要用途:

    1. 硬件/协议数据映射
    
        ```c
        // 网络协议头
        struct __attribute__((packed)) eth_header {
            uint8_t dst_mac[6];
            uint8_t src_mac[6];
            uint16_t eth_type;
        };
        ```

    2. 节省内存空间

        在内存受限的嵌入式系统中减少内存占用。

    3. 数据序列化

        确保结构体布局与外部数据格式完全匹配。

    其他替代方案：

    1. 手动序列化（推荐）

        ```c
        void serialize_eth_header(const struct eth_header *hdr, uint8_t *buffer) {
            memcpy(buffer, hdr->dst_mac, 6);
            memcpy(buffer + 6, hdr->src_mac, 6);
            memcpy(buffer + 12, &hdr->eth_type, 2);
        }
        ```

    2. 使用编译器对齐指令

        ```c
        // 指定最小对齐而非完全取消
        struct __attribute__((aligned(1))) minimal_align {
            // ...
        };
        ```

* `EXIT_FAILURE`

    EXIT_FAILURE 定义在 C 标准库头文件 <stdlib.h> 中，用于表示程序异常终止的退出状态。

    syntax:

    ```c
    #include <stdlib.h>

    // 典型定义示例
    #define EXIT_FAILURE 1    // 常见的实现值
    #define EXIT_SUCCESS 0    // 成功的退出状态
    ```

    example:

    ```c
    #include <stdlib.h>

    // void exit(int status);  // status 可以是 EXIT_SUCCESS 或 EXIT_FAILURE

    #include <stdlib.h>

    void cleanup(void) {
        printf("执行清理操作...\n");
    }

    int main() {
        atexit(cleanup);  // 注册退出时执行的函数
        
        if (some_error_condition) {
            exit(EXIT_FAILURE);  // 退出时会自动调用 cleanup()
        }
        
        return EXIT_SUCCESS;
    }
    ```

* `#if ! MY_MACRO`

    检查宏 MY_MACRO 是否未定义或值为0，如果满足条件，则编译其后的代码。

    `!`与变量之间也可以没有空格：`#if !MY_MACRO`

    等价于`#if !defined(MY_MACRO) || MY_MACRO == 0`

* 两种存储字符串的方式

    ```c
    int main() {
        const char *strs_1[] = {
            "hello",
            "world"
        };

        char strs_2[][8] = {
            "nihao",
            "zaijian"
        };

        return 0;
    }
    ```

    `strs_1`是把字符存在字面常量区，如果不加 const，会报错或报 warning。

    `strs_2`把字符存在栈区，可以不加 const，但是必须写低维度的长度，否则会编译报错。

* const 指针不允许 free()

    example:

    ```c
    #include <stdlib.h>
    #include <stdio.h>

    int main() {
        char *real_path = realpath("./", NULL);
        printf("real path: %s\n", real_path);
        free(real_path);
        return 0;
    }
    ```

    这里`real_path`不能是`const char*`。

* `static`可以只出现在头文件里，不出现在实现文件里，此时实现文件里的函数会被私有化，可以正常编译出`xxx.o`。

* `posix_memalign()`

    一个 POSIX 标准函数, 动态分配一块内存，并保证这块内存的起始地址是对齐在某个特定字节边界上的。

    syntax:

    ```c
    #include <stdlib.h>

    int posix_memalign(void **memptr, size_t alignment, size_t size);
    ```

    params:

    * `size_t alignment`

        指定所需的内存对齐边界，单位是字节。

        这个值必须是 2 的幂次方（如 1, 2, 4, 8, 16, 32, 64, ...），并且必须是 sizeof(void *) 的整数倍。

        常见的值：16 (SSE), 32 (AVX), 64 (AVX-512, 缓存行对齐)。

    * `size_t size`

        指定需要分配的内存块大小，单位是字节。

        注意：分配的内存大小不需要是 alignment 的倍数，但通常你会分配对齐大小的整数倍以确保充分利用。

    返回值：

    * 成功时，返回 0。

    * 失败时，返回一个错误码（不是设置 errno）。常见的错误码有：

        * EINVAL: 参数无效。通常是 alignment 不是 2 的幂次方，或者不是 sizeof(void *) 的倍数。

        * ENOMEM: 内存不足，无法完成分配请求。

    与 aligned_alloc() 的关系：posix_memalign() 是 POSIX 的扩展，而 aligned_alloc() 是 C11 标准引入的函数。两者功能类似，但在参数细节上略有不同。在支持 C11 的环境下，aligned_alloc() 是更可移植的标准选择。

* `aligned_alloc()`

    动态分配一块内存，并且这块内存的起始地址会按照你指定的字节对齐方式对齐。

    syntax:

    ```c
    void *aligned_alloc(size_t alignment, size_t size);
    ```

    * alignment：指定的对齐要求。

        它必须是 2 的幂次方（例如 2, 4, 8, 16, 32, 64 ...）。

        在很多实现中，它必须大于或等于`sizeof(void*)`。

    * size：要分配的内存块大小，单位是字节。

        这个 size 参数最好是 alignment 的整数倍。虽然不是所有标准都强制要求，但这是一个很好的实践，可以确保你分配的内存块末尾之后也有足够的对齐空间，避免潜在问题。

    返回值：成功时返回指向分配内存的指针；失败时返回 NULL（例如，请求的对齐无效或内存不足）。

    example:

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        // 分配 100 字节的内存，并且起始地址保证是 32 字节的倍数
        size_t alignment = 32;
        size_t size = 100;

        // 最佳实践：使分配的大小为对齐的整数倍
        // size_t size = 128; // 这样更好

        int *ptr = (int*)aligned_alloc(alignment, size);

        if (ptr == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }

        // 检查地址是否真的对齐（地址值 % 对齐值 应该为 0）
        printf("Allocated address: %p\n", (void*)ptr);
        printf("Address mod %zu: %zu\n", alignment, (size_t)ptr % alignment); // 应该输出 0

        // 使用内存...
        // ...

        free(ptr); // 记得释放内存！
        return 0;
    }
    ```

    注意事项

    * 释放内存：使用 aligned_alloc() 分配的内存必须使用 free() 来释放。你不能使用 realloc() 直接对其重新分配。

    * 可移植性：这是一个 C11 标准的函数。较老的编译器或库（如 Microsoft VC++ 的 C 库）可能不支持它。在这些平台上，你可能需要使用平台特定的 API（如 _aligned_malloc on Windows, posix_memalign on POSIX systems）或手动进行对齐。

    * 过度使用：除非有明确的对齐需求（如性能分析证明需要或硬件强制要求），否则应优先使用 malloc()，因为它更具可移植性和通用性。

* 内联汇编

    * GCC/Clang（GNU 风格）

        `asm [volatile] ( “汇编模板” : “输出操作数” : “输入操作数” : “Clobbered 寄存器” );`
        
    * MSVC (Microsoft 风格)

        `__asm { 汇编指令 }`

    examples:

    * gpu-style example

        ```c
        #include <stdio.h>

        int add_numbers(int a, int b) {
            int sum;
            // 内联汇编开始
            asm volatile (
                "addl %%ebx, %%eax;"   // 汇编模板：将 ebx 加到 eax
                : "=a" (sum)           // 输出操作数：将 eax 的值输出到变量 sum
                : "a" (a), "b" (b)     // 输入操作数：将 a 放入 eax, 将 b 放入 ebx
                // Clobbered 列表：这里没有显式修改其他寄存器，所以省略
            );
            return sum;
        }

        int main() {
            int result = add_numbers(10, 25);
            printf("The result is: %d\n", result); // 输出： The result is: 35
            return 0;
        }
        ```

        output:

        ```
        The result is: 35
        ```

        代码解释：

        * `asm volatile (...)`: `volatile`关键字告诉编译器不要优化这段汇编代码，确保它按书写顺序执行。

        * 汇编模板`"addl %%ebx, %%eax;"`:

            * 这是实际的汇编指令。addl 表示 “add long”（32位加法）。

            * 在 GNU 语法中，为了区分操作数和寄存器，寄存器名前需要两个 % 符号（%%eax）。单个 % 用于操作数。

        * 输出操作数: `“=a” (sum)`:

            * 格式为 "约束" (变量)。

            * `=a`表示：

                * `=`代表这是一个输出操作数（只写）。

                * `a`是一个约束，要求将变量放入`eax`寄存器。

            * `(sum)`是 C 语言中的变量，结果将从`eax`寄存器写回到这个变量。

        * 输入操作数 : `“a” (a), “b” (b)`:

            格式同上。

            * `“a” (a)`：将变量`a`的值放入`eax`寄存器。

            * `“b” (b)`：将变量`b`的值放入`ebx`寄存器。

            * 编译器负责在汇编代码执行前，将 C 语言变量`a`和`b`的值加载到指定的寄存器中。

* `strdup()`

    动态复制一个字符串。(C23 标准库)

    syntax:

    ```cpp
    #include <string.h>

    char *strdup(const char *str);
    ```

    返回值：

    * 成功：返回指向新字符串副本的指针。

    * 失败（如内存分配失败）：返回 NULL。

    使用`strdup()`后，必须手动释放内存。

* 关于头文件中函数声明的 static

    如果`add()`的声明和定义如下：

    ```cpp
    // lib_1.h
    #ifndef LIB_1_H
    #define LIB_1_H

    static int add(int a, int b);

    #endif

    // lib_1.cpp
    #include "lib_1.h"

    int add(int a, int b) {
        return a + b;
    }
    ```

    那么在编译`g++ -c lib_1.cpp -o lib_1.o`时，`add()`函数的实现会变成不可见的。

    此时别的库去 include `lib_1.h`，比如

    ```cpp
    // lib_2.h
    #ifndef LIB_2_H
    #define LIB_2_H

    int sum(int a, int b, int c);

    #endif

    // lib_2.cpp
    #include "lib_2.h"
    #include "lib_1.h"

    int add(int a, int b);

    int sum(int a, int b, int c) {
        return add(add(a, b), c);
    }
    ```

    编译`g++ -c lib_2.cpp -o lib_2.o`时会 warning `add()`没有函数体：

    ```
    g++ -g main_3.cpp lib_1.o lib_2.o -o main
    In file included from main_3.cpp:1:
    lib_1.h:4:12: warning: ‘int add(int, int)’ used but never defined
        4 | static int add(int a, int b);
          |            ^~~
    ```

    由于编译`lib_2.o`没有涉及到 link 过程，所以到这里仍能生成`lib_2.o`.

    到了编译`main`时，涉及到 link 过程:

    ```cpp
    // main.cpp
    #include "lib_1.h"
    #include "lib_2.h"
    #include "stdio.h"

    int main() {
        int a = 1, b = 2;
        int c = add(a, b);
        printf("%d + %d = %d\n", a, b, c);

        int d = sum(a, b, c);
        printf("%d + %d + %d = %d\n", a, b, c, d);
        return 0;
    }
    ```

    `g++ main.cpp lib_1.o lib_2.o -o main`，会发现仍然找不到`add()`的定义，从而报错：

    ```
    g++ -g main.cpp lib_1.o lib_2.o -o main
    In file included from main.cpp:1:
    lib_1.h:4:12: warning: ‘int add(int, int)’ used but never defined
        4 | static int add(int a, int b);
          |            ^~~
    /usr/bin/ld: /tmp/ccMTEjk1.o: in function `main':
    /home/hlc/Documents/Projects/cpp_test/main.cpp:7: undefined reference to `add(int, int)'
    /usr/bin/ld: lib_2.o: in function `sum(int, int, int)':
    /home/hlc/Documents/Projects/cpp_test/lib_2.cpp:7: undefined reference to `add(int, int)'
    /usr/bin/ld: /home/hlc/Documents/Projects/cpp_test/lib_2.cpp:7: undefined reference to `add(int, int)'
    collect2: error: ld returned 1 exit status
    make: *** [Makefile:10: main] Error 1
    ```

    因此可以得出结论，`static`函数只影响编译对应`.c` / `.cpp`文件的`.o`。

    那么我们一口气把所有 cpp 文件都合一起编译链接会怎样？`g++ lib_1.cpp lib_2.cpp main_3.cpp -o main`的输出仍然报错。看来这个命令有可能只是先生成`.o`，再链接`.o`和`main`的简写版，并不能简化编译过程。

* `static`只能加在函数的实现前，不能加在函数的声明前。

    如果`static`加在头文件的函数声明前，那么将不再检查`xx.c`或`xx.cpp`中的内容，并认为这个函数只有函数头，没有函数体。

    example:

    ```cpp
    // lib_1.h
    static int add(int a, int b);

    // lib_1.cpp
    #include "lib_1.h"

    int add(int a, int b) {
        return a + b;
    }

    // lib_2.h
    int sum(int a, int b, int c);

    // lib_2.cpp
    #include "lib_1.h"
    #include "lib_2.h"

    int sum(int a, int b, int c) {
        return add(add(a, b), c);
    }

    // main.cpp
    #include "lib_1.h"
    #include "lib_2.h"
    #include "stdio.h"

    int main() {
        int a = 1, b = 2;
        int c = add(a, b);
        printf("%d + %d = %d\n", a, b, c);

        int d = sum(a, b, c);
        printf("%d + %d + %d = %d\n", a, b, c, d);
        return 0;
    }
    ```

    Makefile:

    ```makefile
    all: main

    lib_1.o: lib_1.h lib_1.cpp
    	g++ -g -c lib_1.cpp -o lib_1.o

    lib_2.o: lib_2.h lib_2.cpp
    	g++ -g -c lib_2.cpp -o lib_2.o

    main: main.cpp lib_1.o lib_2.o
    	g++ -g main.cpp lib_1.o lib_2.o -o main

    clean:
    	rm -f lib_1.o lib_2.o main
    ```

    `make` output:

    ```
    g++ -g -c lib_1.cpp -o lib_1.o
    g++ -g -c lib_2.cpp -o lib_2.o
    In file included from lib_2.cpp:1:
    lib_1.h:1:12: warning: ‘int add(int, int)’ used but never defined
        1 | static int add(int a, int b);
          |            ^~~
    g++ -g main_3.cpp lib_1.o lib_2.o -o main
    In file included from main_3.cpp:1:
    lib_1.h:1:12: warning: ‘int add(int, int)’ used but never defined
        1 | static int add(int a, int b);
          |            ^~~
    /usr/bin/ld: /tmp/cck01kxj.o: in function `main':
    /home/hlc/Documents/Projects/cpp_test/main_3.cpp:7: undefined reference to `add(int, int)'
    /usr/bin/ld: lib_2.o: in function `sum(int, int, int)':
    /home/hlc/Documents/Projects/cpp_test/lib_2.cpp:5: undefined reference to `add(int, int)'
    /usr/bin/ld: /home/hlc/Documents/Projects/cpp_test/lib_2.cpp:5: undefined reference to `add(int, int)'
    collect2: error: ld returned 1 exit status
    make: *** [Makefile:10: main] Error 1
    ```

* C 语言中，字符串以`\`开头接三位 8 进制数，表示一个字节的 8 进制数字，比如`"\101"`表示`8^2 + 1 = 17`

    字符串以`\x`开头接两位 16 进制数，表示一个字节的 16 进制数，比如`"\x41"`表示`4 * 16 + 1 = 65`。

* `ferror()`

    正常读取文件内容时，如果遇到意外问题，比如磁盘空间不足、硬件故障、权限问题，那么会设置内部的 err 标记，`ferror()`用于检测这个标记。如果要重置这个标记，需要手动调用`clearerr()`。

    返回值	如果错误标识符被设置，返回非零值（真）；否则返回 0（假）

* `feof()`

    `feof()`会判断是否到达文件尾，当上次读取失败后，它会返回 true。

    当使用 fread(), fgetc(), fgets(), fscanf() 等函数读取文件时，如果读取到最后一个字符，那么会读取失败，此时会设置一个内部的“文件结束标识符”。`feof(stdin)`的作用就是去检查这个标识符有没有被设置。

    正确用法：

    ```c
    #include <stdio.h>

    int main() {
        FILE *fp = fopen("example.txt", "r");
        if (fp == NULL) {
            perror("Error opening file");
            return 1;
        }

        char buffer[100];
        // 循环读取，直到fread读不到完整数据
        while (fread(buffer, sizeof(char), sizeof(buffer), fp) > 0) {
            // 处理数据...
            printf("%s", buffer);
        }

        // 循环结束后，用feof判断是否成功到达文件末尾
        if (feof(fp)) {
            printf("Reached the end of file successfully.\n");
        } else {
            printf("An error occurred during reading.\n");
        }

        fclose(fp);
        return 0;
    }
    ```

    错误用法：

    ```c
    while (!feof(fp)) { // 在读取之前就判断，此时标识符可能还没被设置
        fread(...);     // 这次读取可能已经失败了，但循环还会再执行一次
        // ...          // 导致最后一次处理的是无效数据
    }
    ```

    `fgetc()`也是相似的用法：

    1. 方法一

        ```c
        FILE *fp = fopen("test.txt", "r");
        if (fp == NULL) {
            // 错误处理
            return;
        }

        int c;  // 注意：必须是 int，不能是 char！
        while ((c = fgetc(fp)) != EOF) {
            putchar(c);  // 处理读取到的字符
        }

        // 如果需要，可以在这里用 feof() 判断结束原因
        if (feof(fp)) {
            printf("\n成功到达文件末尾");
        } else {
            printf("\n读取过程中发生错误");
        }

        fclose(fp);
        ```

    1. 方法二

        ```c
        int c;
        while (1) {
            c = fgetc(fp);
            if (c == EOF) {
                break;  // 遇到EOF立即退出
            }
            putchar(c);  // 处理有效字符
        }
        ```

    这两种都是正确的。

* C 语言/gdb 中，`(void)`主要用于防止编译器给出 unused variable 的 warning。

* `fileno()`可以获得`FILE*`指针对应的 fd

    syntax:

    ```c
    #include <stdio.h>
    int fd = fileno(file_ptr);
    ```

    example:

    ```c
    #include <stdio.h>

    int main() {
        FILE *f = fopen("msg.txt", "rw");
        if (f == NULL) {
            printf("fail to open file\n");
            return -1;
        }

        int fd = fileno(f);
        printf("fd is %d\n", fd);

        fclose(f);
        return 0;
    }
    ```

    output:

    ```
    fd is 3
    ```

    标准流的描述符：

    ```c
    fileno(stdin)  == 0;  // STDIN_FILENO
    fileno(stdout) == 1;  // STDOUT_FILENO
    fileno(stderr) == 2;  // STDERR_FILENO
    ```

* `strrchr()`

    从右往左搜索指定字符的位置。

    example:

    ```cpp
    #include <string.h>
    #include <stdio.h>

    int main() {
        const char *msg = "hello, world";
        const char *pos_ptr = strrchr(msg, 'r');
        if (pos_ptr == NULL) {
            printf("fail to find pos\n");
            return -1;
        }
        printf("%c is at pos %ld\n", *pos_ptr, pos_ptr - msg);
        return 0;
    }
    ```

    output:

    ```
    r is at pos 9
    ```

* `strstr()`

    syntax:

    ```cpp
    #include <string.h>

    char *strstr(const char *str, const char *sub_str);
    ```

* `do { ... } while(0)`

    常用场景：

    1. 构造单条语句，通常用在宏里，可以把多条代码打包成一条。

        ```c
        #define LOG(msg) do { \
            printf("[LOG] %s\n", msg); \
            write_to_file(msg); \
        } while(0)

        // 使用
        if (error)
            LOG("Error occurred");  // 安全展开为单语句
        else
            recover();
        ```

        注：`LOG("Error occurred");`整体才算一个单语句，如果不加`;`则不算单语句，会报错。

        如果不加`do while`，则有：

        ```c
        #define LOG(msg) { printf("[LOG] %s\n", msg); write_to_file(msg); }

        // 展开后：
        if (error)
            { printf(...); write_to_file(...); };  // 结尾分号导致 else 语法错误！
        else
            recover();
        ```

        其中，`{ printf(...); write_to_file(...); }`表示一条语句，其后的`;`表示第二条语句。

    1. 需要用`break`的场景

        * 替换`goto`语句：

            ```c
            int func() {
                do {
                    if (step1_failed) break;
                    if (step2_failed) break;
                    // ...成功逻辑...
                    return 0;
                } while(0);

                // 统一错误处理
                cleanup();
                return -1;
            }
            ```

        * 简化 if 的逻辑

            ```c
            do {
                // 只执行一次的复杂逻辑
                if (condition) break;
                // ...其他代码...
            } while(0);
            ```

* `strchr()`

    找到一个字符串中某个字符第一次出现的位置。

    头文件：`#include <string.h>`

    syntax:

    ```c
    const char *strchr(const char *str, int c);
    char *strchr(char *str, int c);
    ```

    example:

    ```cpp
    #include <string.h>
    #include <stdio.h>

    int main() {
        const char *msg = "hello, world";
        const char *pos = strchr(msg, 'w');
        printf("idx: %ld, ch: %c\n", pos - msg, *pos);
        return 0;
    }
    ```

    output:

    ```
    idx: 7, ch: w
    ```

    如果未找到，返回`NULL`。

    `strchr()`不支持中文。

* `calloc()`简介

    头文件：`<stdlib.h>`

    `calloc()`与`malloc()`相似，都是分配内存，只不过`calloc()`是按`elm_size * num_elm`的方式计算内存大小，并对内存数据进行置`0`，而`malloc()`使用`buf_size`计算内存大小，并保持内存数据的随机，不进行置`0`。

    example:

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>

    int main() {
        size_t num_elm = 5;
        int *arr = (int*) calloc(sizeof(int), num_elm);
        for (int i = 0; i < num_elm; ++i) {
            printf("%d, ", arr[i]);
        }
        putchar('\n');
        free(arr);
        arr = (int*) malloc(sizeof(int) * num_elm);
        for (int i = 0; i < num_elm; ++i) {
            printf("%d, ", arr[i]);
        }
        putchar('\n');
        free(arr);
        return 0;
    }
    ```

    output:

    ```
    0, 0, 0, 0, 0, 
    -1140679150, 5, 0, 0, 0,
    ```

* `getline()`简介

    `getline()`是 C 的一个 gpu 扩展函数，用于动态申请内存从文件或`stdin`读数据。

    `getline()`在`<stdio.h>`头文件中。

    example:

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        char *line = NULL;
        size_t buffer_size = 0;
        ssize_t n_bytes = getline(&line, &buffer_size, stdin);
        printf("buffer size: %lu\n", buffer_size);
        printf("echo bytes: %ld, %s", n_bytes, line);
        free(line);
        return 0;
    }
    ```

    input:

    ```
    hello
    ```

    output:

    ```
    buffer size: 120
    echo bytes: 6, hello
    ```

    当`getline()`读到文件结尾（EOF）时，会返回`-1`。

    如果发现行的长度大于 120 个字节，那么`getline()`会调用`realloc()`改变 buffer size。

    `getline()`会保留行末尾的`\n`。

    用户必须手动`free(line);`释放内存。

* 编译器内置宏`__FILE__`, `__LINE__`，分别表示当前文件名和行号

* 可变参数宏

    ```cpp
    #include <stdio.h>

    #define INFO(...) printf(__VA_ARGS__)
    #define MSG(msg, val, ...) printf(__VA_ARGS__, msg, val)
    // #define DBG(val_1, ..., val_2) printf(__VA_ARGS__, val_1, val_2);  // error

    int main() {
        INFO("hello, world, %d\n", 123);
        MSG("nihao", 456, "msg: %s, %d\n");
        return 0;
    }
    ```

    output:

    ```
    hello, world, 123
    msg: nihao, 456
    ```

    其中`__VA_ARGS__`表示将`...`中的内容全部转移到当前位置。

    `...`只能作为最后一个参数，不能作为中间参数或第一个参数。

    `...`不能为空，如果可能为空，需要使用`##__VA_ARGS__`（GNU 扩展）或`__VA_OPT__`（C++20 支持）。

* void 在 C 语言中的作用

    对于有返回值的函数，在调用时如果想显式忽略其返回值，可以在调用函数前加`(void)`，如果不加，编译器有可能报 warning。

    ```c
    int get_ret_val() {
        return 0;
    }

    int main() {
        get_ret_val();
        (void) get_ret_val();  // 显式忽略返回值
        return 0;
    }
    ```

    实际测试中，gcc / g++ 并没有报 warning。

    对于 unused variable 同理，可以使用`(void) var_name;`消除编译器的 warning。

    example:

    ```c
    int main() {
        int aaa = 3;
        return 0;
    }
    ```

    compile:

    ```
    gcc -Wall main.c -o main
    ```

    compiling output:

    ```
    main.c: In function ‘main’:
    main.c:2:9: warning: unused variable ‘aaa’ [-Wunused-variable]
        2 |     int aaa = 3;
          |         ^~~
    ```

    将代码改成下面的形式即可消除编译的 warning：

    ```c
    int main() {
        int aaa = 3;
        (void) aaa;
        return 0;
    }
    ```

* c 的变长参数函数无法处理 c++ 的类型

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <stdarg.h>
    using namespace std;

    void func(int a, ...) {
        va_list args;
        va_start(args, a);
        string str = va_arg(args, string);
        va_end(args);
    }

    int main() {
        func(1, string("hello"));
        return 0;
    }
    ```

    在`va_arg(args, string);`这里会提示错误：

    > a class type that cannot be trivially copied cannot be fetched by va_arg

    但是处理指针是可以的：

    ```cpp
    #include <stdio.h>
    #include <string>
    #include <stdarg.h>
    using namespace std;

    void func(int a, ...) {
        va_list args;
        va_start(args, a);
        string *str = va_arg(args, string*);
        printf("%s\n", str->c_str());
        va_end(args);
    }

    int main() {
        string str {"hello, world"};
        func(1, &str);
        return 0;
    }
    ```

    output:

    ```
    hello, world
    ```

    引用也不能处理。

* `memmove()`简介

    `memmove()`在头文件`<string.h>`里，功能和`memcpy()`几乎相同，参数和`memcpy()`完全相同。

    `memmove()`和`memcpy()`的区别是`memmove()`可以处理 dst 区间和 src 区间有交叠的情况，而`memcpy()`不做这个保证。

    当 dst 区间和 src 区间有重叠时，`memmove()`可以保证当 src 区间的中后面的数据还没被处理时，不会被覆盖掉。

    example:

    ```cpp
    #include <stdio.h>
    #include <string.h>

    int main() {
        int arr_1[] = {1, 2, 3, 4, 5};
        memcpy(&arr_1[1], &arr_1[0], sizeof(int) * 4);
        for (int i = 0; i < 5; ++i) {
            printf("%d, ", arr_1[i]);
        }
        putchar('\n');

        int arr_2[] = {1, 2, 3, 4, 5};
        memmove(&arr_2[1], &arr_2[0], sizeof(int) * 4);
        for (int i = 0; i < 5; ++i) {
            printf("%d, ", arr_2[i]);
        }
        putchar('\n');

        return 0;
    }
    ```

    output:

    ```
    1, 1, 2, 3, 4, 
    1, 1, 2, 3, 4, 
    ```

    可以看到，使用 linux 上的 gcc 11.4 编译，这两个函数的实际效果是一样的。

* 使用`fprintf()`向 stderr 输出内容

    ```cpp
    #include <stdio.h>

    int main(int argc, const char **argv) {
        fprintf(stderr, "hello, world\n");
        return 0;
    }
    ```

    output:

    ```
    hello, world
    ```

    如何证明这是 stderr 的内容，而不是 stdout 的内容？我们已知`tee`命令可以把 stdout 的内容输出到文件里，而 stderr 的内容则不会写入到文件里，而是仅输出到 terminal。

    执行`./main | tee out.txt`，`cat out.txt`，文件内容为空。

    而执行`./main 2>& 1 | tee out.txt`，`cat out.txt`，可以看到文件内容：`hello, world`。

    说明`fprintf(stderr, ...)`确实把输出写到了 stderr 中，而不是 stdout。

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

* 大端与小端

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

## Topics

### 格式化输出

* `printf`

    `%3d`可以右对齐输出整数，这个整数占 3 位。

    `%f`表示浮点数。

    `%3.0f`表示浮点数至少共占 3 位，没有小数点和小数部分。也可以写成`%3f`。

    `%6.1f`表示浮点数至少共占 6 位，且小数点后有 1 位数字。

    `%.2f`表示浮点数的宽度没有限制，但是小数点后有 2 位小数。

    `%.0f`表示不打印小数点和小数部分。

    `%o`表示八进制数，`%x`表示十六进制数，`%c`表示字符，`%s`表示字符串，`%%`表示百分号。
    
    `%ld`表示对应的参数是`long`类型。

## other

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

### IO

`<stdio.h>`

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
