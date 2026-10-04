# Gcc Note

## cache

* 目前我的系统只装了 gcc-13，我希望系统里只剩 gcc-11，怎么办

    在 Linux 系统（尤其是 Ubuntu/Debian 系）里，从“只有 gcc-13”变成“只有 gcc-11”，需要分两步走：**先装新的，再删旧的**。直接卸载 gcc-13 可能会破坏系统依赖，非常危险。

    1. 第一步：安装 gcc-11 和 g++-11

        在终端执行以下命令。同时安装 `g++-11` 是为了保证 C++ 编译器版本一致，避免后续编译混乱。

        ```bash
        sudo apt update
        sudo apt install gcc-11 g++-11
        ```

        安装完成后，可以用 `ls /usr/bin/gcc*` 确认一下，应该能同时看到 `gcc-11` 和 `gcc-13` 两个文件。

    2. 第二步：将 gcc-11 设为默认版本

        不要手动去改 `/usr/bin/gcc` 的软链接，推荐用系统自带的 `update-alternatives` 工具来管理。

        这里建议使用 `--slave` 参数，让 `g++` 能跟随 `gcc` 自动联动切换，避免 C 和 C++ 编译器版本不一致引发奇怪的链接错误。

        ```bash
        # 注册 gcc-11 和 g++-11，并设置一个较高的优先级
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
        ```

        如果系统里之前没有注册过 gcc-13，为了完整性，可以顺便把它也注册进去（优先级设低一点，比如 90），这样 `update-alternatives` 的管理列表会更清晰：

        ```bash
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 90 --slave /usr/bin/g++ g++ /usr/bin/g++-13
        ```

        接下来手动切换到 gcc-11：

        ```bash
        sudo update-alternatives --config gcc
        ```

        终端会列出一个菜单，找到 `gcc-11` 对应的编号，输入数字回车即可。最后用 `gcc --version` 验证，确认输出是 11.x 版本。

    3. 第三步：卸载 gcc-13

        **⚠️ 重要提醒**：在 Ubuntu 等系统中，`gcc-13` 很可能是系统的基础依赖包。直接 `apt remove gcc-13` 可能会连带卸载大量系统组件，导致桌面或核心工具无法使用。

        因此，更安全的做法是**保留 gcc-13 但不再使用它**，因为你的目标“系统里只剩 gcc-11”在逻辑上已经通过切换默认版本实现了。

        如果你确实想彻底删除 gcc-13，请先模拟运行卸载命令，看看会牵连哪些包：

        ```bash
        # 模拟卸载，-s 参数不会真正执行删除，仅作演示
        sudo apt remove -s gcc-13 g++-13
        ```

        **观察输出**：如果它只显示要删除 `gcc-13` 和 `g++-13` 本身，那就可以安全执行 `sudo apt remove gcc-13 g++-13`。如果它同时列出了很多其他包（如 `build-essential`、`libc6-dev` 等），**请立即放弃删除**，否则系统可能会损坏。

    ** 验证与排查**

    切换完成后，建议在**新开的终端**里执行以下命令确认：

    ```bash
    # 检查 gcc 和 g++ 是否都指向了 11 版本
    readlink -f $(which gcc)
    readlink -f $(which g++)
    ```

    两个命令的输出结尾都应该是 `-11`。如果 g++ 还是指向旧版本，可能是之前单独注册过，导致 `--slave` 联动失效，需要先清理旧配置再重新注册。

* gcc default 版本如何从 gcc13 切换到 gcc 12

    在大多数基于 Debian 或 Ubuntu 的系统中，将默认的 GCC 从 13 切换到 12，推荐使用 `update-alternatives` 工具来管理。这种方法安全，且便于以后随时切回。

    **操作步骤**

    1. 安装 GCC 12（如果尚未安装）

        首先确保系统中已经安装了 GCC 12。如果还没有，请先安装：

        ```bash
        sudo apt update
        sudo apt install gcc-12 g++-12
        ```

    2. 将 GCC 12 添加到 alternatives 管理列表中

        执行以下命令，将其注册为一个候选版本。这里使用了 `--slave` 选项，可以确保 `gcc` 和 `g++` 版本同步切换，避免版本不一致：

        ```bash
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12
        ```

        > 命令中的 `100` 是优先级。数值越大，在自动模式下越容易被选中。

    3. 切换默认版本

        运行交互式配置命令：

        ```bash
        sudo update-alternatives --config gcc
        ```

        系统会列出所有可用的 GCC 版本。找到对应 `/usr/bin/gcc-12` 的编号，输入该编号并按回车即可。

    4. 验证切换结果

        检查当前版本，确认已成功切换：

        ```bash
        gcc --version
        g++ --version
        ```

    **备选方案：仅临时使用**

    如果你不想改变系统全局的默认设置，只是在某个项目或终端会话中临时使用 GCC 12，可以直接使用完整路径或设置环境变量：

    *   **指定版本**：`gcc-12 -o output source.c`
    *   **Make/CMake**：在编译时指定 `CC=gcc-12 CXX=g++-12`。

* g++ 的头文件搜索过程

    `g++` 拥有一套完整的**头文件搜索优先级（Search Order）**。当你使用 `#include <xxx.h>`（尖括号形式）时，即使没有在 `-I` 中指定路径，编译器也会去其**默认的系统路径**和**环境变量路径**中查找。

    具体来说，`g++` 搜索头文件的顺序如下：

    1. 搜索顺序分解

        1. `-I` 指定的路径：
        
            编译器首先查看你在命令行中通过 `-I` 明确指定的目录。

        2. 环境变量路径：
        
            如果 `-I` 没找到，它会查看环境变量，主要是`CPLUS_INCLUDE_PATH` (对于 C++ 编译) 或 `CPATH`。如果你的系统里设置了这些变量，编译器会自动去这些目录找。

        3. 系统默认标准路径：
        
            这是最常见的原因。即使你什么都不设，`g++` 也会默认搜索以下目录：

            * `/usr/local/include`

            * `/usr/include`

            * `g++` 自带的内部头文件路径（例如：`/usr/lib/gcc/x86_64-linux-gnu/9/include` 等）。

    2. 如何验证这个文件到底在哪？

        如果你想确定编译器到底是从哪里把这个 `xxx.h` 给“挖”出来的，可以使用以下命令查看完整的搜索链：

        方法 A：查看详细搜索路径

        运行编译命令时加上 `-v` 选项：

        ```bash
        g++ -v -c main.cpp
        ```

        在输出信息中，找到 `#include <...> search starts here:` 下方的列表，那里就是编译器实际扫描的所有目录。

        方法 B：查看文件依赖关系

        如果你只想知道 `xxx.h` 的绝对路径，使用 `-H` 选项：

        ```bash
        g++ -H -c main.cpp
        ```

        这会在终端打印出所有被包含头文件的层级结构及其实际位置。

    3. 常见误区：`#include ""` vs `<>`

        * **`#include "file.h"`**：先在**当前源文件所在目录**找，找不到再按上述 `<>` 的顺序找。

        * **`#include <file.h>`**：直接跳过当前目录，从 `-I` 路径开始找。

* makefile 中 -Ipath 会递归搜索 path 以及其子文件夹吗？

    只会搜索 path 目录下文件，不会递归搜索 path 子目录下的文件。

    **解决方案：可以使用 `shell` 函数自动获取所有子目录**

    如果你有大量的子文件夹，不想手动维护，可以使用 Makefile 的 `shell` 命令和 `find` 工具来动态生成这些路径：

    ```makefile
    # 自动查找 include 文件夹下的所有子目录并加上 -I 前缀
    INC_DIRS := $(shell find include -type d)
    CFLAGS += $(addprefix -I, $(INC_DIRS))
    ```
* gcc `-I<inc_path>`不会递归搜索`<inc_path>`下的子文件夹

    自动生成包含路径的方法

    方法1：使用 find 命令（shell 方式）

    ```makefile
    # 查找 include 目录下的所有子目录
    INCLUDE_DIRS := $(shell find include -type d)
    INCLUDE_FLAGS = $(addprefix -I,$(INCLUDE_DIRS))
    ```

    方法2：使用 wildcard 函数（Makefile 方式）

    ```makefile
    # 查找第一级子目录
    INCLUDE_DIRS := include $(wildcard include/*/)
    INCLUDE_FLAGS = $(addprefix -I,$(INCLUDE_DIRS))
    ```

    方法3：递归查找所有子目录

    ```makefile
    # 使用 find 查找所有目录（包括嵌套目录）
    INCLUDE_DIRS := $(shell find include -type d 2>/dev/null || true)
    INCLUDE_FLAGS = $(foreach dir,$(INCLUDE_DIRS),-I$(dir))
    ```

    注意事项

    * 性能考虑

        ```makefile
        # 警告：如果目录树很大，这可能会影响性能
        # INCLUDE_DIRS := $(shell find . -type d -name "include" -o -name "inc")

        # 更好的做法：限制搜索深度
        INCLUDE_DIRS := $(shell find . -maxdepth 4 -type d -name "include" -o -name "inc")
        ```

    * 避免重复

        ```makefile
        # 去重
        INCLUDE_DIRS := $(sort $(INCLUDE_DIRS))
        ```

* `#pragma`

    #pragma 是一个预处理器指令，其核心作用是向特定的编译器发出特殊的命令或指示，从而控制编译器在编译过程中的特定行为。

    在一个编译器（如GCC）中有效的 #pragma 指令，在另一个编译器（如MSVC）中可能完全无效，或者具有不同的含义。

    如果希望代码能在多种编译器上编译，需要为不同的编译器提供相应的 #pragma 指令，通常配合 #ifdef 等宏来判断编译器。

    **常见用途举例:**

    * 防止头文件被重复包含

        这是最经典、最跨平台的用法之一。虽然可以用 #ifndef、#define、#endif 宏来实现，但 #pragma once 更简洁。

        ```cpp
        // 在 header.h 的开头写上
        #pragma once
        // ... 头文件的内容
        ```

        作用：告诉编译器，这个文件只被包含一次。可以有效避免因头文件被多次包含而引发的重定义错误。

    * 警告管理

        在大型项目中，有时需要暂时禁用某些编译器警告。

        ```cpp
        // 保存当前的警告状态
        #pragma warning(push)
        // 禁用第 4996 号警告（例如：不安全的函数如 strcpy 的警告）
        #pragma warning(disable: 4996)
        // 这里写会触发警告的代码，比如使用 strcpy
        // ...
        // 恢复之前保存的警告状态
        #pragma warning(pop)
        ```

        作用：精准地控制编译器在特定代码段发出或忽略哪些警告，保持代码整洁。

    * 指定代码对齐方式

        控制结构体或变量的内存对齐方式，这对底层硬件操作或网络传输很重要。

        ```cpp
        // 指定按 1 字节对齐，取消填充字节
        #pragma pack(push, 1)
        struct MyStruct {
            char a;    // 1 byte
            int b;     // 4 bytes
            // 如果没有 #pragma pack，这里可能会有 3 字节的填充，总大小为 8 字节。
        };
        #pragma pack(pop) // 恢复之前的对齐方式
        ```

        作用：确保结构体在内存中的布局是紧凑且可预测的，总大小在这里是 1+4=5 字节。

    * 优化提示

        给编译器提供优化建议。

        ```cpp
        // 提示编译器这个循环的次数很少，值得做循环展开等优化
        #pragma loop(short_loops)
        for (int i = 0; i < 4; i++) {
            // ...
        }
        ```

    * 指定代码节

        将函数或数据放入特定的段（Section）。

        ```cpp
        // 将初始化代码放入名为 "INIT" 的段
        #pragma code_seg("INIT")
        void InitializeHardware() {
            // 硬件初始化代码
        }
        ```

        作用：在系统编程中，常用于控制不同功能的代码在内存中的布局。

* `gcc main.c -Wl,-rpath,'$ORIGIN/../libs'`

    $ORIGIN 是一个特殊的变量，它在运行时会被解析为可执行文件（ELF文件）自身所在的目录的绝对路径。

    命令中是`-Wl,-rpath,'$ORIGIN/../libs‘`。使用单引号（`'`） 至关重要，因为它防止 Shell（如 bash）在命令执行前就解释 $ORIGIN。我们需要将字面意义上的 $ORIGIN 这个字符串传递给链接器，并最终写入可执行文件，而不是在编译时就被 Shell 替换成某个值（如果 $ORIGIN 这个 Shell 变量未定义，它会被替换成空字符串）。

    -Wl,-rpath,'$ORIGIN/../libs’ 创建了一种可重定位（Relocatable） 的部署方案。只要可执行文件和 libs 文件夹的相对位置不变（例如，整个项目文件夹被移动到其他位置），程序就总能找到它依赖的库，无需手动设置 LD_LIBRARY_PATH。因为搜索路径是程序自身的一部分，不依赖于用户如何设置环境变量。这对于分发和部署程序非常有用，可以确保程序总能找到它自带的库。

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
