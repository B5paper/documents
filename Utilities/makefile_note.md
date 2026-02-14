# Makefile Note

## cache

* makefile 中，不能使用`$(command)`得到 command 的输出，必须使用`$(shell command)`才可以。

* makefile 中，如何将 include_dirs = aaa bbb ccc 替换成 -Iaaa -Ibbb -Iccc

    在 Makefile 中，有几种方法可以将 include_dirs = aaa bbb cccc 替换成 -Iaaa -Ibbb -Icccc：

    方法1：使用 patsubst 函数（推荐）

    ```makefile
    include_dirs = aaa bbb cccc
    INCLUDE_FLAGS = $(patsubst %,-I%,$(include_dirs))

    all:
        @echo "原始: $(include_dirs)"
        @echo "转换后: $(INCLUDE_FLAGS)"
    ```

    方法2：使用 addprefix 函数

    ```makefile
    include_dirs = aaa bbb cccc
    INCLUDE_FLAGS = $(addprefix -I,$(include_dirs))

    all:
        @echo "原始: $(include_dirs)"
        @echo "转换后: $(INCLUDE_FLAGS)"
    ```

    方法3：使用 foreach 函数

    ```makefile
    include_dirs = aaa bbb cccc
    INCLUDE_FLAGS = $(foreach dir,$(include_dirs),-I$(dir))

    all:
        @echo "原始: $(include_dirs)"
        @echo "转换后: $(INCLUDE_FLAGS)"
    ```

    完整示例

    ```makefile
    # 定义包含目录
    include_dirs = src/include libs/math libs/utils

    # 方法1：使用 patsubst
    INCLUDES1 = $(patsubst %,-I%,$(include_dirs))

    # 方法2：使用 addprefix（更简洁）
    INCLUDES2 = $(addprefix -I,$(include_dirs))

    # 编译器设置
    CC = gcc
    CFLAGS = -Wall -O2 $(INCLUDES2)

    # 示例目标文件
    OBJS = main.o utils.o

    all: program

    program: $(OBJS)
        $(CC) -o $@ $^ $(CFLAGS)

    %.o: %.c
        $(CC) -c $< $(CFLAGS)

    clean:
        rm -f *.o program

    debug:
        @echo "包含目录: $(include_dirs)"
        @echo "编译选项: $(CFLAGS)"
    ```

    更复杂的处理（处理带空格的路径）

    如果路径中包含空格，可以使用引号：

    ```makefile
    include_dirs = "path with spaces" "another path" standard_path
    INCLUDE_FLAGS = $(addprefix -I,$(include_dirs))
    ```

    或者使用引号处理：

    ```makefile
    include_dirs = path\ with\ spaces another\ path standard_path
    INCLUDE_FLAGS = $(addprefix -I,$(include_dirs))
    ```

    推荐

    推荐使用方法2（addprefix），因为它最简洁明了：

    * 代码可读性好

    * 函数名语义清晰（添加前缀）

    * 执行效率高

    这样你就可以在编译命令中直接使用 $(INCLUDE_FLAGS) 了：

    ```makefile
    gcc $(CFLAGS) $(INCLUDE_FLAGS) -c source.c
    ```

* makefile 中可以对未定义的变量进行`+=`操作

    ```makefile
    aaa = hello
    bbb = world
    ccc += ${aaa}
    ccc += ${bbb}

    main:
    	echo ${ccc}
    ```

    run: `make`

    output:

    ```
    echo hello world
    hello world
    ```

* makefile 中，`@echo "The process ID in Shell is: $$PPID"`，$ 给 Shell, Shell 看到的是 $PPID

    output:

    ```
    The process ID in Shell is: 427531
    ```

* `@echo "The process ID in Make is: $(shell echo $$PPID)"`

    显示当前执行 make 进程的父进程 ID

    其中，`$$`被转义为`$`, `$PPID`是 shell 变量，表示父进程 ID。

    如果直接使用单个美元符号，`@echo "PID: $(shell echo $PPID)"`，那么会：

    1. make 首先展开：`$P`被 make 当作变量`P`的引用

    2. make 看到`$P`：尝试展开 make 变量`P`（如果未定义就是空）

    3. 最终执行：`echo （空字符串）+ PID`

    make 不区分单双引号，都当作普通字符

* `OBJS = $(SRCS:.c=.o)`

    将 SRCS 变量中的所有 .c 后缀文件名替换为 .o 后缀，并将结果赋值给 OBJS 变量。

    SRCS：通常是一个包含所有 C 源文件名的变量，例如`SRCS = main.c utils.c helper.c`

    example:

    ```makefile
    SRCS = main.c utils.c helper.c
    OBJS = $(SRCS:.c=.o)

    myprogram: $(OBJS)
        gcc -o myprogram $(OBJS)

    main.o: main.c
        gcc -c main.c

    utils.o: utils.c
        gcc -c utils.c

    helper.o: helper.c
        gcc -c helper.c
    ```

    等价写法：

    `OBJS = $(patsubst %.c,%.o,$(SRCS))`

* makefile 中可以使用`$(shell )`帮助实现 bash 中拼接字符串的效果

    example:

    ```makefile
    var_1 := hello, $(uname -r)
    var_2 := hello, $(shell uname -r)

    test:
    	@echo ${var_1}
    	@echo $(var_2)
    ```

    output:

    ```
    hello,
    hello, 6.8.0-79-generic
    ```

* makefile 中的变量与 shell 变量

    ```makefile
    makefile_var := hello, makefile

    test:
    	@echo $(makefile_var)
    	@bash_var="hello, bash"; \
    	echo $${bash_var}
    ```

    `make` output:

    ```
    hello, makefile
    hello, bash
    ```

    `$()`可以引用到前面定义的 makefile 变量。

    makefile target 中的每行都是一个独立 bash 环境，如果需要定义 bash 变量，那么可以使用`; \`将多行 bash 命令看作一行，或者直接写成一行的形式。

    调用 bash 变量时，必须使用两个美元符号`$$`，比如`$$shell_var`, `$${shell_var}`。make 会将两个美元符号变成一个，然后传给 shell。

    说明：

    1. shell 变量不可以使用圆括号`$$()`

    1. shell 写成两行是不可以的，因为算作两个独立的 bash 环境

        ```makefile
        target:
        	@bash_var="hello, bash"
        	@echo $${bash_var}
        ```

        输出为空。

    * Make 变量 在规则的目标、依赖和整个 Makefile 的顶层使用 $(...) 来引用。

    * Shell 变量 在规则的命令部分（以 Tab 开头的行）中使用，但需要使用两个美元符号 $$ 来转义。

* `patsubst`

    在一个以空格分隔的单词列表中，查找并替换符合特定“模式”（Pattern）的文本。

    syntax:

    ```makefile
    $(patsubst PATTERN,REPLACEMENT,TEXT)
    ```

    * PATTERN： 需要被替换的文本模式。它可以包含一个通配符 %，代表任意长度的任何字符。

    * REPLACEMENT： 替换后的文本模式。它同样可以使用 % 通配符，这个 % 会代表 PATTERN 中 % 所匹配到的内容。

    * TEXT： 需要进行处理的原始文本（通常是一个由空格分隔的列表）。

    返回值： 函数会返回一个经过替换处理后的新列表。

    example:

    ```makefile
    SRCS = main.c helper.c utils.c
    OBJS = $(patsubst %.c,%.o,$(SRCS))
    ```

    处理过程：

    1. 取出 main.c，模式 %.c 匹配成功，% 代表 main。替换为 %.o，即 main.o。

    2. 取出 helper.c，模式匹配成功，% 代表 helper。替换为 helper.o。

    3. 取出 utils.c，模式匹配成功，% 代表 utils。替换为 utils.o。

    example 2:

    ```makefile
    # 原始列表
    FILES = foo.txt bar.log baz.txt

    # 目标：将所有 .txt 文件名的前缀改为 “output-”，如 output-foo.txt output-baz.txt
    NEW_FILES = $(patsubst %.txt,output-%.txt,$(FILES))

    # 结果：NEW_FILES 的值为 `output-foo.txt bar.log output-baz.txt`
    # 注意：bar.log 不符合 %.txt 模式，所以原样保留。
    ```

    `%`，匹配任意数量（0个或多个）的任意字符。匹配是“贪婪”的：% 会尽可能多地匹配字符。

* `$(shell command)`

    `$(shell command)`是一个 Makefile 函数，它在 Makefile 解析阶段，执行一个 Shell 命令，并将其标准输出（stdout）的结果作为字符串返回，然后赋值给一个变量或直接展开使用。

    常见使用场景:

    * 动态获取文件列表

        ```makefile
        # 获取当前目录下所有的 .c 文件
        SOURCES := $(shell find . -name "*.c")

        # 将 .c 文件列表转换为 .o 文件列表
        OBJECTS = $(SOURCES:.c=.o)
        ```

    * 获取系统信息或环境变量

        ```makefile
        # 获取当前用户名
        WHOAMI := $(shell whoami)

        # 获取当前工作目录的绝对路径
        CURRENT_DIR := $(shell pwd)

        # 获取 Git 提交哈希或版本号
        GIT_HASH := $(shell git rev-parse --short HEAD)
        ```

    * 检查环境或工具是否存在

        ```makefile
        # 检查是否安装了某个工具（例如 'pandoc'）
        PANDOC_EXISTS := $(shell command -v pandoc 2> /dev/null)

        ifndef PANDOC_EXISTS
        $(error "Error: pandoc is required but not installed.")
        endif
        ```

    * 生成版本号或构建时间

        ```makefile
        BUILD_DATE := $(shell date +%Y-%m-%d_%H:%M)
        VERSION := 1.0.$(shell git rev-list --count HEAD)
        ```

    * 处理文件名

        ```makefile
        # 获取当前目录名，用于命名目标文件
        DIR_NAME := $(shell basename $(CURDIR))
        TARGET = program_$(DIR_NAME)
        ```

    example:

    ```makefile
    # 使用 shell 命令动态获取所有 .c 文件
    SRCS := $(shell find src -name "*.c")
    # 将 .c 文件名转换为 .o 文件名
    OBJS = $(SRCS:src/%.c=obj/%.o)
    # 获取当前时间作为构建版本
    BUILD_TIME := $(shell date)

    # 最终目标
    myapp: $(OBJS)
    	$(CC) -o $@ $^ $(LDFLAGS)

    # 编译规则
    obj/%.o: src/%.c | obj
    	$(CC) $(CFLAGS) -c $< -o $@

    # 创建 obj 目录的规则
    obj:
    	mkdir -p obj

    # 打印一些信息
    print-info:
    	@echo "Sources: $(SRCS)"
    	@echo "Build time: $(BUILD_TIME)"

    clean:
    	rm -rf obj myapp

    .PHONY: clean print-info
    ```

    注意事项:

    1. 通常与 :=（立即展开赋值）一起使用，确保 shell 命令只执行一次。如果使用 =（递归展开），它可能会在每次变量被展开时都执行一次 Shell 命令，导致性能下降。

    1. 错误处理: 如果执行的 Shell 命令失败（返回非零状态码），make 通常会停止执行并报错。可以使用 Shell 的逻辑操作来避免这个问题（例如 command 2>/dev/null || echo "default"）。

    1. 空格处理: Shell 命令的输出会原样返回，包括换行符。有时可能需要使用 $(strip ...) 函数来去除多余的空白字符。

* makefile 模式规则

    模式规则是一种通用模板，它告诉 make 如何基于文件名模式来编译一类文件。它使用通配符 % 来匹配任意非空字符串。

    example:

    ```makefile
    %.o: %.c
    	$(CC) -c $(CFLAGS) $< -o $@
    ```

    解释：

    * `%.o`： 这是目标模式。它匹配任何以 .o 结尾的文件名（例如 main.o, utils.o, foo.o）。

    * `%.c`： 这是依赖模式。它匹配任何以 .c 结尾的文件名。这里的 % 与目标模式中的 % 代表相同的字符串。

    * `$(CC) -c $(CFLAGS) $< -o $@`： 这是规则要执行的命令。

        * `$(CC)`： 通常是编译器，如 gcc 或 clang。

        * `-c`： 告诉编译器只编译不链接，生成目标文件（.o）。

        * `$(CFLAGS)`： 传递给编译器的选项（如 -Wall -g -O2）。

        * `$<`： 一个自动化变量，代表规则中的第一个依赖项的名字。在这个例子中，就是那个匹配到的 .c 文件（例如 main.c）。

        * `-o $@`： 另一个自动化变量，$@ 代表规则中的目标文件名。在这里就是那个 .o 文件（例如 main.o）。

* `$(MAKE)`与`make`的区别

    只使用`make`的问题：

    * 不可移植：不同的系统可能使用不同的 make 程序名称。例如，BSD 系统通常使用 bmake，而 GNU Make 可能被安装为 gmake。如果你的 Makefile 里写死了 make，在这些系统上就会执行失败。

    * 忽略命令行选项：当你使用一些命令行选项（如 -k, -s, -t）调用顶层的 make 时，在递归调用中直接使用 make 会丢失这些选项。子 make 进程不会继承父进程的 flags，导致行为不一致。

    * 无法传递 -j (并行编译) 选项：这是最致命的问题之一。如果你使用 make -j8 启动并行编译，但在 Makefile 内部递归调用时使用的是 make，那么这个子 make 将会是串行执行的（-j1），无法利用多核优势，严重拖慢编译速度。

    MAKE 是一个 Makefile 内置的宏（变量），它的值就是当前正在执行的 make 程序的完整路径名（例如 /usr/bin/make）。并且可以解决上面列出的问题。

    （如何验证`$(MAKE)`可以继承命令行选项？）

* `subst`

    用于在 makefile 字符串中进行文本替换。它可以将一个字符串（或变量）中所有出现的指定子字符串，替换为另一个指定的子字符串。

    syntax:

    ```makefile
    $(subst FROM,TO,TEXT)
    ```

    * FROM：你希望被替换掉的子字符串。

    * TO：你希望用来替换 FROM 的新子字符串。

    * TEXT：需要进行替换操作的原始字符串或变量。

    注意：参数之间用逗号 , 分隔，并且不能有空格，否则空格会被当作字符串的一部分。

    example:

    ```makefile
    # 定义一个变量
    ORIGINAL = foo bar baz foo

    # 使用 subst 将所有的 "foo" 替换为 "qux"
    RESULT = $(subst foo,qux,$(ORIGINAL))

    all:
    	@echo "Original: $(ORIGINAL)"
    	@echo "Result:   $(RESULT)"
    ```

    output:

    ```
    Original: foo bar baz foo
    Result:   qux bar baz qux
    ```

    其中，`RESULT`还可以写成`RESULT = $(subst foo,qux,"foo bar baz foo")`, `RESULT = $(subst foo,qux,foo bar baz foo)`, 但是不能写成

    `RESULT = $(subst foo,qux, foo bar baz foo)`, 否则会多出一个空格：

    ```
    Original: foo bar baz foo
    Result:    qux bar baz qux
    ```

    常见应用场景:

    * 修改文件后缀

        这是 subst 最经典的用法之一，用于生成目标文件列表。

        ```makefile
        SOURCES = main.c utils.c helper.c
        # 将 .c 替换为 .o
        OBJECTS = $(subst .c,.o,$(SOURCES))

        all: $(OBJECTS)
            # ...

        # 这条规则会尝试编译 main.o, utils.o, helper.o
        ```

    * 调整路径格式

        例如，将空格路径转换为适合某些命令行工具的格式。

        ```makefile
        PATH_WITH_SPACES = /path/with\ spaces/file.txt
        # 将空格替换为转义空格（或其他字符）
        ESCAPED_PATH = $(subst \ ,\\ ,$(PATH_WITH_SPACES))
        ```

    * 简单的字符串修正

        任何需要批量修改字符串内容的地方。

        ```makefile
        MY_MSG = This is a test string.
        # 将所有的空格替换为连字符
        HYPHENATED = $(subst ,-,$(MY_MSG))
        # HYPHENATED 的值变为：This-is-a-test-string.
        ```

* `g++ *.o main.cpp -o main`在 makefile 中的问题

    ```makefile
    main: *.o main.cpp
        g++ *.o main.cpp -o main
    ```

    存在的问题：

    1. command 中的`*.o`会作为 bash 命令展开，如果当前目录没有`.o`结尾的文件，那么会报错

        example:

        ```
        (base) hlc@hlc-VirtualBox:~$ ls *.aaa
        ls: cannot access '*.aaa': No such file or directory
        (base) hlc@hlc-VirtualBox:~$ echo $?
        2
        (base) hlc@hlc-VirtualBox:~$ touch b.aaa
        (base) hlc@hlc-VirtualBox:~$ ls *.aaa
        b.aaa
        (base) hlc@hlc-VirtualBox:~$ echo $?
        0
        ```

    1. target dependency 中的`*.o`会在 makefile 规则分析时展开，由于并没有`*.o`文件的 target，所以触发 makefile 的隐式规则，使用`g++ -c xxx.cpp -o xxx.o`生成`.o`文件。

        注意此时不会生成`g++ -c main.cpp -o main.o`文件。

        ```makefile
        all: main

        main: *.o main.cpp
        	g++ *.o main.cpp -o main

        clean:
        	rm -f *.o main
        ```

        project dir:

        ```
        (base) hlc@hlc-VirtualBox:~/Documents/Projects/makefile_test$ ls
        lib.cpp  lib.h  main.cpp  Makefile
        (base) hlc@hlc-VirtualBox:~/Documents/Projects/makefile_test$ make
        g++    -c -o *.o lib.cpp
        g++ *.o main.cpp -o main
        (base) hlc@hlc-VirtualBox:~/Documents/Projects/makefile_test$ ls
         lib.cpp   lib.h   main   main.cpp   Makefile  '*.o'
        ```

    1. makefile 的内置隐含规则会生成`'*.o'`文件

        make 的隐含规则为：

        ```makefile
        # Make内置的隐含规则（近似于）
        %.o: %.cpp
            $(CXX) $(CXXFLAGS) -c $< -o $@
        ```

        检测到`*.o`不存在，会执行：

        `g++ -c -o *.o *.cpp`

        根据 bash 的规则，`*.cpp`会展开为`lib.cpp`，而`-o *.o`保持 raw string，因此生成的文件也叫`'*.o'`。Shell只在命令参数中展开通配符，不在选项参数中展开。

* `wildcard`

    用于匹配指定模式的文件名.

    syntax:

    ```makefile
    $(wildcard PATTERN...)
    ```

    根据给定的模式 PATTERN，返回当前目录下 符合模式的文件列表（以空格分隔）。

    如果没有文件匹配，则返回空字符串。

    常见用法：

    * 获取某类型的源文件

        `SRC := $(wildcard *.c)`

        返回当前目录下所有 .c 文件，例如：main.c util.c test.c

    * 结合 patsubst 生成目标文件列表

        ```makefile
        SRC := $(wildcard *.c)
        OBJ := $(patsubst %.c, %.o, $(SRC))
        ```

        把所有 .c 文件转成对应的 .o 文件列表

    * 递归目录（需要配合 wildcard 和 foreach）

        `SRC := $(wildcard src/*.c lib/*.c)`

        (这个没看明白, chatgpt 的输出不完整？)

* makefile 中特殊的自动变量

    这些自动变量主要用于表示目标和先决条件（依赖）。

    * `$@`: 表示规则中的目标文件名。

        在规则`main.o: main.c`中，`$@`的值就是`main.o`

        避免重复书写目标名。如果你想重命名目标，只需修改规则开头即可，命令部分无需改动。

    * `$<`: 表示规则中的 第一个先决条件 的名称。(第一个依赖)

        在规则`main.o: main.c header.h`中，`$<`的值就是`main.c`。

        在编译源文件（.c -> .o）时，我们通常只需要将源文件（.c）传递给编译器，头文件（.h）是通过 #include 指令包含的，不需要直接出现在编译命令中。$< 完美地提供了这个源文件。

    * `$^`: 表示规则中 所有不重复的先决条件，以空格分隔。（所有依赖）

        在`myapp: main.o utils.o`规则中，`$^`就是`main.o utils.o`。

        在链接生成最终可执行文件时，需要将所有目标文件（.o）都传递给链接器。使用 $^ 可以确保一个不漏。

    * `$?`: 表示所有 比目标更新的先决条件，以空格分隔。（更新的依赖）

        如果`header.h`被修改而`main.c`没有，在重建`main.o`时，`$?`的值就是`header.h`。

        这个变量在某些高级场景下很有用，例如当你需要只对发生变化的文件执行某些特殊操作（比如生成日志）时。在普通的编译命令中较少使用。

    * `$*`: 表示与目标匹配的 茎（即`%`匹配的部分）。常用于隐含规则和模式规则中。（茎，Stem）

        在模式规则`%.o: %.c`中，如果目标是`dir/foo.o`，则`$*`的值就是`dir/foo`。

        可以用来生成与源文件同名但扩展名不同的文件，或者在命令中使用匹配部分的名字。

    * `$+`: 类似于`$^`，但它 包含所有重复的先决条件。

        在规则`main.o: main.c header.h main.c`中，`$^`是`main.c header.h`，而`$+`是`main.c header.h main.c`。

    常用场景：

    * 编译单个源文件：使用`$<`(源文件) 和`$@`(目标文件)。

    * 链接多个目标文件：使用`$^`(所有目标文件) 和`$@`(可执行文件)。

* makefile 中的子文件夹与`.PHONY`

    假如当前的工程目录为：

    ```
    - proj
        my_lib.h
        my_lib.cpp
        Makefile
        - tests
            xxx.h
            xxx.cpp
            Makefile
        - imported_libs
            yyy.h
            yyy.cpp
            Makefile
    ```

    如果我们希望`proj`文件夹中的 makefile 可以进入到子文件夹`tests`和`imported_libs`中执行 make 进行子模块的编译，那么我们写出的`proj/Makefile`文件可能是这样的：

    ```makefile
    all: libs imported_libs tests
    	@echo "in all target"

    libs:
    	@echo "in libs target"
    	touch libs.txt
    	# g++ -c my_lib.cpp -o my_lib.o

    imported_libs:
    	@echo "in imported_libs target"
    	$(MAKE) -C imported_libs

    tests:
    	@echo "in tests target"
    	$(MAKE) -C tests

    clean:
    	$(MAKE) -C tests clean
    	$(MAKE) -C imported_libs clean
    	rm -f libs.txt
    ```

    `make`输出如下：

    ```
    in libs target
    touch libs.txt
    # g++ -c my_lib.cpp -o my_lib.o
    in all target
    ```

    可以看到虽然``all`的依赖目标中包含有`imported_libs`和`tests`，但是这两个根本没执行。因此已经有同名的文件夹存在。

    此时需要`.PHONY`来解决这个问题：

    ```makefile
    .PHONY: tests imported_libs

    all: libs imported_libs tests
    	@echo "in all target"

    libs:
    	@echo "in libs target"
    	touch libs.txt
    	# g++ -c my_lib.cpp -o my_lib.o

    imported_libs:
    	@echo "in imported_libs target"
    	$(MAKE) -C imported_libs

    tests:
    	@echo "in tests target"
    	$(MAKE) -C tests

    clean:
    	$(MAKE) -C tests clean
    	$(MAKE) -C imported_libs clean
    	rm -f libs.txt
    ```

    `make`的 output:

    ```
    in libs target
    touch libs.txt
    # g++ -c my_lib.cpp -o my_lib.o
    in imported_libs target
    make -C imported_libs
    make[1]: Entering directory '/home/hlc/Documents/Projects/makefile_test/imported_libs'
    in imported_libs dir...
    touch imported_libs.txt
    make[1]: Leaving directory '/home/hlc/Documents/Projects/makefile_test/imported_libs'
    in tests target
    make -C tests
    make[1]: Entering directory '/home/hlc/Documents/Projects/makefile_test/tests'
    in tests dir...
    touch tests.txt
    make[1]: Leaving directory '/home/hlc/Documents/Projects/makefile_test/tests'
    in all target
    ```

* makefile 中的依赖机制

    1. `target`如果没有依赖项，那么检测名为`target`的文件/文件夹是否存在，若不存在，则执行`target`，否则不执行

    2. 若`target`有依赖项`dep`，那么判断`dep`文件是否比`target`新，如果是，那么执行`target`，如果`dep`只是目标，不是文件，那么无论`dep`是否执行，总是认为`dep`比`target`新

    3. 如果有`.phony: target dep`存在，那么认为`target`和`dep`都只是目标，不是文件

* makefile 中的`export`

* makefile 里，编译时依赖 a.o 和 b.o 文件，但是这两个文件不在当前目录里，如何在 g++ 命令里方便地把它们加上去？

    目前有三种方案：

    1. 创建一个`a.o`所在目录的路径变量，后面使用`${OBJ_DIR}/a.o`, `${OBJ_DIR}/b.o`的方式把它们添加到 g++ 的编译命令里：

        ```makefile
        OBJ_DIR = ../obj
        OBJS = $(OBJ_DIR)/a.o $(OBJ_DIR)/b.o
        TARGET = main

        $(TARGET): $(OBJS)
        	g++ $^ -o $@
        ```

        其中，

        * `$^`：代表所有依赖文件（即`a.o`和`b.o`）

        * `$@`：代表目标文件（即`main`）

    2. 直接使用绝对路径

        ```makefile
        main: ../path/to/a.o ../another/path/to/b.o
        	g++ $^ -o $@
        ```

    3. 使用`patsubst`命令做字符串替换

        ```makefile
        OBJ_NAMES = a b
        OBJ_PATH = ../myobjs

        # 使用 patsubst 函数将 a b 转换为 ../myobjs/a.o ../myobjs/b.o
        OBJS = $(patsubst %,$(OBJ_PATH)/%.o,$(OBJ_NAMES))

        main: $(OBJS)
        	g++ $^ -o $@
        ```

* makefile 文件名可以是`Makefile`，也可以是`makefile`,但是不能是`makefilE`。

    也就是说，文件名只对第一个字母大小写不敏感。

## topics

### 变量与赋值

* makefile 中的`?=`

    ?= 被称为 条件赋值符 或 默认赋值符。只有当这个变量之前没有被赋值过（是空的），才会给它赋值。如果已经有值了，那么就忽略这次赋值。

    如果我们在 makefile 中写`CFLAGS ?= -Wall -O2`，那么在外部执行`make CFLAGS="-g"`时，makefile 发现`CFLAGS`已经被赋值了，那么`CFLAGS ?= -Wall -O2`将不再生效。

    makefile 中的四种赋值方法：

    | 操作符 | 名称 | 作用 |
    | - | - | - |
    | `=` | 递归赋值 | 在变量被使用时才会展开并确定值。 |
    | := |	直接赋值 |	在变量被定义时就立即展开并确定值。 |
    | ?= |	条件赋值 |	如果变量为空，则赋予等号右边的值。 |
    | += |	追加赋值 | 将等号右边的值追加到现有变量之后。 |

* makefile 中的`+=`

    在 Makefile 中，`+=`是追加赋值运算符，用于向变量追加新内容，而不是替换原有内容。

    `+=`会忽略左右的空格，甚至会忽略右侧的所有空白分隔符（空格，制表以及换行）

    example:

    ```makefile
    msg = nihao
    msg += hello world

    msg_2 = nihao
    msg_2 +=hello world

    msg_3 = nihao
    msg_3 +=    hello world

    msg_4 = nihao
    msg_4 = hello	world\
    zaijian


    test:
    	@echo $(msg)
    	@echo $(msg_2)
    	@echo $(msg_3)
    	@echo $(msg_4)
    ```

    output:

    ```
    nihao hello world
    nihao hello world
    nihao hello world
    hello world zaijian
    ```

* makefile 中，可以使用`:=`避免`=`递归变量引用的报错

    ```makefile
    my_str := hello
    # my_str = hello  # OK
    my_str := $(my_str) world
    # my_str = $(my_str) world  # error

    test:
    	@echo $(my_str)
    ```

    output:

    ```
    hello world
    ```

* makefile 中的`var = val`更像是一个宏，它只记录表达式，不记录值，当用到`var`时，再去展开表达式，计算值是多少。这是一种 lazy evaluation 的方式。

* makefile recipe 中，无法直接对变量赋值

    在Makefile的recipe中，不能直接对Makefile变量进行赋值。Recipe中的每一行都是在独立的shell进程中执行的，而且Makefile变量解析在recipe执行之前就已经完成了。

    比如：

    ```makefile
    base_dir := /path/to/base_dir_1
    file_path = $(base_dir)/hello

    test:
    	@echo $(file_path)
    	base_dir := /path/to/base_dir_2  # error
    	@echo $(file_path)
    ```

    报错：

    ```
    /path/to/base_dir_1/hello
    base_dir := /path/to/base_dir_2
    make: base_dir: No such file or directory
    make: *** [Makefile:17: test] Error 127
    ```

    但是可以使用 shell 变量：

    ```makefile
    base_dir := /path/to/base_dir_1
    file_path = $(base_dir)/hello

    test:
    	@echo $(file_path)
    	$(eval base_dir := /path/to/base_dir_2)
    	@echo $(file_path)
    ```

    output:

    ```
    /path/to/base_dir_1/hello
    /path/to/base_dir_2/hello
    ```

* makefile 与 bash 中变量等号左右的空格

    makefile 会自动删除等号左右的空格：

    ```makefile
    var_1 := hello world

    var_2 =  hello world

    var_3=hello world

    var_4:=  hello world

    test:
    	@echo "var_1: [$(var_1)]"
    	@echo "var_2: [$(var_2)]"
    	@echo "var_3: [$(var_3)]"
    	@echo "var_4: [$(var_4)]"
    ```

    run: `make`

    output:

    ```
    var_1: [hello world]
    var_2: [hello world]
    var_3: [hello world]
    var_4: [hello world]
    ```

    但是 makefile 不会删变量最右侧的空格：

    ```makefile
    var_1 := hello world    

    test:
    	@echo "var_1: [$(var_1)]"
    ```

    output:

    ```
    var_1: [hello world    ]
    ```

    bash 要求等号左右不允许有空格：

    ```bash
    var_1=hello
    var_2 = hello
    var_3= hello
    var_4 =hello
    var_5=hello world
    var_6=hello    # 有后缀空格
    var_7=" hello   "

    echo var_1: [${var_1}]
    echo var_2: [${var_2}]
    echo var_3: [${var_3}]
    echo var_4: [${var_4}]
    echo var_5: [${var_5}]
    echo var_6: [${var_6}]
    echo var_7: [${var_7}]
    echo var_7: ["${var_7}"]
    ```

    output:

    ```
    main.sh: line 2: var_2: command not found
    main.sh: line 3: hello: command not found
    main.sh: line 4: var_4: command not found
    main.sh: line 5: world: command not found
    var_1: [hello]
    var_2: []
    var_3: []
    var_4: []
    var_5: []
    var_6: [hello]
    var_7: [ hello ]
    var_7: [ hello   ]
    ```

    解释：

    * `var_1=hello`

        没有问题，正常的赋值变量的方式。

    * `var_2 = hello`

        bash 会将`var_2`作为一个 command，` = hello`作为 command 的第一个和第二个参数。

    * `var_3= hello`

        `var_3`为空字符串，`hello`是一个 command。

        这个模式类似于`LD_LIBRARY_PATH=xxx ./main`

    * `var_4 =hello`

        `var_4`是一个 command，`=hello`是其第一个参数。

    * `var_5=hello world`

        这个与`var_3= hello`同理。

    * `var_6=hello    # 有后缀空格`

        忽略字符串后的空格，认为这些空格是空白分隔符。

    * `var_7=" hello   "`

        使用双引号将空格也算在字符串内。

        但是在打印的时候出了问题。

        对于`echo var_7: [${var_7}]`，bash 会将`${var_7}`替换为` hello   `，因此实际执行的命令为`echo var_7: [ hello   ]`，`echo`会认为`var_7:`, `[`, `hello`, `]`是 4 个独立的字符串，两个字符串之间的的空格都为 1.

        对于`echo var_7: ["${var_7}"]`，经过 bash 替换变量后为`echo var_7: [" hello   "]`，echo 会认为`var_7`是第 1 个字符串，`[`, `" hello   "`, `]`分别是第 2，3，4 个字符串，但是这些字符串紧挨着，中间没有空格。

* makefile 中，变量与定义间的空格

    ```makefile
    VAR=foo       # 值是 "foo"
    VAR =foo      # 值是 "foo"
    VAR= foo      # 值是 " foo"（前面多了一个空格！）
    VAR = foo     # 值是 " foo"（同样多一个空格）
    ```

    推荐写法:

    ```makefile
    KERNEL_DIR := /usr/xxx   # 立即展开赋值
    ```

* 在 Makefile 中，`$(VAR)`和`${VAR}`在功能上是完全相同的，可以互换使用。

    使用 makefile 的内置函数时，必须使用圆括号，比如`$(subst from,to,text)`

    综合看来，makefile 中使用圆括号较多，使用花括号`${VAR}`比较少见。

    makefile 中，不允许使用`$VAR`。只会解析`$V`。

    example:

    ```makefile
    NAME = MyApp
    VAR = wrong_value
    V = correct_value

    test:
    	@echo "你想输出 MyApp, 但实际会输出: $NAME"
    	@echo "解析后相当于: $(V)NAME"
    	@echo "而变量 V 的值是: $(V)"
    ```

    output:

    ```
    你想输出 MyApp, 但实际会输出: AME
    解析后相当于: correct_valueNAME
    而变量 V 的值是: correct_value
    ```

## note

* hello world example

    `Makefile`:

    ```makefile
    hello:
    	echo "hello, world"
    ```

    run: `make`

    output:

    ```
    echo "hello, world"
    hello, world
    ```

    其中`hello`是一个 target，makefile 中的第一个 target 是默认 target，当执行`make`命令时，会执行默认 target。

    target 跟的是 command，所有的 command 都以`\t`缩进，不能是空格，否则会报错。

* target 与文件

    > A target is usually the name of a file that is generated by a program; examples of targets are executable or object files. A target can also be the name of an action to carry out, such as ‘clean’ (see Phony Targets).

    每个 target 都与当前目录下的一个文件相对应，如果名字为`<target>`的文件不存在，则执行 target 下面的 command。如果文件存在，那么不执行命令。

    example:

    files in current dir:

    ```
    hello  Makefile
    ```

    ```makefile
    hello:
    	echo "hello, world"
    ```

    run: `make`

    output:

    ```
    make: 'hello' is up to date.
    ```

* target 与前置条件（prerequisite）

    在 target 的冒号后跟其他 target，可以指定当前 target 的依赖。

    example:

    ```makefile
    main: main.c
    	gcc -g main.c -o main
    ```

    run: `make`

    output:

    ```
    gcc -g main.c -o main
    ```

## 其他

Ref: <https://www.gnu.org/software/make/manual/html_node/index.html#SEC_Contents>

> When make recompiles the editor, each changed C source file must be recompiled. If a header file has changed, each C source file that includes the header file must be recompiled to be safe. Each compilation produces an object file corresponding to the source file. Finally, if any source file has been recompiled, all the object files, whether newly made or saved from previous compilations, must be linked together to produce the new executable editor. 

rules:

```Makefile
target … : prerequisites …
        recipe
        …
        …

```

A prerequisite is a file that is used as input to create the target. A target often depends on several files.

A recipe is an action that make carries out. A recipe may have more than one command, either on the same line or each on its own line. Please note: you need to put a tab character at the beginning of every recipe line! This is an obscurity that catches the unwary. If you prefer to prefix your recipes with a character other than tab, you can set the .RECIPEPREFIX variable to an alternate character (see Other Special Variables). 

Usually a recipe is in a rule with prerequisites and serves to create a target file if any of the prerequisites change. However, the rule that specifies a recipe for the target need not have prerequisites. For example, the rule containing the delete command associated with the target ‘clean’ does not have prerequisites. 

A rule, then, explains how and when to remake certain files which are the targets of the particular rule. make carries out the recipe on the prerequisites to create or update the target. A rule can also explain how and when to carry out an action. See Writing Rules.

A makefile may contain other text besides rules, but a simple makefile need only contain rules. Rules may look somewhat more complicated than shown in this template, but all fit the pattern more or less. 

**A Simple Makefile**:

```Makefile
edit : main.o kbd.o command.o display.o \
       insert.o search.o files.o utils.o
        cc -o edit main.o kbd.o command.o display.o \
                   insert.o search.o files.o utils.o

main.o : main.c defs.h
        cc -c main.c
kbd.o : kbd.c defs.h command.h
        cc -c kbd.c
command.o : command.c defs.h command.h
        cc -c command.c
display.o : display.c defs.h buffer.h
        cc -c display.c
insert.o : insert.c defs.h buffer.h
        cc -c insert.c
search.o : search.c defs.h buffer.h
        cc -c search.c
files.o : files.c defs.h buffer.h command.h
        cc -c files.c
utils.o : utils.c defs.h
        cc -c utils.c
clean :
        rm edit main.o kbd.o command.o display.o \
           insert.o search.o files.o utils.o
```

When a target is a file, it needs to be recompiled or relinked if any of its prerequisites change. In addition, any prerequisites that are themselves automatically generated should be updated first.

A recipe may follow each line that contains a target and prerequisites. These recipes say how to update the target file. A tab character (or whatever character is specified by the `.RECIPEPREFIX` variable; see Other Special Variables) must come at the beginning of every line in the recipe to distinguish recipes from other lines in the makefile. (Bear in mind that make does not know anything about how the recipes work. It is up to you to supply recipes that will update the target file properly. All make does is execute the recipe you have specified when the target file needs to be updated.) 

Targets that do not refer to files but are just actions are called phony targets. See Phony Targets, for information about this kind of target.

By default, `make` starts with the first target (not targets whose names start with ‘.’ unless they also contain one or more ‘/’). This is called the default goal. (Goals are the targets that make strives ultimately to update. You can override this behavior using the command line (see Arguments to Specify the Goals) or with the .DEFAULT_GOAL special variable (see Other Special Variables). 

The recompilation must be done if the source file, or any of the header files named as prerequisites, is more recent than the object file, or if the object file does not exist. 

`make`会检测文件的修改时间，如果 prerequisites 的时间比 target 的时间要新，那么就重新构建 target。所以只要把`.c`，`.cpp`，`.h`等文件放到 prerequisites 中，那么就可以每次修改完源代码，就自动构建依赖了。

Variables allow a text string to be defined once and substituted in multiple places later (see How to Use Variables). 

It is standard practice for every makefile to have a variable named objects, OBJECTS, objs, OBJS, obj, or OBJ which is a list of all object file names.

Example:

```Makefile
objects = main.o kbd.o command.o display.o \
          insert.o search.o files.o utils.o
```

Then, each place we want to put a list of the object file names, we can substitute the variable’s value by writing ‘$(objects)’

Example:

```Makefile
objects = main.o kbd.o command.o display.o \
          insert.o search.o files.o utils.o

edit : $(objects)
        cc -o edit $(objects)
main.o : main.c defs.h
        cc -c main.c
kbd.o : kbd.c defs.h command.h
        cc -c kbd.c
command.o : command.c defs.h command.h
        cc -c command.c
display.o : display.c defs.h buffer.h
        cc -c display.c
insert.o : insert.c defs.h buffer.h
        cc -c insert.c
search.o : search.c defs.h buffer.h
        cc -c search.c
files.o : files.c defs.h buffer.h command.h
        cc -c files.c
utils.o : utils.c defs.h
        cc -c utils.c
clean :
        rm edit $(objects)
```

it has an implicit rule for updating a ‘.o’ file from a correspondingly named ‘.c’ file using a ‘cc -c’ command.

When a ‘.c’ file is used automatically in this way, it is also automatically added to the list of prerequisites. We can therefore omit the ‘.c’ files from the prerequisites, provided we omit the recipe. 

Example:

```Makefile
objects = main.o kbd.o command.o display.o \
          insert.o search.o files.o utils.o

edit : $(objects)
        cc -o edit $(objects)

main.o : defs.h
kbd.o : defs.h command.h
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h

.PHONY : clean
clean :
        rm edit $(objects)
```

## Quickstart

### 1. echo hello world

在一个空文件夹下新建`Makefile`文件，内容如下：

`Makefile`:

```makefile
hello:
	echo "hello world"
```

运行`make`命令，可以看到输出：

```
echo "hello world"
hello world
```

注意，`echo`前是一个`tab`，而不是 4 个空格，这是 Makefile 的规定。如果写成空格，会报错。

Makefiles must be indented using TABs and not spaces or `make` will fail.

上面的代码中，`hello`被称为 target，下面的`echo "hello world"`被称为 recipes 或 rules。他们的意思也很明显了：我们使用 recipe 来构建 target。

### 2. compile hello world

在一个空文件夹下新建`main.cpp`文件，内容如下

`main.cpp`:

```cpp
#include <iostream>
using namespace std;

int main()
{
	cout << "hello world" << endl;
	return 0;
}
```

然后新建`Makefile`文件，内容如下：

```Makefile
main:
	g++ main.cpp -o main
```

编译：

```bash
make
```

编译输出：

```bash
g++ main.cpp -o main
```

此时可以看到编译出的文件：`main`

运行：

```bash
./main
```

输出：

```
hello world
```

Makefile 会把 target 看作一个文件，如果 target 不存在，那么使用 recipe 进行构建。在这个例子中，Makefile 先检测到`main`文件不存在，然后使用`g++ main.cpp -o main`进行构建。

在前面的 echo hello world 例子中，target 是`hello`，由于`echo "hello world"`总是无法生成`hello`文件，所以每次运行`make`都会执行`echo "hello world"`。在当前例子中，由于`g++`命令执行后可以生成`main`文件，所以第二次执行`make`命令，将不会执行任何操作，并输出`make: 'main' is up to date.`。

### 3. dependency

[to do]

`make`并没有版本控制系统，所以如果我们此时修改了`main.cpp`中的内容，再执行`make`，那么`make`检测到`main`文件存在后便不再执行任何操作

## Rules

通常每

rules:

```makefile
targets: prerequisites
    command
    command
    command
```

* The `targets` are file names, separated by spaces. Typically, there is only one per rule.

* The `commands` are a series of steps typically used to make the target(s). These need to start with a tab character, not spaces.

* The `prerequisites` are also file names, separated by spaces. These files need to exist before the commands for the target are run. These are also called *dependencies*.

Example:

```makefile
blah: blah.o
	cc blah.o -o blah  # Runs third

blah.o: blah.c
	cc -c blah.c -o blah.o  # Runs second

blah.c:
	echo "int main() { return 0; }" > blah.c  # Runs first
```

依赖链分析：`blah`的生成需要`blah.o`，`make`程序往下找到`blah.o`目标，发现它依赖于`blah.c`；然后`make`程序发现`blah.c`什么都不信赖，所以直接就执行对应的命令了。

`make`程序会将`Makefile`中的第一个 target 作为 default target。

如果某个 target 已经被创建，那么`make`就会跳过这个 target。

如果某个 target 不创建任何文件，那么每次`make`，它被依赖时，总是执行。常见的`clean`选项就利用了这个特点：

```makefile
some_file:
    touch some_file

clean:
    rm -rf some_file
```

此时如果执行`make`，那么会执行`some_file` target。如果执行`make clean`，那么会执行`clean` target。

使用变量：

```makefile
files := file1 file2
some_file: $(files)
	echo "Look at this variable: " $(files)
	touch some_file

file1:
	touch file1
file2:
	touch file2

clean:
	rm -f file1 file2 some_file
```

取变量值可以使用`$()`，也可以使用`${}`。

说明：

1. 这个例子里变量用`:=`赋值了，也可以使用`=`。

另外一个例子：

```makefile
x := dude

all:
	echo $(x)
	echo ${x}

	# Bad practice, but works
	echo $x 
```

自己写的一个例子：

```makefile
lib_src = hello.c

main: main.c hello.o
	cc main.c hello.o -o main

hello.o: hello.c
	cc -c ${lib_src} -o hello.o

clean:
	rm -f main

echo_lib_src:
	echo ${lib_src}
```

**打印一个变量**：

```makefile
$(info    VAR is $(VAR))
```

Ref: <https://stackoverflow.com/questions/16467718/how-to-print-out-a-variable-in-makefile>

打印变量这个功能挺有用的，有时间了看下。

**使用 eval 改变变量的值**

```makefile
$(eval COMPONENTS = opengl)
```

说明：

1. 目前不清楚使用 eval 对变量赋值和直接对变量赋值有什么不同

**执行多个 targets**:

```makefile
all: one two three

one:
	touch one
two:
	touch two
three:
	touch three

clean:
	rm -f one two three
```

multiple targets:

```makefile
all: f1.o f2.o

f1.o f2.o:
	echo $@
# Equivalent to:
# f1.o:
#	 echo f1.o
# f2.o:
#	 echo f2.o
```

其中`$@`表示 target name。

`make`的执行结果：

```
echo f1.o
f1.o
echo f2.o
f2.o
```

说明：

1. 如果执行`make`或`make all`，那么会分别执行`make f1.o`和`make f2.o`。如果执行`make f1.o`，那么只会执行`make f1.o`；`f2.o`同理。如果执行`make f1.o f2.o`，那么效果同`make`和`make all`。

**Wildcard**

常用的 wildcard 有`*`和`%`。

* `*` wildcard

## Conditional

```makefile
libs_for_gcc = -lgnu
normal_libs =

foo: $(objects)
ifeq ($(CC),gcc)
        $(CC) -o foo $(objects) $(libs_for_gcc)
else
        $(CC) -o foo $(objects) $(normal_libs)
endif
```

匹配一个空变量：

```makefile
ifeq ($(TOP),)
$(error TOP not defined: Was preconfig.mk included in root makefile?)
endif
```

## Functions

Function call examples:

```makefile
$(function arguments)
${function arguments}
```

调用内置函数的一个例子：

```makefile
comma:= ,
empty:=
space:= $(empty) $(empty)
foo:= a b c
bar:= $(subst $(space),$(comma),$(foo))
# bar is now ‘a,b,c’.
```

## Variables

**在 makefile 中尝试输出当前的所有变量**

Ref: <https://stackoverflow.com/questions/7117978/gnu-make-list-the-values-of-all-variables-or-macros-in-a-particular-run>

上一级 make 和下一级 make 间的环境变量继承关系是怎样的？

如果下一级的 make 继承上一级的 make，那么就可以在执行下一级 make 前，把所有变量保存下来，从而可以单独运行下一级 make。这样就能独立编译一些模块了。

**makefile 中`:=`和`=`的不同**

`:=`是一次性赋值，`=`是递归赋值，一个变量改变，所有与之相关的变量都受其影响。

有空了看一下。

Ref: <https://stackoverflow.com/questions/4879592/whats-the-difference-between-and-in-makefile>

## Miscellaneous

1. 在 makefile 里使用`cd`进入其他目录

    ```Makefile
    target:
        cd xxx; run xxx;
        // or use this:
        cd xxx && run xxxx
    ```

    Ref: <https://stackoverflow.com/questions/1789594/how-do-i-write-the-cd-command-in-a-makefile>

* 在 make 时报错`*** missing separator.  Stop.`

    很有可能是`\t`被換成了空格。

* 如果一个 makefile 里的一个 target 只使用`$()`调用函数，不执行 bash 命令，那么它被视作什么也不做

    这时候会产生一条`make: xxx is up to date`的提示。

* 待处理的笔记（需要把 qa 完善，才能处理这些笔记）

    每个 makefile 都包含下面这五种成分： explicit rules, implicit rules, variable definitions, directives, and comments.

	其中 explicit rules 告诉 make 程序如何构建一个 target（通常是一个文件），implicit rules 告诉 make 如何构造链式依赖（主要是那些 prerequisites）

	一个命令只能做三件事：

	* Reading another makefile (see Including Other Makefiles).

	* Deciding (based on the values of variables) whether to use or ignore a part of the makefile.

	* Defining a variable from a verbatim string containing multiple lines.

	可以使用`-f name` or `--file=name`执行指定的 makefile。

	**include**

	将 make 程序从当前行停下来，先读取其他 makefile

	```makefile
	include filenames…
	```

	`filenames` can contain shell file name patterns. If `filenames` is empty, nothing is included and no error is printed. If the file names contain any variable or function references, they are expanded.

	Example:

	```makefile
	include foo *.mk $(bar)
	```

	include 文件的搜索顺序：

	1. current directory

	2. 使用`-I` or `--include-dir`指定的目录

	3. `/usr/local/include`, `/usr/include`

	The `.INCLUDE_DIRS` variable will contain the current list of directories that make will search for included files.

	You can avoid searching in these default directories by adding the command line option `-I` with the special value `-` (e.g., `-I-`) to the command line. This will cause make to forget any already-set include directories, including the default directories. 

	如果找不到 include 后面的文件，make 不会立即报错，只有当读完 makefile 所有内容，仍找不到构建 include 所需的 makefile 的方法后，才会报 fatal error。

	如果想让 make 在找不到 include 文件时不报错，可以使用`-include filenames…`。

	注：

	1. 这个`-I-`不知道怎么用，目前看来也没有什么用

	**MAKEFILES**

	If the environment variable MAKEFILES is defined, make considers its value as a list of names (separated by whitespace) of additional makefiles to be read before the others. 

	搜索的路径与`include`相同。

	看不出来有啥用。

	小知识：`%`会 matches any target whatever.

	**make的两个工作流程**

	GNU make does its work in two distinct phases. During the first phase it reads all the makefiles, included makefiles, etc. and internalizes all the variables and their values and implicit and explicit rules, and builds a dependency graph of all the targets and their prerequisites. During the second phase, make uses this internalized data to determine which targets need to be updated and run the recipes necessary to update them. 

	这两个阶段直接决定了 when variable and function expansion happens

	We say that expansion is immediate if it happens during the first phase: make will expand that part of the construct as the makefile is parsed. We say that expansion is deferred if it is not immediate. Expansion of a deferred construct part is delayed until the expansion is used: either when it is referenced in an immediate context, or when it is needed during the second phase. 

	* Variable Assignment

			```makefile
			immediate = deferred
			immediate ?= deferred
			immediate := immediate
			immediate ::= immediate
			immediate :::= immediate-with-escape
			immediate += deferred or immediate
			immediate != immediate

			define immediate
			deferred
			endef

			define immediate =
			deferred
			endef

			define immediate ?=
			deferred
			endef

			define immediate :=
			immediate
			endef

			define immediate ::=
			immediate
			endef

			define immediate :::=
			immediate-with-escape
			endef

			define immediate +=
			deferred or immediate
			endef

			define immediate !=
			immediate
			endef
			```

			For the append operator ‘+=’, the right-hand side is considered immediate if the variable was previously set as a simple variable (‘:=’ or ‘::=’), and deferred otherwise.

			For the immediate-with-escape operator ‘:::=’, the value on the right-hand side is immediately expanded but then escaped (that is, all instances of $ in the result of the expansion are replaced with $$).

			For the shell assignment operator ‘!=’, the right-hand side is evaluated immediately and handed to the shell. The result is stored in the variable named on the left, and that variable is considered a recursively expanded variable (and will thus be re-evaluated on each reference). 

	* Conditional Directives

			Conditional directives are parsed immediately. This means, for example, that automatic variables cannot be used in conditional directives, as automatic variables are not set until the recipe for that rule is invoked. If you need to use automatic variables in a conditional directive you must move the condition into the recipe and use shell conditional syntax instead. 

	* Rule Definition

			```makefile
			immediate : immediate ; deferred
					deferred
			```

			That is, the target and prerequisite sections are expanded immediately, and the recipe used to build the target is always deferred. This is true for explicit rules, pattern rules, suffix rules, static pattern rules, and simple prerequisite definitions. 

	**How Makefiles Are Parsed**

	1. Read in a full logical line, including backslash-escaped lines (see Splitting Long Lines).

	2. Remove comments (see What Makefiles Contain).

	3. If the line begins with the recipe prefix character and we are in a rule context, add the line to the current recipe and read the next line (see Recipe Syntax).

	4. Expand elements of the line which appear in an immediate expansion context (see How make Reads a Makefile).

	5. Scan the line for a separator character, such as ‘:’ or ‘=’, to determine whether the line is a macro assignment or a rule (see Recipe Syntax).

	6. Internalize the resulting operation and read the next line. 

