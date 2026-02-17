# 临时项目管理

这个文件中列出的项目，每天过一遍。

每天应该先启动任务执行系统，如果任务列表为空，那么跳转到这里，收集当天要执行的任务。

将任务管理系统作为一个基本的项目，每周至少整理一次。

每个项目都尽量遵循 复习（随机检测 qa） -> 学习 -> 调研 的步骤，不一定按这个顺序，但这三项起码要有。

有关“学习”：

1. 这里的学习是一个专有概念，其行为包括但不限于，阅读学习材料，做实验验证学习资料上的内容，做实验验证自己的猜想，记笔记，写一个新的 qa，完成任务列表上的任务

目前遇到的最大问题就是，有些问题不清楚能否解决，不清楚靠什么解决，只有在尝试解决的时候才会遇到很多新问题，并不断用已有知识去解决新问题。这样解决问题的思路就像一个无底洞，无法预知解决的时间，只能靠不断往其中投入时间。这类问题目前仍没有很好的解决方案。

不做任务执行系统，就没有顿挫的感觉，就不可能有高效率。只与项目池打交道，会产生无穷无尽的任务，从而无法分辨任务的边界，导致效率越来越低。

* 每天在使用项目管理系统获得新任务之前，都必须将之前的每日任务列表进度同步到任务管理系统中，并加上`[processed]`标记

    因为任务执行系统会改变项目管理系统中的内容

    项目管理系统最好是当天结束时更新，这样才能不对第二天收集任务清单造成影响

    如果前一天的 task list 不被处理，那么第二天的项目池几乎不可能正常运行

	因为每天临时的变动太大了，远远超过按周处理的频率。

## 本周提醒

* 总是对自己熟悉的领域过于骄傲，总以为自己的进度领先，不愿投入时间。其实对自己熟悉的领域，更应该舍得投入时间才对。

* 不应该想着有万全准备了才开始某项任务，应该想着只要不明显阻碍当前的其他任务，就应该开始这项任务。

    实际执行时，可以将这项任务拆解为更小的，便于启动的任务。

## cache

* [new] 调研斐波那契数列的并行方法

    ```py
    # 串行计算（必须依赖前值）
    def fib_serial(n):
        if n <= 1: return n
        return fib_serial(n-1) + fib_serial(n-2)  # 必须等待前两个值

    # 但是！我们可以用矩阵快速幂并行化
    def fib_parallel(n):
        # 虽然还是顺序计算，但每步可以并行处理多个不相关的计算
        return matrix_power([[1,1],[1,0]], n)[0][1]
    ```

* project pool 中的任务数量先增大（广度优先搜索），再减少。说明复杂度存在一个最大值。如果我们的处理系统/脑子达不到处理这个最大值的能力，那么说明我们的处理系统/脑子无法处理这么复杂的问题。

* 长期任务`{ } xxx`是不是只需要在下面标注日期就可以了？

    比如：

    ```
    * { } xxx

        * 02.01  30 mins

            feedback:

            * [ ] yyy

        * 02.02  30 mins
    ```

* c语言中，调试字符串相关的代码，经常有拿到某个字符对应的位置、以及拿到某个位置对应的字符的需求，自己动手数比较麻烦，是否有工具可以解决这个问题

    是的，完全有工具可以解决这个手动数位置的问题！以下是几种实用的解决方案：

    1. 在线字符串可视化工具

        推荐工具：

        * CodeBeautify String Viewer - 显示每个字符的索引位置

        * Online String Tools - 支持各种字符串分析

        * strview - 专门为C语言开发者设计

        功能特点：

        ```text
        字符串: "Hello, World!"
        索引:    0123456789012
        字符:    Hello, World!
                 ^    ^ ^    ^
                 |    | |    |
        位置:    0    5 7    12
        ```

        实时显示每个字符的索引位置和ASCII值

    2. IDE/编辑器插件

        VS Code 扩展：

        * String Manipulation - 高亮显示字符位置

        * Rainbow CSV - 对字符串进行彩色标记

        * Bracket Pair Colorizer - 帮助识别字符串边界

        配置方法：

        ```json
        // VS Code settings.json
        {
            "editor.renderWhitespace": "all",
            "editor.renderControlCharacters": true,
            "editor.wordWrap": "off"
        }
        ```

    3. C语言调试技巧

        使用gdb调试：

        ```c
        // 示例代码
        #include <stdio.h>
        #include <string.h>

        int main() {
            char str[] = "Debug this string";
            
            // 使用gdb时：
            // print str[5]      // 查看第6个字符
            // x/s &str[0]       // 查看整个字符串
            // print &str[7] - &str[0]  // 计算偏移量
            return 0;
        }
        ```

        gdb命令：

        ```bash
        (gdb) p str
        (gdb) p str[3]        # 位置3的字符
        (gdb) x/20cb str      # 以字符形式查看内存
        (gdb) printf "%d\n", strchr(str, 's') - str  # 查找's'的位置
        ```

    4. 自制辅助函数

        ```c
        #include <stdio.h>
        #include <string.h>

        void print_string_with_indices(const char *str) {
            printf("字符串: \"%s\"\n", str);
            printf("索引:    ");
            for(int i = 0; i < strlen(str); i++) {
                printf("%d", i % 10);
            }
            printf("\n字符:    ");
            for(int i = 0; i < strlen(str); i++) {
                printf("%c", str[i]);
                if(i < strlen(str)-1) {
                    // 对齐空格
                    int spaces = 1;
                    if(i >= 10) spaces = 2;
                    for(int j = 0; j < spaces; j++) printf(" ");
                }
            }
            printf("\n");
            
            printf("\n详细分析:\n");
            for(int i = 0; i < strlen(str); i++) {
                printf("位置 %2d: '%c' (ASCII: %3d)\n", 
                       i, str[i], (int)str[i]);
            }
        }

        int main() {
            char test_str[] = "C语言调试";
            print_string_with_indices(test_str);
            return 0;
        }
        ```

    5. 推荐组合方案

        日常开发：

        * VSCode + 字符串插件 - 编写时实时查看

        * 在线工具 - 快速分析复杂字符串

        * gdb - 运行时调试

        实用技巧：

        ```c
        // 快速定位技巧
        #define DEBUG_CHAR(str, pos) \
            printf("str[%d] = '%c' (ASCII: %d)\n", \
                   pos, str[pos], str[pos])

        // 使用示例
        DEBUG_CHAR("Hello World", 6);  // 输出: str[6] = 'W' (ASCII: 87)
        ```

    6. 特殊字符处理

        这些工具特别有助于处理：

        * 转义字符：\n, \t, \\

        * Unicode/中文字符（注意：中文字符占多个字节）

        * 空终止符 \0

    最佳实践：对于中文字符串调试，建议使用支持UTF-8的在线工具，并注意一个中文字符可能对应2-4个字节的存储位置。

* async chain 的实现非常重要，有时间了研究一下

	* js 中的 promise 的原理

	* rust 中 async 的原理

	* c++ 中协程的原理

	* libhv 中事件驱动的原理

* 香山 riscv 源代码 repo

    <https://gitee.com/OpenXiangShan/XiangShan/>

* 调研 MLIR, IREE

* 学习资源的调研主要用于完成下面几件事情

    1. 发现学习资源里，是否有需要做实验验证的地方，是否可以用实验验证他说的，是否有可以用实验澄清的模糊不清的表达

    2. 语言表达是否有歧义

    3. 未完待续

## cached tasks

这里主要是暂时难以分类的任务。

* [ ] 量化想法：针对不同板块的 etf，对 dmi 指标进行矩阵回测，找到最好用的 dmi 周期超参数。

* [ ] 调研 Computer algebra system

    <https://en.wikipedia.org/wiki/Computer_algebra_system#>

    自动求导、符号求导等相关知识可能和这个概念有关。

* [asso] 写一个数字图像处理库，暂时不包含深度学习

* [asso] 调研多 gpu inference

    把不同的 layer 放到不同的 gpu 上，使用一个 layer 的 output 作为另一个 layer 的 input。

* [ ] 如果有$x_1$，$x_2$两个变量，是否优化目标为修改曲面在指定点处的高度？

* [ ] 需要一个 graph 工具，建立不同的东西之间的连接

    stack 工具只适合任务的 trace

* [ ] v2ray + http 代理是否可以代理 udp？如果不可以那么如何代理 udp？

* [ ] 调研如何判断一个新的 example 是否已经在 qa 文件中

    可能的方向：向量数据库，llm + rag

    目标是在 qa file 中找到与 example 相似或相近的 unit

* [ ] 如何使用模板（卷积）的方式识别缺陷？

## cached questions

* [new] 思考：文件中使用引用标记来引用其它文件中的内容

    比如

    output

    ```
    <dir/file_name.log>
    ```

    直接在这里放整段 log 内容太长，需要把 log 保存到外部文件中。

    还需要一种机制，可以快速跳转到外部文件，或者在渲染时显示外部文件中的内容。可以考虑折叠－展开机制，输入某个快捷键，则显示文件内容，再按一次快捷键，将文件内容重新变回引用。

    还需要一种机制，检测一个外部文件是否被引用，如果没有被引用，那么可以删除。

    还需要一种机制，保证当 note 文件和 log 文件路径改变时，引用记录不会失效。

* [思考] 无论是 qa 还是 note，除了添加内容，更需要优化。这个优化该如何进行？

    如果只添加内容，不优化内容，那么东西越来越多，而且组织混乱。

    目前正在进行的一个优化是将按时间顺序写下的笔记，重新整理成按结构存储的数据。

## english words

english words 由 { } reorg: english words 进化而来。 

### cache

* 感觉还是写成一个 pat 比较好

* 由于 word 后面可能跟 prn.，也可能跟 exp.，所以 word 本身在哪结尾无法风开始就确定，需要先确定 prn. 和 exp.，再选一个较小的 start 作为 word 的 end。或者如果发现有 prn. 则直接用 prn. 的 start 作为 word 的 end，如果没有 prn，则使用 exp 的 start 作为 word 的 end。

### tasks

* [ ] 增加英语单词的 qa

## reorg

### cache

* 贝叶斯网引论下载到`Documents/Projects/book`目录里了。还下载了些其他书，有时间了看看

* 贝叶斯网的笔记不应该太简洁，应该在概念下给出大量 example，给出注释，给出引申的猜想。

* java 的 note 大部分已经空间化了，没有什么需要 reorg 的

* 清理了 latex 和 opengl 的笔记，但是这两份笔记只有几行，并没有什么需要整理的

    最好还是先让程序扫描一遍目录，然后根据规则排除一些文件，然后再跟上一次使用的数据库做对比，看看新扫描的增加了哪些文件，缺少了哪些文件，手动确认是否更新数据库。

    然后根据更新完的数据库，重新计算每个文件被选中的概率权重，最后根据权重，随机选择一个文件。

    如果被选中的文件没有什么好 reorg 的，那么可以手动调低权重，让它下次出现的概率变低。

    如果一个文件整理完后，发现还有许多要整理的地方，那么就手动升高其权重，增加它出现的概率。

    其实降低别某一个文件概率就是变相地升高其他文件的概率，所以也可以不去设置手动调高概率。

* [o] 使用 python ＋ re 写一个英语单词的 parser，每次随机检测指定数量个单词，保存索引，后面每次复习时检测上次抽取的单词 + 融合前几次抽取的单词，时间越久的单词出现的概率越小。

    feedback:
    
    1. 尝试了下，直接写 parser 有难度，目前可以用这种方式

        ```python
        def main_2():
            txt = 'endeavour prn. [ɪnˈdevər] v. 努力'
            pat_pronunciation = r'(?<= )prn\..+'
            pat_explanation = r'(?<= )v\..+'

            m = re.search(pat_pronunciation, txt)
            if m is None:
                print('fail to match')
                return
            selected_txt = txt[m.start():m.end()]
            print(selected_txt)

            m = re.search(pat_explanation, selected_txt)
            if m is None:
                print('fail to match exp')
                return
            selected_txt = selected_txt[m.start():m.end()]
            print(selected_txt)

            return
        ```

        output:

        ```
        prn. [ɪnˈdevər] v. 努力
        v. 努力
        ```

        这样只需要每次都只处理最前面的一点就可以了。生成多个 pat，用 for 轮流检测。

### tasks

* { } reorg: restricted

    * 02.03  30 mins

* [O] CUDA Core Compute Libraries (CCCL)

    <https://github.com/nvidia/cccl>

    feedback:

    1. This repository unifies three essential CUDA C++ libraries into a single, convenient repository:

        Thrust (former repo)
        CUB (former repo)
        libcudacxx (former repo)

    2. 这个库后续有兴趣了再调研，感觉目前使用 cupy 已经够用了

* [ ] qa 中的代码片段越来越长，手动编辑和翻页很慢，需要写一个程序专门管理 qa 中 unit 的添加和查看

    还需要这个程序具备以下功能：为所有 unit 添加或删除某个属性`[xxx]`，如果缺失，那么就返回 (empty)。我们可以在代码里直接 hardcode 编码所有可用属性，否则就需要一个 meta info 文件，比较麻烦。

* [ ] 调研 freedos live iso

    res:

    1. video tutorial
    
        <https://www.youtube.com/watch?v=xXkmOwLPpcg>

    2. official site
    
        <https://www.freedos.org/>

    3. Get Started with FreeDOS
    
        <https://www.freedos.org/books/get-started/>

* [ ] 调研 freedos bonus iso

* { } reorg: projects

* { } reorg: documents

* { } windows 文件整理

    目前主要整理`D:\Documents\xdx_res`, `D:\shared_folder\ai_resources`, `D:\shared_folder\Downloads`, `D:\Documents\res_processing`这四个文件夹。

* { } 《github入门与实践》

    看到 P7

* [ ] 增加 cimg note qa，并加入 test collect 里

    1. 增加 qa unit，打开图片，获取指定位置的像素 rgb 值

    2. 保存图片

    可以参考`ref_5`

* [ ] 在 v100 5.15 系统下安装 docker，并尝试透传 nvidia gpu device

* [P] 增加一项功能：是否答对，并将结果记录到 qa 中。

    如果一个 unit 答对的频率较高，那么它被选择的概率变小。

    如果一个 unit 距离上次回答的时间较长，那么它被选择的概率变大。

    feedback:

    * [ ] py 中，如果有函数 A(), B()，并且 B() 在 A() 的下面，那么 A() 如何调用到 B()？

    * 不能按所有历史记录去计算，因为一年前答题好坏不代表现在答题好坏。必须使用滑动窗口，回顾过去 N 天的记录。

        但是过去 N 天可能一次都没有记录。我们可以设定一个阈值，比如过去 180 天如果这个 unit 被提问的次数小于等于 3 次，那么可以认为答对率无参考意义。

        在一个 qa 文件中，如果过去 180 天这个问题未被抽取次数小于等于 3 次，那么按上次提问日期进行倒序排列，然后线性化到 0 ～ 1，得到的数值即作为分数。如果大于 3 次，那么将错误率作为分数，比如错误率为 80%，那么得分为 0.8。然后按得分倒序排列，优先选得分靠前的。

        并且结果不能记录到 qa 中，必须记录到 record 中，每次列举过去 180 天的答题情况。

* [ ] 将 project pool 中常用到的 pdf 等 resources 打包成 zip，发送到邮箱里

* [ ] 找到贝叶斯网引论 pdf，上传到邮箱里

* [ ] powershell 调研<https://learn.microsoft.com/en-us/powershell/scripting/samples/sample-scripts-for-administration?view=powershell-7.4>

    目前看完了`Working with objects`中的`Viewing object structure`。

* [asso] 调研自己搭建 ftp 服务器，是否能用 aria2 实现多线程下载？

* [asso] 调研 http 服务器是否支持 aria2 多线程下载文件？

* [asso] python 简易 http 服务器

    `python3 -m http.server 8000 -d /usr/local/cuda/doc/thrust/html/`

* [asso] Dash/Zeal（离线API查看器）

* [asso] cppman

## qa

### cache

* 一些 gnu 工具入门级的 guideline，废话有点多

    <https://thevaluable.dev>

* 使用 bfs 算法对 qa units 从易到难检测

* 现在主要需要实现 new, prev rand 的功能。等这两个实现后，需要实现 dependency 的功能，根据一个条目，可以查到它的依赖条目，根据依赖条目。

* cmake qa:

    修改`使用 cmake 根据下面的代码编译出`libhmath.so`库，然后编译出可执行文件。`的`[u_0]`。

    增加`mymath.h`和`mymath.cpp`相关内容。

* 如果一些知识点正处于 cache 状态，未变成基于空间结构的数据，但是任务中又要用到，任务会依赖一些 qa，该怎么办？

* 如果在 collect 文件里新增加一个 qa 文件，权重该如何设置？

### Tasks

* [v] exam 在显示 unit 时，显示 idx, idx 以及其所对应的 qa 文件名

* [ ] opengl add qa: 请使用 shader 画一个彩色的 cube，并使之旋转。

* [O] 调研 <https://thevaluable.dev/regular-expression-basics-vim-grep/> 12.15

    目前看到
    
    > Quantifiers

* [o] 给每个 unit 设置一个比重，在抽取随机数时按比重抽取

    感觉比较熟悉的，之前重复出现过的 unit，可以把比重设置得低一点

    feedback:

    * 目前只实现了给每个 qa file 增加一个比重，并且保证在选择时，采样不到重复的 qa file

        或许应该实现 qa file 可以相同，但是 unit 需要保证不同？

* [ ] 如果观察的是一个连续量，比如随机摘一株草，观察其长度，那么是否无法写出样本点？是否必须以变量 + 区间 + 叉乘的形式写出样本空间？

* [ ] 调研 qa parse 与 rewrite 时是否保留了 unit 的`[dep]`信息

* 修改 opengl note qa 中的`请给一个三角形加上纹理贴图。`，在 glsl 代码前加上

    `#version 330 core`

    否则跑不通。

* [ ] fix bug: 保存最新 qa record 时，不能删除旧的

* 假如一个集合有 10 个 0.1，现在只允许每个元素对自身除以 2，再平均到 1，这个集合构造出的数是有限的还是无限的？这些数的取值的概率密度是怎样的？

* [v] qa: 4 units 12.18

    正确率：3 / 4

    feedback:

    3. 动态的 review 间隔确定：通过即时复述，确定记忆量；间隔一段时间，比如早上到晚上，或者早上到第二天早上，再次复述，达到 90% 暂定）以上

* [ ] `使用 element draw 画一个 cube`增加 deps:

    1. load shader

* [ ] 记录每个 uni 的多个历史完成时间，如果平均时间大于 N 分钟，那么标记该 unit 应该被拆分成 deps。

* [ ] 调研 mysql

* [ ] 调研：unit 增加 hint 字段，显示完`[u_0]`后，输入`h`显示一条 hint 

    或许应该把 d 命令和 h 命令合成一个功能？

* [ ] qa: exam 中调换 clear previous qa_record 和 save this exam record 的位置

* [ ] qa: 增加 openshmem 的 qa

* [ ] 创建一个`qa_utils`文件下，下设`opengl`, `opencl`等文件夹，文件夹中再设`triangle.c`, `vec_add.c`等文件或文件夹，与 dep 任务相对应。

    最好可以把功能写成`.h`的形式，在使用时直接`#include "qa_utils/opengl/load_shader.h"`就可以。

    实在写不成`.h`的形式，再写成`.c`或`.cpp`的形式，如果连这种形式也写不成，就创建文件夹工程目录。

    dep 不一定要实现完整工程，只需要写出核心的逻辑就可以。

* [ ] exam 程序应该先打印 deps，如果有 deps，必须依赖 deps 进行开发

* [asso] 写一套文件操作 API，分别使用文件函数和 mmap 函数打开文件，实现功能如下：

    1. 从指定位置开始，读指定字节的数据，若字节数为 <= 0，则读取所有数据

    1. 实现一个 generator，每调用一次读一行，文件的最后一行如果只有 EOF 没有 \n，那么也算一行。如果 \n 后紧接 EOF，那么算同一行

    1. 实现一个函数，将指定字节处（从 0 开始索引）的字节替换成指定字节

* [asso] 调研文件编辑器如何实现插入/删除功能？

* [asso] 调研 qemu edu driver 将寄存器`mmap()`到用户态，使用 polling 的方式代替中断

* [asso] `cublasSgemm`

* [asso] latex/pdflatex/xelatex

* [asso] 终端图片显示工具（如 chafa, img2sixel, Kitty 的 icat）

* [asso] convert (ImageMagick) 

* [asso] latex2img 脚本或 catimg（简单图片显示

* [asso] texmath（Haskell），pandoc（可以转换格式）

* [asso] Vim/Neovim： 插件如 vim-markdown-composer 或 markdown-preview.nvim

* [asso] Emacs： 功能强大，通过 org-mode 或 latex-preview-pane 等可以在编辑器内渲染公式（通常是生成图片覆盖在文本上）。

* [asso] mdmath + 浏览器

    ```bash
    # 将 Markdown 转换为 HTML 并在浏览器中打开
    pandoc math.md -o math.html --mathjax && xdg-open math.html
    ```

* [asso] pandoc + lynx

    ```bash
    # 转换为文本格式查看
    sudo apt install pandoc lynx
    pandoc document.md -t plain | less
    ```

* [asso] 图片变成字符画

    chafa：强烈推荐。功能非常强大，支持多种输出格式（字符、符号、六角形等），色彩还原好，性能高。是目前最好的选择之一。

    catimg： 简单易用，专门用于显示图片，对彩色图片支持不错。

    img2txt（来自 caca-utils 包）： 老牌工具，也能生成字符画。

    ```bash
    # 安装 chafa (Ubuntu/Debian)
    sudo apt install chafa

    # 查看图片
    chafa photo.jpg
    # 或指定大小为终端宽度的一半
    chafa -s 80x40 photo.jpg
    ```

* [asso] 图形协议方式

    这种方法利用终端仿真器支持的特殊协议，直接在其文本网格中渲染图形。效果最好，能显示真彩色的原图。

    * Kitty 图形协议： Kitty 终端自带的协议，效率很高。

        工具： Kitty 终端自带的 icat 命令。

        ```bash
        # 只在 Kitty 终端中有效
        kitty +kitten icat image.png
        ```

    * Sixel： 一种较老的协议，但被许多终端支持（如 XTerm, WezTerm, Mintty）。

        工具： img2sixel, chafa（也支持 Sixel 输出）。

        ```bash
        # 安装 ImageMagick (通常包含 `convert`，可用于生成 sixel)
        brew install imagemagick
        # Ubuntu: sudo apt install imagemagick

        # 转换为 sixel 格式并显示
        convert image.jpg sixel:-
        # 或者使用专门工具
        img2sixel image.jpg
        ```

    * iTerm2 内联图片协议： 专用于 iTerm2 终端。

        iTerm2 提供了 imgcat 脚本。

        ```bash
        # 通常在 iTerm2 中，可以这样使用
        ~/.iterm2/imgcat image.png
        ```

* [asso] `xdg-open`

* [asso] Kitty, WezTerm

* [asso] Sixel 或 Kitty 的图形协议

* [asso] texmath

* [asso] MathGL： 一个用于绘制数学数据的科学图形库，它可以在终端中绘制函数图像，但并非渲染任意公式。

* [asso] `docker start -ia` 其中`-ia`是什么含义？

    如果只有`-i`没有`-a`会发生什么，如果只有`-a`没有`-i`会发生什么？如果两个都没有会发生什么？

* [asso] `sysfs_create_group()`, `sysfs_remove_group()`

* [asso] `^(?!\./).*\..*`  

    排除以 ./ 开头

* [asso] `(^|/)\.[^/].*`

    隐藏文件匹配

* [asso] `(^|/)\.[^/]*(?=/|$)`, `(?:^|/)(?:\.[^/]+)(?=/|$)`, `(?:^|/)(?:\.[^/]+)+(?=/|$)`, `(^|/)\.([^/]+/)+`

    匹配隐藏目录

* [asso] catwalk

* [asso] linux 中的管道（pipe）底层是如何实现的？

## cache tabs / process urls / process tab

* 需要消化 cached urls

* 处理 url 的时间比想象中要长，每一条大概需要 30 mins

* 可以使用 youtube 学一些英语课，比如 julia，octave 等，这样既锻炼了英语，也学到了东西

### tasks

* [ ] 确保正确的内核配置

    ```bash
    # 检查内核配置中是否启用了 DRM 子分配器
    zcat /proc/config.gz | grep CONFIG_DRM_GPUVA_MGR
    # 或
    grep CONFIG_DRM_GPUVA_MGR /boot/config-$(uname -r)
    ```

* [ ] 强制正确的模块依赖顺序

    ```bash
    # 在 /etc/modules-load.d/ 中确保先加载 DRM 相关模块
    echo "drm" | sudo tee -a /etc/modules-load.d/drm.conf

    # 或使用 modprobe 显式指定依赖
    sudo modprobe drm
    sudo modprobe sipu
    ```

* [ ] 如果使用 DKMS

    ```bash
    # 重新注册和构建 DKMS 模块
    sudo dkms remove sipu/version -k $(uname -r)
    sudo dkms add /path/to/sipu
    sudo dkms build sipu/version -k $(uname -r)
    sudo dkms install sipu/version -k $(uname -r)
    ```

* [ ] 忽略缺失的符号

    ```bash
    # 忽略缺失的符号（有风险）
    sudo insmod /path/to/sipu.ko allow_unsupported=1
    ```

* [v] `let save_reg = @"`, `let @" = save_reg`

* [ ] `let start_line = line("'<")`

* [ ] vim 中列表的用法`let lines = []`

* [ ] vim `s` 命令与`c`命令

    s```<C-r>"``` 

    映射：

    vnoremap <leader>` c```<C-r>0```<Esc>

* [v] vim `I` 命令与`A`命令

    ```vim
    qq
    I```<Esc>
    A```<Esc>
    q
    ```

    `:'<,'>normal @q`或`:'<,'>norm @q`

* [ ] vim 命令

    :'<,'>s/.*/```&```/

    解释：

        '<,'>：表示选中的范围

        s/.*/```&```/：将整行替换为三个反引号 + 原内容 + 三个反引号

        & 表示匹配到的整个文本

* [ ] 调研 vim-surround 插件

    <https://github.com/tpope/vim-surround>

    安装插件后

    选中文本，输入：
    
    ```vim
    S```
    ```

    就会自动在选中文本两侧添加三个反引号

* [v] `git submodule update --init --progress`

    `git clone --recurse-submodules ../repo-server .`

* [ ] 调研下述说法是否正确

    ❌ Gemini 的说法

    .gitmodules 里的 url = ../repo-server-2 是相对于主仓库的远程地址（origin URL）

    这句话 只有在“主仓库是远程 URL（如 https://、ssh://）”时才成立，
    在你这个 case（本地路径 clone）里是错误的理解延伸。

* [ ] 调研下述说法是否正确

    Git 的规则是：

    子模块的相对 URL 是相对于“父仓库的 superproject URL”解析的

    而你的 superproject URL 是：

    git clone ../repo-server .


    也就是说：

    superproject URL = ../repo-server

* [ ] `git ls-files --stage`

* [ ] `git config --get-all submodule.repo_B.url`

* [ ] `git clone --recursive ../repo-server .`

* [ ] 手动初始化并指定 URL：

    ```bash
    git submodule init
    # 如果自动获取的 URL 不对，可以手动设置（注意路径）
    git config submodule.repo_B.url ../repo-server-2 
    git submodule update
    ```

* [ ] `fatal: transport 'file' not allowed`

    err info:

    ```
    (base) hlc@hlc-VirtualBox:~/Documents/Projects/git_test/repo-contains-submodule$ git submodule update 

    Cloning into '/home/hlc/Documents/Projects/git_test/repo-contains-submodule/repo-2'...

    fatal: transport 'file' not allowed

    fatal: clone of '/home/hlc/Documents/Projects/git_test/repo-contains-submodule/../repo-server-2' into submodule path '/home/hlc/Documents/Projects/git_test/repo-contains-submodule/repo-2' failed

    Failed to clone 'repo-2'. Retry scheduled

    Cloning into '/home/hlc/Documents/Projects/git_test/repo-contains-submodule/repo-2'...

    fatal: transport 'file' not allowed

    fatal: clone of '/home/hlc/Documents/Projects/git_test/repo-contains-submodule/../repo-server-2' into submodule path '/home/hlc/Documents/Projects/git_test/repo-contains-submodule/repo-2' failed

    Failed to clone 'repo-2' a second time, aborting 
    ```

    1. 为什么会报错？

    为了防御安全漏洞（CVE-2022-39253），Git 在较新版本中默认禁用了子模块的 本地文件传输协议 (File Protocol)。

    当你使用相对路径（如 ../repo-server-2）且 Git 识别到这指向本地磁盘路径时，它会出于安全考虑拦截这个操作，防止恶意仓库通过子模块手段读取你机器上的敏感本地文件。
    2. 解决方法

    你需要通过 Git 配置显式允许使用 file 协议。你有两种选择：
    方案 A：单次命令允许（推荐）

    如果你只是临时测试，不想修改全局安全设置，可以使用 -c 参数：
    Bash

    git -c protocol.file.allow=always submodule update

    方案 B：修改本地仓库配置（长期有效）

    仅针对当前这个项目开启：
    Bash

    git config protocol.file.allow always
    git submodule update

    方案 C：全局配置（方便但略降低安全性）

    如果你在本地经常做这种多仓库嵌套实验，可以全局开启：
    Bash

    git config --global protocol.file.allow always

* [ ] `git config --file=.gitmodules submodule.repo_B.url /home/hlc/Documents/Projects/git_test/repo-server-2`

* [ ] 调研下述用法

    ```bash
    # 检查是否有子模块相关记录
    git show HEAD:.gitmodules 2>/dev/null

    # 查看提交历史中是否有 .gitmodules
    git log --oneline -- .gitmodules
    ```

* [ ] 使用 ncat 进行 ssh 的 http 代理

    ```bash
    ssh -o ProxyCommand="ncat --proxy-type http --proxy proxy.company.com:3128 %h %p" myuser@server.example.com
    ```

    * %h 和 %p 是 SSH 的占位符，会自动替换为目标主机和端口。

    * 如果代理需要认证，可以使用 --proxy-auth username:password 参数，但不建议在命令中直接输入密码，有安全风险。

    配置文件：

    ```conf
    Host *
        # 如果你的代理是全局需要的，可以放在最前面
        # 但通常更建议针对特定域名配置

    # 匹配所有以 .example.com 结尾的主机，通过公司代理连接
    Host *.example.com
        User myuser # 设置默认用户名
        IdentityFile ~/.ssh/id_rsa # 指定私钥
        ProxyCommand corkscrew proxy.company.com 3128 %h %p

    # 如果代理需要认证，使用以下格式（注意安全！）
    Host *.example.com
        User myuser
        IdentityFile ~/.ssh/id_rsa
        ProxyCommand corkscrew proxy.company.com 3128 %h %p ~/.ssh/proxy_auth

    # 使用 ncat 的配置示例
    Host myserver
        HostName server.example.com
        User myuser
        ProxyCommand ncat --proxy-type http --proxy proxy.company.com:3128 --proxy-auth user:pass %h %p
    ```

* [asso] 调研`-t`是否可以让 ssh 保持连接

    > 保持连接：如果需要保持 SSH 连接，可以添加 -t 参数：

    `ssh -t user@hostname "cd my_dir && ./main"`

* [asso] 调研 screen 和 tmux

    `ssh user@host "cd my_dir && screen -dmS my_session ./main"`

* [asso] 调研`./main & disown`这个命令中，`&`的含义，以及 disown 是否可以这样写（语法是否正确）

* [O] 调研 PyTorch Loss Functions 11.22

    15:01 ~ 15:31, 23:10 ~ 23:43

    <https://www.geeksforgeeks.org/deep-learning/pytorch-loss-functions/>

    * 目前看到

        > Negative Log Likelihood Loss

* [v] 调研`torch.max()`以及`_, predicted = torch.max(outputs.data, 1)`

    为什么返回值会有两个，outputs.data 又是什么？

* [ ] Feedforward Neural Network

    <https://www.geeksforgeeks.org/nlp/feedforward-neural-network/>

    文本中的链接和下面的 explore 可以看看，正文本身没什么要看的。

* [v] Introduction to Deep Learning

    <https://www.geeksforgeeks.org/deep-learning/introduction-deep-learning/>

    页面最下方的 explore 看一下。

    feedback:

    * [ ] Introduction to Convolution Neural Network

        <https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/>

    * [ ] Introduction to Recurrent Neural Networks

        <https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/>

    * [ ] Generative Adversarial Network (GAN)

        <https://www.geeksforgeeks.org/deep-learning/generative-adversarial-network-gan/>

    * [ ] Autoencoders in Machine Learning

        <https://www.geeksforgeeks.org/machine-learning/auto-encoders/>

    * [ ] Transformers in Machine Learning

        <https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/>

    * [ ] Artificial intelligence vs Machine Learning vs Deep Learning

        <https://www.geeksforgeeks.org/artificial-intelligence/artificial-intelligence-vs-machine-learning-vs-deep-learning/>

    * [ ] Deep Learning Examples: Practical Applications in Real Life

        <https://www.geeksforgeeks.org/deep-learning/deep-learning-examples/>

    * [ ] Challenges in Deep Learning

        <https://www.geeksforgeeks.org/deep-learning/challenges-in-deep-learning/>

    * [ ] Why Deep Learning is Important

        <https://www.geeksforgeeks.org/deep-learning/why-deep-learning-is-important/>

    * [ ] What is a Neural Network?

        <https://www.geeksforgeeks.org/machine-learning/neural-networks-a-beginners-guide/>

    * [ ] Types of Neural Networks

        <https://www.geeksforgeeks.org/deep-learning/types-of-neural-networks/>

    * [ ] Layers in Artificial Neural Networks (ANN)

        <https://www.geeksforgeeks.org/deep-learning/layers-in-artificial-neural-networks-ann/>

    * [ ] Activation functions in Neural Networks

        <https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/>

    * [ ] Backpropagation in Neural Network

        <https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/>

    * [ ] Convolutional Neural Network (CNN) in Machine Learning

        <https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-machine-learning/>


    * [ ] What is LSTM - Long Short Term Memory?

        <https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/>

    * [ ] Gated Recurrent Unit Networks

        <https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/>

    * [ ] PyTorch Tutorial

        <https://www.geeksforgeeks.org/deep-learning/pytorch-learn-with-examples/>

    * [ ] Gradient Descent Algorithm in Machine Learning

        <https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/>

    * [ ] Momentum-based Gradient Optimizer - ML

        <https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/>

    * [ ] Adagrad Optimizer in Deep Learning

        <https://www.geeksforgeeks.org/machine-learning/intuition-behind-adagrad-optimizer/>

    * [ ] RMSProp Optimizer in Deep Learning

        <https://www.geeksforgeeks.org/deep-learning/rmsprop-optimizer-in-deep-learning/>

    * [ ] What is Adam Optimizer?

        <https://www.geeksforgeeks.org/deep-learning/adam-optimizer/>

    * [ ] Lung Cancer Detection using Convolutional Neural Network (CNN)

        <https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-convolutional-neural-network-cnn/>

    * [ ] Cat & Dog Classification using Convolutional Neural Network in Python

        <https://www.geeksforgeeks.org/deep-learning/cat-dog-classification-using-convolutional-neural-network-in-python/>

    * [ ] Sentiment Analysis with an Recurrent Neural Networks (RNN)

        <https://www.geeksforgeeks.org/python/sentiment-analysis-with-an-recurrent-neural-networks-rnn/>

    * [ ] Text Generation using Recurrent Long Short Term Memory Network

        <https://www.geeksforgeeks.org/machine-learning/text-generation-using-recurrent-long-short-term-memory-network/>

    * [ ] Deep Learning Interview Questions

        <https://www.geeksforgeeks.org/deep-learning/deep-learning-interview-questions/>

    * [ ] Machine Translation with Transformer in Python

        <https://www.geeksforgeeks.org/nlp/machine-translation-with-transformer-in-python/>

* [ ] `qemu-system-x86_64 -enable-kvm -device pci-bridge,id=mybridge -device e1000,bus=mybridge,addr=0x1`

* [ ] 调研 ds 写的添加虚拟 pci 设备的代码（未验证）

    ```c
    // 文件名: vpci_device.c
    #include <linux/module.h>
    #include <linux/pci.h>
    #include <linux/device.h>

    // 定义虚拟 PCI 设备的 Vendor ID 和 Device ID
    #define VENDOR_ID 0x1234
    #define DEVICE_ID 0x5678

    static struct pci_dev *vpdev = NULL;

    static int __init vpci_init(void) {
        struct pci_bus *bus;
        int ret = -ENODEV;

        // 获取第一个 PCI 总线 (例如: 0000:00)
        bus = pci_find_bus(0, 0); 
        if (!bus) {
            printk(KERN_ERR "无法找到 PCI 总线\n");
            return ret;
        }

        // 动态分配一个 pci_dev 结构体
        vpdev = pci_alloc_dev(bus);
        if (!vpdev) {
            printk(KERN_ERR "无法分配 PCI 设备\n");
            return -ENOMEM;
        }

        // 设置设备基本信息
        vpdev->vendor = VENDOR_ID;     // 厂商 ID
        vpdev->device = DEVICE_ID;     // 设备 ID
        vpdev->devfn = PCI_DEVFN(0, 0); // 设备号 (Bus 0, Device 0)
        vpdev->class = PCI_CLASS_NETWORK_ETHERNET; // 设备类别 (示例: 网络设备)

        // 初始化设备并添加到 PCI 子系统
        pci_bus_add_device(vpdev);
        ret = pci_add_device(vpdev);
        if (ret < 0) {
            printk(KERN_ERR "无法添加 PCI 设备\n");
            pci_free_dev(vpdev);
            return ret;
        }

        printk(KERN_INFO "虚拟 PCI 设备已创建: %04x:%04x\n", VENDOR_ID, DEVICE_ID);
        return 0;
    }

    static void __exit vpci_exit(void) {
        if (vpdev) {
            pci_stop_and_remove_bus_device(vpdev); // 从总线移除设备
            pci_free_dev(vpdev);                   // 释放设备内存
            printk(KERN_INFO "虚拟 PCI 设备已移除\n");
        }
    }

    module_init(vpci_init);
    module_exit(vpci_exit);
    MODULE_LICENSE("GPL");
    ```

    若需要设备支持完整的配置空间操作（如 BAR 映射、中断等），需扩展模块代码：

    ```c
    // 在 vpci_init 中补充资源配置
    vpdev->resource[0].start = 0x1000;  // BAR0 起始地址 (虚拟)
    vpdev->resource[0].end = 0x1FFF;    // BAR0 结束地址
    vpdev->resource[0].flags = IORESOURCE_MEM; // 内存资源类型

    // 启用设备
    pci_enable_device(vpdev);
    ```

* [ ] 调研`-kernel`使用 qemu 时，是否有 console 输出？如果加上`-append 'console=ttyS0'`是否会有 console 输出？

* [ ] `setitimer()`

* [ ] `timer_create()`

* [v] `signal()`

* [ ] `sigaction()`

* [ ] python `threading.Timer`, `sched`, `asyncio.sleep()`

* [ ] Node.js

* [ ] 调研`gcc -static`静态链接

* [ ] 调研`nm -D`

    `nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep strtol`

* [ ] 调研`objdump -T`

    `objdump -T /lib/x86_64-linux-gnu/libc.so.6 | grep strtol`

* [ ] `gcc --wrap`

* [ ] `gcc -ldl`

* [ ] `getconf`

* [ ] `source`

* [ ] 调研 Floyd 最短路径

* [ ] 调研图的遍历

* [ ] 调研通过矩阵幂计算路径数量

* [ ] 调研度矩阵

* [ ] 调研拉普拉斯矩阵

* [ ] 调研图神经网络

* [ ] 调研Dijkstra

* [ ] nvidia-smi

    <https://www.chenshaowen.com/blog/basic-usage-of-nvidia-smi.html>

    未处理完，目前看到了

    > 常用参数

* [ ] a100 spec

    <https://datacrunch.io/blog/nvidia-a100-gpu-specs-price-and-alternatives>

    未处理完，目前看到了

    > A100 Data Sheet Comparison vs V100 and H100

    不明白 TPCs 是什么意思。

* [o] process 1 url 10/01

    Resetting remote to a certain commit: <https://stackoverflow.com/questions/5816688/resetting-remote-to-a-certain-commit>

    feedback:

    4. 调研`ORIG_HEAD`, `git show ORIG_HEAD`

    5. 调研`git update-ref `

    6. 调研`git log --graph --all --oneline --decorate`

* [o] process 1 url 10.06

    <https://www.baeldung.com/linux/last-directory-file-from-file-path>

    feedback:

    2. 没处理完，有时间了接着处理

    3. 这篇博客的思维方式也很好，先处理简单的情况，再处理 corner case，下次学习一下

* [o] process 1 url  10.23

    <https://cloud.tencent.com/developer/article/1805119>

* [ ] 调研 bash array <https://www.gnu.org/software/bash/manual/html_node/Arrays.html>

* [o] process 1 url 10.09

    <https://linuxhint.com/trim_string_bash/>

    feedback:

    2. 虽然

        ```bash
        a="hello, world"
        b=${a%world}  # hello, 
        ```

        可以 trim `world`，但是如果是

        ```bash
        a="hello, worldworld"
        b=${a%%world}  # hello, world
        ```

        那么只能删减一个 world。
        
        该如何删减全部的两个 world？

* [P] cache tabs 06.04

    feedback:

    * <https://www.geeksforgeeks.org/binary-search/>

        目前看到 Recursive Binary Search Algorithm:

* [ ] 调研 boost 库的 lexical_cast

* [ ] 调研 frpc 自动重连 (Service Health Check)

    ```toml
    # frpc.toml

    [[proxies]]
    name = "test1"
    type = "tcp"
    localPort = 22
    remotePort = 6000
    # Enable TCP health check
    healthCheck.type = "tcp"
    # TCPing timeout seconds
    healthCheck.timeoutSeconds = 3
    # If health check failed 3 times in a row, the proxy will be removed from frps
    healthCheck.maxFailed = 3
    # A health check every 10 seconds
    healthCheck.intervalSeconds = 10
    ```

* [ ] grdctl

* [ ] wlroots, Sway

* [ ] rdesktop

* [ ] remmina

* [ ] freerdp2-wayland

* [ ] freerdp2-shadow-cli

* [ ] wlfreerdp

* [ ] 调研括号 `( ... )` 是子 shell，`{ ...; }` 是当前 shell 里的复合命令

    `nohup bash -c "{ sleep 5 && echo hello; }"`

* [ ] 调研 vscode 的"代码片段（Snippet）"

* [ ] dns 的`MX`, `NS`记录类型

* [ ] 既然 epoll 可以监控 fd，那么除了 socket fd 外，epoll 是否也可以监控普通文件的改动？

* [ ] `grub2-mkconfig`

* [ ] `/boot/grub/grub.cfg`

* [ ] SYSLINUX/ISOLINUX/PXELINUX 系列引导程序

    ISOLINUX (用于光盘): isolinux/isolinux.cfg

    SYSLINUX (用于 FAT 文件系统): syslinux/syslinux.cfg

    PXELINUX (用于网络启动): pxelinux.cfg/default

* [ ] 调研`OBJS = $(SRCS:src/%.c=obj/%.o)`处理字符串

* [ ] 调研`./configure`运行的是什么程序？如何配置？

* [asso] `truncate()`

* [asso] `std::ios::trunc`

* [asso] 什么是 tmpfs，和普通的文件系统在实现上有什么区别？

* [asso] 调研 SIMD 指令：SSE, AVX, NEON

* [asso] 调研不使用 swap 的 tmpfs

    挂载时指定 nr_blocks=0 和 nr_inodes=0 选项

    `sudo mount -t tmpfs -o size=2G,nr_blocks=0,nr_inodes=0 tmpfs /mnt/shm`

* [asso] 调研临时修改系统的 swappiness 参数

    ```bash
    # 查看当前值（通常为 60）
    cat /proc/sys/vm/swappiness

    # 设置为 0：内核会尽量避免使用交换空间。
    # 设置为 100：内核会积极使用交换空间。
    sudo sysctl vm.swappiness=0
    ```

* [asso] 调研永久修改系统的 swappiness 参数

    `/etc/sysctl.conf`:

    ```conf
    vm.swappiness=0
    ```

    `sudo sysctl -p`

* [asso] 调研`vm.overcommit_memory`

* [asso] 编译器特定的扩展（如 GCC/Clang 的 `__attribute__((aligned))` 或 `_aligned_malloc` on MSVC）。

* [asso] `ipcrm -m <shmid>`手动删除内存段。

* [asso] 调研 6.12.2 Extended Asm - Assembler Instructions with C Expression Operands

    <https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html>

* [asso] 调研`/etc/fstab`的格式

* [asso] 调研 bash 技巧

    ```bash
    set +o history  # 禁用历史记录
    # 敏感操作
    set -o history  # 重新启用
    ```

* [asso] `sendfile()`

* [asso] 调研 SG-DMA（Scatter-Gather DMA）实现零拷贝的 example

* [asso] Kafka

* [asso] Netty

* [asso] 调研 Perl

* [asso] 调研 python `regex`及其递归匹配

* [asso] 调研正则表达式`(?:...)`

* [asso] `if constexpr`

    这个似乎可以编译期让不同类型走不同的 if 分支

    ```cpp
    #include <type_traits>
    #include <iostream>

    class MyClass {
    public:
        template <typename T>
        MyClass(T value) {
            if constexpr (std::is_same_v<T, int>) {
                std::cout << "int constructor: " << value << std::endl;
            } else if constexpr (std::is_same_v<T, float>) {
                std::cout << "float constructor: " << value << std::endl;
            } else {
                std::cout << "generic constructor: " << value << std::endl;
            }
        }
    };

    int main() {
        MyClass a(10);      // int 版本
        MyClass b(3.14f);   // float 版本
        MyClass c("hello"); // 通用版本
    }#include <type_traits>
    #include <iostream>

    class MyClass {
    public:
        template <typename T>
        MyClass(T value) {
            if constexpr (std::is_same_v<T, int>) {
                std::cout << "int constructor: " << value << std::endl;
            } else if constexpr (std::is_same_v<T, float>) {
                std::cout << "float constructor: " << value << std::endl;
            } else {
                std::cout << "generic constructor: " << value << std::endl;
            }
        }
    };

    int main() {
        MyClass a(10);      // int 版本
        MyClass b(3.14f);   // float 版本
        MyClass c("hello"); // 通用版本
    }
    ```

* [asso] C++20 Concepts

* [asso] 调研`as`

    GNU Assembler (as)

    * 准备树莓派开发板，使用 ssh 进入系统

        或使用 qemu 模拟一个 arm 环境，安装 ubuntu 系统，使用 ssh 进入系统

    * 调研系统中是否有 as，如果没有，安装

    * 跑通第一个 arm 汇编的 hello world 程序

    * 使用汇编实现循环求和 1 + ... + 10

    * 使用汇编实现自定义函数的调用，要求有 2 个输入参数

    * 使用汇编实现斐波那契数列

* [asso] 调研 QEMU 的 VirGL 或 Wine 的 DXVK

* [asso] 调研串流技术 Parsec、Moonlight 等

* [asso] 调研 virt-manager ssh 远程连接主机

    ```bash
    virt-manager -c qemu+ssh://user@linux-host/system
    ```

* [asso] 调研 qemu 的 efi 启动方法

    `qemu-system-x86_64 -bios /usr/share/OVMF/OVMF_CODE.fd -hda disk.vdi`

* [asso] 调研 qemu -hda 显式声明格式

    `qemu-system-x86_64 -drive file=disk.vdi,format=vdi`

* [asso] 调研`make -d 2>&1 | grep -A5 -B5 "INCLUDE_DIRS"`

* [asso] 调研`g++ $(addprefix -I,$(subst :, ,$(INCLUDE_DIRS))) main.cpp -o main`

    重点是`addprefix`的用法。

* [asso] 调研`g++ $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) main.cpp -o main`

    重点是`foreach`的用法。

* [asso] 调研 makefile 中的 shell 变量

    ```makefile
    base_dir := /path/to/base_dir_1
    file_path = $(base_dir)/hello

    test:
    	@echo $(file_path)
    	@base_dir="/path/to/base_dir_2"; \
    	echo $${base_dir}/hello
    ```

    调研其中的`@base_dir="/path/to/base_dir_2"; \`的写法。

* [asso] 调研 makefile 中其他修改变量的方法

    感觉这两种方法比较冷门，有空了再看吧。

    * 使用递归Make调用

        ```makefile
        base_dir := /path/to/base_dir_1
        file_path = $(base_dir)/hello

        test:
        	@echo $(file_path)
        	@$(MAKE) test-inner base_dir=/path/to/base_dir_2

        test-inner:
        	@echo $(file_path)
        ```

    * 使用条件赋值

        ```makefile
        base_dir ?= /path/to/base_dir_1
        file_path = $(base_dir)/hello

        test:
        	@echo $(file_path)
        	@base_dir=/path/to/base_dir_2 $(MAKE) test-inner

        test-inner:
        	@echo $(file_path)
        ```

* [asso] Huber Loss 的表达式是如何构建出来的？

* [asso] 使用 stat 命令查看创建时间

    `stat -c "%n %W" * | sort -k2 -n`

    使用 stat 查看访问时间并手动排序:

    `stat -c "%n %x" * | sort -k2 -r`

* [asso] 使用 find 按创建时间排序

    `find . -maxdepth 1 -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f2-`

    （注：find 默认不支持 birth 时间，但可以按 mtime 排序）

* [asso] `if constexpr`

* [asso] 调研命令后写`&`的写法

    ```bash
    while sleep 1; do
        echo "后台任务持续运行..."
    done &
    ```

* [asso] 使用 nc 的 -X 代理选项 + 跳板机SOCKS 代理 SSH 代理转发，进行跳板机中转连接到 target

    `ssh -o ProxyCommand="ssh -q -X jumpuser@bastion.example.com nc -x localhost:1080 %h %p" targetuser@target.internal.com`

* [asso] 调用 Squid、CCProxy 等 http proxy server

* [asso] corkscrew

* [asso] 使用 corkscrew 进行 ssh 代理

    ```bash
    ssh -o ProxyCommand="corkscrew <代理服务器IP> <代理服务器端口> <目标主机IP> <目标主机SSH端口>" <用户名>@<目标主机IP>
    ```

    参数解释：

        corkscrew: 核心工具。

        <代理服务器IP> 和 <代理服务器端口>：你的 HTTP 代理服务器的地址和端口（例如 proxy.company.com:8080）。

        <目标主机IP> 和 <目标主机SSH端口>：你最终想要连接的 SSH 服务器的地址和端口（通常是 22）。

    example:

    `ssh -o ProxyCommand="corkscrew proxy.company.com 3128 server.example.com 22" myuser@server.example.com`

    带有用户名和密码的版本：

    `ssh -o ProxyCommand="corkscrew proxy.company.com 3128 server.example.com 22 ~/.ssh/proxy_auth" myuser@server.example.com`

    保存用户名和密码：

    ```bash
    echo "proxy_username:proxy_password" > ~/.ssh/proxy_auth
    chmod 600 ~/.ssh/proxy_auth # 非常重要！确保只有你能读这个文件
    ```

* [asso] 调研 Winget

* [asso] 调研 Chocolatey

* [asso] GitHub Desktop

* [asso] SourceTree

* [asso] `:s/foo/bar<lt>CR>/g`中的`<lt>`是什么意思？

* [asso] `:s/foo/bar\r/e`中`/e`是什么意思？

## matplotlib

### tasks

* [ ] `plt.Circle()`, `ax.add_patch()`

* [ ] `plt.Line2D()`, `ax.add_line()`

* [ ] `ax.text()`

* [new] `ax.set_xlim()`, `ax.set_ylim()`, `ax.set_aspect()`, `ax.axis()`

* [new] `fig.patch.set_facecolor()`, `plt.figtext()`, `plt.figtext()`

* [new] `ax.text()`, `ax.annotate()`

## fnirs

### tasks

* [v] MNE-NIRS

* [new] pyserial, struct

* [ ] NIRS-Toolbox

* [new] OpenBCI GUI、LSL Lab Streaming Layer

* [asso] `print(dir(mne_nirs.preprocessing))`

* [asso] 查看所有可用的方法（过滤掉内置方法）

    ```py
    methods = [method for method in dir(raw) if not method.startswith('_')]
    print("可用的方法：", methods)
    ```

* [v] matlab `cellfun`

    ```matlab
    % 或者使用 cellfun
    file_cell = {'file1.txt', 'file2.csv'};
    names = cellfun(@(x) fileparts(x), file_cell, 'UniformOutput', false);
    ```

## Torch

系统地学一遍 pytorch.

resources:

1. Welcome to PyTorch Tutorials

    <https://pytorch.org/tutorials/>

    主要看 learn the basics 和 learning pytorch with examples

2. PyTorch documentation

    <https://pytorch.org/docs/stable/index.html>

    可以看下下面的 Developer Notes 主题，重点看一看模型压缩，混合精度以及并行训练／推理

### cache

* 一个训练 cifar-10 的 example

    见`ref_40`。

* ImageNet, CIFAR, and COCO

* [ ] 使用 dataset 和 dataloader 在 cpu 环境下拟合 sine 曲线

* [ ] 调研 v100 部署 pytorch 的小模型（CV or NLP）

* [ ] 调研 llama 部署

* [ ] 调研 llama 在 cpu 上的部署

### tasks

* [O] torchmetric中micro precision计算详解

    * 目前看到

        > 那么在 torchmetric 中，在多分类任务中，如果我想获得一个指定类别的 precision / recall / f1 该怎么办？

* [ ] 关键阈值：秩（Rank）

    矩阵的秩决定了信息保留的程度：

    ```py
    import torch
    import numpy as np

    def analyze_layer(W):
        """分析权重矩阵的信息传递能力"""
        rank = torch.linalg.matrix_rank(W).item()
        full_rank = min(W.shape)
        rank_ratio = rank / full_rank
        
        # 奇异值分解（SVD）
        U, S, V = torch.svd(W)
        condition_number = S[0] / S[-1] if S[-1] > 0 else float('inf')
        
        print(f"权重形状: {W.shape}")
        print(f"矩阵秩: {rank}/{full_rank} ({rank_ratio:.1%})")
        print(f"条件数: {condition_number:.2e}")
        print(f"信息保留能力: {'高' if rank_ratio > 0.8 else '中' if rank_ratio > 0.3 else '低'}")
        
        # 测试不同输入的信息保留
        y_random = torch.randn(100, W.shape[1])
        z = y_random @ W.T
        input_var = y_random.var().item()
        output_var = z.var().item()
        print(f"输入方差: {input_var:.3f}")
        print(f"输出方差: {output_var:.3f}")
        print(f"方差保留率: {output_var/input_var:.1%}")
        print("-" * 50)

    # 不同情况的对比
    print("=== 不同权重矩阵的信息传递能力 ===")

    # 1. 全零矩阵（最坏情况）
    W_zero = torch.zeros(10, 5)
    analyze_layer(W_zero)

    # 2. 低秩矩阵（部分信息丢失）
    W_low_rank = torch.randn(10, 5)
    W_low_rank[:, 2:] = 0  # 只保留前2列有效
    analyze_layer(W_low_rank)

    # 3. 满秩随机矩阵（正常情况）
    W_full_rank = torch.randn(10, 5)
    analyze_layer(W_full_rank)

    # 4. 单位矩阵（理想情况）
    W_identity = torch.eye(5, 5)
    analyze_layer(W_identity)
    ```

    输出结果会显示：

    * 全零矩阵：秩=0，方差保留率=0%

    * 低秩矩阵：秩<满秩，部分信息丢失

    * 满秩矩阵：秩=满秩，信息基本保留

* [ ] `FashionMNIST`, `KMNIST`, `EMNIST`, `QMNIST`

* [ ] 在 py 中构建与稀疏矩阵等价的稠密矩阵，比较稀疏乘法和稠密乘法的时间

* [ ] py 可能有内部优化，使用手写的纯 c / c++ 版本再比较一版

* [O] 稀疏矩阵如何加速矩阵乘？

* [O] PyTorch Functional Transforms for Computer Vision

    16:48 ~ 17:20

    <https://www.geeksforgeeks.org/computer-vision/pytorch-functional-transforms-for-computer-vision/>

    feedback:

    * 目前看到

        > 2. adjust_contrast() function:floating-point

* [ ] 带动量的SGD（Momentum）

* [ ] 带动量和权重衰减的SGD

* [ ] 调研 Reshaping a Tensor in Pytorch

    <https://www.geeksforgeeks.org/python/reshaping-a-tensor-in-pytorch/>

* [ ] 调研 pytorch-tutorial

    <https://github.com/yunjey/pytorch-tutorial?tab=readme-ov-file>

* [ ] 调研 Learn the Basics

    <https://docs.pytorch.org/tutorials/beginner/basics/intro.html>

* [ ] gpu 如何对 transpose 加速？或者说，高性能 transpose 如何实现？

* [ ] 调研 SG-DMA

* [ ] 调研散列表（Scatterlist）

    ```c
    struct scatterlist sg;
    sg_init_one(&sg, user_buf, size); // 初始化散列表项
    dma_map_sg(dev, &sg, 1, DMA_TO_DEVICE); // 映射物理地址
    ```

* [ ] 调研`from typing import Optional, List, Dict, Tuple, Set`

* [v] `np.loadtxt()`

* [ ] `nn.ReLU(True)`

    true 是什么意思？

* [ ] `nn.Unflatten()`

* [ ] `nn.ConvTranspose2d()`

* [ ] `nn.BatchNorm2d()`

* [ ] `sg_dma_address()`, `sg_dma_len()`

* [ ] `mlock()`

* [ ] `get_user_pages()`

* [ ] 调研自旋锁、互斥锁的汇编实现

    `LDREX`和`STREX`（独占加载/存储）

* [ ] 调研内存屏障指令

    * DMB (Data Memory Barrier)：确保屏障前的内存操作在后续操作之前完成。

    * DSB (Data Synchronization Barrier)：比 DMB 更严格，等待所有内存操作完成。

    * ISB (Instruction Synchronization Barrier)：清空流水线，确保新指令的执行。

* [asso] 调研`mypy`

* [asso] `DataLoader`是如何实现 shuffle 的？先按照 dataset 的 length 生成 range，然后 random permute 吗？

* [asso] python 的实例既然可以定义成员变量，那么可以定义成员函数吗？

* [asso] Albumentations

* [asso] 调研这个网站下面的 tutorial 目录，看看其他部分

    <https://www.geeksforgeeks.org/python/datasets-and-dataloaders-in-pytorch/>

* [asso] Python Introduction

    <https://www.geeksforgeeks.org/python/introduction-to-python/>

    可以看看下面的 explore 部分。

* [asso] 调研 python 中 list 的方法`.sort()`，如果 list 中存储的对象是 datetime 对象或者其他类型的元素，sort() 方法会如何处理？如果是混合类型的元素，又会如何处理？

## Machine Learning

### cache

* pytorch 的 torchtext 已经在 24 年停止维护了，不要再用了

### tasks

* [ ] 《利用Python进行数据分析》（Python for Data Analysis）

    作者: Wes McKinney（Pandas 的创始人）

    内容: 从 Pandas 基础到实战，是公认的入门圣经。

* [ ] polars

* [O] Apply a 2D Convolution Operation in PyTorch

    23:51 ~ 00:09

    <https://www.geeksforgeeks.org/computer-vision/apply-a-2d-convolution-operation-in-pytorch/>

    目前看到

    > 2D convolution in PyTorch 

* [ ] 1D卷积处理时间序列方法详解

    目前看到

    > 从不同角度的推导

* [O] Apply a 2D Max Pooling in PyTorch

    <https://www.geeksforgeeks.org/computer-vision/apply-a-2d-max-pooling-in-pytorch/>

    目前看到

    > 2d Max pooling

* [O] Difference Between "Hidden" and "Output" in PyTorch LSTM

    <https://www.geeksforgeeks.org/deep-learning/difference-between-hidden-and-output-in-pytorch-lstm/>

* [O] Generative Adversarial Networks (GANs) in PyTorch

    17:05 ~ 19:56

    <https://www.geeksforgeeks.org/deep-learning/generative-adversarial-networks-gans-in-pytorch/>

    feedback:

    * [ ] 调研：根据反卷积性质，如果我们正向卷积一个矩阵，可以构造出一个 W，那么如何使用这个 W 的转置 W^T 还原输出 X？

* [ ] Implementing an Autoencoder in PyTorch

    <https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/>

* [ ] Transfer Learning with Fine-Tuning in NLP

    <https://www.geeksforgeeks.org/nlp/transfer-learning-and-fine-tuning-in-nlp/>

* [ ] Transfer Learning for Computer Vision

    <https://www.geeksforgeeks.org/computer-vision/transfer-learning-for-computer-vision/>

* [ ] How to implement transfer learning in PyTorch?

    <https://www.geeksforgeeks.org/deep-learning/how-to-implement-transfer-learning-in-pytorch/>

* [O] 调研 matplotlib 画 surface

    ~ 18:03

* [ ] 调研 matplotlib 基本和常见的 example

* [ ] 调研论文《Finding Structure in Time》 (1990) by Jeffrey L. Elman

    <https://crl.ucsd.edu/~elman/Papers/fsit.pdf>

* [ ] LSTM（长短期记忆网络）

* [ ] GRU（门控循环单元）

* [ ] 调研 词嵌入（Word Embeddings）

* [v] torch tensor `.float()`

* [ ] 调研 LSTM

    example 代码，尝试跑通

* [ ] 调研 AG News 新闻分类数据集

    包含超过100万篇新闻文章，任务是将新闻分类到四个顶级类别：世界、体育、商业、科技/科学。

    ```py
    from datasets import load_dataset
    dataset = load_dataset('ag_news')
    train_data = dataset['train']
    ```

    res: <https://arxiv.org/abs/1509.01626> （论文中有下载链接）

* [ ] SimpleBooks / TinyShakespeare 数据集

    SimpleBooks：包含来自古登堡计划的纯文本书籍。

    TinyShakespeare：包含莎士比亚的所有作品，约1MB的文本。

* [ ] Cornell Movie Dialogs Corpus (电影对白语料库)

    包含大量电影角色之间的对话，非常适合训练一个简单的聊天机器人或对话生成模型。

    res: <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>

* [ ] 调研 GRU

    example 代码，尝试跑通

* [ ] 调研 BERT

    example 代码，尝试跑通

* [ ] 调研 GPT

    example 代码，尝试跑通

* [ ] 调研 NLP From Scratch: Classifying Names with a Character-Level RNN

    <https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>

* [ ] 调研论文 Seq2Seq

    Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In NeurIPS. 

* [ ] 调研掩码自注意力（Masked Self-Attention）

* [ ] 调研 LLaMA， BERT, T5, BART

* [ ] Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)

    GPT-1

* [v] 调研 decoder-only 架构

    feedback:

    1. 调研论文

        * Language Models are Few-Shot Learners (GPT-3, Brown et al., 2020)

        * LLaMA: Open and Efficient Foundation Language Models (Meta, 2023)

        * Attention Is All You Need (Vaswani et al., 2017)

            提出 Encoder-Decoder 架构，但解码器使用掩码自注意力实现自回归生成。

        * BART: Denoising Sequence-to-Sequence Pre-training (Lewis et al., 2019)

            结合双向编码器 + 自回归解码器的预训练框架。

        * T5: Text-to-Text Transfer Transformer* (Raffel et al., 2020)

            统一任务为文本生成，采用 Encoder-Decoder 自回归架构。

        * Encoder-Agnostic Adaptation for Conditional Language Generation (Edunov et al., 2020)

            分析 Encoder-Decoder 与 Decoder-Only 在生成任务中的差异。

        * The Trade-offs of Large Scale Language Models (Bender et al., 2021)

            讨论不同架构的计算效率与生成质量权衡。

    1. 调研掩码自注意力相关论文

        * Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)

            提出相对位置编码，改进 Decoder 的长期依赖建模。

        * Generating Long Sequences with Sparse Transformers (Child et al., 2019)

            讨论掩码注意力的稀疏化扩展。

    1. 调研数学基础

        自回归生成的数学基础：

        * Neural Autoregressive Distribution Estimation (Larochelle & Murray, 2011)

            早期关于自回归概率建模的理论。

        * Decoder-Only 的扩展应用：

            Zero-Shot Text-to-Image Generation (DALL·E, Ramesh et al., 2021)

                将 Decoder-Only 架构用于跨模态生成。

* [ ] 调研 VAE with a VampPrior, Jakub M. Tomczak, Max Welling

    <https://arxiv.org/abs/1705.07120>

* [ ] 调研 Towards Causal Representation Learning, Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, Yoshua Bengio

    <https://arxiv.org/abs/2102.11107>

* [ ] 调研 Generative Adversarial Networks

* [v] 调研 yoshua bengio 和他的贝叶斯推理与神经网络

    feedback:

    1. 调研 Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference

    1. 调研下面这句话展开的含义
    
        > 传统机器学习（如深度学习）擅长拟合数据相关性（P(Y|X)），但无法回答干预性（P(Y|do(X))）或反事实（What if?）问题。

    1. 调研下面几个基本因果算法
    
        * 结构因果模型（SCM）（Pearl, 2009）

        * 潜在结果框架（Rubin Causal Model）

        * 因果发现算法（如PC算法、神经因果模型）

    1. 调研贝叶斯派的常用算法

        变分推断（VI）（如VAE）

        马尔可夫链蒙特卡洛（MCMC）

        概率图模型（PGM）

    1. 调研因果推断的未来几个方向

        神经因果模型（如Neural Causal Models）

        因果表征学习（Bengio团队重点方向）

        因果强化学习（如DeepMind的“因果RL”）

        基于梯度的因果结构学习（如DAG-GNN）

        因果发现+自监督学习（如ICRL, 2023）

        鲁棒因果估计（如Double Machine Learning）

        因果迁移学习（跨领域因果泛化）

        扩散模型+贝叶斯优化（如Diffusion Bayesian Networks）

        量子计算加速（如量子MCMC）

        稀疏贝叶斯学习（如Bayesian Pruning）

        概率Transformer（如Bayesian Attention）

        Judea Pearl 的 Do-Calculus

        Bengio 的 因果生成模型

        主动因果学习（Active Causal Learning）

        多模态数据融合（如结合文本、图像、时间序列）

        近似推断的优化（如Normalizing Flows）

        硬件加速（TPU/GPU定制化计算）

        AutoML + 因果（如AutoCausal）

        因果AI的可视化工具（如DoWhy库的推广）

    1. 调研标准化工具链（如Pyro、TensorFlow Probability的普及）

* { } 调研《Python机器学习》

    feedback:

    1. 目前看到 pdf P54

    2. 不清楚为什么`self.w_ = np.zeros(1 + X.shape[1])`要`1 +`。

* [ ] 调研 三维的 Swiss Roll

* [ ] 调研论文 The Perceptron,a Perceiving and Recognizing Automaton

* [ ] 调研论文 A Logical Calculus of the Ideas Immanent in Nervous Activity

* [ ] 调研 numpy ndarray 的`.item()` method

* [ ] 调研 Zico Kolte Linear Algebra Review and Referen

* [ ] 调研牛顿法求导法迭代找一个函数的极值

    可能和数值计算方法和凸优化有关

    feedback:

    1. 凸优化似乎主要讲的是线性规划，多目标优化，梯度下降之类的

    2. 考虑一个最简单的情况：寻找$y = x^2$的最小值，初始位置为$x = -1.5$

    3. 不需要考虑系统地学一门课程，可以先考虑从零散的方法学起来，再慢慢整理成系统的知识

        因为书籍本来也不是直接成型的，而是一篇论文一篇论文积累起来的。

    4. 调研 Stephen Boyd - 《Convex Optimization》

* [ ] 调研证明书上给出的优化方法和求导法本质上相同

* [v] 调研《Python机器学习》 11.10

* [v] 调研 <https://pytorch.org/tutorials/>

    feedback:

    1. 目前看到<https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>

    2. `main.py`未处理完，临时记录放在`temp.md`文件中

        理论上可以自己写一个 mnist 数据集的训练、测试代码了。

* [ ] 调研 matplotlib

* [ ] 构建数据集以及数据集的代码

    先从 sin, cos 入手，再做鸢尾花，葡萄洒等传统机器学习数据集，再做 mnist, minist-fashion，再然后是自然语言数据集，最后是综合数据集，比如视频，imagenet 等。

* [asso] 调研`plt.rcParams`的基本用法

    如何查看当前安装了哪些字体？

* [asso] MATLAB: sparse() 函数

* [asso] `cmap='viridis', marker='o'`

* [asso] eigen 如何创建稀疏矩阵（静态大量，动态添加）？

* [asso] eigen 中稀疏矩阵与稠密矩阵如何转换？

* [asso] eigen 如何打印出一个稀疏矩阵中的非零值？如何打印完整矩阵？如何读取或修改指定位置的值？

* [asso] eigen 支持的矩阵最高到几维？是否支持类似 torch 的 4 维矩阵？

* [asso] eigen 中的稀疏矩阵支持哪些运算？矩阵乘法？还有其他什么运算？

* [asso] `plt.grid()`画格子时，目前是长方形，如何使 grid 变成正方形？

## Mathematics

resources:

1. 《高等数学》同济大学应用数学系

### cache

* ML-Prerequests: 机器学习的预备知识（矩阵论，概率论，凸优化等）

    <https://github.com/robinluodh/ADMM_learning_NJU_HBS>

* ADMM_learning_NJU_HBS: 凸优化、变分法、ADMM 资料学习。来自南大何炳生教授主页。

    <https://github.com/robinluodh/ADMM_learning_NJU_HBS>

### tasks

* [ ] 多元函数的连续性

* [ ] 调研使用梯度法求二无函数$f(x, y)$的最值

* [ ] 矩阵微积分 / 矩阵求导

* [ ] 复分析 / 复变函数论

* [asso] Wirtinger 微积分 / 复变函数求导

* [asso] 论文: 《The Complex Gradient Operator and the CR-Calculus》 by Kreutz-Delgado. 这是该领域的经典入门文献。

* [asso] 《Matrix Algebra Useful for Statistics》 by Searle and Khuri. 有章节涉及矩阵求导。

* [asso] 《Matrix Differential Calculus with Applications in Statistics and Econometrics》 by Magnus and Neudecker. 这是矩阵微积分的权威著作，虽然主要针对实数，但其思想可以扩展到复数。

* [asso] JAX

    ```py
    import jax.numpy as jnp
    from jax import grad

    # 定义一个实值损失函数，输入是复数矩阵
    def loss(Z):
        return jnp.real(jnp.trace(Z.conj().T @ Z))  # ||Z||_F^2

    # 计算梯度
    Z = jnp.array([[1+2j, 3j], [4-1j, 5+0j]])
    gradient = grad(loss)(Z)
    print("梯度:\n", gradient)
    # 理论上，loss(Z)关于Z的梯度应该是 2 * Z_conj，但JAX等框架会处理好定义。
    ```

* [asso] SymPy

    ```py
    from sympy import symbols, I, conjugate, diff, Matrix

    # 定义符号
    z11_r, z11_i, z21_r, z21_i = symbols('z11_r z11_i z21_r z21_i', real=True)
    z11 = z11_r + I*z11_i
    z21 = z21_r + I*z21_i
    Z = Matrix([[z11], [z21]])

    # 定义一个函数，例如 f = |z11|^2 + |z21|^2 = z11*conjugate(z11) + ...
    f = conjugate(z11)*z11 + conjugate(z21)*z21

    # 对实部求导（等价于一种处理方式）
    diff(f, z11_r)
    ```

* [asso] Amir Beck 的 Beck & Teboulle 算法

    著名的 FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）

* [asso] Amir Beck 的《First-Order Methods in Optimization》（2017）

    系统介绍一阶优化方法的权威教材, 涵盖梯度方法、次梯度方法、近端算法等

## Linux Driver

### cache

* 一个带有 dev file, pci driver init 过程的 edu driver

    见`ref_41`。

* [ ] 调研 pcie 的中断是否不需要修改中断向量表，这个中断号是否由操作系统提供？

* 调研`INIT_WORK`, `DECLARE_WORK`, `struct work_struct task;`, `queue_work`

    * `kthread_should_stop`
    
    * `msleep`
    
    * `likely`, `unlikely`
    
    * `orderly_poweroff`

    * `iommu_domain_alloc`, `iommu_group_get`, `iommu_attach_group`

    * `dev_to_node`, `kzalloc_node`

    * `idr_init_base`

    * `spin_lock_init`, `spin_unlock_irqrestore()`

    * `mdev_register_device()`

* 调研一下`KBUILD_MODNAME`的含义。

resources:

* Linux Kernel Development, 3rd Edition

### tasks

* [asso] `devm_kobject_create_and_add()`

* [asso] `devm_device_add_groups()`

* [ ] 设备如何通过CPU控制的PIO（编程I/O）方式来访问内存（尽管可能是低效的）？

* [ ] 调研标准亲和性 (smp_affinity)

* [v] `device_unregister()`

* [v] `put_device()`

* [ ] `device_initialize()`

* [ ] `-serial mon:pty`

* [ ] `minicom`

* [ ] `-serial mon:/path/to/file`

    `-serial file:<filename>`

* [ ] `-serial mon:tcp:0.0.0.0:2345,server,nowait`

    `-serial udp:<host>:<port>[,<localaddr>:<localport>]`

* [ ] `telnet <host-ip> 2345`

* [ ] `-serial mon:unix:/path/to/socketfile,server,nowait`

    `-serial unix:<path>[,server|,client][,nowait]`

* [ ] `-serial mon:null`

* [ ] `-append 'console=ttyS0'  # 告诉内核使用第一个串口作为控制台`

* [ ] `-serial vc[:WxH]`

* [ ] `-serial pipe:<basename>`

* [ ] `-serial null`

* [ ] `-serial chardev:<id>`

* [ ] 调研 makefile 中 target 的执行机制

* [ ] 调研 USB 协议

* [ ] 调研 I2C 协议

* [ ] 调研 SPI 协议

* [ ] `device_register()`

* [ ] `device_attach()`

* [ ] `linux/list_lru.h`, `linux/list_sort.h`

* [ ] `list_add_rcu()`

* [ ] `list_lru_add()`

* [ ] `kmalloc_array()`, `kmalloc_caches()`

* [ ] `devm_kmalloc()`, `devm_kzalloc()`

* [ ] `class_create_file()`

* [ ] `class_device_destructor()`

* [ ] `class_dev_iter`

* [ ] 如果写成`module_param_array(m_arr, int, NULL, 0766);`，那么无法通过静态检查，从而通不过编译，为什么？

    `0766`不可以，`0755`可以。

* [ ] `unregister_module_notifier()`

* [ ] `Big Kernel Lock (BKL)`

* [ ] `module_param_array(m_arr, int, NULL, 0755);`, `755`报 warning

    ```
    [ 4358.400458] Attribute m_arr: Invalid permissions 0755
    ```

    为什么？

* [ ] `obj-m += hello.o`是什么含义？字符串`obj-m`添加空格后再添加`hello.o`？

* [ ] `module_param_call()`

* [ ] `module_param_named()`

* [ ] `module_param_string()`

* [ ] 调研`request_threaded_irq()`

* [ ] `vm_area_struct()`

* [ ] 调研 PTE（Page Table entry）, 进程的页表

* [ ] 后备存储（Backing Store）

* [ ] 什么是虚拟页？和物理页有什么不同？

* [ ] 调研`fdatasync()`

* [ ] `setvbuf()`, `setbuf()`

* [ ] `fflush()`是否基本等价于调用系统调用`write()`？

* [ ] 页面缓存（Page Cache）

* [ ] radix tree

* [ ] `dup2()`

* [ ] Btrfs、ZFS的COW机制

* [ ] 页帧分配、页表管理、换入换出（Swapping）

* [ ] 调研 thread pool

    ```cpp
    // 使用第三方线程池库（如 BS::thread_pool）
    #include "BS_thread_pool.hpp"

    BS::thread_pool pool;
    auto future = pool.submit(task); // 明确使用线程池

    // 或者使用 C++17 的并行算法
    #include <execution>
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::for_each(std::execution::par, data.begin(), data.end(), process);
    ```

* [ ] 调研`exec()`

* [ ] 调研`dup()`

* [ ] 调研`RLIMIT_DATA`

* [ ] 调研`ACPI`

* [ ] 调研IO队列是什么（可能和NVMe控制器相关）

* [ ] 调研`create_workqueue`

* [ ] 调研`container_of()`或`list_entry()`的实现，尝试手动实现一下

* [ ] 调研 linux 中的完成量（completion）

* [ ] `LIST_POISON1`, `LIST_POISON2`

* [ ] 调研如何多线程读写同一个链表，比如一个线程在循环遍历，另一个在随机添加/删除节点。

* [ ] 调研 rcu 链表

* [ ] 调研`list_lru.h`, `struct list_lru`, `list_lru_del()`

* [ ] `spin_trylock()`

* [ ] `spin_lock_irq()`, `spin_unlock_irq()`

* [ ] `spin_lock_irqsave()`, `spin_unlock_irqrestore()`

* [ ] 给出一个造成 spin lock 死锁的代码

* [ ] `spin_lock_bh()`, `spin_unlock_bh()`

* [ ] `struct task_struct`

* [ ] `schedule()`

* [ ] `spin_lock_irqsave()`, `spin_unlock_irqrestore()`

* [ ] 调研无锁单向链表`llist`

* [ ] `WRITE_ONCE()`

* [ ] 调研无锁（lock-free）操作

* [ ] `kfree_rcu`

* [ ] `list_move()`

* [ ] `list_cut_position()`

* [ ] 调研链表移动元素：list_move(), list_move_tail()

* [ ] 调研旋转链表：list_rotate_left()

* [ ] 调研分割链表：list_cut_position()

* [ ] 调研 为什么侵入式链表（数据包含链表节点而非相反）可以避免内存分配和指针间接寻址的开销？

* [ ] 调研`DEFINE_SPINLOCK()`

* [ ] 调研高级可编程中断控制器（APIC）, IO-APIC

* [ ] `devm_request_irq()`

* [ ] `platform_get_irq()`

* [ ] 调研`cat /proc/interrupts`的输出里，`2-edge`，`9-fasteoi`这些代表什么意思

* [ ] 调研中断描述符表（IDT）

* [ ] 调研`/dev/input/eventX`, 输入子系统接口在内核中注册一个事件处理器

* [ ] 中断流处理程序（flow handler）

* [ ] `pci_get_device()`

* [ ] `pci_get_domain_bus_and_slot()`

* [ ] `dev->msix_entries`

* [ ] `pci_dev->irq`

* [ ] `disable_irq()`, `enable_irq()`

* [ ] `pci_dev_msi_enabled()`

* [ ] 调研：如果`request_irq()`中，`dev_id`填`NULL`会发生什么？

* [ ] `free_irq()`为什么需要传入 dev_id？其返回值`void*`又是什么含义？

* [ ] 调研`pci_msix_vec_count`

* [ ] Root Port, Switch, Endpoint

* [ ] `pci_info()`

* [ ] `pci_enable_msix()`

* [ ] `/var/log/messages`

* [ ] `/var/log/syslog`

* [ ] `journalctl`

* [ ] 买 fpga 学习 pcie 设备及驱动

    deps:

    1. [ ] 学习 fpga 与基本 verilog 开发

* [ ] 调研 AXI4-Stream

* [ ] `__iomem`

* [ ] `request_mem_region()`

* [ ] `ioremap()`

* [ ] `release_mem_region()`

* [ ] `pci_resource_flags()`

* [ ] 调研 内核的虚拟地址是如何构成的？

* [ ] 调研 inb(), outb(), inl(), outl()

* [O] 调研在 kmd 上使用 mmio

    feedback:

    1. 调研平台设备（Platform Device）

    1. 还是要从嵌入式开发板看起。

        如果直接上 pc，那么比较简单的是 pci转串口（16550 uart）的驱动，网卡驱动，SATA控制器（块设备驱动）

        比较复杂的是 fpga pcie 开发板

        那么还不如先看看 arm 开发板的常见驱动写法，再转到更复杂的 pcie。

* [ ] 调研《Linux Device Drivers》，《PCI Express System Architecture》

* [ ] 调研 linux 的`drivers/pci/`目录

* [ ] 调研 linux `drivers/misc/`可能有简单PCI驱动示例

* [ ] 调研`setpci`命令

* [ ] 调研`minicom`命令

* [ ] 调研`picocom`工具

* [ ] 调研`screen`命令

* [ ] 调研驱动程序的`.remove()`和`.shutdown()`函数

* [ ] 调研 pci_request_region 时，操作系统（内核）负责分配这些地址范围，并维护一个全局的“资源树”来记录哪些地址区域已经被哪些设备占用，其中的资源树指的是什么？

* [ ] `dma_set_mask_and_coherent()`

* [ ] `pci_iomap()`

* [ ] `pci_set_master()`

* [O] 调研 qemu edu driver

    尝试跑通 example

    example: <https://github.com/kokol16/EDU-driver/tree/main>

    doc: <https://www.qemu.org/docs/master/specs/edu.html>

    res: <https://jklincn.com/posts/qemu-edu-driver/>

    feedback:

    1. 执行后输出为

        ```
        Factorial: 0
        Stored d027828c : Hello World
        Loaded 8e8fa6b0 : 
        Stored d0278288 : Wat
        Loaded 8e8fa6b0 :
        ```

        按道理`Factorial`应该是`8! = 40320`才对。

    1. 使用`sudo mknod /dev/edu c 241 0`创建`/dev/edu`后，使用`sudo ./test`可以得到输出：

        ```
        Factorial: 40320
        Stored 4e2d733c : Hello World
        Loaded 4d0416b0 : Hello World

        ```

        但是到这里整个 qemu 会卡住。目前不清楚原因。

    1. `depmod`

    1. `udev`, uevent, systemd-udevd

    1. vscode 里`pci_register_driver()`会报 warning：

        > unrecognized tokenC/C++(7)

        在`pci.h`里，`pci_create_slot()`也会报 warning:

        > declaration is incompatible with "struct pci_slot *pci_create_slot(struct pci_bus *parent, int slot_nr, const char *name, struct hotplug_slot *hotplug)" (declared at line 1129)C/C++(147)

        上面两个仅报 warning，但是不影响编译。

* [ ] 调研在`MKDEV()`前，哪些设备号是已经被占用的？

* [ ] 调研 I2C 驱动

* [ ] 还是先把 linux driver 开发看完比较好

    先看 qa，再看网站

* [ ] 调研为什么 ioctl 的 cmd 要靠`#define WR_VALUE _IOW('a','a',int32_t*)`这个构造

* [ ] 调研 sysfs 读写，sync

* [ ] 调研`mutex_lock`, `mutex_unlock`, `mutex_destroy`

* [ ] `kzalloc_node()`

* [ ] `kfree_rcu()`

* [ ] `kfree_bulk()`

* [ ] `kfree_const()`

* [ ] `kfree_sensitive()`

* [ ] 调研`select`的用法

* [ ] socket 调研：为什么`accept()`的第三个参数是一个长度指针，它有什么用？

* [ ] 调研：实现一个仅使用 read device 触发的中断程序

* [ ] sync socket programming

* [ ] `pci_iomap_range()`

* [ ] `pci_iomap_wc()`, `pci_iomap_wc_range()`

* [ ] 调研什么是可预取（Prefetchable）？

* [ ] `getdents`

* [ ] 调研`phys_addr_t`

* [ ] 调研流式（Streaming）DMA

    用于大数据块的单向传输。CPU或设备一方完成访问后另一方再访问，需要软件手动处理缓存同步（dma_sync_*函数）。

* [ ] `virt_to_phys()`

* [ ] `PAGE_OFFSET()`

* [ ] `get_free_pages()`

* [ ] `vmalloc_to_page()`

* [ ] `dma_map_sg()`

* [ ] `get_user_pages()`

* [ ] `flush_cache_all()`

* [ ] `invalidate_cache_all()`

* [ ] 调研 IRQ 2 与 irq 1 的级联中断

* [ ] 调研 irq p 系统定时器与 irq 8 实时时钟有什么区别？

* [asso] 调研 silent data corruption（静默数据损坏）

* [asso] 调研换出（paged out）

* [asso] 调研 VFIO 以及其 example

* [asso] 调研`mb()`, 全屏障

* [asso] 调研头文件 C `#include <stdatomic.h>`, c++ `#include <atomic>`

* [asso] 调研`memory_order_relaxed`, `memory_order_acquire`, `memory_order_release`, `memory_order_seq_cst`

* [asso] 调研`__sync_synchronize();`, `__atomic_thread_fence(__ATOMIC_SEQ_CST);`

* [asso] c++ 中`<atomic>`, std::memory_order 枚举

    `memory_order_seq_cst`, `memory_order_acq_rel`, `memory_order_relaxed`

* [asso] `/proc/<pid>/maps`

* [asso] DMA控制器芯片（如Intel的8237）

* [ ] linux 的 interruptible sleep 是如何实现的？

* [asso] pci_get_slot(), pci_get_device()

* [asso] kaggle pandas tutorial

    <https://www.kaggle.com/learn/pandas>

* [asso] pandas 实践平台

    * Kaggle: 上面有成千上万的真实数据集和别人的代码（Kernels/Notebooks），是学习 Pandas 高级用法和数据分析思路的最佳场所。

    * LeetCode 题库之 Pandas: 专门用 Pandas 解决数据处理问题，非常适合面试准备和巩固基础。

## CCL

### cache

* vllm pynccl 中目前看来改动的文件是`/home/test/miniconda3/envs/vllm/lib/python3.10/site-packages/vllm/distributed/parallel_state.py`

    看起来比较重要的几段代码：

    ```python
    with self.pynccl_comm.change_state(enable=True, stream=torch.cuda.current_stream()):
        self.pynccl_comm.send(tensor, dst=self.ranks[dst])
    ```

    ```python
    with self.pynccl_comm.change_state(enable=True, stream=torch.cuda.current_stream()):
        self.pynccl_comm.recv(tensor, src=self.ranks[src])
    ```

    ```python
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and not pynccl_comm.disabled:
        pynccl_comm.send(tensor, dst)
    else:
        with xxxx

    # torch.distributed.send(tensor, self.ranks[dst], self.device_group)
    ```

* nccl tmp

    * 多卡之间如何 reduce copy?

    * 如何动态确定多 srcs / dsts?

    * cu host alloc 的调用路径

        `ncclAsyncJobMain()` -> `ncclCommInitRankFunc()` -> `initTransportsRank()` -> `devCommSetup()` -> `devCommSetup()` -> `ncclCudaHostCalloc()`

* 目前看到 nccl 的 rank 是由 mpi 分配的，并未给 gpu 分配 rank。一个 rank 上的 gpu 按照 dev 0, dev 1 等方式进行区分。

* graph 应该以功能为评价标准，来考察形式

    一个图常用的几个功能：

    * 初始化，一次性导入大量节点和边，I/O；添加、删除少量节点，添加、删除边

    * bfs, dfs 搜索 path

    * dijkstra, floyd 算法，旅行商问题

    * query
    
        * 一个节点的边

        * 2 个节点之间是否有边，如果有边，边的权重是多少
        
        * 两个节点之间的 path

* set path nic 之后，siccl 与 nccl 输出一致。50 机器上没有 nvswitch，因此先跳过，后面在 135 机器上再测。

* 在 trim system 后，有些 net -> gpu 的 path 被清除掉了，影响到后面的 ring / tree 生成。

* `ncclTopoSearchRec()`函数参数为`int *index`，传入数据为`localNets+localNetCount`,其中`localNets`是`int*`, `localNetCount`是`int`，这个操作并不是把`int* + int`相加后生成一个匿名对象，再把匿名对象看作一个指针传入函数，而是将`localNets`看作一个数组，而`localNets+localNetCount`是对数组中某个元素取址。

* siccl 疑似在`generate_coll_graph()`的

    ```cpp
    if (coll_graph.nChannels * coll_graph.bwInter >= topo_system.totalBw)
        goto done;
    ```

    处触发`goto done;`，而 nccl 并没有。是因为 siccl 没有调用 search init 函数，导致 system->totalBw 为 0

* 目前看来 siccl 和 nccl 的 net 输出都是相同的，net idx 都为 1，0, net id 总为 2, 1

* `ncclTopoGetLocalNet()`返回的 net id 是 1，因为当`net = 1`, `localNetCount = 2`, `localGpuCount = 1`时，根据下面的规律，可看出当`channel = 0`时，`net`最终算出来为`1`。

    ```
    gpu idx: 0, channel: 0, net before: 1, net after: 1

    gpu idx: 0, channel: 1, net before: 1, net after: 2

    gpu idx: 0, channel: 2, net before: 1, net after: 1

    gpu idx: 0, channel: 3, net before: 1, net after: 2

    gpu idx: 0, channel: 4, net before: 1, net after: 1

    gpu idx: 0, channel: 5, net before: 1, net after: 2
    ```

* `ngpus`经过 trim system 后，就从 2 变成 1 了，后面一直是 1.

    典型的计算方式为

    `int ngpus = system->nodes[GPU].count;`

* 调用`ncclTopoSearchRec()`时，第二个参数是`&tmpGraph`, 第三个参数是`graph`，然而`ncclTopoSearchRec()`的参数列表是

    `ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {`

    也就是说，在函数内部改变的，其实是`saveGraph`的`nChannels`。

* 在`ncclTopoSearchRecNet() -> ncclTopoSearchTryGpu()`第一次调用时，channel 数就已经为 2 了。此时 net id = 1，后续会搜索 net id = 2 的情形。

    有时候 net id 会变为 2，1.

* `ncclTopoSearchRecGpu() -> graph->nChannels++;`

    `memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));`

    应该是在这个位置`saveGraph`的 channel 变为 2 的

* `ncclTopoSearchRecGpu()`是多层调用的，在最内部调用完`ncclTopoSearchRecNet()`后，channel 数就变成了 2.

* `ncclTopoSearchRec()`中，`invoke_cnt = 33`, `depth = 3`

    在这之前 nchannels 就已经是 2 了。

* `#define ASSERT(x) (void)(x)`这个宏的作用有可能是

    ```cpp
    #ifdef DEBUG
        #define ASSERT(x) assert(x)  // 调试阶段启用真实断言
    #else
        #define ASSERT(x) (void)(x) // 发布阶段无害化
    #endif
    ```

    总之是占位符性质。

* nccl 在 trim system 中的 domain 划归算法，或许未来可以用到 siccl 的 switch 节点发现上。

* gpu 1 -->SILINK--> swi 0 -->SILINK--> gpu 2

    未来可能形成这种形式，swi 只是一个虚拟节点，但是可能会有多个 switch，比如 gpu 1 通过两层 switch 才连接到 gpu2，但是在拓扑层中，两层 switch 只表现为一个 swi 拓扑节点，这样一来，如果我们在统计 gpu 1 -> gpu 2 的延迟和带宽时，如果只计算两个 silink 的延迟，肯定有问题。

    目前能想到的只有 3 点：

    1. 如果直接放弃 swi 节点， 使用 gpu 1 --SILINK--> gpu 2，那么倒是可以解决不同物理拓扑下不同连接的不同延迟问题，但是 topo system 的表示比较复杂

    2. 使用多个 gpu 互相印证：

        测试 gpu 1 --> gpu 2 的延迟 d_1，得到 x + y = d_1

        测试 gpu 1 --> gpu 3 的延迟 d_2，得到 x + z = d_2

        此时我们得到 y - z = d_1 - d_2

        测试 gpu 2 --> gpu 3 的延迟 d_3，得到 y + z = d_3

        此时我们得到 2 * y = d_1 - d_2 + d_3，由此可分别解出 x, y, z

        那么 x 就是 gpu 1 到 swi 0 的延迟，y 就是 gpu 2 到 swi 0 的延迟，z 就是 gpu 3 到 swi 0 的延迟。

    3. 测试出 gpu 1 --> gpu 2 的延迟后，直接除以 2，作为 SILINK 1 和 SILINK 2 的延迟。

* a100 4 gpu 环境为什么 gpu 到 gpu 的 path type 是 8 而不是 1，是因为在`compute_path()`函数中的

    ```cpp
    PeerInfo* dstInfo = &comm.peerInfo[topo_system.nodes[GPU][vert_idx_1]->gpu.rank];
    for (int p = 0; p < topo_system.nodes[GPU].size(); p++) {
        if (p == vert_idx_1) {
            continue;
        }
        PeerInfo* srcInfo = &comm.peerInfo[topo_system.nodes[GPU][p]->gpu.rank];
        int p2p = 0;
        // ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, system, NULL, srcInfo, dstInfo);
        if (p2p == 0) {
            int shm = 0;
            // NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, system, NULL, srcInfo, dstInfo));
            if (shm == 0) {
                // Mark this peer as inaccessible. We'll trim it later.
                topo_system.nodes[GPU][p]->paths[GPU][vert_idx_1].type = PATH_NET;
            }
        }
    }
    ```

    这里没有对 p2p 和 shm 进行检测，导致进入了`if (shm == 0)`分支。

* `xml_tag_to_topo_system()`中的`topo_system_connect_nodes()`之前的 bw 计算得不对，这个 bw 是根据 cpu model 计算出来的。16 指的是 amd 的 cpu。

* `ncclTopoSearchRecGpu()`中的

    ```cpp
    } else if (step == backToFirstRank) {
        // Find first GPU and loop back to it
        int p;
        ret = system->find_node((size_t*) &p, GPU, graph->intra[graph->nChannels*ngpus]);
        if (ret != 0) {
            printf("fail to find gpu node\n");
            return -1;
        }
        // getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p);
        TopoNode* firstGpu;
        ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu);
        if (firstGpu) {
            ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time);
            ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu);
        }
    ```

    其中`getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p);`这个函数看起来应该是使用 gpu 的 dev id 找到 gpu 的 idx。如果这个假设是对的，那么`graph->intra`存储的就应该是 dev id（待验证）。并且……想不起来了。

* g++ 和 ld 环境不一样，需要`export PATH=/usr/bin:/usr/local/bin:$PATH`.

* 因为 siccl topo 中无法 include nccl topo header，所以无法直接在 siccl topo 中引用 nv 的 struct。也无法直接复制一份 struct，因为有依赖，而且会重名。因此只能在 shim 层做转换。

* 不引入 nccl 头文件编译 siccl 不可能实现，因为 nccl 的 struct 依赖的 struct 太多，并且分散在不同文件里。

* Halving-Doubling 拓扑, 2D-Torus拓扑

### tasks

* [ ] 手动实现一下 ring + chunk 方式做 broadcast，对比直接调用 mpi 的 broadcast 函数，看看哪个比较快。

* [ ] pci busid 改成 uuid

* [ ] 在 data 目录中构建 load topo 对应的几种情况

    1. 1 cpu + 2 sipu + 2 usipu per sipu

    2. 2 cpu + 2 sipu + 2 usipu per sipu

    3. 2 cpu + 4 sipu + 1 usipu per sipu

    4. 1 cpu + 2 sipu + 2 usipu per sipu + socket net

    5. 1 cpu + 2 sipu + 2 usipu per sipu + rdma net

* [ ] 整理 xml.cpp，添加前缀

* [ ] `ncclSocketStateConnectPolling` socket 也支持 polling 吗？

* [ ] `cudaThreadExchangeStreamCaptureMode()`

* [ ] `cudaPointerAttributes()`

* [ ] C++ 中，下面的情况如何写成智能指针？

    ```cpp
    struct A;

    struct Parent {
        A* childs[4];
    };

    struct A {
        Parent *parent;
    };
    ```

    循环引用问题？编译器不知道 struct 的大小的问题？

* [ ] `struct alignas(32) ncclIbNetCommBase`中 alignas 是什么意思？

* [ ] 调研是否可以区分三种模式，switch, p2p same host, p2p not same host

* [ ] 调研可视化的方案

* [ ] 以 uuid 为入口重构 topo layer 代码

* [ ] 调研 nccl graph 中 sameChannels 的含义

* [ ] 调研`OneCCL`, `RCCL`, `Gloo`

* [ ] 调研 SCCL 的动态路径选择和故障恢复机制

* [ ] 调研`crossNic`什么时候变成的 2？

* [ ] 调研`ncclTopoSearchRecNet()`

* [ ] 调研除了 nccl 外的其他 ccl 库

* [ ] 调研 cuda memcheck tool / compute-sanitizer

    res:

    1. <https://stackoverflow.com/questions/75973717/where-is-cuda-memcheck#comment136638567_75973968>

    1. <https://docs.nvidia.com/cuda/archive/9.1/cuda-memcheck/index.html>

* [ ] reorg: linux socket programming

* [ ] 调研`inotify_init()`, `inotify_add_watch()`

* [ ] 调研 POSIX 标准

* [ ] 调研 Boyer-Moore 算法

* [ ] 调研`wchar_t`，`wcschr()`

* [ ] 调研`setlocale()`

* [ ] 调研`ICU`或`libiconv`

* [ ] 调研实现`ncclTopoCheckGdr()`

* {O} 调研`ncclTopoTrimSystem()`

    feedback:

    1. 调研为什么 trim，怎么 trim，trim 了哪些

* [ ] 调研`ncclTopoGetPxnRanks()`

* [ ] 调研`ncclParamNetGdrRead()`

* [ ] 调研`ncclGetLevel()`中，old level 和 new level 是如何映射的

* [ ] 调研`ncclTopoSelectNets()`

* [ ] 调研`#define COUNT_ARGS(...) sizeof((int[]){__VA_ARGS__}) / sizeof(int)`是如何统计参数个数的

* [ ] 调研为什么 gpu vert 1 不等于 vert 2 时，vert 1 到 vert 2 的 path 类似为`PATH_PHB`。

* {O} 调研`ncclTopoCheckP2p()`

* [ ] 调研 NCCL_COLLNET 是干嘛用的

* [ ] 调研 cmake FetchContent_Declare

* [ ] 调研 cmake `ExternalProject`

* [ ] 调研如果构造函数有多个参数，那么`explicit`有意义吗？

* [ ] 调研`opt.then()`, `opt.transform()`

* [ ] 调研 c++ 中如何知道数组有几个维度

* [ ] 调研`rsync -v`, `--info=progress2`

* [ ] 调研`std::reference_wrapper`

* [ ] 调研 gdb `info registers`

* [ ] 调研 gdb `set $my_var = $ `

* [ ] 调研`Coverity`

* [ ] 调研`Clang Static Analyzer`

* [ ] 调研 gdb 中 p 命令调用函数`(gdb) p (void)printf("Hello, GDB\n")`
    
* [ ] 调研 gdb `/x`用法

    `(gdb) call/x my_func()            # 十六进制显示结果`
    
* [ ] 调研 gdb `call system()`用法

    `(gdb) call system("ls /tmp")      # 可能影响外部环境`

* [ ] 调研 gdb call 调用构造函数和析构函数

    ```
    (gdb) call obj->method()
    (gdb) call ptr->~MyClass()  # 析构函数
    ```

* [ ] 调研`RLIMIT_NPROC`

* [ ] 调研`sched_getaffinity()`

* [ ] 调研`taskset`

* [ ] 调研`std::atomic`

* [ ] 调研`memory_order_relaxed`

* [ ] 调研`numactl --hardware`

* [ ] 调研`ncclLoadParam()`

* [ ] 调研`NCCL_PARAM()`

* [ ] 调研：当初为什么放弃了 idx + type 的形式？

* [ ] 调研 siccl 在构建 gpu tag 时传入 uuid 或 bus id + micro id

    不能传入 bus id，因为这样会构建两个或多个 gpu tag

    目前的方案是在 merge tag 时，filter out smae tag，但是这个方案毕竟不够完美

* [ ] 调研 filter out invalid silink

* { } 调研 ptx 指令集

    resources:

    1. <https://docs.nvidia.com/cuda/parallel-thread-execution/>

    current progress:

    目前看到

    <https://docs.nvidia.com/cuda/parallel-thread-execution/#device-function-parameters>

    5.1.6.4. Device Function Parameters 

* { } 调研 nccl 最小可验证集

    基于 socket + shm, simple protocol 实现 send, recv, reduce copy，实现拓扑功能，实现 ring。

    目前的项目：

    * `ptx_test`用于做 ptx 指令的实验，解决 reduce copy 中常见的 sync 机制，以及 load step value 时的 sync 机制；重写 load / store 相关的指令

    * `kern_nccl_test`：从 work batch, prims, reduce copy 等几个 layer 进行 kernel 功能的验证。因为有些混乱，所以慢慢可能要放弃这个工程。

    * `reduce_copy_test`: 专门拿来测 reduce copy，目前可以跑通。

    * `nccl_asymptotic_reproduce`: 由于前面的项目代码比较混乱，所以这里创建了一个渐近功能测试。最底层是 load store asm 级别的测试，再往上是 reduce copy，再往上是 prims，再往上是 work 等等。

        目前在致力于建设这个模块。

    feedback:

    1. 目前已完成 load 相关的指令集的编译与运行

        目前已完成两卡之间的 reduce opy，但是需要打开 p2p 功能才行。

    1. 使用 asm 做 load store，似乎真没有比直接解引用或数组索引快多少。

        或许单机多卡的跨卡通信上 asm 有优势？

    1. 目前看到无法是否禁用 shm，在已知的 host alloc 和 malloc 中，都没有 4M buffer 的申请。shm 有可能走的 fifo buffer。但是 socket 也没有 4M buffer申请，有点奇怪。

    1. 打印 cuda host alloc 的地址，打印 reduce copy 中的 dst 和 src 的地址，作对比，如果 src/dst 在 host alloc 的地址范围内，那么说明确实使用 host alloc 确定的 uva。否则什么也确定不了。

        socket send 的时候也可以打印下地址看看。

        使用自定义的 test case，在 cuda malloc 时打印出两个 deivce 的 addr 范围。

    1. 目前可以确定，在 shm 禁用的情况下，数据是通过 socket 传输的，socket 的 buffer size 是 128 KB，但是这个数据并不一定是 malloc 时正好 malloc 128 KB，而可能根据 nccl 环境变量`NCCL_P2P_NET_CHUNKSIZE`得到的。

        目前在 host alloc 和 malloc 中没看到这个 buffer addr。这个 buffer addr 的后 5 位总是 0，猜测可能做了 align alloc。

* { } 调研 nccl app 的写法

    目前看到<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html>
    
        Finalizing a communicator

    feedback:

    1. 自定义版的 nccl 有太多 debug 输出，因此重新编译了原版的 nccl，搭建了 nccl test case 调试环境。

    1. 目前 53 机器不走代理，因此只能访问国内网站。如果 50 机器需要访问 github，可以用笔记本 ssh -R 反向代理。

    1. [ ] 调研`ncclCommGetAsyncError()`

* [ ] 调研如何实现 grep 搜索包含 N 个关键词中的 M 个的行？

* [ ] qa: bash 30 mins

* [ ] 调研 apt 包`libgoogle-glog-dev`

* [ ] 调研 apt 包`libgtest-dev`, `libiberty-dev`

* [ ] 调研`python3 -m venv`

* [ ] 调研`>> $LOG_FILE 2>&1`

* [ ] 调研`if [ ! -d sipu_sw ];`

* [ ] 调研`from_chars()`, `atoi()`

* [ ] 调研 c/c++ 中 8 进制和 2 进制的字面常量怎么写，有解析这样字符串的函数吗？

* [ ] 调研`find . -type f -name '*config*.xml' -exec grep -l 'database' {} +`

    调研`find . -type f -regex '.*/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}.*\.log' -exec grep -l 'error' {} +`

    grep 在匹配文件名时，只支持 glob，如果想使用 regex 匹配文件名，那么必须将 find 和 grep 结合起来使用。

* [ ] 调研`cudaMallocManaged()`

* [ ] 调研 c++ string 使用正则表达式

* [ ] vim 中如何实现撤销操作？

* [O] 调研 qemu 添加 pci 设备

    feedback:

    1. [ ] 调研 qemu 使用`-kernel` + qemu gdb server 进行 debug

* [ ] 调研构建 graph 的 benchmark

* [ ] 调研`realpath()`, `tolower()`

    `realpath()`是否可以保证线程安全？

* [ ] 调研常用 c 标准库，linux 常用库

* [ ] 调研`perror()`

* [ ] c++ 中, string + char * 得到的是什么？如果改变运算顺序，char* + char* + string，又会得到什么？

* [ ] 调研 c++ string 与 int, float 的转换，调研 c string 与 int float 的转换

* [ ] 调研`__cpuid()`

* [ ] 调研 bootstrap 中 unique id 的生成方式，以及这个 id 有什么用？

* [ ] 调研 tcp 如何在 listen 时，bind 一个未使用过的 port？或者如何让系统自动分配一个 port？

* [ ] 调研 epoll

* [ ] 调研`recvmsg()`, `recvmmsg()`

* [ ] 如果未建立连接就 send / recv，或者如果建立了连接后，但是对方没有 send / recv 时就 recv / send，会发生什么？

* [ ] 调研：可以在一个 thread 中打开另一个 thread 吗？

* [ ] 调研常见的基于 poll 的异步事件中心的写法

* [ ] 尝试使用 cuda host malloc 实现基于 host 中转的 send / recv，需要在两个 node 上跑通。   

* [ ] 如果 rdma 中使用的是 va + offset，那么还可以 remote write 吗？此时该如何查表？

* [ ] 如果在 cond wait 的时候 destroy mutex，是否会导致程序出错？

* [ ] 在 for 和 while 中`int a = 0;`，然后再修改`a`的值，`a`第二次循环时是 0 还是修改过的值？

    同理，在循环中`MyClass my_obj;`，是否会多次调用构造函数和析构函数？

* [ ] 调研`barrierAny()`

* [o] 调研`asm volatile("barrier.sync.aligned`

    feedback:

    1. ptx 中没有直接对应的指令，不如从头看整个 ptx 指令集

* [o] 调研 barrier red or

    ```cpp
    __device__ inline bool barrier_red_or(bool vote, int name, int nThreads) {
      int ans;
      asm("{ .reg .pred p;"
          "  setp.ne.s32 p, %1, 0;"
          "  barrier.red.or.pred p, %2, %3, p; "
          "  selp.s32 %0, 1, 0, p; }"
          : "=r"(ans) : "r"((int)vote), "r"(name), "r"(nThreads) : "memory");
      return bool(ans);
    }
    ```

    feedback:

    1. 目前只看到`barrier.red.or.pred p`。

* [ ] 调研 nccl 中的 asm 语句 <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints>

* [ ] `fstatat()`

* [ ] `renameat()`

* [ ] `chrpath`, `patchelf`悠 RUNPATH

* [ ] `readelf -d my_program | grep -E '(RUNPATH|RPATH)'`

* [ ] `od -t c`是干嘛用的？

* [asso] 解析`readelf -l <bin_file>`输出中各个字段的含义

* [asso] `readelf -d <文件名> | grep NEEDED`

* [asso] `patchelf`

* [asso] 调研 elf dynamic section

* [asso] 调研下面这个命令，看不懂

    ```bash
    # 查找美化打印脚本
    find /usr -name "python*" -type d 2>/dev/null | xargs -I {} find {} -name "libstdcxx*" 2>/dev/null
    ```

* [asso] 调研下面这个命令，看不懂

    ```bash
    # 检查 GDB 是否能加载美化打印
    gdb -nx -ex "set verbose on" -ex "source /usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25-gdb.py" -ex "quit"
    ```

* [asso] 下面这两种都无法实现鼠标悬停看到数据，需要调研 gdb pretty print

    ```cpp
    #include <memory>
    #include <string>
    #include <vector>
    #include <stdio.h>

    using namespace std;

    struct MyStruc {
        int val = 123;
        string msg = "hello, world";
    };

    int main() {
        vector<unique_ptr<MyStruc>> objs;
        objs.push_back(make_unique<MyStruc>());

        const string &msg = "hello, world";
        printf("msg: %s\n", msg.c_str());
        
        return 0;
    }
    ```

* [asso] 调研 lldb

* [asso] 调研 vim 中如何开分栏？如何同步滚动左右两个分栏？

* [asso] 调研 vscode Synced Scroll 插件

## linux maintain

### cache

* [new] `curl -I`

    `curl -I https://gitlabsoft.siorigin.com/ccl/siccl_utils`

* [new] 调研 ssh remote forward 配置

    ```conf
    Host remote-host
        HostName your.server.com
        RemoteForward 12345 localhost:12345
    ```

* [new] 调研 OSC 52 协议

    程序直接操作本地剪贴板：

    ```vim
    " ~/.vimrc
    function! Osc52Yank()
        let buffer=system('base64 | tr -d "\n"', @")
        let data='\e]52;c;'.buffer.'\x07'
        let silent=system('printf "'.data.'" > /dev/tty')
    endfunction
    map <leader>y :call Osc52Yank()<CR>
    ```

    使用时：yy 后按 `<leader>y` 即可复制到本地剪贴板。

* [new] 调研 linux clipboard

    ```bash
    cat /tmp/copy.txt | xclip -selection clipboard  # Linux
    ```

    ```bash
    # 在本地终端运行
    nc -l 12345 | xclip -selection clipboard
    ```

* [ ] cmake 环境变量的用法

* [ ] 调研 bash 的`exec`命令

* [ ] 调研 bash 中的特殊变量`%%`（当前作业）

* [ ] 调研 rsync `--backup`的用法

* [ ] `less -j 10 +G filename  # 在某些系统中结合使用`

* [ ] `fzf < filename`

* [ ] 调研 less 搜索

    从当前行开始新搜索:

    ```text
    //pattern
    ```

    在搜索模式中使用正则表达式:

    ```text
    /[0-9]+\.txt
    ```

* [ ] `ripgrep (rg)：rg -i pattern filename`

* [new] 调研 vim 映射

    ```vim
    " 在 normal 模式下，按 ; 重复上一次的数字动作
    nnoremap <silent> ; :<C-u>call RepeatMove()<CR>

    function! RepeatMove()
        if v:count > 0
            execute "normal! " . v:count . "j"
        else
            normal! .
        endif
    endfunction
    ```

* [new] 调研 iptables

    ```bash
    # 查看当前规则
    sudo iptables -L -n

    # 手动添加规则到 fail2ban-sshd 链
    sudo iptables -I f2b-sshd -s 192.168.1.100 -j REJECT --reject-with icmp-port-unreachable

    # 或使用 DROP
    sudo iptables -I f2b-sshd -s 192.168.1.100 -j DROP
    ```

* [new] 调研`ssh -R 127.0.0.1:2222:127.0.0.1:22 -o GatewayPorts=no ali`

    通常 GatewayPorts 是在 server 端指定的，这个命令在 client 端指定，有用吗？

* [ ] 调研 makefile 中的`=`递归展开（lazy evaluation）

* [ ] 调研 makefile 中的`:=`立即展开（simple evaluation）

* [ ] 调研 makefile 中的`?=`若未定义则赋值

* [ ] 调研 makefile 中的`+=`追加

* [asso] makefile `filter`, `filter-out`

* [asso] `grep -P`

* [ ] 调研 rdp 远程桌面如何降低画质、提升速度

* [asso] 调研 ramfs

    `sudo mount -t ramfs -o size=2G ramfs /mnt/shm`

* [asso] 调研`df -T /tmp`

* [ ] `make -p`

* [asso] `make --print-data-base`

* [asso] 调研`tmpfs /dev/shm tmpfs defaults,size=2G 0 0`

    以及`/etc/fstab`的文件格式。

* [asso] `sudo mount -o remount,size=2G /dev/shm`

    以及除了`size=2G`外，其他的`remount`常跟的选项。

* [ ] `disown`不加参数的用法

* [ ] `dig`

* [ ] `nslookup`

* [ ] `setsid`

* bash 命令中`&&`和`;`的区别

    * `&&`： 只有前一个命令成功执行（返回0）才会执行下一个命令

    * `;`： 无论前一个命令是否成功都会执行下一个命令

* [new] `file $(which xfreerdp)`这个是什么意思？

* [new] `dpkg -L freerdp2-x11`

* [new] 调研 strace

    ```bash
    # 查看运行时的后端选择
    strace -e openat xfreerdp /v:dummy 2>&1 | grep -i "lib.*backend"
    ```

* [new] `update-alternatives --config xfreerdp`

* [new] 调研 PipeWire

* [new] 调研 gsettings 相关的配置，调研 netstat -tlnp，调研 grep -E，example 如下：

    ```bash
    # 查看 GNOME 远程桌面支持的协议
    gsettings get org.gnome.desktop.remote-desktop.rdp enable
    gsettings get org.gnome.desktop.remote-desktop.vnc enable

    # 查看正在运行的远程桌面服务
    netstat -tlnp | grep -E '3389|5900'  # RDP端口3389, VNC端口5900+
    ```

* [new] `sudo lsof -p $(pgrep gnome-remote-desktop) | grep freerdp`

* [new] `systemctl --user restart gnome-remote-desktop`

    `--user`是什么意思？

* [asso] 使用 lastb 命令查看最近的失败登录尝试

* [asso] denyhosts 自动封禁多次失败的IP

* [asso] 设置日志监控（如使用 logwatch 或 auditd），对异常登录尝试发出告警。

* 如果你觉得 fail2ban 太复杂，可以考虑：

    CrowdSec：更现代、配置更简单的替代品

    firewalld 动态规则：对于简单场景可能够用

    自定义脚本：用简单脚本+iptables/ufw 实现基础防护

* [asso] 调研 OpenRC

    类 Systemd 的现代初始化系统

    被以下 linux 发行版采用：Gentoo, Alpine Linux, Artix Linux

* systemd 的 alternative

    * runit (Void Linux 默认)

        特点：简单直接，基于监督树（supervision tree）

        设计哲学：保持简单，只做进程监控

        组件：runsvdir, runsv, sv 命令

    * s6 和 s6-rc

        特点：Unix 哲学，组件化设计

        优势：非常灵活，可组合性强

        作者：Laurent Bercot

    * Dinit

        特点：C++ 编写，配置简单

        目标：提供 systemd-like 功能但更简单

    * 传统 SysV init

        ```bash
        # 经典的系统V风格
        /etc/rc.d/
        /etc/init.d/
        ```

        特点：简单脚本，顺序启动

        仍在维护：Slackware, Devuan (可选)

    * 基于监督的初始化系统
    supervisord (Python)

        特点：专注于进程管理而非系统初始化

        场景：常用于容器或特定服务管理

    daemontools / DJB's supervise

        特点：最早的监督式 init 系统之一

        设计者：Daniel J. Bernstein

    * 针对特定场景的替代

        * BusyBox init

            特点：嵌入式系统的极简方案

            场景：资源受限环境，小型 Linux 发行版

        * minit (CRUX 使用)

            特点：极简设计

            配置：简单的纯文本文件

    * 容器/云原生环境
    不用 init 系统

        容器直接运行应用进程

        使用 tini 或 dumb-init 处理信号

    * systemd 轻量模式

        systemd 本身也有轻量版本

    各替代方案比较
    名称	学习曲线	复杂性	资源占用	主要优势
    OpenRC	中等	中等	低	平衡性最好，兼容性好
    runit	简单	低	极低	简单可靠，启动快
    s6	较陡	高	低	极其灵活和强大
    Dinit	简单	低	低	现代但简单
    SysV init	简单	低	极低	极其简单
    选择建议
    推荐 OpenRC 如果你：

        想要现代化功能但避免 systemd 复杂性

        需要良好的软件包支持

        来自 systemd 背景，想平滑过渡

    推荐 runit 如果你：

        追求极简主义

        在容器或嵌入式环境

        喜欢 "do one thing well" 哲学

    推荐 s6 如果你：

        是高级用户，需要最大灵活性

        不介意手工配置

        构建定制系统

    发行版支持情况

        完全避免 systemd 的发行版：

            Devuan (Debian 衍生)

            Artix Linux (Arch 衍生)

            Void Linux (runit 默认)

            Alpine Linux (OpenRC 默认)

            Gentoo (OpenRC 默认)

        提供选择的发行版：

            Slackware (传统 SysV)

            CRUX (minit)

* [new] 调研`env`命令

    比如`env LD_LIBRARY_PATH=/path/to/libs bash run_main.sh`

* SSH密钥常用的位元长度和类型

    * RSA 密钥

        ```bash
        # 生成RSA密钥
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_4096
        ```

        位元长度	安全性	兼容性	推荐程度
        RSA 1024	低	极高	❌ 已不安全，禁用
        RSA 2048	中	很高	⚠️ 最低要求
        RSA 3072	高	高	✅ 推荐
        RSA 4096	很高	高	✅ 推荐

    * ECDSA 密钥

        ```bash
        # 生成ECDSA密钥
        ssh-keygen -t ecdsa -b 256 -f ~/.ssh/id_ecdsa_256
        ```

        曲线类型	等效RSA强度	安全性	推荐程度
        ECDSA 256	RSA 3072	高	✅ 推荐
        ECDSA 384	RSA 7680	很高	✅ 良好
        ECDSA 521	RSA 15360	极高	✅ 良好

    * Ed25519 密钥

        ```bash
        # 生成Ed25519密钥（固定256位）
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
        ```

        特性	说明	推荐程度
        固定256位	等效RSA 3072	✅ 强烈推荐

* 为什么选 ECDSA 521，而不是 2 的整数幂，比如 512？

    1. 基于素数域的特性

        ```py
        # 不是 2^521，而是基于素数：
        p_521 = 2^521 - 1  # 这是一个梅森素数

        # 对比：
        p_512 = 2^512 - 1  # 这不是质数！
        ```

    2. NIST标准曲线

        ECDSA使用的不是任意位数，而是特定的标准化曲线：
        曲线名称	字段大小	安全强度
        NIST P-256	256位	128位
        NIST P-384	384位	192位
        NIST P-521	521位	256位

    3. 521位的数学优势

        ```py
        # P-521曲线的素数域：
        p = 2^521 - 1
        ```

        这个素数的特性：

        - 是一个梅森素数
        - 支持高效的模运算
        - 提供恰好256位的安全强度

    安全强度匹配

        P-256: 256位字段 → 128位安全强度

        P-384: 384位字段 → 192位安全强度

        P-521: 521位字段 → 256位安全强度

* check cipher key info

    ```bash
    # 查看密钥信息
    ssh-keygen -l -f ~/.ssh/id_ecdsa
    ```

* ssh 中是否可以指定任意整数的位？

    不能指定任意整数的位元，SSH密钥生成有严格的限制。

    ```bash
    # 尝试生成非常规位数的密钥
    ssh-keygen -t ecdsa -b 512 -f test_key  # ❌ 会失败！
    ssh-keygen -t ecdsa -b 500 -f test_key  # ❌ 会失败！
    ssh-keygen -t ecdsa -b 300 -f test_key  # ❌ 会失败！
    ```

    各类型密钥的有效位元:

    * RSA 密钥

        ```bash
        # 有效位元：必须是较大数字，通常 ≥1024
        ssh-keygen -t rsa -b 1024    # ⚠️ 最小（不安全）
        ssh-keygen -t rsa -b 2048    # ✅
        ssh-keygen -t rsa -b 3072    # ✅
        ssh-keygen -t rsa -b 4096    # ✅
        ssh-keygen -t rsa -b 8192    # ✅（但性能差）

        ssh-keygen -t rsa -b 1000    # ❌ 无效！
        ssh-keygen -t rsa -b 512     # ❌ 太弱，现代SSH会拒绝
        ```

    * ECDSA 密钥

        ```bash
        # 有效位元：只有3个固定值
        ssh-keygen -t ecdsa -b 256   # ✅ NIST P-256
        ssh-keygen -t ecdsa -b 384   # ✅ NIST P-384  
        ssh-keygen -t ecdsa -b 521   # ✅ NIST P-521

        ssh-keygen -t ecdsa -b 512   # ❌ 无效！
        ssh-keygen -t ecdsa -b 300   # ❌ 无效！
        ```

    * Ed25519 密钥

        ```bash
        # 固定位元：不能指定位数
        ssh-keygen -t ed25519        # ✅ 固定256位
        ssh-keygen -t ed25519 -b 512 # ❌ 忽略-b参数
        ```
        
* ECDSA

    ECDSA 密钥强度对比
    密钥类型	等效RSA密钥长度	安全性	性能
    ECDSA 256	RSA 3072	高	快
    ECDSA 384	RSA 7680	很高	较快
    ECDSA 521	RSA 15360	极高	中等

    ```bash
    # 您的521位ECDSA密钥提供极高的安全强度
    ssh-keygen -t ecdsa -b 521 -f ~/.ssh/id_ecdsa_521
    ```

    ECDSA 521属于NIST P-521曲线（secp521r1）, 被FIPS 186-4标准认可, 广泛支持于现代SSH客户端和服务器.

    兼容性 check:

    ```bash
    # 检查SSH客户端支持情况
    ssh -Q key
    # 应该包含: ecdsa-sha2-nistp521

    # 检查SSH服务器支持
    ssh -Q key-sig
    ```

    性能:

    比ECDSA 256稍慢，但差异很小, 在大多数场景下感知不到性能影响

* [asso] investigate fail2ban filter

    ```conf
    # /etc/fail2ban/filter.d/ssh-kex.conf
    [Definition]
    failregex = ^%(__prefix_line)serror: kex_exchange_identification:.* from <HOST>
                ^%(__prefix_line)serror: kex_exchange_identification: Connection closed by remote host.* from <HOST>
    ```

### tasks

* [ ] 官网介绍说，只需要使用`aria2c -x 2 <url>`就可以多线程下载，不知道真假

* [v] bash 中的分号是什么意思？

    比如`export LD_LIBRARY_PATH=/path/to/libs; bash run_main.sh`

* [asso] `env`命令

* [v] `wc`是否可以统计汉字的字节数，单词数？

* [ ] `who`, `w`, `last`这三个命令是什么意思？

* [ ] 为什么`g++ -g main.cpp -lvulkan -lglfw -o main`可以通过编译，`g++ -g -lvulkan -lglfw main.cpp -o main`就不行？

* [ ] 调研`grep -A`

    ```bash
    # 显示匹配行及其后2行
    grep -A 2 -E "keyword1|keyword2" file.txt
    ```

* [v] 调研`grep -w`

    > -w 选项匹配整个单词

* [ ] `ls -lt`, `ls -i`, `ls -n`

* [ ] `ls --color=auto`

* [ ] `ls -alht`, `ls -lhR`

* [ ] 调研`:tag function_name` - 跳转到指定标签

* [ ] `readelf -d <文件名> | grep NEEDED`

* [ ] 调研：为什么`grep -r siDeviceGet(`不能有左小括号？

* [ ] 调研`ip rule`

* [ ] 调研 crontab 系统级定时任务

* [ ] 调研`lsyncd`

    这个工具似乎是 inotifywait 和 rsync 的结合，是个比较成熟的工具。

* [ ] 调研`inotifywait -m /path/to/dir  # 持续监控目录`中`-m`的含义

* [ ] 调研`mpg123`, `vlc`, `paplay`音乐播放器

* [ ] 在 crontab 中，无法通过 mpv 播放音乐

    即使设置`DISPLAY=:0`和`DBUS_SESSION_BUS_ADDRESS`也不行。根据日志看起来像是 alsa 初始化失败。

    使用`mpv --ao=pulse`选择 pulse audio 也不行，日志提示未找到 pulse audio 的驱动。

    除了 mpv，其他的方案未尝试。

* [ ] 调研`aplay`，`paplay`, `cvlc`, `ffplay`

* [ ] 调研使用 bash 实现一个定时器任务管理工具

* [ ] 调研 bash 中的`REPLY`变量

* [ ] 调研`if [[ $char == $'\0' ]]`与`if [  ]`有何不同

* [ ] 调研`$char == $'\0'`是否可以写成`$char==$'\0'`

* [ ] `read -r`以及有哪些常见的反斜杠转义字符？

* [ ] `od -A x -t x1 test.bin`

* [ ] `echo "Hello" | od -t x1c`

* [ ] 调研 openssl, gpg

* [ ] 调研`dmenu`

* [ ] 调研`ksshaskpass`

* [ ] 调研`GIT_ASKPASS`

* [ ] 调研 PTY 与 tty 有何不同

* [ ] 调研`huponexit`

* [ ] `chdir()`

* [ ] 调研`gpg -dq ~/.ssh/password.gpg`

* [ ] 调研

    ```bash
    echo 60 > /proc/sys/net/ipv4/tcp_keepalive_time
    echo 10 > /proc/sys/net/ipv4/tcp_keepalive_intvl
    echo 3 > /proc/sys/net/ipv4/tcp_keepalive_probes
    ```

    与其他系统的 tcp 配置

* [ ] 调研

    ```bash
    # 查看当前SSH连接参数（客户端）
    ssh -vvv user@example.com 2>&1 | grep Keepalive

    # 服务端日志（需启用Debug模式）
    tail -f /var/log/auth.log | grep Keepalive
    ```

* [ ] 调研`csh`, `tcsh`

* [ ] 调研 sudoers 中的全局配置

    ```
    Defaults	env_reset
    Defaults	mail_badpass
    Defaults	secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
    Defaults	use_pty
    ```

* [ ] 调研 dnf 与 yum 的异同

* [ ] 调研`vim -X`非交互运行

* [ ] 调研如果在非终端（非交互）模式下运行top、htop、tmux、screen 等工具，会发生什么

* [ ] 调研 tty ? 是什么意思

* [ ] 调研`htop`

* [ ] 调研`ps -e -o pid,cmd`

* [ ] `ps aux --sort=-%cpu`, `ps aux --sort=-%mem`

* [ ] 调研`strings`工具

* [ ] 调研`xargs -0`

* [ ] 调研`ps -p <PID> -o cmd`

* [ ] 调研`pgrep -a <PATTERN>`

* [ ] 调研`/proc/<PID>/comm`

* [ ] 调研`readlink /proc/1234/cwd`

* [ ] 调研`/proc/<PID>/status`

* [ ] 调研`env KEY=value command`

* [ ] 调研 tr 能否处理汉字？如果不能，那么是否有能处理汉字的 tr like 软件。

* [ ] 调研`expect`脚本

* [ ] 调研`type <command>`命令

* [ ] 调研`grep *.txt`使用通配符搜索多个文件，对比与`include file`有什么不同？

* [ ] 调研`cut -d: -f1) app.log`

* [ ] 调研`sed`, `awk`

* [ ] 调研`vim +$(grep -n "error" app.log`

* [ ] 调研`grep --color=auto`

* [ ] 调研 gdb `x`命令

* [ ] 调研`rsync --filter`

    `rsync -av --filter='protect /destination/keep_this.txt' /source/ /destination/`

* [ ] 调研`rsync -n`或`rsync --dry-run`

* [ ] 调研`rsync --delete-excluded`

* [ ] 调研`rsync --max-delete`

* [ ] 调研 conan

* [asso] 调研 ssh 心跳保持

    client 端：

    ```conf
    # 每60秒发送心跳包
    ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -R 8080:localhost:80 user@remote-host

    # 或者写入配置文件 ~/.ssh/config
    Host remote-host
        HostName your-server.com
        User username
        ServerAliveInterval 60
        ServerAliveCountMax 3
        RemoteForward 8080 localhost:80
    ```

    server 端：

    ```conf
    # 在服务端 /etc/ssh/sshd_config 中配置
    ClientAliveInterval 60
    ClientAliveCountMax 3
    TCPKeepAlive yes
    ```

* [asso] 调研 frp 加密

    ```conf
    [common]
    tls_enable = true
    ```

* [ ] 调研 v2ray 按域名选择流量出口

* [ ] 调研 virtual box 的虚拟机如何使用两块显示器／如何开两个 display 窗口／如何登录另一个 linux 窗口

* [ ] 调研 meson, ninja

* 多线程调试时锁定单线程

    GDB scheduler-locking 命令详解

    <https://www.cnblogs.com/pugang/p/7698772.html>

* [ ] 调研 docker 中 app 的调试方法

* [ ] 调研 git reset 查看不同版本的 linux kernel version

* [asso] 调研 Remmina 的功能，是否支持 wayland

* [asso] vim 中是否有 shift + tab 来回退 4 个空格？

* [asso] setsid 是否可以在 bash 退出时，给 ssh 发送 sighup 信号？

* [asso] 什么是进程组？bash 中打开的进程在哪个进程组？bash 中打开的 ssh 在哪个进程组？进程组和父进程/子进程有什么关系？

* [asso] `ps -o pid,ppid,pgid,sid,command -p $$,$SSH_PID`, 这里 -o 是什么意思？

* [asso] 什么是 TTY 和 pts？

* [asso] `chroot <dir>`是什么意思？

* [asso] systemd 的`Wants=`, `Requires=`, `After=`有什么不同？

* [asso] systemd 的`Type=forking`是什么意思？

* [asso] systemd 中，`StartLimitInterval=300`, `StartLimitBurst=5`这两个是什么意思？

* [asso] systemd 除了`After=syslog.service`, `After=network.target`，还有什么常用的 service？

* [asso] `Restart=on-failure`, `Restart=Always`，这些有什么不同？还有什么可取的值？是否大小写敏感？

* [asso] 调研 iptables

    ```bash
    # 记录所有访问22端口的连接
    sudo iptables -I INPUT -p tcp --dport 22 -j LOG --log-prefix "SSH_CONN: "

    # 查看iptables日志
    sudo tail -f /var/log/kern.log | grep SSH_CONN
    ```

    ```bash
    # 使用iptables限制连接频率
    iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
    iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
    ```

* [asso] 调研 tcpdump 的用法

    `sudo tcpdump -i any port 22 -n`

* [asso] 调研 ufw 防火墙

    `ufw allow from YOUR_IP to any port 22`

* [asso] investigate port knocking

    端口敲门 (port knocking)

## gpu driver

### cache

* vscode 多线程调试: <https://zhuanlan.zhihu.com/p/704723451>

* GDB scheduler-locking 命令详解

    <https://www.cnblogs.com/pugang/p/7698772.html>

* csdn 上大部分文章只介绍了 nvlink 的作用和速度，并没有介绍协议细节

* load, store 看起来比较有用的几个网站

    * SCALING WITH DENSE NODES

        <http://nowlab.cse.ohio-state.edu/static/media/workshops/presentations/exacomm17/exacomm17-invited-talk-chris-newburn.pdf>

    * NVSHMEM Memory Model

        <https://docs.nvidia.com/nvshmem/api/gen/mem-model.html>

    * Load/Store over ETH 乎？

        <https://zhuanlan.zhihu.com/p/717851262>

    * HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA 

        <https://mp.weixin.qq.com/s/qLRC3dv4E93LwWXtuhQcsw>

    * AI fabric is a bus or a network？

        <https://zhuanlan.zhihu.com/p/708602042>

* vscode 多线程调试: <https://zhuanlan.zhihu.com/p/704723451>

* 50 机器物理机上仍需要在`paths.cc`文件中的`ncclTopoCheckP2p()`函数里添加`path->type = PATH_PIX;`，重新编译 nccl，才能使用 pcie p2p，否则只设置 nccl 环境变量无法跑通 p2p.

    同时，不能设置`NCCL_P2P_LEVEL`环境变量。把它设置为`PIX`也跑不通。

* [ ] 调研`cudaStreamBeginCapture()`, `cudaStreamBeginCaptureToGraph()`, `cudaStreamEndCapture()`

* cuda stream management

    <https://docs.nvidia.com/cuda/archive/9.1/cuda-runtime-api/group__CUDART__STREAM.html>

    一共没几个函数，有空了可以看看，找找 example。

* 简单看下 cuda stream csdn 上的资料

    <https://blog.csdn.net/huikougai2799/article/details/106135203>

    <http://turing.une.edu.au/~cosc330/lectures/display_notes.php?lecture=22>

* 在 gdb 设置 schedule locking 时，其他线程会被 freeze。

    是否可以让其他线程也运行，但只在当前线程触发断点？

* tenstorrent 使用分布式的处理器和内存，强调互联，文档给得不是很全，可以直接看代码。

    或许可以从 pytorch 接入那部分开始看，但是首先需要弄明白 pytorch 模型的保存格式。

* `initTransportsRank()`这个看起来挺重要的。`p2pSendSetup()`这个也比较重要。`ncclTransportP2pSetup()`这个看起来也很重要。

* 在`ld_volatile_global()`中设置断点，经过半个多小时后，断点被 hit

    看到的调用栈如下：

    ```
    ld_volatile_global() - op128.h:295
    Primitives::loadStepValue() - prims_simple.h:106 -> 116
    Primitives::loadRecvConn() - prims_simple.h:477 -> 496
    Primitives::Primitives() - prims_simple.h:574 -> 646
    RunWorkBatch::run() - common.h:280 -> ?
    RunWorkBatch<AllReduce_Sum_f32_RING_SIMPLE>.run() -> build/obj/device/gensrc/all_reduce_sum_f32.cu:15
    ncclKernelMain() - common.h:312 -> 369
    AllReduce_Sum_f32_RING_LL build/obj/device/gensrc/all_reduce_sum_f32.cu:3
    ```

    hit 断点时，output 界面仍有 perf data 输出，但 bandwith 等数据都是 0. 说明 perf data 和 ld_volatile_global() 可能是异步执行的，并且 perf data 可能会用到 ld volatile global 的数据。

* 猜想：nccl 的底层通信可以走 host 中转，也可以走 pcie p2p，无论走哪种方式，一定是 launch kernel 去处理的通信，launch kernel 一定会直接处理 va。因此如果是 p2p 通信，那么这里的 va 就是 peer device bar 空间的 va；如果是走 host 中转，那么这里的 va 就是 host memory 的 va，此时 host memory 作为 buffer。

* 在 nvidia open source kdm 里面有 nvlink, nvswitch 相关的代码

    <https://github.com/NVIDIA/open-gpu-kernel-modules>

### tasks

* { } cuda programming guide 12.30

    10:16 ~ 14:31

    cuda programming guide website: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

    目前看到

    > 2.2.1. Thread Block Clusters

    feedback:

    1. 调研`cudaMalloc3DArray()`, `cudaMallocArray()`

    1. [ ] 调研 C 中的二维数组，是一维数组的 transpose，还是一堆不连续的一维数组？

    1. [ ] 调研：是否可以将`n_rows`, `n_cols`作为 typename，从而用模板的方式实现任意类型，任意 size 的矩阵乘法？

* { } 调研 cuda gdb

    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>
    
    目前看到 3.3.2. Multi-GPU Debugging

* { } 抽取 nccl LL 协议

    tmp:

    * work 是一个什么样的类型？

    * 问题： aligned 的条件是什么？

    * 每个 chunk 中包含有多个 slice，每个 slice, 中包含有多个 step，每个 step 又有 stepSize

        这个和 warp, hunk 又是什么关系？

    * load step value 是否和这里的 step 有关系？

    * waitPeer()  ->  here 3, 4, 5, 7

        waitPeer 的机制是怎样的？是死循环等待条件吗？看起来不像。

    * `postPeer()`和`waitPeer()`分别有什么作用？

    * `genericOp()`的 DirectSend1, DirectRecv1, Send, Recv 取 0 或 1, SrcBuf, DstBuf 取 0，1 或 -1，Input 总为 0，Output 总为 1

    * 当 send 或 direct send 时，SrcBuf 为 Input，当 send from output 时，SrcBuf 为 Output。无论是哪一种 send，DstBuf 总为 -1。

        当 copy send / direct copy send 时，SrcBuf 和 DstBuf 都会被填，分别被填 Input 和 Output。
        
        看来 Input 和 Output 分别指明了 SrcBuf 和 DstBuf 的用途，并没有其他特别的含义。问题：这样模板化有什么好处？

        Input 和 Output 在 Primitives 类中被写死。无论是哪种协议，Input 总为 0，Output 总为 1.

        当 recvSend 时，SrcBuf 和 DstBuf 都是 -1，看起来是不做缓存，接收到数据后直接再传输出去？

    * DirectSend 一定是 Send，Send 不一定是 DirectSend。Recv 同理。

    * conn 是 struct ncclConnInfo 类型的对象，被赋值的地方为`conn = &peer->recv[connIndex];`

        问题：`peer`是什么？`peer->recv`在何时被赋值？

    * redop 就是 redfn，都在`src/device/reduce_kernel.h`里

        常用的有`FuncSum`，`FuncCopy`等。

    * IntBytes 用来指定使用什么 int 类型，比如可以指定为`size_t`，`int`, `long`等等。

        目前看来指定为`size_t`后，不影响功能。

    * 调研 redArg 是否可以拆开

        redArg 似乎没有被用上，即使设置成 0 也没有什么 bug 产生。

    * `PreOpSrcs` 的实测大小是多少？

        这个似乎也用不到。

    * byte pack 相关的都在`op128.h`里，ld st 相关的东西也在`op128.h`里

    * 一个函数同时有类模板参数和函数模板参数，该如何调用？

        比如 struct 中的 static 成员函数？

    * 当 redop 是 copy 时，multi src 和 multi dst 该如何处理？

    * sendPeers 和 recvPeers 中填的是什么内容？是 rank 号吗？

    * 断点 + cuda gdb print 发现 ring->prev 和 ring->next 都是 0

        2025/01/23/00: 存疑。记得有一次看到 ring->prev 是 0，ring->next 是 1

    * `Primitives -> setDataPtrs()`的作用？看起来有点像是通过网络拿到 peer 的信息后，填写当前 primitive 扮演的 role 所对应的 buffer 地址。

    * Input 总为 0，Output 总为 1，那么 SrcBuf 和 DstBuf 又是在哪赋的值？

        在`genericOp` -> `send() / recv()`时赋的模板参数值，并且被赋为 Input 或 Output。

        * 猜想：每一个 primitive 有两块 buffer，一个叫 src，另一个叫 dst，这两块 buffer 可能不在当前 primitive 上。根据不同的 role，猜测两块 buffer 的作用如下：

            * `send()` -> `genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);`

                src 作为 input，dst 没有用到。从 src 读数据，直接走 transport 层发出去。

            * `sendFromOutput()` -> `genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);`

                src 作为 output，函数的名称又叫 send from output，那么能想到的一种解释是 src 不在当前 device 上，可能是 peer device 的 output，我们直接从它的 output 读数据，并走 transport 层发送。

            * `recv()` -> `genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);`

                没有用到 src buf，说明从 transport 层拿到数据后，直接写入到 dst buffer 里。

            * `directRecvCopy()` -> `genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, /*postOp=*/false);`

                这个比较奇怪，src buf 没用上，但是 inpIx 用上了。那我们猜测它是从 transport 层把数据复制到 dst buffer。

            * `copySend()` -> `genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);`

                input 和 output 都用到了，是否可以理解为直接使用 cuda memcpy 把数据从 src 搬运到 dst？

            * `recvSend()` -> `genericOp<0, 0, 1, 1, -1, -1>(-1, -1, eltN, postOp);`

                input 和 output 都没用到，说明是从 transport 层拿到数据后，直接发送给另一个 transport 层，不把数据存储到 gpu memory 里。

            * `recvCopySend()` -> `genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);`

                input 没用上，input idx 也没用上，那么这个和`directRecvCopy()`有什么区别？

                凡是有 copy 的，output 都会用到。

        * 要求`if (Dst)`才会填`ncclShmem.groups[group].dsts[0]`的内容，但是`Dst`之前是 0，说明 dsts 的数据在前面就已经被填了。

        * Send 任务要求至少有一个 dst ptr，这个 dst ptr 默认是 next gpu 的 buffer，如果用户提供了 DstBuf，则额外增加一个 dst ptr；
        
            同理，Recv 任务要求至少有一个 src ptr，这个 src ptr 默认是 prev gpu 的 buffer，如果用户提供了 SrcBuf，则额外增加一个 src ptr。

        * 如果填写了多个 src ptr 或 dst ptr，不一定会用到所有的 ptr

    feedback:

    1. [ ] 调研 cuda shmem

    1. [ ] 不使用 simple 协议，把 primitives 改成 ll 协议

    1. [ ] 调研`extern __shared__ xxx;`申请动态的 shard 数据

    1. [v] 调研 cuda `__shared__`

* [ ] 调研`cudaMalloc3D()`

* [ ] 调研 riscv 模拟／仿真，调研指令集如何扩展

* [ ] 调研 pytorch load/save 支持哪些格式，`.pth`的格式

* [ ] 调研制作 docker image: 透传一个 nvidia device 可以成功跑通 cuda test

* [ ] 调研制作 docker image
    
    2. 透传两个 nvidia gpu，调研是否能跑通 nccl
    
    3. 调研 2 个 gpu 的通信方式

        1. shared host memory

        2. pcie p2p

        3. socket

        4. nvlink

* [ ] 调研 openmpi 对 mellanox, cuda, rocm 的支持

* [ ] 调研 mpi err handler，这个概念是否可以认为是 c 版本的 c++ try catch 错误捕捉机制？

* [v] 调研`MPI_Probe`, <https://mpitutorial.com/tutorials/dynamic-receiving-with-mpi-probe-and-mpi-status/>

    feedback:

    2. add reference 也应该先放到 cache 里，再添加到 note 里。不然无法保证写入 note 里的都是经过验证的。
    
    3. 调研`MPI_Get_count()`, `MPI_Probe()`, `MPI_Cancel()`，增加 example 和 ref doc

    4. 调研`MPI_ANY_SOURCE`, `MPI_ANY_TAG`，写 example 验证其功能

    5. 调研<https://www.mpi-forum.org/docs/>

    6. 未处理完，需要继续处理 Using MPI_Probe to find out the message size

        <https://mpitutorial.com/tutorials/dynamic-receiving-with-mpi-probe-and-mpi-status/>

        这个可能是防止 mpi recv 报错，每次先看一下有多少数据，提前分配好内存，再去 mpi recv 接收数据。

        （如果 recv 端的 buffer 有限，无法一次接收完，该怎么办？是否有循环接收的机制？）

* [ ] 调研 nccl p2p 的调用流程

* [ ] 调研 nvshmem API，重点看 n_pes 相关的函数和说明

* [ ] 调研 py 调用 C 库函数

* [v] 调研 vllm 中 nccl 的用法

    feedback:

    1. vllm 的 pynccl 主要 wrapper 了 all reduce, send, recv 三个函数，all reduce 的算子似乎是直接调用 pytorch 的算子，并没有额外实现一版。

        传入参数中有 cuda stream，但是没有看到 sync 的函数。因此理论上是支持异步的，不知道具体怎么个调用法。

    1. 调研 py 类型提示`group: Union[ProcessGroup, StatelessProcessGroup]`

    1. async 相关的可能在`cuda_wrapper.py`里

        这里面可以重点看下`cudaDeviceSynchronize()`是怎么 wrapper 的，在哪里调用的。

    1. 调研 cuda 函数`cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`

* [ ] 调研 nccl 中的 asm 语句 <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints>

* [ ] 调研尝试在 nccl 中把 p2p 传输的代码剥离出来，使用单独的一份代码跑通 nvlink + p2p

[ ] 继续调研 nccl 源码，看是否有 put get 相关的函数

* [v] 调研 linux nvidia kmd 中与 nvlink 相关的部分

    feedback:

    2. 入口函数`nv.c`: `nvlink_drivers_init(void)`

        `nvlink_linux.c`: `int __init nvlink_core_init(void)`

* [ ] 构建正则表达式的 note 和 qa

* [o] 整理使用 cuda async copy p2p 的代码

    要求如下：

    1. 使用 nvml 判断当前日环境是否支持 nvlink

    1. 判断当前环境是否支持 p2p copy

    1. 不 enable p2p，直接使用 p2p copy，测速

    1. enable p2p，使用 p2p copy，测速

    feedback:

    1. 调研 cuda event，cuda api

    1. 为什么 cuda malloc 的 va 可以区分不同的设备？

    1. timeit 增加  TIMEIT_END_WITH_SECS

* [ ] 调研 nvlink 的 kmd，尤其是 ioctl 里的各种命令，尝试找到 mmio 配置寄存器的地方

* [ ] 调研 sglang，尝试跑通 example

* [ ] 调研 linux ubuntu 环境下，nvidia 的 profiling 工具，比如 nsight 之类的

* [ ] 调研关于`CUDA_VISIBLE_DEVICES`的疑问：假如在启动 cuda-gdb 之前指定这个环境变量，host code 只能看到指定的 device；假如在启动 cuda-gdb 后，改变`CUDA_VISIBLE_DEVICES`，是否只会在指定的 device 上 hit 到断点？

* [v] 调研 cuda gdb

    17:16 ~ 17:36

    feedback:

    3. 调研 CUDA Quick Start Guide
    
        <https://docs.nvidia.com/cuda/cuda-quick-start-guide/>

    4. nvidia 的 cuda gdb 似乎是开源的，有时间了调研一下

        <https://github.com/NVIDIA/cuda-gdb>

    5. 未解决的问题

        cuda-gdb 如何切换 kernel 线程？如何 schedule lock 到一个线程上？

* [ ] 调研 python 中 ctypes 的用法

* [v] 调研 pynccl 的用法

    feedback:

    2. 目前看到`tests/test_1_init.py`的 53 行

* [v] 调研 cuda gdb

    feedback:

    1. 其他的 cuda kernel 都可以正常断点，但是 nccl 中的 kernel 无法打断点。

        start gdb:

        `LD_LIBRARY_PATH=/home/huliucheng/Documents/Projects/nccl/build/lib NCCL_DEBUG=INFO NCCL_MAX_NCHANNELS=1 NCCL_SHM_DISABLE=1 /usr/local/cuda/bin/cuda-gdb --args /home/huliucheng/Documents/Projects/nccl_test_3/main`

        output 如下：

        ```
        assign cuda mem data for dev 1
        cu dev 0:
        	cubuf_A: 3.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 
        	cubuf_B: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        cu dev 1:
        	cubuf_A: 1.0, 3.0, 2.0, 2.0, 4.0, 0.0, 1.0, 2.0, 
        	cubuf_B: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        [New Thread 0x7fffc3250000 (LWP 54269)]
        [New Thread 0x7fffc3a51000 (LWP 54270)]
        [New Thread 0x7fff598fe000 (LWP 54271)]
        zjxj:54233:54270 [0] NCCL INFO Channel 00/0 : 1[1] -> 0[0] [receive] via NET/Socket/1
        zjxj:54233:54270 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via NET/Socket/1
        [New Thread 0x7fff589fc000 (LWP 54272)]
        zjxj:54233:54269 [1] NCCL INFO Channel 00/0 : 0[0] -> 1[1] [receive] via NET/Socket/2
        zjxj:54233:54269 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] [send] via NET/Socket/2
        zjxj:54233:54270 [0] NCCL INFO Connected all rings
        zjxj:54233:54269 [1] NCCL INFO Connected all rings
        [Thread 0x7fffc3a51000 (LWP 54270) exited]
        [Thread 0x7fffc3250000 (LWP 54269) exited]
        [Switching focus to CUDA kernel 0, grid 2, block (0,0,0), thread (0,0,0), device 1, sm 0, warp 1, lane 0]

        ```

        会一直在这里卡着。

* [ ] 在 nccl 中自己写 kernel；显式写 comm kernel，不使用 nccl 中的 template

* [v] 调研 nccl launch kernel 与 cudaMemcpyAsync() 的时机

    feedback:

    1. 初步判断 cuda memcpy 是复制 conn info 数据。

    2. 调研`asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");`命令

    3. 调研 cuda 的 ptx 指令集

        <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/>

    4. 需要看断点的调用栈的上下文判断任务实际是如何执行的

* [ ] 调研`cudaGetDevice`和`cuDeviceGet`有什么区别？

* [ ] 调研 cuda samples 代码

* [ ] 调研 nccl 中 va 是何时被映射的

* [ ] 调研`cudaMemcpyDefault`

* [ ] 调研`cudaHostAlloc()`

* [ ] 调研 sglang start up

* [v] 调研 nccl 中 task planner 是如何组合 transport 和 launch kernel 的

    feedback:

    1. `proxy.cc`, `ncclProxyProgress()`, `progressOps()`

    2. `bootstrap.cc`, `socket.cc`, `bootstrapRoot()`, `socketTryAccept()`

        socket accept 在这里等待接受连接。

    3. `net_socket.cc`, `recvProxyProgress()`

* [ ] 调研 pytorch load/save 支持哪些格式，`.pth`的格式

* [o] 调研 nccl 中的 task planner

* [ ] 调研 cuda-gdb 当进入 cuda kernel 代码中后，是否还能查看 host code 的变量

* [ ] 调研`cuLaunchKernelEx()`，为自己在 nccl 里写 kernel 做准备。

* [o] 调研 LL 协议的最简实现

* [ ] 调研：对于同一个 warp 中的不同线程，汇编中的`%1`，`%2`等寄存器对于这些线程来说是相同的吗？

* [ ] 调研 hugging face，看看比 mmdetection 多了什么东西

## HPC comm

* [v] 调研 pci host bridge

    feedback:

    5. cached tabs

        * NCCL源码解析①：初始化及ncclUniqueId的产生

            <https://zhuanlan.zhihu.com/p/614746112>

        * NCCL源码解析②：Bootstrap网络连接的建立

            <https://zhuanlan.zhihu.com/p/620499558>

        * NCCL源码解析③：机器内拓扑分析

            <https://zhuanlan.zhihu.com/p/625606436>

        * NCCL源码解析④：建图过程

            <https://zhuanlan.zhihu.com/p/640812018>

        * NCCL源码解析⑥：Channel搜索

            <https://zhuanlan.zhihu.com/p/653440728>

        * NCCL源码解析⑦：机器间Channel连接

            <https://zhuanlan.zhihu.com/p/658868934>

        * NCCL的不足，集合通信库初步调研 NCCL、BCCL、TCCL、ACCL、HCCL

            <https://blog.csdn.net/lianghuaju/article/details/139470668>

    8. gdb+vscode进行调试12——使用gdb调试多线程 如何实现只对某个线程断点，其他线程正常运行

        <https://blog.csdn.net/xiaoshengsinian/article/details/130151878?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-130151878-blog-140669886.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-130151878-blog-140669886.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=1>

## 概率图 Probability Graph

* { } sync 贝叶斯网

## 控制

* { } 调研《控制之美》

## rdma

### cache

* 调研 rdma link

* 调研

    * `rdma_create_event_channel()`

    * `rdma_create_id()`

    * `rdma_listen()`

* RDMA exmaple

    <https://github.com/animeshtrivedi/rdma-example?tab=readme-ov-file>

    实现了一个使用 verbs 实现的 client server 模型。

    还给出了基于 iWARP 的 rdma 的安装方法。

* rdma repo

    调研一下 makefile 中 KERNEL，BINARY 这些变量的含义。是否还有其他的特殊变量

* 调研`ENOMEM`

* 调研`spin_lock_init()`，自旋锁相关

* 调研`bitmap_zalloc()`, `bitmap_free()`

* 调研`spin_lock_irqsave()`, `spin_unlock_irqrestore()`

* 调研`find_first_zero_bit()`

* 调研`bitmap_set()`, `bitmap_clear()`

* 调研`dma_alloc_coherent()`, `dma_free_coherent()`

* 调研`ida_alloc()`, `ida_free()`

* 调研`usleep_range()`

* 调研`debugfs_create_dir()`, `debugfs_remove()`

* 调研`rcu_read_lock()`, `rcu_read_unlock()`

* 调研`list_for_each_netry_rcu()`

* 调研`spic_lock_init()`, `INIT_LIST_HEAD()`

* 调研`snprintf()`, `pci_name()`

* 调研`pci_enable_msic_range()`

* 调研`module_pci_driver()`, `MODULE_DEVICE_TABLE()`

* 调研`module_auxiliary_driver()`, `auxiliary_device`

* 调研`be64toh()`, `endian.h`

* 调研 ibv ping pong

    Error: `No space left on device`

* 对于主动采样操作的 log 格式参考

    1. 当成功 poll 到 valid event 时，显示在这之前 poll empty event 多少次，并且清零 empty event 的计数

    2. 没有 poll 到 valid event 时，每 1000 次显示一次结果

* <https://docs.nvidia.com/networking/display/mlnxofedv531001/ethernet+interface(base)>指出，ports of connectx-4 adapter cards and above can be individually configured to work as infiniband or ethernet ports.

    if you wish to change the port type, use the mlxconfig script after the driver is loaded.

    这个配置方案说是在<https://www.mellanox.com>上，但是这个网站目前已经失效了。

* ofed installer 中似乎没有配置 roce 的相关文档和脚本

* 设置 cma default roce mode: <https://enterprise-support.nvidia.com/s/article/howto-set-the-default-roce-mode-when-using-rdma-cm>

    看起来是更改 cma 的默认 roce 设备的。在我们的 test case 里，直接使用 open device 打开指定设备，不存在默认设备这一说。因此应该和这个默认设备配置关系不大。

* mellanox 网卡配置 roce, 一个可能有用的网站

    <https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/configuring_infiniband_and_rdma_networks/configuring-roce_configuring-infiniband-and-rdma-networks#configuring-roce_configuring-infiniband-and-rdma-networks>

* mellanox 网卡配置 roce, 可能有用的一组命令

    `mlxconfig`, 

    ```
    (base) hlc@zjxj:~$ mlx
    mlxburn                mlxfwstress            mlxptrace
    mlxburn_old            mlxfwstress_ext        mlxptrace_ext
    mlxcableimgen          mlxgearbox             mlxptrace_int
    mlxcables              mlxi2c                 mlxreg
    mlxcables_ext          mlxlink                mlxreg_ext
    mlxconfig              mlxlink_ext            mlxtokengenerator
    mlxdpa                 mlxlink_plane_wrapper  mlxtrace
    mlxdump                mlxmcg                 mlxtrace_ext
    mlxdump_ext            mlxmdio                mlxuptime
    mlx_fs_dump            mlxpci                 mlxvpd
    mlxfwmanager           mlxphyburn             
    mlxfwreset             mlxprivhost
    ```

    上面是`mlx`开头的一组命令，可能有用。

* 目前看到的信息是 cx5 的网卡支持的协议有 ib 和 roce v1，cx4 网卡支持的协议是 roce v2。

* switch 上的 perftest 差不多能跑到 97 Gb/s

### tasks

* [ ] 调研：`ibv_get_cq_event()`会不会消耗`ibv_poll_cq()`的 wc？

* [ ] 调研为什么 cable 不支持高速率

* [ ] 调研 PCI relaxed ordering

* [ ] 调研 ibv cmd req 中的 driver data

* [ ] 增加 client remote write to server test case

* [ ] 调研 open mpi 的 scatter, gather C 程序

* [ ] 调研 rdma repo 中 pcie driver

* [ ] 调研 iwarp 实现的 ibverbs

* [ ] 调研 spdx

* [ ] 调研 poc-a pcie 地址映射和寄存器配置流程

* [ ] 调研使用 mmap 维护 cqe

* [ ] 调研 kgdb

## riscv

* [ ] 调研 chisel 的编译和项目开发环境

## kicad + npspice

### cache

* ngspice 仿真电路

    install: `sudo apt install ngspice`

    仿真一个简单的电阻分压电路：

    1. 新建工程目录，新建一个文件：

        `netlist_1.cir`:

        ```ngspice
        voltage divider netlist
        V1 in 0 1
        R1 in out 1k
        R2 out 0 2k
        .end
        ```

    2. 启动 ngspice，进入命令行界面

        `ngspice`

        ```
        (base) hlc@hlc-VirtualBox:~/Documents/Projects/ngspice_test$ ngspice
        ******
        ** ngspice-36 : Circuit level simulation program
        ** The U. C. Berkeley CAD Group
        ** Copyright 1985-1994, Regents of the University of California.
        ** Copyright 2001-2020, The ngspice team.
        ** Please get your ngspice manual from http://ngspice.sourceforge.net/docs.html
        ** Please file your bug-reports at http://ngspice.sourceforge.net/bugrep.html
        ** Creation Date: Mon Mar 11 21:44:53 UTC 2024
        ******
        ngspice 1 -> 
        ```

    3. 使用`source`加载网表文件

        ```bash

        ```

* kicad note

    * install

        ```bash
        sudo apt install kicad
        ```

* ngspice note

    * install

        ```bash
        sudo apt install ngspicd
        ```

    * ngspice official site

        <https://ngspice.sourceforge.io/tutorials.html>

* wikipedia 上列出的常用 spice 的资料

    * <https://en.wikipedia.org/wiki/List_of_free_electronics_circuit_simulators>

### tasks

* [v] 调研 kicad 仿真电阻分压电路

    14:01 ~ 16:53

    feedback:

    1. guideline 可以参考这个：<https://www.instructables.com/Simulating-a-KiCad-Circuit/>，讲得挺详细的

    2. cached tabs:

        * SPICE Simulation

            <https://www.kicad.org/discover/spice/>

    3. deps

        1. [ ] ngspice 仿真出一个简单的网表

            <https://ngspice.sourceforge.io/ngspice-tutorial.html>

        2. [ ] 学习 kicad 的 tutorial

* [v] 调研 ngspice 仿真一个简单电路

    14:08 ~ 

    feedback:

    1. [ ] 调研 gpulot

* [ ] 调研 sbt

## qemu

### cache

* PCI passthrough of devices with QEMU

    <https://www.theseus-os.com/Theseus/book/running/virtual_machine/pci_passthrough.html>

* PCI passthrough via OVMF

    <https://wiki.archlinux.org/title/PCI_passthrough_via_OVMF>

* GPU passthrough with libvirt qemu kvm

    <https://wiki.gentoo.org/wiki/GPU_passthrough_with_libvirt_qemu_kvm>

* QEMU/virtual machines with GPU pass through even possible on Debian based system? 

    <https://www.reddit.com/r/linux4noobs/comments/15vtwgt/qemuvirtual_machines_with_gpu_pass_through_even/>

* Non-GPU PCI Passthrough 

    <https://www.reddit.com/r/VFIO/comments/rivik0/nongpu_pci_passthrough/>

* QEMU Virtual Machine PCIe Device Passthrough Using vfio-pci 

    <https://null-src.com/posts/qemu-vfio-pci/>

* Chapter 15. PCI passthrough

    <https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/5/html/virtualization/chap-virtualization-pci_passthrough>

* Configuring pass-through PCI devices

    <https://www.ibm.com/docs/en/linux-on-systems?topic=vfio-pass-through-pci>

* Assigning Host Devices to Virtual Machines

    <https://documentation.suse.com/smart/virtualization-cloud/html/vm-assign-pci-device/index.html>

* VFIO Device Passthrough Principles (2)

    <https://www.openeuler.org/en/blog/wxggg/2020-11-29-vfio-passthrough-2.html>

* lisovy/qemu_pci_pass_through.txt

    <https://gist.github.com/lisovy/1f737b1db2af55a153ea>

* qemu VM device passthrough using VFIO, the code analysis 

    <https://terenceli.github.io/%E6%8A%80%E6%9C%AF/2019/08/31/vfio-passthrough>

* 如果在编译内核时使用 ctrl + C 或者关机强制中断，那么第二次继续编译的时候有可能会编译不通过，此时可以`make clean -f`，再去编译就好了。

* 使用自己编译的 6.8.0 内核后，使用`qemu-system-x86-64`启动虚拟机时，使用`-vga std`或`-vga virtio`都可以启动起来，但是会黑屏几分钟。

    看起来像是内核在等什么东西，目前不清楚。

### task

* [v] 公司电脑开启 kvm feature

    feedback:

    1. 公司笔记本 cpu 为 i7-1360P，联想 BIOS 中不支持 VT-x 虚拟化，因此无法使用 kvm 或 hyper-v。

    2. 由于公司电脑这个反例，所以即使使用命令行强制开启了 virtual box 的嵌套 kvm 特性，也不一定生效。

        具体要看虚拟机窗口右下角的运行引擎，如果是一个乌龟图标，那就表明用得是 native API。

    3. 即使 host 没有启动 kvm 虚拟化，virtual box 的半虚拟化接口也会为虚拟机里的 qemu 提供一个虚拟的 kvm 接口。

        在 qemu 虚拟机中使用`lsmod | grep kvm`可以看到`kvm_amd`是启用状态。

        这个当然没啥用，没有嵌套虚拟化，qemu 虚拟机速度很慢。

* [v] 调研 shrink qcow2 image

* [ ] 调研 virt-sparsify

    ref: <https://serverfault.com/questions/432119/is-there-any-way-to-shrink-qcow2-image-without-converting-it-raw>

* [o] 调研 qemu 命令行如何使用 bridge

    feedback:

    1. 调研 qemu network config

        * <https://amoldighe.github.io/2017/12/20/kvm-networking/>

    2. 在 virtualbox 7.0 启动的虚拟机 ubuntu 22.04 中使用命令行 qemu 启动 ubuntu 22.04 虚拟机时，需要指定`-vga std`，或者直接省略这个配置。选用`-vga virtio`和`-vga qxl`都会出现鼠标位置偏移的问题。

        有可能是 virtualbox 7.0 虚拟机 ubuntu 22.04 开启了 scaling 125%，但是不确定。

* [v] 调研 ubuntu 24.04 自定义内核对 qemu vga 的支持

    feedback:

    3. 可以调研下内核源码中 menuconfig 中有什么支持 vga 的选项

## GUI

* [P] 调研 tk app 开发

    10:14 ~ 11:01

    尝试实现一个定时器

    feedback:

    1. 中文版教程：<https://blog.csdn.net/vor234/article/details/134761002>

    1. 英文版教程

        * <https://tkdocs.com/>

        * <https://tkdocs.com/tutorial/install.html>

## 操作系统 OS Linux 运维

### cache

* ibus 中的`mod`键是 alt 键

* 经常出现 terminal 没有登录虚拟机 ssh 的情况

    有没有什么办法可以避免这个问题？

* 如何为一个用户增加指定目录的各种权限？

* 猜想：如果一个文件夹中已经有内容，那么使用`mount`, `mount -t nfs`, `sshfs`挂载设备或远程目录时，不会删除文件夹下的内容，而是暂时覆盖文件夹下的内容

* 对于字符串`/tmp/dir/target`，如果我们想提取出最后的`target`，可以用下面几种方法

    ```bash
    sed 's.*/##' <<< "/tmp/dir/target"
    target

    awk -F'/' '{print $NF}' <<< "/tmp/dir/target"
    target

    grep -o '[^/]*$' <<< "/tmp/dir/target" 
    target
    ```

    也可以直接使用 bash 脚本：

    ```bash
    INPUT="/tmp/dir/target"
    echo ${INPUT#*/}
    target
    ```

* [ ] ln 是否能创建文件夹的 hard link?

* [ ] 调研`ssh-add`，`ssh-agent`的作用

* [ ] 调研 makefile 的 submodule

* [ ] 调研 diff 命令的用法

* [ ] 调研 linux `time` command

    `time ./my_app`

    调研`nvprof ./vector_add`如何给出性能数据

    调研`cudaMallocManaged(&x, N*sizeof(float));`

### tasks

## 分布式计算调研

tasks:

* [ ] 调研 mpi 如何给不同的 node 配置不同的环境变量？

## GPU Perf

cache:

* 调研算子的优化

* 无论是 windows 还是 ubuntu，rgp 都无法正常启动。

    尝试了各种 app，rgp 都无法 capture profiling。

tasks:

* [ ] gui 增加下拉菜单选择曲线的方式

    选择 2 条不同 scaling 的曲线进行处理

* [ ] 使用 ring buffer 代替 vector

* [ ] 增加历史数据的扁平小窗显示

* [v] 调研 gui 增加下拉菜单 30 mins

    feedback:

    1. cached task

        * 调研使用 slint interpreter 索引到子组件

* [ ] 调研 glx

* [x] 调研 gpa 对一个 opencl command queue 能拿到什么数据

* [ ] 给定 json 格式的 glmark 2 baseline

    一些模式：

    1. n 机 1 core (n = 1, 2, 3, 4)

    2. n 机 2 core (n = 1, 2)

    3. totally passthrough

* [ ] perf: 调研 gpa 源码

    看一下抓数据时使用的是什么接口

* [ ] 处理`slint/examples/plotter`中的代码

    reset 成原来的样子

## markdown renderer

使用 electron + markdown parser + mathjax 实现 markdoen renderer。

tasks:

* [ ] 调研 js

* [ ] 调研 electron 中的 lifecycle, emitter

    <https://www.electronjs.org/docs/latest/tutorial/tutorial-first-app>

* [v] 调研 electron

## GPU virt

cache:

* virt leraning roadmap

    1. 对Linux kernel有全面了解，关键模块有理解（走读kernel源码，对流程有印象）。推荐书籍：深入Linux内核架构（+1）或者深入理解LINUX内核。

    2. hypervisor虚拟化, Intel的《系统虚拟化》，很老很实用，看Qemu，KVM，Xen代码。4.容器虚拟化，读cgroup,lxc,docker代码。

* 英文网站上对 iommu 的介绍很少，只有 linux kernel docs 上的资料多一些，另外就是一些零散的 ppt slides。

    知乎上对 iommu 的介绍很多。

    下载了一些 linux kernel programming 的书，也基本没提到 iommu。

    不清楚这个知识点来自哪里。

    关于某个具体的函数`iommu_domain_alloc`，全网几乎都没有资料。但是代码里确实用到了。

    假如未来也遇到了这样的问题，似乎只有看源码才能解决了。

resources:

* professional linux kernel architecture

* 王道考研视频

* 《操作系统设计与实现》

### tasks

* [v] 看 linux 内核那本书

    feedback:

    * current progress: P36 1.3.1 Processes, Task Switching, and Scheduling

    * Microkernels 的 os 都有哪些？

* [v] 调研中断的处理过程

    猜想：如果虚拟机卡死，那么一定存在一个命令发送完后，接收不到下一个中断

    假设的模型：pcie 发送包 -> 等待中断 -> pcie 接收包

    feedback:

    3. 调研 kmd 中设备的创建流程，中断设置以及 pcie 配置

    4. 调研 kmd 中的地址转换，尝试使用 mmio 读取 gpu 中寄存器的值

    5. pci 的中断向量配置本身很简单，现在需要先 sync 一些 pci 驱动的知识

    6. 调研 msix 的含义

* [ ] 调研 pci_register_driver 等相关的 pci 驱动知识

* [ ] 调研 c++ asio，调研操作系统抢占式任务，调研异步的实现

* [ ] 调研 share mode，虚拟机中跑分，是否存在可以调度的地方

## 编译器 Compiler

**cache**

Resources:

* 《现代编译原理C语言描述》

* Compilers Principles Techniques and Tools (2nd Edition) 

* 120 Introducing bison.pdf

    介绍了 bison 程序的简单用法，并给出了一个 example。

    但是缺少很多编译的基础知识，这部分需要去看书才行。

调研 bision

Tasks:

* [v] 解释名词 3.1.3 二义性文法

* [v] 解释文件结束符

    feedback:

    1. 这个问题显然不是先调研再决定解决的，因为非常简单。

        以后的 task 尽量先调研再尝试解决。

* [ ] 2.4.2 将 nfa 转换为 dfa

* [ ] 2.2 bc 小题

* [ ] 解释 3.2 预测分析和递归下降

* [ ] 调研，3.2.2 构造一个预测分析器

## Vulkan 学习 [0]

**cache**:

* to complete:

    1. modern c++, rewrite the ray tracing program

    3. performance analyzing

    4. vulkan compute shader

* glfw compiling error: `Undefined reference to XOpenDisplay`

    Adding `-lX11` to the the Makefile solved this issue.

* 需要继续看书，弄明白 command buffer 和 subpass 的关系。

* `VkSubmitInfo`中`pWaitDstStageMask`是什么意思？这个不填的话，validation layer 会报错。

* vulkan 使用 uniform buffer 改变三角形的颜色 final code

    见`ref_1`。

* cached question: 尝试用这段代码拿 vulkan 的 physical device 的 queue family properties，

    ```cpp
    for (int i = 0; i < phy_dev_cnt; ++i)
    {
        vkGetPhysicalDeviceQueueFamilyProperties(phy_devs[i], &queue_family_cnt, nullptr);
        queue_familys.resize(queue_family_cnt);
        vkGetPhysicalDeviceQueueFamilyProperties(phy_devs[i], &queue_family_cnt, queue_familys.data());
        printf("%d-th physical device has %d queue families:\n", i, queue_family_cnt);
        for (int j = 0; j < queue_family_cnt; ++j)
        {
            printf("    %d: queue count: %d, queue falgs: %d, time stamp: %d\n", j, queue_familys[j].queueCount,
                queue_familys[j].queueCount, queue_familys[j].timestampValidBits);
        }
    }
    ```

    输出：

    ```
    0-th physical device has 3 queue families:
        0: queue count: 1, queue falgs: 1, time stamp: 64
        1: queue count: 4, queue falgs: 4, time stamp: 64
        2: queue count: 1, queue falgs: 1, time stamp: 64
    ```

    为什么 0 号 queue family 和 2 号 queue family 是一模一样的？

* `vkGetDeviceQueue`指定的 queue 有数量总数吗？

* 为什么<https://github.com/Overv/VulkanTutorial/blob/main/code/22_descriptor_set_layout.cpp>没有用到 layout pool ?也没有用到 allocate，直接就是 create，为什么？

参考资料：

1. pdf 电子书《vulkan_tutorial_en》

2. 配套代码 <https://github.com/Overv/VulkanTutorial/blob/main/code/22_descriptor_set_layout.cpp>

3. pdf 电子书 vulkan programming guide

依赖项目：

1. 英语单词

目前主要在看 descriptor set，先能独立画出一个彩色三角形。

然后看一下 uniform buffer 相关的，让三角形动起来。

再看一看 model loading，学会加载模型。

接下来看 compute shader 相关的东西。

添加一个 resource，官方的 vulkan programming guide。

任务列表：

* [v] 添加 vulkan 的 qa： 给出画一个三角形的流程步骤

* [v] 调研 vulkan tutorial: texture mapping

    feedback:

    1. current progress: P210 Combined image sampler

    2. deps:

        vk 画一个三角形

        需要 sync

* [ ] 学会传入一张图片作为 framebuffer resource。

* [ ] vulkan pipeline synchronization

* [v] 调研 vulkan programming guide

	feedback:

	1. 整理 descriptor set 的笔记，创建一个 qa

	2. 可以在 vulkan programming guide 里参考一下 sampler，descriptor set 的 api，重点还是 vulkan tutorial 的 texture mapping 这一章

* vulkan programming guide

    侧重一下 renderpass 和 subpass 的关系，barrier 对 image layout 的转换，以及 copy image from buffer

    sparse memory 是干嘛用的，看书上的意思，是可以让资源对象在运行时重新绑定到一块不同的内存上？

    1. P47 ~ P52

        feedback:

        1. allocator 明显目前用不到，而且也没有做实验的时间，也不能抄书做 note，因此这部分就应该跳过。

            那么这样的内容明显就不是一次能消化完的，只能做个大概的总结，需要下次任务再看一遍。

        2. 找一个 allocator 的 example

    2. P52 ~ P58
    3. P58 ~ P71
    4. P71 ~ P77
    5. P77 ~ P78
    6. chapter 3 and later
    7. moving data

## OpenCL 学习 [1]

参考资料：

1. pdf 电子书《pdfcoffee.com_opencl-programming-guidepdf-pdf-free》

2. 其他网上的资料

基本的东西会了，剩下的系统地看一看吧，查漏补缺。

重点看一看内置函数。

### cache

* cached question:

    * 不同 shape 的矩阵，分别使用 cpu, gpu 计算，哪个速度更快一些？是否有一些估算方法？能否总结出一套经验公式估计不同操作的时间？

    * 对于 1025 等无法被 2 整除的 size，该如何处理多余的量？

    * 由于 local memory 的大小无法在运行时改变，该如何才能设置一个合理的值？

* opencl 好像有个 sub group 的概念，这个是干嘛用的？

* 尝试 opencl 的 aync 写法。

* 使用`async_work_group_copy()`可以实现将数据从 global 复制到 local 吗？如果能那么速度比直接使用指针解引用快多少？

* opengl `请画一个彩色正方体，并使之绕 y 轴旋转。`

    修改 dep，先按 ind 画出正方体。

* 调试 opencl kernel 代码时，可以先把 work size 调整成 1，看看局部的情况

* opencl 中`uint8`指的并不是占 8 位的 int 值，而是有 8 个`uint`的 vector。

	这点和 C 语言不同，需要注意。

	如果需要用到单字节的数据类型，可以使用`uchar`或`char`。

	cached task:
	
	1. 调研不同数值类型的长度，总结从 8 位一直到 128 位的不同数据类型。

	2. opencl 中 char4 和 int 是否等价？或者说，它们的位存储情况是否相同？

* opencl cached question: 如何在 kernel 中创建一个临界区？

	```c
	__kernel void test_print(__global char* src, __global int* increment)
	{
		printf("%c\t", src[0]);
		if(atomic_add(increment, 1)==get_global_id(0))
		{
			src[0]++;
		}
		printf("%c\n", src[0]);
	}
	```

	这样写似乎不太行，因为`atomic_add()`执行完后，假如`if`语句块有多条语句，那么会有多个线程进入语句块中。

* simple opencl 增加 offset copy mem 功能

* 如何在 kernel 中创建一个 float3 向量？

* opencl 的笔记缺少很多基础东西，需要慢慢补全

* 如果有多个`xxx.cl`程序组成一个大的项目，host 又该如何编译？

* 中值滤波算子已经完成代码，详见`ref_5`。

    同样一张图片执行中值滤波，cpu 使用的 clock 时间为 13163，gpu 使用的时间为 1137。

### 任务列表

* [ ] 下载一本数字图像处理的书比较好，网上找的资料太碎片了

* [ ] 单通道图像求中值的代码可以参考`ref_0`

    下次可以直接试试单张图片。

* [o] 使用 cpu 实现中值滤波

    目前只写完了读取图片。见`ref_3`。

* [ ] sync 一下 opencl

* [v] 调研第 5 章 current progress: P201 Image Read and Write Functions

	feedback:

	1. 增加每天的英文阅读量

	2. 需要提出一种任务切换需要满足的最小条件，防止任务切换过于频繁

	3. dependency

		需要看 chapter 8: images and samplers

* [v] 调研 reduce sum, reduce max

* [v] 调研虚拟机内运行 opencl

    feedback:

    1. 调研`simple_ocl.hpp`中获取设备的逻辑改为优先 GPU，没有 GPU 可使用时，再使用 CPU

## 算法 algorithms 与 leetcode

cache:

* 不要忽略题目的暴力写法

    暴力写法是基础。

    在写出暴力写法的基础上，再去分析改进。这样的技能才是通用的。

* 力扣题：为什么做智力题无法从前往后递推，打家劫舍就可以？智力题的递归该怎么写？记忆化搜索该怎么写？总结智力题的双重循环搜索和从前往后的剪枝。强行单循环从前往后搜索的解法就不看了。

* 做一道 leetcode 题差不多要一个小时

* 图论里，如果已知一个 node，需要知道和这个 node 相连的边的情况，那么用链表是最优选择。

    如果已知两个 node，需要知道这两个 node 之间的边的情况，那么用矩阵是最优选择。

    如果既需要知道某个 node 有哪些边，又需要快速查询两个 node 之间是否相连，可以同时使用这两种结构。

tasks:

* [v] 调研《fundamentals of computer algorithms》

	feedback:

    * 这本书没有提到平衡树，红黑树，B+树之类的高级算法，字符串和图论的大部分算法也没有提到。

		但是讲了很多有关算法的基础分析知识，二分搜索，回溯，图的遍历等算法。

		总得来说非常值得一读，认真把题做一做。

* [v] algorithm: 写一道图论题

* [v] 解释 P11 ~ 12 TowersOfHan 递归算法

* [ ] P13 习题 1

* [ ] P24 ~ 25, 解释 step table 1.1, 1.2, 1.3

* [ ] 解释 P29 1.3.3 asymptotic notation $\Omega$, $\Theta$

* [v] leetcode 1 题  05/25

* [ ] 调研：大于等于指定元素的第一个元素的二分查找写法

    以及小于等于指定元素的第一个元素的二分查找写法

## awk

cache:

* awk 使用`#`标记注释

resources:

1. <https://www.gnu.org/software/gawk/manual/html_node/index.html#SEC_Contents>


## OpenGL

有时间就看看 opengl 各种效果的实现，主要是光影，贴图，动画等。

* cache

    1. 画 cube 不加旋转等于没画

        因为画出来的图形的正面总是一个正方形，无法判断其他几个面画得对不对。

        不将不同面涂成不同的颜色也不行，因为只涂一个颜色无法分辨立体感。

## 计算机图形学 学习 [2]

### cache

* 计算机图形学的任务列表太长了，有时间了整理一下

参考资料：

1. 虎书《Fundamentals-of-Computer-Graphics-Fourth-Edition》

2. Games 101

3. 其他网上的资料

依赖项目：

1. 英语单词

数学基础目前还算够用，每天记录一点点公式就可以了，一个概念，一段话，一个公式，不需要学习太多。

目前已经看完了第四章 ray tracing，并做了代码的实现。现在想先看一看光栅化的 pipeline，自己去尝试光栅化一条直线，一个三角形。

后续的安排是调研一下第 13 章，慢慢学一些折射，透明的效果。

目前未解决的问题：采样与 gamma 校正。

### tasks

* [ ] 调研 compute shader

* [v] 调研图形学

    feedback:

    1. 再找一本图形学的书，看看辐射度量学是怎么讲的

* [v] 找一本新的有关计算机图形学的书

* [v] 调研 Computer Graphics Principles and Practice

	feedback:

	1. 感觉像是部字典书，有空的话可以翻翻

		它的光追部分都排到很后面了，等学完不知道啥时候了。

* [ ] P18 ~ P20 三角形定义与常用的三角函数性质

	看了反三角函数，其中最有用的是 atan2。实际上也确实是这样。不过 atan2 的两个参数是 (y, x) 还是 (x, y)？

* P22 2.4.2 Cartesian Coordinates of a Vector

* P24 2.4.4 Cross Product

* P30 2.5 Curves and Surfaces

* 如何以一组向量为基，表示任意一个向量？

* 光栅化 pipeline 部分

    1. [v] P161 8.1.1 line drawing，用代码实现一下

    1. [ ] 直线的光栅化还有一个用增量的方式判断，简单了解下记个笔记就行，不用实现

    1. P164 8.1.2 Triangle Rasterization 尝试学习，理解

1. 光追渲染部分

    1. [v] 调研折射效果

    1. [ ] P324 ~ P327 13.1 折射，尝试实现一个透明球体

        已经实现了求折射射线，接下来尝试求反射光强度$R(\theta)$。

        对于一个透明物体，尝试找到出射点。

        使用 ri ratio 来判断是否进入或离开物体可行吗？如果摄像机在水中，这个方法是否会失效？

        如果这种方法失效，是否有更好的解决方案？

    1. P327 ~ P328 13.2 实例化，更方便地采样

        这一段及后面的慢慢看

    1. P328 ~ P329 13.3 使用布尔操作构造新物体

    1. P329 13.4 distribution ray tracing，目前还没仔细调研

1. 辐射度量学 Light 章节

    1. [ ] 重新从头整理一遍辐射度量学的公式和笔记 60mins

        要求书上任意一个地方都可以找到对应的笔记。

        要求可以解释**笔记中**任意一个字母的含义，公式的推导。

        已经整理完了 irradiance 相关的东西，接下来整理 radiance 相关的内容。在笔记里想把 radiance 和$L_f$调换一下位置，同时弄清楚 radiance 的具体含义。

        任务中止原因：达到了最大知识容纳限度。

1. Color 章节

    这一章讲怎么把辐射转换成人眼可见的颜色。

## 解析几何 学习

参考资料：

1. 《Gordon Fuller - Analytic Geometry-ADDISON-WESLEY @ (1954)》

依赖项目：

1. 英语单词

希望能学到些有关参数方程的知识，其他的知识就当作查漏补缺。

任务列表：

1. [ ] 调研全书，列出几个感兴趣的部分，继续调研

## 基于路径追踪的全局光照

参考资料：

1. firefox 上打开的 tab

1. Games101 的 slides

    <https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html>

1. 别人基于 games101 的一个 c++ 实现

    <https://github.com/zhiwei-c/Monte-Carlo-Path-Tracing/tree/main>

1. 路径追踪的知乎专栏

    <https://zhuanlan.zhihu.com/p/475547095>

不需要深度调研，简单跑跑 demo，研究下代码就可以了。

依赖项目：

1. 计算机图形学

    必须熟悉完计算机图形学 Light 和 Color 两个章节后，再来做这个项目。

任务列表：

1. 调研 Global Illumination project，尝试照着他的思路，实现一个差不多的渲染器

1. 实现 refraction 效果

    1. 实现 reflectance

    1. 计算折射衰减

1. [v] 屏蔽 refraction 效果，先实现一个 path tracing

1. [v] 创建一个房间场景，让光在房间中弹射

1. [v] 对接场景文件

1. 将任务拆分并并行化，使用 opencl 加速

1. 实现更多的效果，比如贴图采样，抗锯齿

1. 完成 blender, maya 插件开发

    1. [ ] [调研] 调研 blender 插件开发的 sdk，看看能不能找到类似场景文件的东西或与渲染器输入输出相关的资料。

## 英语单词

参考资料：

1. `english_words.md`

这个项目作为被依赖的项目而启动。

## 调研 kmd 在 container 中运行的可行性

1. 使用`sed`替换整行，替换的字符串中，有`$()`命令得到的系统信息

1. 在 44 机器上搭建 kmd 编译环境，成功编译出 xdx 的 kmd

1. 在 server 上拉取 umd 的代码，使用上海镜像

1. 在 44 机器上搭建 umd 的编译环境，成功编译出 umd

1. 调研 kmd 代码，查看其构成，尝试找到负责与 pci 交互的那一部分

1. 调研 modprobe 和 depmod 的作用，并写代码实践

## C/C++

主要任务是学完 modern c++，即 c++23 及之前的内容，找一些开源库看一看。

### cache

* [asso] c / c++ 中是否有类似 argparse 的库，或者其他处理参数的库？

* [ ] 调研写法`char str_1[]{ "Hello !!, GeeksforGeeks" };`, `char str{ "Muddy" };`

* 异或操作的特性：

    交换律：hash(a) ^ hash(b) = hash(b) ^ hash(a)

    自反性：a ^ a = 0

    这可能不是最佳组合方式，因为 (a,b) 和 (b,a) 会产生相同哈希值

    更好的方法：

    ```cpp
    struct VertexPtrHash {
        size_t operator()(const pair<Vertex*, Vertex*>& src_dst) const {
            // 使用旋转和组合减少碰撞
            size_t h1 = std::hash<void*>()(src_dst.first);
            size_t h2 = std::hash<void*>()(src_dst.second);
            return h1 ^ (h2 << 1) ^ (h2 >> (sizeof(size_t)*8 - 1));
        }
    };
    ```

    或者直接使用 boost 库：

    ```cpp
    struct VertexPtrHash {
        size_t operator()(const pair<Vertex*, Vertex*>& src_dst) const {
            size_t h1 = std::hash<void*>()(src_dst.first);
            size_t h2 = std::hash<void*>()(src_dst.second);
            // 使用 boost::hash_combine 类似的方法
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
    ```

* 写递归还是有点难度，可以先将一个完整结构分成 2 部分或 3 部分，然后分别写出 2 部分或 3 部分的处理方式，比如第 2 部分需要遍历，第 3 部分需要给出当前的 position 等。如果在第 2 部分需要遍历，那么判断下是先序遍历还是后序遍历（先遍历完子节点再处理当前节点）。

    做题时候的后序遍历，通常是使用 int 来返回一个值。实际项目中，如果遍历子节点时需要用到 parent 信息，而且又需要返回一些处理完后的信息，那么函数的设计就比较复杂了。这个时候究竟应该以 parent 的角度遍历子节点，还是在递归的开头直接处理当前节点，如果遇到空节点则返回？这个问题有时间了可以讨论下。

* string view learning material:

    <https://how.dev/answers/what-is-the-cpp-string-view-in-cpp-17>

* `(void) getHostName(hostHash, sizeof(hostHash), '\0');`

    前面这个`(void)`是干嘛用的？

* 调研`snprintf(node->attrs[index].value, MAX_STR_LEN, "%#lx", value);`中`%#`的用法

* 以后再做 c++ 的 qa，要么写一点点语法，要么就是做题

* 找一个 c++ 学习资料，系统学一下比较好

* 存储 image 可以有两种方式

	一种是 continuous，即开辟一块连续的内存，每次二维索引都转换成一维。

	另外一种是 separate，即使用`vector<vector<float>>`来存储。这样可以直接索引，其实性能也差不到哪。
	
	这样的话维度其实是固定死的，好在大多数图像都是二维的。如果使用`vector<vector<vector<float>>>`，或者`vector<vector<vector<vector<float>>>>`来存储三维或四维的图片或者数组，会不会损失性能，或者对内存造成浪费？

	默认就以 seperate 形式存储吧，将 2d 图片，3d 图片存储成不同类型的。

	存储图像时，可以使用 int 也可以使用 float。

* 指针的指针，区分指针的指针所指的内容不能被修改，指针所指的内容不能被修改（指针的指针本身不能被修改），指针本身不能被修改

    分别如何 new, malloc

* modern c++

    <https://github.com/federico-busato/Modern-CPP-Programming>


* 学习一下 c++ make uniquue

    可以自定义一个类型，然后使用 make unique 创建一个固定长度的数组。

    ```cpp
    // c++14
    auto constantContainer = std::make_unique<YourType []> ( size );

    // c++11
    std::unique_ptr<YourType[]> constantContainer {new YourType[ size ]};


    // Access
    constantContainer[ i ]
    ```

    有空了研究一下这段代码，分析一下利弊。

* `fprintf()`向 stdin 中写数据，似乎没有什么反应。

### tasks

* [new] `perror`

* [new] `std::expected`

    std::expected (C++23)

    ```cpp
    #include <expected>

    std::expected<int, std::string> safe_divide(int a, int b) {
        if (b == 0) {
            return std::unexpected("除零错误");
        }
        return a / b;
    }

    // 使用
    auto result = safe_divide(10, 0);
    if (!result) {
        std::cout << "错误: " << result.error() << "\n";
    } else {
        std::cout << "结果: " << *result << "\n";
    }
    ```

* [new] extern C 是什么含义？

    ```cpp
    extern "C" void c_function() {
        // 不能在这里抛出C++异常
    }
    ```

* [ ] 调研 c++ `inner_product`, `adjacent_difference`

* [ ] 调研 c++ `reduce`, `ranges::fold_left`

* [ ] c 语言中`#defile`中`##`的用法？

* [ ] 调研 c++ 的 enumerate

* [ ] C++自定义哈希函数解析与优化

    目前看到

    > 5. std::function（C++11起）

* [ ] 调研 c++ 中成员函数的指针和普通函数的指针有何不同。

* [ ] `std::get_if<T>()`

* [ ] 调研`std::holds_alternative`

* [ ] 调研`std::get_if`, `std::get`

* [ ] 调研 c++ 泛型 lambda

* [P] 调研`std::visit`

* [ ] 为什么`vector<unique_ptr<XmlTag>> root_tags;`无法`root_tags.push_back(new XmlTag);`，但是可以`root_tags.emplace_back(new XmlTag);`？

* [ ] 调研 c++ 20 的 format

* [ ] 调研 elements 的 Design Aspects

        <https://cycfi.github.io/elements/elements/aspects.html>

* [ ] 调研 elements 的 layout

        <http://cycfi.github.io/elements/elements/layout.html>

* [ ] 调研 elements 的代码

* [ ] 调研：在 .h 文件里定义 edge type id to str 是否会引起多重定义的问题？

* [v] 调研`fprintf(stderr," Internal error, existing.\n");`的用法

* [ ] 调研`partial_sum()`和`accumulate()`有什么区别？

* [ ] 调研 c++ string 从指定位置开始 assign 另一个字符串

* [ ] 调研 c++ thread 如何拿到线程函数的返回值？

* [ ] 调研：c++ `lower_bound()`, `upper_bound()`作用
    
    以及如果找不到元素或元素在数组边界时的情况

* [ ] 调研：c++ 迭代器为什么`upper_bound()`的返回值减去`lower_bound()`的返回值等于数组的长度？

* [ ] 调研 c++ 迭代器，increase 相关

## Vim

本项目的目标是学完 vim，可以使用 vim 代替 vscode 进行日常的代码开发和调试，以及文本文档的编辑。

* vim 中的 redo 是哪个？

### cache

* [new] 调研 vim 命令

    ```vim
    :10,20w /tmp/copy.txt  # 将10-20行保存到临时文件
    ```

    ```vim
    :'<,'>w !nc localhost 12345  # 将选中内容发送到本地
    ```

### tasks

* [new] 调研 vim 设置

    `:set term=ansi`, `:set term=xterm`, `:set term=putty`, `set term=xterm-256color`

    ```vim
    set nocompatible
    set backspace=indent,eol,start
    set term=xterm
    ```

    ```vim
    :set term=builtin_ansi
    :set term=builtin_xterm
    ```

* [new] 调研 windows 下的包管理器，以安装 vim 为例

    ```powershell
    # 在远程机器上安装 vim 的完整版本
    choco install vim  # 如果使用 Chocolatey
    # 或
    scoop install vim  # 如果使用 Scoop
    ```

* [new] 调研 vim 中 <A-xxx> 代表 alt 快捷键吗？

    `nnoremap <A-v> <C-v>`

* [new] 调研 vim 剪贴板管理插件

    ```vim
    Plug 'svermeulen/vim-easyclip'
    ```

* [new] 调研 vim 命令

    标记要选中的行（例如第1、3、5行）：

    `:1,5g/^/if line('.')%2==1|exe "normal! mV"|endif`

* [new] 调研 vim 命令

    `:for i in range(1, line('$'), 2) | call setpos('.', [0, i, 1, 0]) | normal V | endfor`

* [new] 调研 vim 实现标记多个行，然后调用宏或函数

    主要思路是，normal 模式下，使用快捷键，比如 <leader> + L （未检测这个是否与已有快捷键冲突）标记行，这些行号记录在脚本变量中，然后使用 <leader> + C 准备好调用，用户输入调用指令（比如 @x 代表调用宏， func_name 表示调用函数）后，对刚才保存在脚本中的行应用这些调用。调用完成后，清空脚本变量。

    可能的问题：
    
    假如脚本中涉及到了行的增删，记录的行号会变动，如何解决？

* [v] 调研 vim `?`搜索

    `?{           " 搜索上一个 {`

    feedback:

    * [ ] `?\<vim\>   " 精确匹配单词"vim"`

    * [ ] 搜索后可使用 cgn 进行替换操作

    * [ ] 结合 q/ 或 q? 查看和编辑搜索历史

    * [ ] 搜索历史可通过 :history / 查看

* [v] vim `noremap!`

* [ ] vim `s`命令：替换并插入

* [asso] 调研 vim 命令

    ```vim
    " 修复终端退格键映射
    set backspace=indent,eol,start
    ```

* [asso] vim `set smarttab`

    功能是否为 让退格键一次删除 4 个空格？

    如果是，那么这个功能是否和 softtabstop 冲突了？

* [asso] vim :normal 命令

    ```vim
    :normal i 
    ```

    （注意最后的空格）

* [asso] vim 的启动参数

    `vim -n "set nobackup nowritebackup noundofile" file.txt`

* [ ] 调研 vim-unimpaired

    提供 [<Space> 和 ]<Space> 分别在上方/下方插入空行

* [ ] 调研 vim-addon-mw-utils

* [ ] 调研 vim 一次性添加多行

    ```vim
    " 添加3个空行在下方
    :put =repeat([\"\"], 3)<CR>

    " 或使用函数
    function! AddLinesBelow(count)
        for i in range(a:count)
            put! ''
        endfor
    endfunction
    :call AddLinesBelow(3)
    ```

* [ ] 调研 vim 添加行的映射

    ```vim
    " 快速插入空行（保持普通模式）
    nnoremap [<Space> :<C-u>put! =repeat([''], v:count1)<CR>
    nnoremap ]<Space> :<C-u>put =repeat([''], v:count1)<CR>

    " 或者更简单的映射
    nnoremap <silent> go :<C-u>put =''<CR>
    nnoremap <silent> gO :<C-u>put! =''<CR>
    ```

    这样你就可以：

    * `3]<Space>` - 在下方插入3个空行

    * `2[<Space>` - 在上方插入2个空行

    * `go` - 在下方插入一个空行

    * `gO` - 在上方插入一个空行

* [ ] 调研 vim 用法

    ```vim
    nnoremap ]w :call search('^\s*\S', 'W')<CR>
    ```

* [asso] 调研插件`vim-sneak'`

    ```vim
    " 使用插件如 vim-sneak
    Plug 'justinmk/vim-sneak'

    " 搜索上一个 {
    s{  
    ```

* [asso] 调研下面这段 vim 代码

    ```vim
    " 使用 coc.nvim 或其他 LSP 的函数跳转
    nnoremap [[ :call CocAction('jumpDefinition', 'prev')<CR>
    ```

* [ ] 调研 vim `gd / gD 跳转到局部/全局定义`跳转

* [ ] 调研 vim masks

    ```vim
    ma                   " 在当前函数打标记 a
    // 光标移到下面
    'a                   " 跳回标记 a
    ```

* [new] 调研 vim 测试文件

    ```bash
    # 创建一个干净的测试环境配置
    cat > ~/vim_plugin_test.vim << 'EOF'
    " 最小化配置，只保留必要选项
    set nocompatible
    filetype off

    " 设置你的 runtimepath
    set rtp=~/.vim,/usr/share/vim/vimfiles,/usr/share/vim/vim82,/usr/share/vim/vimfiles/after,~/.vim/after

    " 在这里加载你要测试的插件
    set rtp+=~/my_vim_plugin

    " 启用文件类型检测
    filetype plugin indent on
    syntax enable

    " 添加你的调试配置
    let g:my_plugin_debug = 1
    EOF
    ```

* [new] 调研 vim 调试模式

    ```vim
    # 启动时启用详细日志
    vim -V9vimlog -u test_vimrc.vim

    # 或者在 Vim 中开启
    :set verbose=9
    :set verbosefile=/tmp/vim.log
    ```

* [new] 调研 vim 临时加载插件

    ```vim
    " 在测试配置文件中这样写：
    function! LoadPluginForTest()
        " 临时修改 runtimepath
        set rtp+=~/projects/my_vim_plugin
        
        " 加载插件
        runtime plugin/myplugin.vim
        
        " 设置测试用的全局变量
        let g:myplugin_test_mode = 1
    endfunction

    call LoadPluginForTest()
    ```

* [new] 调研 vim -c 的用法

    ```bash
    # 一次性命令测试
    vim -u NONE -c "source myplugin.vim" -c "echo MyPluginFunction()" -c "qa"

    # 交互式测试
    vim -u ~/test_vimrc.vim +"MyPluginCommand"
    ```

* [new] 调研 vim Vader

    ```bash
    # 安装 Vader.vim
    mkdir -p ~/vim_test_env/pack/test/start
    cd ~/vim_test_env/pack/test/start
    git clone https://github.com/junegunn/vader.vim

    # 运行测试
    vim -u test.vim -c "Vader test/*.vader" -c "qa"
    ```

* [new] vim `execute`

* [new] 调研 vim 命令

    `filetype plugin indent on`

    `filetype on`

    `set term?`: 检查颜色支持. 如果显示 term=builtin_gui 或 term=win32 是正常的。如果是 term=dumb，可能需要调整终端设置。

    `set filetype?`: 查看当前文件类型

    `:version  " 查看 Vim 版本和功能`

    `:scriptnames  " 查看加载的脚本`

    ```vim
    :colorscheme default  " 设置默认配色方案
    :syntax enable
    ```

    `set encoding=utf-8`

## urls

### cache

### tasks

#### processing

* { } 旅行商问题（枚举，回溯，动态规划，贪心，分支界限）

    <https://blog.csdn.net/m0_64372178/article/details/134541003>

    feedback:

    1. 2025.08.03 目前看到

        > 暴力枚举
        > 
        > 使用 dfs 枚举每一个点，不适用剪枝的话就是暴力枚举法

#### tracking

## Network / Proxy

### cache

* 多级代理

    多级代理跳转是指数据包通过多个代理服务器依次转发，形成一条代理链（Proxy Chain）。每个代理只知道它的前一个节点和后一个节点，不知道完整路径。

    原理：

    以 socks 代理为例，socks 协议支持发送两种包，一种是控制包，用于建立连接，另一种是数据包。多级代理的实现细节如下：

    socks 的 CONNECT 命令中，有一段数据如下：

    ```
    +----+-----+-------+------+----------+----------+
    |VER | CMD |  RSV  | ATYP | DST.ADDR | DST.PORT |
    +----+-----+-------+------+----------+----------+
    | 1  |  1  | X'00' |  1   | Variable |    2     |
    +----+-----+-------+------+----------+----------+
    ```

    此即控制包，其中 `DST.ADDR` 和 `DST.PORT` 可以是任意目标，包括另一个代理服务器！

    1. 客户端向代理 1 发送 socks 控制包

        ```
        [客户端端口] -> [代理1:1080]
        数据包内容:
            SOCKS5版本: 0x05
            认证方法数: 0x01  
            认证方法: 0x00 (无认证)
            命令: 0x01 (CONNECT)
            保留: 0x00
            地址类型: 0x01 (IPv4)
            目标地址: 代理2的IP
            目标端口: 1080
        ```

    2. 代理 1 向客户端发送 socks ack 包

        ```
        [代理1:1080] -> [客户端端口]
            SOCKS5版本: 0x05
            回复: 0x00 (成功)
            保留: 0x00
            地址类型: 0x01
            绑定地址: 代理1的IP
            绑定端口: 新分配的端口
        ```

        此时客户端和代理 1 建立起一个 socket 连接，客户端以后从这个 socket 发数据，都是直接发 payload （实际发出的信息可能会经过加密），不再有任何代理相关的信息。

    3. 客户端向代理 2 发送 socks connect 包

    4. 此时代理 1 看到数据包是个控制包，读取其中的 DST IP （代理 2）及 DST PORT，建立一个 TCP 连接，然后告诉客户端此包已成功处理

        此时整个代理的结构如下：

        ```
        客户端:12345 → 代理1:1081（进程）
                          ↓ (进程内socket对)
                   → 代理1:12346 → 代理2:1082
        ```

    5. 按照上面的步骤建立代理 2 -> 代理 3 的 socks 连接

        此时客户端已经完成客户端 -> 代理 1 -> 代理 2 -> 代理 3 的建链，从此客户端发送的数据不再包含 socks connect header，而是只有 payload 信息，比如 http 包，或 tcp 包

    6. 客户端直接在与代理 1 连接的 socket 中发送 http payload (假设要访问 baidu.com)

    7. payload 传递到代理 3 处，代理 3 向 baidu.com 的 443 端口建立 TCP 连接

        代理 3 读取代理 2 发过来的请求：`SOCKS5 CONNECT example.com:443`，建立起最后一个 TCP 连接。

        至此，整个代理链路完成，客户端与网站间都是只考虑传输 payload 就可以，中间的所有代理过程都是透明的。

    整个代理过程中，存在两种建立TCP连接的方式，一种是读取SOCKS CONNECT控制报文，另一种是payload的HTTP或者TCP报文。

    在所有的代理连接建立初期，上游必须给下游发送 socks connect 控制报文，告诉下游上游所充当的角色，之后上游的角色不再更改。因此整个角色的确立过程是静态的，不能在传输途中动态改变。这样可以使后续的所有传输都是纯 payload，提高传输效率。

* 代理协议：

    Proxy Protocol：传递原始客户端信息

    HTTP/2 CONNECT：更高效的隧道协议

    QUIC over Proxy：基于UDP的多路复用

    混淆技术：

        TLS包装代理流量

        流量整形避免特征识别

        随机填充数据包

## git

### cache

* [new] `GIT_SSH`, `SSH_AUTH_SOCK`

* [new] `GIT_SSH_COMMAND`

    ```bash
    # 使用 -v 参数查看详细的 SSH 连接过程
    GIT_SSH_COMMAND="ssh -v" git clone web@ali:hd-dot-web
    ```

    ```bash
    # 查看 Git 的 SSH 配置
    git config --global --get core.sshCommand

    # 如果需要，显式设置 SSH 命令
    git config --global core.sshCommand "ssh -o IdentitiesOnly=yes -i ~/.ssh/id_rsa"
    ```

    ```bash
    # 方法3：指定 SSH 密钥
    GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa" git clone web@ali:hd-dot-web
    ```

* [new] `git submodule update --remote --recursive`

* [new] `git config --local`

    `git config --local --remove-section submodule.src/collectives/device/utils 2>/dev/null || true`

* [new] `git pull --recurse-submodules`

* [new] `git ls-tree`

    `git ls-tree HEAD src/collectives/device/utils`

* [ ] 调研`git revert -n <commitToRevet>`, `git revert --no-commit <commitToRevet>`, `git revert HEAD~x`

* [ ] 调研`git cherry-pick`

* [ ] 调研`git reflog`

* [ ] 调研`git rebase --onto`

* [ ] 调研`git merge --no-ff`

### tasks

* [ ] 调研`git bisect`

* [ ] 调研`git reset --soft`

* [ ] 调研`git merge --squash`

* [ ] 调研 git

    这个网站<https://www.atlassian.com/git/glossary#terminology>上关于 git 的资料还挺多的，有时间了看看

* [new] `git clone --filter=blob:none --no-checkout <repo-url>`

* [new] `git fetch --deepen 5  # 再获取5个历史提交`

* [new] `git checkout --orphan new-history  # 创建无父提交的新分支`

* [new] `git fetch origin v2.0 --depth 1`

    这里是不是少写了个`tag`？

* [new] 调研这段代码的含义

    ```bash
    git checkout --orphan compressed-history
    for commit in $(git log --reverse --format="%H" --all); do
        git checkout $commit -- .
        git add -A
        git commit -m "版本 $(git describe --tags $commit 2>/dev/null || echo $commit)"
    done
    ```

* [ ] `git log main..origin/main`

    ```bash
    git fetch origin
    git log main..origin/main --oneline  # 查看远程有哪些新提交
    ```

* [v] 调研`git merge v1.0 v2.0 v3.0`

* [v] 使用 git remote -v 查看配置的远程仓库

* [v] 调研`git fetch origin tag v1.0`

    * git fetch origin 获取最新远程分支信息

* [ ] 增加 git 的 qa

## java

### tasks

* [ ] java 找 package 去哪里找？

* [ ] java 是否有 package manager?

* [ ] java 中多线程中的 mutex 与 semaphore 如何使用？

* [ ] java 如何进行 socket 编程？

* [ ] java 是否有 gui 的库？

## 其他 Miscellaneous

* [new] `\mbox{}`

* [new] 模运算一般用 \bmod 或 \pmod{}

    ```latex
    \[
    a \bmod b, \quad n \equiv 1 \pmod{2}
    \]
    ```

* [new] 调研 imagemagick 

    `convert -density 150 input.pdf -set filename:f "%t" +adjoin "pdf_pages/%[filename:f]-%03d.png"`

* [new] 调研 parallel 用法

    `parallel 'convert {} -resize 1024x768 web_images/{.}.webp' ::: *.jpg`

* [new] 调研`mutool`(mupdf工具)

* [new] 调研`pdf2image`(Python库)

* 简述 Dask 和 apache spark 的用法

    Dask 和 Apache Spark 都是分布式计算框架，用于处理大规模数据，但设计理念和使用场景有所不同。以下是它们的用法简述：

    Dask

    特点

    * Python原生：与NumPy、Pandas、Scikit-learn等生态无缝集成。

    * 动态任务调度：适用于复杂、非线性的计算任务。

    * 灵活性：支持单机多核或分布式集群。

    核心用法

    * 并行化集合：

        * Dask Array：模拟NumPy，分块处理大型数组。

            ```python
            import dask.array as da
            x = da.random.random((10000, 10000), chunks=(1000, 1000))
            result = x.mean().compute()
            ```

        * Dask DataFrame：模拟Pandas，处理表格数据。

            ```python
            import dask.dataframe as dd
            df = dd.read_csv('large_file.csv')
            result = df.groupby('column').mean().compute()
            ```

        * 延迟执行（delayed）：

            ```python
            from dask import delayed
            @delayed
            def slow_function(x):
                return x * 2
            tasks = [slow_function(i) for i in range(10)]
            results = dask.compute(*tasks)
            ```

    机器学习（dask-ml）：

    ```python
    from dask_ml.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    ```

    部署集群：

    ```python
    from dask.distributed import Client
    client = Client('scheduler-address:8786')  # 连接集群
    ```

    Apache Spark

    特点

    * JVM生态：基于Scala，支持Java、Python、R。

    * 内存计算：通过RDD/DataFrame实现高性能迭代计算。

    * 全栈式：支持SQL、流处理、机器学习、图计算。

    核心用法

    * 初始化SparkSession：

        ```python
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("example").getOrCreate()
        ```

    * 数据处理：

        * RDD（弹性分布式数据集）：

        ```python
        rdd = spark.sparkContext.parallelize([1, 2, 3])
        rdd.map(lambda x: x * 2).collect()
        ```

    * DataFrame（主要API）：

        ```python
        df = spark.read.csv('large_file.csv', header=True)
        df.groupBy('column').agg({'value': 'mean'}).show()
        ```

    * SQL 查询：

        ```python
        df.createOrReplaceTempView("table")
        spark.sql("SELECT * FROM table WHERE column > 0").show()
        ```

    * 流处理（Structured Streaming）：

        ```python
        stream_df = spark.readStream.format("kafka").option("subscribe", "topic").load()
        query = stream_df.writeStream.outputMode("append").start()
        ```

    * 机器学习（MLlib）：

        ```python
        from pyspark.ml.classification import LogisticRegression
        model = LogisticRegression().fit(train_df)
        ```

    对比总结

    | 维度 | Dask | Apache Spark |
    | - | - | - |
    | 语言生态 | Python原生 | JVM为主（支持Python/Java/Scala/R） |
    | 部署难度 | 较轻量，易上手 | 需配置集群（如YARN/K8s） |
    | 适用场景 | 中等数据、复杂工作流、Pandas扩展 | 超大数据、ETL流水线、全栈分析 |
    | 内存管理 | 依赖 Python GC | 自主内存优化（Tungsten引擎） |
    | 生态工具 | 集成 Python 科学栈 | Spark SQL/Streaming/MLlib/GraphX |

    选择建议

    * 若团队以Python为主，数据量在TB以下，且需要灵活的任务调度，选Dask。

    * 若处理PB级数据，需企业级稳定性和全栈分析（SQL/流/ML），选Spark。

* [new] 调研 python 命令

    ```bash
    # 查看 pip 安装包的位置
    python -m site
    # 或
    python -c "import sys; print(sys.path)"
    ```

* A Comprehensive Guide to Using pathlib in Python For File System Manipulation

    <https://www.datacamp.com/tutorial/comprehensive-tutorial-on-using-pathlib-in-python-for-file-system-manipulation>

* pathlib --- 面向对象的文件系统路径

    <https://docs.python.org/zh-cn/3.13/library/pathlib.html>


* 看起来，`ref_14`对应的问题是：对于一个数字 n，给定一系列数字`arr = [a, b, c, ...]`，现在从 arr 中选取 m 个数字，使得这 m 个数字的和为 n。现在需要使得 m 最小，如果无法找到 m 个数字和为 n，那么 m 取 -1。

* 计算机底层只有 01 真值表，逻辑判断，流程控制和存储空间，为什么可以发展出复杂的数据结构？

	比如树，图，等等。为什么简单的概念蕴藏了如此大的复杂性？

	假如一个内存只有 4 bits，我们几乎什么都干不了，不可能形成图，树。

	假如有 4KB，可以做一些简单的程序。

	假如有 4GB，那么就可以写很复杂的数据结构。

	在这中间，红黑树，quich sort 等算法，所需要的最小内存数是多大？

	也就是说，至少需要多少存储，才可能发展出更加复杂的概念？

* 先使用 python + numpy 实现一个自动求导，再使用 c++ 实现一个自动求导

	可能需要用到 c++ 版的 numpy

	最终的目标是实现一个类似 pytorch 的框架

* [ ] 调研 onnx 模型的格式，搭建一个小型全连接网络，尝试用 opencl 跑通。

* linaro 是 arm 发起的开源非盈利组织，主要做物联网方向

    可以在这个页面<https://www.linaro.org/downloads/>找 arm gcc 编译器的下载。

* [new] 调研分子动力学

    做什么的？roadmap 是什么？世界上有哪些知名机构/研究所/实验室？论文有哪些？有哪些常用软件/工具？是否有和生物/医学/化学的交叉研究？

    目前的难点在哪里，未来有哪些前进的方向？

* android connectbot 中，端口转发中断通常是因为省电策略

    后台程序的小锁是只代表不 kill，不代表不会 freeze。指定后台程序无省电策略限制，才是不会 freeze。

    一旦 freeze，ssh 的 tunnel 即关闭，必须重新连接 ssh 才能打开，但是 ssh session 本身不会退出，仍可正常发送命令。这样会造成迷惑。

* performance 的工具

    * `timeit_ms()`

        区间计时。

    * 需要热启动的框架

        `init_env()`, `timeit_ms_N()`, `exit_env()`

        calculate mean time consumption and std.

        warm up time and dextroy env time

    * 随机填充一个数组/矩阵

    * 标准参考答案

* 数据库书籍

    * 《数据库系统概念》（Database System Concepts）作者：Abraham Silberschatz, Henry F. Korth, S. Sudarshan。这本书是数据库领域的权威教材，适合初学者入门。

    * 《数据库管理系统》（Database Management Systems）作者：Raghu Ramakrishnan, Johannes Gehrke。它详细介绍了数据库系统的关键概念和技术，适合作为深入学习的教材。

    * 《数据库系统实现》（Database System Implementation）作者：Hector Garcia-Molina, Jeffrey D. Ullman, Jennifer Widom。如果你想了解数据库系统的内部工作原理，这本书非常合适。

    * 《数据库系统概论》（An Introduction to Database Systems）作者：C.J. Date。这是另一本经典的数据库教材，内容详尽。
    如果你是中文读者，《数据库原理及应用》作者：王珊、萨师煊，这本教材结合了理论与实践，通过实例讲解数据库的设计和开发过程。

* c++ 版本的 opencl 入门

    <https://objectcomputing.com/resources/publications/sett/july-2011-gpu-computing-with-opencl>

* libeigen

    other linger algebra libraries: <https://stackoverflow.com/questions/1380371/what-are-the-most-widely-used-c-vector-matrix-math-linear-algebra-libraries-a>

    以 libeigen 为主，其他的库对比着学下。

    * numcpp

        矩阵运算库

        <https://github.com/dpilger26/NumCpp>

        <https://zhuanlan.zhihu.com/p/341485401>

