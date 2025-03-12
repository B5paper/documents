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

* log 不应该与 task 放在同一个文件里，翻来翻去太费时间

## cache

* 如果一个任务被标记为`o`（有思路，但未完成），那么应该在任务池中将其修改为一个长期任务`{ }`。

* 如果有 new task，可以添加到当日的 task list 里，但是必须添加`[new]`标记

    比如：

    ```
    * [ ] task 1
    
    * [ ] task 2

    ...

    * [new] new task 1

    * [new] new task 2

    ...
    ```

    添加到当日的 task list，可以防止

* 交互式地阅读材料并不能解决效率逐渐降低的问题，只能缓解。如果想要解决，还是得靠真正的输出。

* dfs 停止的条件

    阅读材料时，每次开始认为是一个 block，每遇到一个无法解释的点，则记 uncomprehensive point +1，记录够 N 个后，则认为这个 block 到此已经无法再理解了。可以尝试从下面开始，或者往下跳几段/几章，开始一个新的 block。如果新的 block 也无法理解，则记录 uncomprehensive block +1，攒够 N_2 个后，则认为整个 material 是无法理解的。

* 对于当前节点，必须反复提问自己：上一个节点在哪里？我是从哪里来的？

* 概念重定义

    对于一个所有概念都清晰定义的系统，如果剔除某个概念 A 的定义，只保留其他概念对 A 的使用，那么根据这些使用的描述，我们能从什么程度上反推出 A 的定义？

    example:

    > 为了获得高带宽，shared Memory被分成32（对应warp中的thread）个相等大小的内存块，他们可以被同时访问。不同的CC版本，shared memory以不同的模式映射到不同的块（稍后详解）。如果warp访问shared Memory，对于每个bank只访问不多于一个内存地址，那么只需要一次内存传输就可以了，否则需要多次传输，因此会降低内存带宽的使用。

    上面是某个博客的一段文字，其中用到了 bank 的概念，但是并没有给出 bank 的定义。我们是否能仅通过类似这样描述性的文字，推测出 bank 的定义或含义？

* 遗忘点

    每次 qa 时，把遗忘点记录下来，再针对遗忘点进行巩固，或许效果会好些。

    这些遗忘点可以作为 hint，在回忆时，优先看 hint，如果看 hint 回忆不起来，再看 material。

* 关于自适应

    对于一个 token 序列，我们希望找到和它相关的其他 token，一个直观的想法是对 token 的集合进行遍历，计算 token 相和度，给定一个阈值然后筛选（或者直接排序），选出合适的 token。此时阈值成为了一个超参数，无法做到真正的自适应。但是现代的做法是直接使用两个矩阵相乘，计算出了任意两个 token 的相和度。

    从阈值控制到自适应，这两种框架的特点有什么不一样？如何把任意一个使用阈值的框架变成一种自适应的方法？

* 猜想：存在仅靠在当前水平下的猜想（构造假设空间）和假设推断无法彻底弄明白的复杂系统。

    假如复杂系统中有逻辑的简并，那么有很大概率无法搞明白它原本想表达的含义。

    假如这个猜想为真，那么我们在尝试解开复杂系统时，应该首先以明显的类比为手段，如果这个手段不好用，再尝试看看有没有明显的猜想，如果都没有，最后再尝试构建假设空间和猜想。

    如果这一套尝试都不奏效，那么 dfs 就到此停止。此时应该去看一看其他内容，增加信息源等等。

* 概念漫步

* 尽管现在做的努力都是尽量让所有非线性的行为变成线性，但是真当所有行为都变成线性时，又怀念非线性的失重感，尝试在线性之外找到非线性的领域

* 可解构与复杂性

    假如一个复杂的模块可以被解构成子模块，那么我们就认为它是简单的，是复杂性可降解的。但是还存在一些模块，无法被解构为子模块，当我们想处理这样的模块时，必须把这个模块相关的所有内容都加载到缓存中，这样的模块越不可解构，复杂性越高。

* 输入与输出

    输入的信息越多，就越乱。猜想：可以使用输出来平衡。可以是文字，也可以是绘画，音乐等。

* 断点的行数和实际断到的函数不对应，猜想可能的原因是模板函数在展开时行数无法完全对应

* 电脑前集中注意力的方法

    1. 解释看到的每一个词语/代码变量的意思，如果可以解释，那么解释一句话/一行代码是什么意思。如果又可以解释，那么尝试解释一段话/一段代码的含义以及为什么要写这段话/这段代码。

    2. 如果在解释的过程中遇到困难，那么尝试去解决问题

    3. 如果代码都可以解释，那么尝试复现代码，重写一遍

* 未知概念的非线性搜索范围

    一个初步的方案：假如概念 1 没有理解，可以暂时跳过，继续看后面的东西。假如看到了概念 2，概念 2 依赖于概念 1，概念 3 也依赖概念 1，但是概念 3 与概念 2 是独立的，那么可以借助概念 2 和 3 理解概念 1. 假如概念 4 依赖概念 2，而概念 2 还没弄明白，那么就应该停止了。

* 推理 nccl 有点像玩扫雷

* 非 cache 笔记（顺序笔记）的原则：假设自己对所有的概念都一无所知，又假设所有后续的笔记都依赖前面的笔记。

* 描述，猜想，问题与实验

    * 实验由对比，或推理加实验的方式组成，并且有结论，是最强的证据

    * 描述是充满可能性的表达

    * 猜想是一个无端的可能性

    * 问题是完全的不确定性

    探索的部分可以由这几部分记录。

* 如果一个任务连续出现 2 次，3 次，依然没有完成，那么就可以考虑把这个任务变成一个长期任务

* `{ }`类型的任务不应该另起一个`[ ]`任务，应该全部合并到`{ }`任务里

* 假设生成 -> 内部自洽 -> 外部验证

    如果内洽与外延都是正确的，那么假设空间可以认为是一个等价映射。

* 如果有$x_1$，$x_2$两个变量，是否优化目标为修改曲面在指定点处的高度？

* [ ] 调研 matplotlib 画 surface

* 对于已完成的 task，feedback 部分向下添加；对于未完成的 task，deps 部分向上添加

* 我们的对抗并不是谁把任务做完，谁投入时间长谁就能取胜的，而是大家都把当前工作做到精致的情况下，继续对未来的可能性进行探索、冒险、赌博和选择。

    按部就班的回答不会有奇迹。

* 非线性变化

    假如现在有一团热气流和冷气流相遇，如果在对气流一无所知的情况下去做仿真，大概率得到的结果是冷气流与热气流的温度逐渐趋于相同，锋面的温度变化最快，远端的温度变化最慢。我们大概率不能直接仿真出龙卷风，台风。

    假如我们提前知道了冷暖气流相遇有可能出现台风，那么大概率会修改我们的仿真算法，使结果中一定几率出现台风。

    对于云层也是一样，假如我们对云的知识一无所知，那么大概率只能仿真出一块一块的云，但是自然界有卷云、排成阵列的云、下击暴流的云，如果我们提前不知道这些概念，那么它们几乎不会在仿真里出现。

    假如我们称冷暖空气的温度变化、天空中的白云为线性变化，台风、阵列云就是非线性变化，那么问题是：非线性变化是什么带来的？我们在仿真时如何才能尽可能地捕捉到这种变化？

* 如果 doc 或者资料里只有一部分是能看懂的，其他的看不懂，那么在写笔记时只记录看懂的，做过实验验证的，不懂的另起新的调研 task／项目

* 假如感觉每天完成 task 困难，那么在 reorg 和 qa 后，只设立 1 个 task，这样总能完成吧。

* 关于 dl 的一些问题

    * 问题 1: 假如底层的卷积实现的是 filter，中层的卷积实现的也是 filter，那么是否可以认为卷积神经网络只是在匹配模式，并没有做出创新的能力？

    * 问题 2：假如单靠 filter 就可以实现思维，那么 filter 的数量，size 与 depth 是否存在一个最小的必须值？

    * 问题 3：假如思维的底层确实是 filter，那么是否存在区别于 SGD 的优化方法？因为大脑不做反向传播。

    * 问题 4：创新是否是可以分解的，是否也可以被 filter 模仿？

* incontext learning 与复杂系统的猜想

    猜想：假如大模型可以靠 inference 完成复杂问题的推理，那么说明大模型在训练时不光学到了指定词组/句子后的可能模式，还学到了不同知识节点之间的复杂相互作用关系。

    验证：假如只保留复杂关系，对前端表示做替换，那么不影响最终的输出结果。

* 有关化学物质帮助度过临时的困境

    虽然说奶茶和咖啡只是临时兴奋神经，假如有一天状态非常不好，喝了咖啡后可以有精神地干完一件事，这件事后来会影响到人生的进程，那么是否可以认为“治标”和“治本”同样重要？

    命题：治本很重要，治标则填平了生活中每天的沟壑，也是非常重要的。

* 想法：用摄像头判断自己在执行一项 tasks 时拿起手机的次数

* 非线性项目

    如果完成一个项目用到的知识无法全部从 qa 和 note 中得到，并且这个项目有时间约束，那么这个项目就是一个非线性项目。

* 必须增加时间的约束

    不可能只靠流程安排得到最优效率，因为假如一件任务的使用 30 分能完成 70%，60 分能完成 90%，我们可能更想要 70% 的完成度。除非持续有新的发现，随着时间增长，成果线性增长。

* 时间的进退

    增加时间的约束后，并不是简单地增加了一个条件，而是整体的规划都会参照时间因素进行优化。这些优化肯定有些与流程的优化相冲突。如果遇到了冲突无法调和的矛盾，那么就需要两次将时间的因素撤出，等系统平稳后，再将时间加入。

* 两种稳定：基于循环反馈的稳定，基于多吸引子的稳定

* 一些想法：

    * 一个会跳舞的机器人 -> 一个机器人游乐园/农场

    * 一个机械肾脏科技公司

    * 一家动画公司，探索情感的控制与释放

    * 一个人文科学研究所

* 灵敏度分析与 partial fix，filter 重塑

    分组对 parameter 进行灵敏度测试，观察 acc，找到最不灵敏的一组 param，此时 fix 其他 param，将这组 param re-init 重置，并开始训练。反复重复这个过程，直到模型整体的 acc 不再变化。

* 挪瓦咖啡的加浓生椰拿铁太苦了

* 数学在离散算法领域并不总是能帮上忙的，比如各种智能优化算法，神经网络算法

* ai 的一个需要解决的问题是内部认识的协调性，我们的 target 可能并不总是在外部

* 如果一个任务没有指定具体的类型，那么就默认认为它是个调研任务

* 在开始一项任务前我们真的需要最佳状态吗？如果最小状态可以满足，那么就认为任务可以开始。

* [ ] 调研 v2ray 按域名选择流量出口

* [ ] 调研 virtual box 的虚拟机如何使用两块显示器／如何开两个 display 窗口／如何登录另一个 linux 窗口

* 调研写一个计时 ＋ 闹钟工具

* 如果使用 https 克隆 github repo 失败，可以试试 ssh

* 什么时候需要开一个新函数？

    当变量名混乱有冲突，不好起新名字的时候可以尝试开一个新函数。

* 对于一项任务，如果我们可以想象模拟出执行它的全过程，并可以预测所有可能的结果，那么它就不是一个“调研”

    比如“记忆”，就不是一个调研；随机检测 qa 也不是一个调研。

* [ ] 调研 meson, ninja

* 多线程调试时锁定单线程

    GDB scheduler-locking 命令详解

    <https://www.cnblogs.com/pugang/p/7698772.html>

* [v] c 语言中 static 全局变量和不加 static 的全局变量有什么不同？

* 笔记与中间结果

    有些需要记录的内容明显是中间结果而不是笔记，如果把中间结果当成笔记来记，那么在归类的时候就不知道该把归类到什么地方去。

    中间过程只是记录，不是结论，因此很难复用。这些不应该出现在笔记里。

* [ ] 调研 docker 中 app 的调试方法

* [ ] 调研 makefile 的 submodule

* [ ] 调研 diff 命令的用法

* [ ] 调研`fprintf(stderr," Internal error, existing.\n");`的用法

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

* [ ] ln 是否能创建文件夹的 hard link?

* [ ] 调研`ssh-add`，`ssh-agent`的作用

* 晚上吃饭不要吃到撑，不然会特别困，几乎没有精力继续学习。吃个半饱就可以了。

* 任务完不成应该分两种情况处理，一种是有极大可能完成，只不过时间不够，另一种是还在调研搜集信息阶段，不清楚是否能完。显然这两种情况的处理方式应该是不同的。

* 在准备去执行一项任务时，不应该考虑当前状态是否为最佳，而应该考虑当前状态是否满足最低要求

* 需要一个 graph 工具，建立不同的东西之间的连接

    stack 工具只适合任务的 trace

* 一个比较好的 explore 的想法是先从 amazon 上搜索书籍，然后在 zlib 或 libgen 上下载

* 笔记的结构

    先记录单个独立主题，再记录 topic，topic 中是多个独立主题的组合

* 如果学新概念/知识时，所有的新概念都可以从已知的知识轻松推导出来，那么就称这种学习过程为线性学习

    如果在学习一块新知识时，新知识中的一部分或全部无法通过已知概念推导出来，那么就称这种学习过程为非线性学习

* 非线性学习的一些方法

    * 总会有一些东西是可以一眼看懂的（从旧知识中推导出来），对这些一眼可以看懂的知识进行条目化总结。

        sync 这些新概念，直到可以用它们解释其他的新知识

    * 猜想-验证，对于无法理解的概念，先给出自己的一个猜想的解释，然后做出一些预测，再去验证，最后修正自己的猜想

        难点在于，有时候需要同时对大量的新概念提出猜想，变量过多，不容易修正自己的猜想。

    * 孤岛信息的连结

        如果有一段一两句话的知识点，虽然看不懂，但是可能在新知识体系中有用，不需要理解，但是需要知道它出现过，以后可能用得到，这种孤岛信息可以选择性地收集起来，以备后面使用。

* `http://security.ubuntu.com/ubuntu/ jammy-security restricted multiverse universe main`的 ip 为`1.1.1.3`，属于 cloudflare 的机器，国内不一定能访问到。

    如果在`apt update`时无法访问这个 ip 的 80 端口，可以考虑在`/etc/apt/source.list`里把这一行注释掉。

* 笔记的结构

    先记录单个独立主题，再记录 topic，topic 中是多个独立主题的组合

* 如果学新概念/知识时，所有的新概念都可以从已知的知识轻松推导出来，那么就称这种学习过程为线性学习

    如果在学习一块新知识时，新知识中的一部分或全部无法通过已知概念推导出来，那么就称这种学习过程为非线性学习

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

* 调研 Computer algebra system

    <https://en.wikipedia.org/wiki/Computer_algebra_system#>

    自动求导、符号求导等相关知识可能和这个概念有关。

* 猜想：如果一个文件夹中已经有内容，那么使用`mount`, `mount -t nfs`, `sshfs`挂载设备或远程目录时，不会删除文件夹下的内容，而是暂时覆盖文件夹下的内容

* 当我们说一个任务无法完成时，意味着我们必须要完成其他前置任务，依赖任务

* [ ] 有时间了调研一下`https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.lib_ref/topic/a/asprintf.html`，这好像是个 c api 的文档

* 如何评价 2024 年阿里全球数学竞赛决赛试题？难度怎么样？

    <https://www.zhihu.com/question/659607083>

* 香山 riscv 源代码 repo

    <https://gitee.com/OpenXiangShan/XiangShan/>

* 如何为一个用户增加指定目录的各种权限？

* [ ] 调研 git reset 查看不同版本的 linux kernel version

* 节点的独立分解

    1. 使用多种语义场景对节点的含义坐标进行定位

        <div>
        <img src='../../Reference_resources/ref_24/pics/pic_1.png'>
        </div>

    2. 调节单节点的值，保证在调整的过程中其他节点不受影响

        <div>
        <img src='../../Reference_resources/ref_24/pics/pic_2.png'>
        </div>

    3. 复杂网络的映射

        <div>
        <img src='../../Reference_resources/ref_24/pics/pic_3.png'>
        </div>

* [ ] 调研 u 盘里的资料

* performance 的工具

    * `timeit_ms()`

        区间计时。

    * 需要热启动的框架

        `init_env()`, `timeit_ms_N()`, `exit_env()`

        calculate mean time consumption and std.

        warm up time and dextroy env time

    * 随机填充一个数组/矩阵

    * 标准参考答案

* 不要忽略题目的暴力写法

    暴力写法是基础。

    在写出暴力写法的基础上，再去分析改进。这样的技能才是通用的。

* dfs 学习的两个常用方法

    1. 栈式（链式）问题回溯

    2. 猜想，假设，跳跃

* 调研

    `create_workqueue`, `inb`, 
    
    `INIT_WORK`, `DECLARE_WORK`, `struct work_struct task;`, `queue_work`

* 经常出现 terminal 没有登录虚拟机 ssh 的情况

    有没有什么办法可以避免这个问题？

* to complete:

    1. modern c++, rewrite the ray tracing program

    3. performance analyzing

    4. vulkan compute shader

* qa 频率可以降低到每周一次

* ibus 中的`mod`键是 alt 键

* 看手机的时间定在完成一项任务后，或者至少离上次看手机过去 xxx 分钟后

    一旦任务启动就不能再轻易看手机。可以写一个程序控制一下。

    保持在执行任务时的注意力。

* cached tasK: 写一个接收 cgtn 广播的 python 程序，实现断线自动重连

* 基于 dfs 的学习法，每次遇到无法理解的问题就卡在那里，浪费很多时间。或者是都能理解，但是非常非常不常用，进入到陌生的领域。或者说虽然有用，但是需要做实验验证才能完全理解，但是这时没有完善的实验条件。或者虽然是有用的知识，并且可以做实验，但是与目标任务无关。显然基于 dfs 的学习法是有问题的，我们不可能把一本书/一套视频/一个网站全部看完才去看另一个信息源。

    基于 bfs 的学习法才是正确的方法，遇到问题就停下来，看看其他地方，每次都只看自己可以看懂的，有用的，螺旋积累知识。

    虽然这么说，但是 dfs 仍然是必要的，难以理解的东西可能稍微往后看一点就能理解了，或者复杂的东西稍微推导一下就看明白了。

    这两者的平衡目前还是需要靠直觉去把控。

* 笔记中想法、批注的更新该如何表示？

    必须要保留旧的想法记录，因为这样可以形成一串思考线索，不然只能看到最终的思考结果。

    2024/05/07/00: 有两个形式可以参考，一个是邮件列表 forward，另一个是贴吧。目前先采用贴吧的形式，使用日期加冒号作为新一个批注的开始，每次都只追加，不修改，不删除。如果遇到必须解决的问题再修改这个形式。同一天的不同想法使用最后两位数字进行区分。

* 有一些知识、猜想、推理是正确但暂未理解的，即发现它只在一个陌生的上下文里正确，但是暂时或永远无法对到已有的知识体系上

    这种知识是否应该写入到笔记中？

* 如果一段资料，有 50% 和当前任务相关，另外 50% 和当前任务无关，但是也是还算有用的知识，那么这另外的 50% 是否应该继续学下去

    2024/05/07/00: 不应该学，应该缓存起来。然后让缓存资料留一个随机检测的通道，隔段时间就再来看看这段知识是否是必要的。如果缓存的难度（比如视频，纸质书籍）比较大，该怎么办？

* 无法否认，对于需要频繁交互才能继续下去的任务，目前仍然是主要靠直觉和“灵机一动”推动任务进行，并没有什么其他比较好的方式

* 将笔记未经验证的部分作为 cache 抽离出来也是非常重要的

* 一个 resource/reference 每次只能提取一点知识，未来可能需要多次提取。是否该使用一个进度标识来标记一个知识来源被处理了多少？

* 这个网站<https://www.atlassian.com/git/glossary#terminology>上关于 git 的资料还挺多的，有时间了看看

* [ ] 调研`git bisect`

* sync 只提出了标准，但是没有提出该如何 sync

    一个可能的 sync 方案：

    1. 先过一遍 qa，如果完不成 unit，那么找 dep，划分出可以完成的 unit，看了答案后可以完成的 unit，以及看了答案也无法完成的 unit

    2. 再过一遍笔记，尝试提取刚才没见过的 qa unit

        将笔记分为两部分，一部分是可以被验证的，另一部分是未被验证的

    3. 清理 test 和 qa 文件夹

    4. 将 qa 中过于复杂的拆分成 dep，无法理解的以及 empty 的都删掉

    5. 将笔记中未被验证的部分作为 cache，等待接受调研

    至此，sync 工作完成。

* linaro 是 arm 发起的开源非盈利组织，主要做物联网方向

    可以在这个页面<https://www.linaro.org/downloads/>找 arm gcc 编译器的下载。

* 对材料的思考：名词解释，名词联想（在图中，联想一个节点相连接的其他节点）

    只机械地整理资料不思考，就不会消耗脑力。只有消耗脑力才能学到东西。这里提供一些消耗脑力的常见问题，需要时可以尝试应用。

* [ ] 调研 compute shader

* [ ] java 找 package 去哪里找？

* [ ] java 是否有 package manager?

* [ ] java 中多线程中的 mutex 与 semaphore 如何使用？

* [ ] java 如何进行 socket 编程？

* [ ] java 是否有 gui 的库？

* [ ] 调研多 gpu inference

    把不同的 layer 放到不同的 gpu 上，使用一个 layer 的 output 作为另一个 layer 的 input。

* 相信自己的判断 (2024.04.19)

* 如何将一个子线程的 stdout 重定向到父程序的指定缓冲区？

* 如果一个调研任务没有完成，该怎么办？

* [ ] clear cmake_test project

* 需要频繁交换数据的两个对象必须离得近，无论是在物理上还是在逻辑上

* 与其说 nv 在搞 AI，不如说 nv 发现了一种新的计算范式，一种处理计算的新思路，而这个并行计算范式恰好符合了 AI 的需求

    是否还有其他的计算范式未被发现呢？

* connection 非常重要

    必须有一些必做的任务，可以通向以前的问题，以前的缓存，以前的记忆，以前的工程。

    必须有一条路径通向过去。

    目前这些必做的任务就是 reorg 和 qa。

* 串行的任务执行对提高效率非常重要，因为大脑频繁切换任务会降低效率

    可能训练汉语／英语文字阅读也是必要的。

    猜想：大脑的各个模块各个细胞都是通过规律的频率和谐地交互信息，才能达到高度专注的效果。切换任务会切换细胞之间互相配合的模式和频率，导致模式的调整，这个过程会导致注意力无法集中，处理问题的效率降低。

    使任务串行，一个非常大的挑战就是在规定的时间内使用手机，在执行任务时不看。

* 调研 zig

* 要想对一个东西有熟悉的理解和熟练的运用，必须将它作为更复杂的东西的一部分

* 如果前一天的任务清单上任务没有完成该怎么办？

    目前的方案是全部判为未完成，在第二天重新从项目管理系统中获取新的任务，不再考虑这些任务。

* [ ] 写一个数字图像处理库，暂时不包含深度学习

* 如果在任务列表里只说了“调研”，没有明确的调研任务，那么就是从 resource，笔记，qa，搜索，多方面开始调研。

* 所有文档笔记中的图片资源，都应该由 resource 文件夹统一管理

	如果每个笔记都创建一个图片的相对目录，那么当笔记转移来转移去时，就必须让图片也随着文件夹移动，很麻烦。

* 调研 onnx 模型的格式，搭建一个小型全连接网络，尝试用 opencl 跑通。

* 每次想玩手机时，可以喝一口水代替

* 先使用 python + numpy 实现一个自动求导，再使用 c++ 实现一个自动求导

	可能需要用到 c++ 版的 numpy

	最终的目标是实现一个类似 pytorch 的框架

* 应该增加各个项目 sync 的频率

* 遇到了 vscode 中 markdown 图片显示不出来的问题，chrome 显示正常

* async chain 的实现非常重要，有时间了研究一下

	* js 中的 promise 的原理

	* rust 中 async 的原理

	* c++ 中协程的原理

	* libhv 中事件驱动的原理

* 计算机底层只有 01 真值表，逻辑判断，流程控制和存储空间，为什么可以发展出复杂的数据结构？

	比如树，图，等等。为什么简单的概念蕴藏了如此大的复杂性？

	假如一个内存只有 4 bits，我们几乎什么都干不了，不可能形成图，树。

	假如有 4KB，可以做一些简单的程序。

	假如有 4GB，那么就可以写很复杂的数据结构。

	在这中间，红黑树，quich sort 等算法，所需要的最小内存数是多大？

	也就是说，至少需要多少存储，才可能发展出更加复杂的概念？

* 如果已经处理过某一天的 task 列表，那么应该标记下已经处理过

	这里的处理包括：处理过 feedback，同步过任务进度

	如果一个 task 列表被标记为处理过，那么它就只剩下一个记录的作用了。一周总结一次，如果没什么用就可以删了。

* “构造”很重要，或者说，生成假设空间的能力很重要。

	先有假设，然后才能推理分析。这是创造力的来源。

	“构造”很难。推理谁都会，但是构造很难。

* [ ] 增加 git 的 qa

* 在调研时，可以将笔记分成几个部分

	* 猜测（或者假设）

		给出自己的猜测，不需要验证

	* 疑问

		提出的问题

	* 验证

		对猜测的验证

	* 已经可以确定的笔记

		经过验证的猜测或假设，需要背会。

* 线下的思考很有可能是低工作量向高工作量突破的关键

    虽然使用任务管理系统可以达到很高的工作效率，但是一天的工作量并不是很大。

    猜测主要原因是浪费了很多的可利用时间，比如吃饭，走路，挤地铁等等。如果把这些时间拿来思考 cached questions，那么就有可能大量地提高工作量。

* 是否该引入每周的计划预期？

    目前只完成了每天的计划列表，但是每周完成的任务量太少。

* 无序列表（cache）的弱排序

    将已经完成的任务放到列表的最上面，比较重要的任务紧接着放到已经完成的任务的下面。

    这样无序列表会有一种“滚动”的效果，旧的 cache 慢慢消失，新的 cache 不断出现。

* 为了实现短 item 的原则，不要把长代码放到项目管理或任务列表里

    应该额外创建一个文件，然后在任务列表或项目管理里引用这个文件。

* 创建 qa 一定是从笔记里直接复制的，不要从头开始写

* 冒泡

    对于 cached 的想法、任务、问题，哪个更重要，更需要优先解决？

    每次向任务列表合并新的 cache 时，如果出现了重复的条目，可以将这个条目在 cache 中上移，表示需要优先处理。这个过程称为冒泡。

    通过不断冒泡，就可以以一种贪心的算法复杂度，对任务的重要性做出排序。

    为什么要这样？因为在 md 文件中，anchor 只能定位到标题，离标题越近的肯定越先被处理。如果我们把重要的都放在离 anchor 最近的地方，那么每次优先处理的一定都是重要的。

    使用 bookmark 确实也可以做到快速跳转，但是 bookmark 只能作为一种临时的手段，因为它会破坏顺序结构。

* cache 是缓存各种混沌内容的地方，其中包含有一些 questions

    另外再创建一个 cached questions，作为一个项目，将一些问题从 cache 中拿出来，汇集到一起。

    每天走路，发呆的时候，可以想一想这些问题，看看有没有灵感，给出一个初步的方案。

* 用`[o]`表示任务完成了一半，还正在进行

    用`[P]`表示任务进入暂停 pending 状态，需要行凶执行依赖任务，再执行当前任务。

* 每天结束的时候要列出来一个明天改进什么地方

* 调研 MLIR, IREE

* 如果以后在执行任务时遇到很多未知的问题，需要及时停止，分解任务。

    保证任务系统的正常运行才是效率最高的最优解。

* cacHed task

    cmake 环境变量的用法

* 暴露的接口越多，代码越底层，可组合的方式就越多，功能越强大，编程越繁琐难用

    可见代码量也是评价一个库是否好用的参考标准

    反过来想，如果要实现一个功能，拆分了代码后，并不能增加可组合的方式，那么它就一定是需要优化的。

    比如 vulkan，虽然比 opengl 繁琐，但是完全支持异步，多线程，这就是拆分功能的代价。

* 对于黑洞任务，应该用时间去限制

    比如只执行 30 分钟，然后整理一下已经得到的信息，解决的问题，尝试对未来的进度进行估计。

    这样的过程应该被称为采样。

* 低频（主能量）信息对应的是直觉，高频（低能量）信息对应的是强推理

* 应该三天整理一次 log

    五天整理一次 log 会花将近一个小时

* 临时项目管理中的项目不应该超过 7 个, 不然就管理不过来了

* 每天离开工作区之前应该把 task 中的 feedback 整合到临时项目管理中

* 深度学习和数学最本质的区别是，深度学习只告诉你他的方法为什么行，不告诉你别人的方法为什么不行，而数学会同时告诉你这两者

* 可以规定一下，如果 2 天（或者 3 天）都能通过某个 domain 的 qa，那么就可以申请增加新的 qa。

* 为什么`g++ -g main.cpp -lvulkan -lglfw -o main`可以通过编译，`g++ -g -lvulkan -lglfw main.cpp -o main`就不行？

* 顺着笔记找 qa 效率并不高，更好的做法是随机在笔记的后面部分找一个问题，然后写到 qa 里，再倒过来找 qa 的 dependency.

* note 的每个 item 也尽量控制得短一些，类似代码的函数长度

* 学习资源的调研主要用于完成下面几件事情

    1. 发现学习资源里，是否有需要做实验验证的地方，是否可以用实验验证他说的，是否有可以用实验澄清的模糊不清的表达

    2. 语言表达是否有歧义

    3. 未完待续

* 其实与 cache 对应的是工作区（working area），而工作区的整洁是 stack　的要求，不然各种回退操作都会混乱，没有办法快速定位到想要的资源

* 需要一个指标来评价当天有多少时间是被任务系统接管的，就叫它接管率好了

    接管率从一定程度止反映了专心程度

## cached questions

* 如果遇到的知识是条目式的，没有理解难度，但是数量比较大，是否该将其列入到记忆任务中？还是说调研时先跳过？

    * append (2024.03.21)

        如果没有理解难度，那么基本上看一遍就能记住，不需要列到记忆任务中。这个效率其实挺高的，不会占太多时间。

* [思考] 无论是 qa 还是 note，除了添加内容，更需要优化。这个优化该如何进行？

    如果只添加内容，不优化内容，那么东西越来越多，而且组织混乱。

    目前正在进行的一个优化是将按时间顺序写下的笔记，重新整理成按结构存储的数据。

## reorg

### cache

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

* 使用`{ }`表示 task 是一个长期任务，每次分派任务时，都从这个长期任务里派生出一个短时间任务，直到长期任务被完成为止

    其实这样的任务也可以单独开一个项目来追踪。

* 在随机选择时，必须把权重加上

    权重的不平衡性太大了。

* reorg 应该分三类

    * project pool

    * documents

    * projects

* qa test 中不能出现选过的 unit

* 看起来，`ref_14`对应的问题是：对于一个数字 n，给定一系列数字`arr = [a, b, c, ...]`，现在从 arr 中选取 m 个数字，使得这 m 个数字的和为 n。现在需要使得 m 最小，如果无法找到 m 个数字和为 n，那么 m 取 -1。

* 彻底抛弃 note 的固定组织，目前的目标的是总是使用相对固定的松散组织

* pathlib --- 面向对象的文件系统路径

    <https://docs.python.org/zh-cn/3.13/library/pathlib.html>

* A Comprehensive Guide to Using pathlib in Python For File System Manipulation

    <https://www.datacamp.com/tutorial/comprehensive-tutorial-on-using-pathlib-in-python-for-file-system-manipulation>

* `glXQueryVersion()`, grep 搜索结果显示出自`/usr/include/GL/glx.h`中。

    ref: <https://www.ibm.com/docs/ro/aix/7.1?topic=environment-glxqueryversion-subroutine>

### tasks

* { } reorg: projects

* { } reorg: documents

* { } reorg: english words 12.24

    feedback:

    1. 感觉还是写成一个 pat 比较好

    2. 由于 word 后面可能跟 prn.，也可能跟 exp.，所以 word 本身在哪结尾无法风开始就确定，需要先确定 prn. 和 exp.，再选一个较小的 start 作为 word 的 end。或者如果发现有 prn. 则直接用 prn. 的 start 作为 word 的 end，如果没有 prn，则使用 exp 的 start 作为 word 的 end。

* { } windows 文件整理

    目前主要整理`D:\Documents\xdx_res`, `D:\shared_folder\ai_resources`, `D:\shared_folder\Downloads`, `D:\Documents\res_processing`这四个文件夹。

* { } 《github入门与实践》

    看到 P7

* [o] process 1 url  10.03

    <https://www.baeldung.com/linux/single-quote-within-single-quoted-string>

    feedback:

    1. 这个 url 未处理结束，下次继续处理

* [ ] 完成程序：遍历索引和目录，找到`ignore.md`中无效的索引和未被收录的目录/文件

* [ ] 调研 git ignore 的实现原理

* [ ] 调研 pcie 的中断是否不需要修改中断向量表，这个中断号是否由操作系统提供？

* [v] 调研 deb 创建安装包

* [v] 调研`glXQueryVersion()`出自哪个头文件

* [ ] 在虚拟机里安装 cpu 版本的 mmdetection，看看能跑通哪些基本功能

* [ ] 调研 hugging face，看看比 mmdetection 多了什么东西

* [ ] 增加 cimg note qa，并加入 test collect 里

    1. 增加 qa unit，打开图片，获取指定位置的像素 rgb 值

    2. 保存图片

    可以参考`ref_5`

* [ ] 在 v100 5.15 系统下安装 docker，并尝试透传 nvidia gpu device

* [ ] 手动实现一下 ring + chunk 方式做 broadcast，对比直接调用 mpi 的 broadcast 函数，看看哪个比较快。

* [ ] 增加一项功能：是否答对，并将结果记录到 qa 中。

    如果一个 unit 答对的频率较高，那么它被选择的概率变小。

    如果一个 unit 距离上次回答的时间较长，那么它被选择的概率变大。

* [x] 调研《github入门与实践》

    feedback:

    1. 找不到 pdf 文件在哪...

* [v] reorg: documents

    feedback:

    1. [ ] 增加正则表达式的 qa

    2. [ ] 增加英语单词的 qa

* [ ] 在 10 个 epoch 内拟合一条 sin 曲线

* [ ] 将 project pool 中常用到的 pdf 等 resources 打包成 zip，发送到邮箱里

* [ ] 为 stochastic exam 增加`--check <qa_file>`功能，检查是否每个 unit 都有 idx, id。

* [ ] 找到贝叶斯网引论 pdf，上传到邮箱里

* [v] reorg: documents 30 mins

    feedback:

    1. 贝叶斯网引论下载到`Documents/Projects/book`目录里了。还下载了些其他书，有时间了看看

    2. 贝叶斯网的笔记不应该太简洁，应该在概念下给出大量 example，给出注释，给出引申的猜想。

* [ ] powershell 调研<https://learn.microsoft.com/en-us/powershell/scripting/samples/sample-scripts-for-administration?view=powershell-7.4>

    目前看完了`Working with objects`中的`Viewing object structure`。

* [v] reorg: documents 30 mins  12.24

    14:22 ~ 14:52

* [v] reorg: documents 30 mins  01.01

    16:23 ~ 17:49

* [v] reorg: documents 30 mins 02.06

    feedback:

    1. bootstrap 下载完后，解压出来文件，可以只使用 header 文件，也可以先编译成 .so，再在编译 main 时链接。（模板类无法实例化，如何编译？）

        可以在 vscode cpp config 的 include path 里添加 bootstrap 的路径，比如`/home/hlc/Documents/Projects/boost_1_87_0`，即可在 main 代码中使用 header file 而不报错：

        `#include <boost/interprocess/sync/interprocess_semaphore.hpp>`

        在编译时需要加上编译参数：`-I/home/hlc/Documents/Projects/boost_1_87_0`

        如果需要编译`.so`文件，可以先运行`./bootstrap.sh`，再运行`./b2`。此时即会开始编译，编译完成后会有提示：

        ```
        ...updated 641 targets...


        The Boost C++ Libraries were successfully built!

        The following directory should be added to compiler include paths:

            /home/hlc/Documents/Projects/boost_1_87_0

        The following directory should be added to linker library paths:

            /home/hlc/Documents/Projects/boost_1_87_0/stage/lib
        ```

* [v] reorg: documents 30 mins 02.07

* [v] reorg: documents 30 mins

    feedback:

    1. 官网介绍说，只需要使用`aria2c -x 2 <url>`就可以多线程下载，不知道真假。

    2. aria2 的源代码使用的是 c++ 11，主要用了 class 和智能指针，有时间了学习下

    3. aria2 文档：<https://aria2.github.io/manual/en/html/index.html>

* [v] reorg: documents 30 mins 02.19

## qa

### cached

* 使用 bfs 算法对 qa units 从易到难检测

* 现在主要需要实现 new, prev rand 的功能。等这两个实现后，需要实现 dependency 的功能，根据一个条目，可以查到它的依赖条目，根据依赖条目。

* opengl add qa: 请使用 shader 画一个彩色的 cube，并使之旋转。

* 一个 qa 文件里所有的 unit 平分 1

    然后每次遇到熟悉的 unit，可以让这个 unit 的权重减少 n% (暂定 10%)，然后重新分配所有权重，总和仍然是 1

* cmake qa:

    修改`使用 cmake 根据下面的代码编译出`libhmath.so`库，然后编译出可执行文件。`的`[u_0]`。

    增加`mymath.h`和`mymath.cpp`相关内容。

目前的 qa 项目： vulkan, c++, vim, cmake, makefile

* 感觉 bash, makefile, cmake, sed, awk, vim 这几个都是挺重要的，把 qa 加上去吧，每天练习

* qa 的每个 unit 设计不应过于复杂。如果实际检测时间超过 30 mins，那么需要拆开。

    这里的拆开指的是写成 dependency 的形式，如果 dependency 之间做过，那么直接复杂 dep 的结果，从而减少当前 unit 的 exam 时间。

* 把 vim 加入到每日 qa 中

* opencl 向量相加基本模板

    1. 在两次使用函数得到资源列表时，容易忘写第二次

    2. 总是忘写`clBuildProgram()`

    3. `fseek()`里第二个参数和第三个参数的位置写反了

* 在一个新的电脑环境上，执行 qa 的前提是有一个可以测试的环境，这个环境的搭建也必须作为 qa 的一部分，并且作为 qa 的 dep 项

* 每次降低权重时直接在原 qa 权重上除以 2，然后重新分配权重。增加权重同理，乘 2.

* 如果一些知识点正处于 cache 状态，未变成基于空间结构的数据，但是任务中又要用到，任务会依赖一些 qa，该怎么办？

* 如果在 collect 文件里新增加一个 qa 文件，权重该如何设置？

### Tasks

* [ ] 调研在 vim 中根据正则表达式搜索指定索引所在的位置

* [ ] 使用`./main --id-to-idx <id> <qa_file>`找到指定哈希值的索引

* [o] 调研 qa unit 中 dep 功能

    feedback:

    1. py 中的 int 是 32 位还是 64 位？如何区别 signed 和 unsigned？假如 signed 和 unsigned 都是 int，那么在 print 时，如何决定是否在前面加负号（-）？

* [ ] 调研：假如 search 和 match 一个是从头开始搜索，一个是从指定位置开始搜索，那么为什么这两个函数函数都有 pos 和 endpos 这两个参数？

* [ ] 在同一次 test 中，不能出现重复的 unit

* [ ] 调研 python 处理 csv 文件

* [o] 给每个 unit 设置一个比重，在抽取随机数时按比重抽取

    感觉比较熟悉的，之前重复出现过的 unit，可以把比重设置得低一点

    feedback:

    2. 调研 python type hint

    3. py 中的`f.write()`接受变参数吗，可以写入多个 str 吗

    4. 调研 python 的去重功能（unique）

        ```py
        arr_1 = ['a', 'b', 'c', 'a']
        arr_2 = [{'a': 1, 'b': 2}, {'a': 1, 'b': 2}]
        ```

        比如这种数据该如何去重？

        目前的做法是`arr = list(set(arr))`，是否还有更好的办法？

    5. 目前只实现了给每个 qa file 增加一个比重，并且保证在选择时，采样不到重复的 qa file

        或许应该实现 qa file 可以相同，但是 unit 需要保证不同？

* [ ] sync bash

* [ ] 如果观察的是一个连续量，比如随机摘一株草，观察其长度，那么是否无法写出样本点？是否必须以变量 + 区间 + 叉乘的形式写出样本空间？

* [ ] 修改 qa 文件的权重范围，所有的权重加起来为 100.00，保留两位小数

* [v] qa: 4 units

    正确率： 2 / 4

    feedback:

    1. 当前的 qa 缺少重复环节，一天基本只能记一遍，无法达到巩固的效果

    1. 必须增加 dep 功能了，不然 qa 没法进行下去

* [ ] exam 在显示 unit 时，显示 idx, idx 以及其所对应的 qa 文件名

* [v] qa: 4 units  12.01

    15:11 ~ 16:10 (59 mins)

    正确率：3 / 4

    feedback:

    1. 不创建 class 时,`/dev`文件夹下不显示设备文件。u0 为`请写出添加及删除 cdev 的最小代码。`的 u1 有问题，有时间了改一下。

* [v] 增加`python main.py --review`功能，复习当天的 units

    10:55 ~ 13:58

    feedback:

    1. 正则表达式中`^`指的是字符串的开头还是`\n`的下一个字符？

* [v] 调研 qa test 增加功能：

    在工程目录下增加`qa_db.txt`文件，每次 qa test 结束后，输入`save`向此文件中写入类似如下的格式：

    ```
    [date-time]
    2024-12-01 16:20
    [units]
    /path/to/qa_file_1,2956213572395325
    /path/to/qa_file_2,2342343294235
    /path/to/qa_file_3,723649235873223
    /path/to/qa_file_4,47567435623432

    [date-time]
    2024-11-30 13:35
    [units]
    xxx
    xxxx
    ...
    ```

    总是保持最新的日期在最上面。

    在每天需要复习的时候，使用`python main.py --review`进入 exam 模式，复习当天的 units。

    目前可以先手动录入`qa_db.txt`文件，程序只负责解析和 review。

    deps:

    1. [v] 把 randexam 改成 selected_units 的形式

    2. [v] 在 randexam 开始之前，对 qa file collect 文件中的所有 qa file 进行 check，检查 id 和 idx 是否完整

* [ ] 调研 qa parse 与 rewrite 时是否保留了 unit 的`[dep]`信息

* [v] qa: 4 units

    正确率：2 / 4

    feedback:

    1. 修改 opengl note qa 中的`请给一个三角形加上纹理贴图。`，在 glsl 代码前加上

        `#version 330 core`

        否则跑不通。

    2. [ ] fix bug: 保存最新 qa record 时，不能删除旧的

    3. 假如一个集合有 10 个 0.1，现在只允许每个元素对自身除以 2，再平均到 1，这个集合构造出的数是有限的还是无限的？这些数的取值的概率密度是怎样的？

* [ ] 调研`register_chrdev_region()`与`register_chrdev()`有什么区别？

* [v] qa: 4 units  12.16

    正确率：3 / 4

    feedback:

    1. 必须先执行`glfwInit()`，等`glfwMakeContextCurrent()`执行后，再执行`glewInit()`，

        没有`glewInit()`，`glCreateShader()`会立即返回失败。

    2. 调研 exam 时显示 unit 的 id 和 idx

* [v] qa: 4 units  12:17

    正确率：2 / 4

    feedback:

    1. 如果**大部分**的 qa file 正确率都很**高**，那么考虑扩充 units；如果大部分的 qa file 正确率都很低，那么调低高正确率 qa file 的选中概率

    2. 如果单个 qa file 的正确率很高，那么降低它出现的概率

* [v] qa: 4 units 12.18

    正确率：3 / 4

    feedback:

    1. 关注 qa file 的正确率，如果正确率高，那么 sync note。sync note 已经完成，那么减小 qa file 的 prob。

    2. 增添新 record 时，不删减以前的 record，每三天 review 一次。

    3. 动态的 review 间隔确定：通过即时复述，确定记忆量；间隔一段时间，比如早上到晚上，或者早上到第二天早上，再次复述，达到 90% 暂定）以上

* [v] qa: 2 units 02.05

    feedback:

    1. opengl routine 遗忘点

        1. 先 init glfw，再 init glew

        2. bind buffer 时，先有`GL_ARRAY_BUFFER`，再有`GL_ELEMENT_ARRAY_BUFFER`.

        3. enable 的是 vertex attrib array, vertex 在位置 1，attrib 在位置 2，array 在位置 3

        4. 解释 array buffer 时，使用的是 vertex attrib pointer，vertex 在位置 1，attrib 在位置 2，pointer 在位置 3

        5. 记得读取 shader 文件中的内容

        6. `glDrawElements()`中填的是使用的 index 的数量，如果画三角形，那么必须为 3 的倍数

* [v] qa: review 02.06

    feedback:

    1. hint point

        1. flex 程序中，初始化的调用是`yylex();`，不是`yyflex();`

        2. 在 glfw get key 之前，需要`glfwPollEvents();`

        3. 在进入 while 循环前，必须执行`glEnableVertexAttribArray(0);`

    2. [ ] `使用 element draw 画一个 cube`增加 deps:

        1. load shader

* [v] qa: 2 units  02.07

    正确率：1 / 2

* [v] qa: review  02.10

* [v] 调研 qa review 增加 clear 功能

    默认情况下 append，当手动指定 clear 后，清空文件。这样可以保持一周的 review 量。

    feedback:

    1. [ ] 记录每个 uni 的多个历史完成时间，如果平均时间大于 N 分钟，那么标记该 unit 应该被拆分成 deps。

    2. [ ] 调研 mysql

* [ ] 调研：unit 增加 hint 字段，显示完`[u_0]`后，输入`h`显示一条 hint 

    或许应该把 d 命令和 h 命令合成一个功能？

* [ ] qa: exam 中调换 clear previous qa_record 和 save this exam record 的位置

* [ ] qa: 增加 openshmem 的 qa

* [v] qa: 2 units 02.19

    正确率： 1 / 2

    feedback:

    1. hint

        1. flex 程序，先是`%{ %}`，再`%% %%`

        2. `yywrap()`, `yylex()`中间没有下划线

* [v] qa: 2 units 02.14

    正确率：0 / 2

    feedback:

    1. hint
        
        1. 获取 key 按键时，函数叫 glfw get key, 不叫 glfw key press

    1. 创建一个`qa_utils`文件下，下设`opengl`, `opencl`等文件夹，文件夹中再设`triangle.c`, `vec_add.c`等文件或文件夹，与 dep 任务相对应。

        最好可以把功能写成`.h`的形式，在使用时直接`#include "qa_utils/opengl/load_shader.h"`就可以。

        实在写不成`.h`的形式，再写成`.c`或`.cpp`的形式，如果连这种形式也写不成，就创建文件夹工程目录。

        dep 不一定要实现完整工程，只需要写出核心的逻辑就可以。

    1. exam 程序应该先打印 deps，如果有 deps，必须依赖 deps 进行开发

* [v] qa: review 02.21

    hint:

    1. bind buffer 时，target 是`GL_ARRAY_BUFFER`，不是`GL_VERTEX_ARRAY`。

* [v] qa: 2 units 03.03

* [v] qa: 1 unit 03.10

* [v] qa: review 30 mins

    feedback:

    1. `glDrawArrays()`完后，还需要`glfwSwapBuffers()`才能显示内容。

    2. `alloc_chrdev_region()`时，第二个参数是 start，第三个参数才是 num dev。

    3. flex 程序在 init 时，是`yylex();`，不是`yyflex();`

    4. 在`\n {return 0;}`规则时，是`return 0;`，不是`return;`

    4. 重新进入已经 stop 的容器，使用的是`docker start`，不是直接`docker -ia`。

## cache tabs / process urls

* 需要消化 cached urls

* 处理 url 的时间比想象中要长，每一条大概需要 30 mins

* 可以使用 youtube 学一些英语课，比如 julia，octave 等，这样既锻炼了英语，也学到了东西

### tasks

* [v] 调研`git revert`的用法

    feedback:

    1. 调研`git revert -n <commitToRevet>`, `git revert --no-commit <commitToRevet>`, `git revert HEAD~x`

    2. 调研`git cherry-pick`

* [ ] 调研`git reflog`

* [o] process 1 url 10/01

    Resetting remote to a certain commit: <https://stackoverflow.com/questions/5816688/resetting-remote-to-a-certain-commit>

    feedback:

    4. 调研`ORIG_HEAD`, `git show ORIG_HEAD`

    5. 调研`git update-ref `

    6. 调研`git log --graph --all --oneline --decorate`

* [o] process 1 url 10.06

    <https://www.baeldung.com/linux/last-directory-file-from-file-path>

    feedback:

    1. deps

        1. awk, sed

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

* [v] cache tabs 11.07

* [v] cache tabs

## markdown renderer

使用 electron + markdown parser + mathjax 实现 markdoen renderer。

tasks:

* [ ] 调研 js

* [ ] 调研 electron 中的 lifecycle, emitter

    <https://www.electronjs.org/docs/latest/tutorial/tutorial-first-app>

* [v] 调研 electron

## Machine Learning

### tasks

* { } 调研《Python机器学习》

    feedback:

    1. 目前看到 pdf P54

    2. 不清楚为什么`self.w_ = np.zeros(1 + X.shape[1])`要`1 +`。

* { } 调研 pytorch

    系统地学一遍 pytorch

    resources:

    1. Welcome to PyTorch Tutorials

        <https://pytorch.org/tutorials/>

        主要看 learn the basics 和 learning pytorch with examples

    2. PyTorch documentation

        <https://pytorch.org/docs/stable/index.html>

        可以看下下面的 Developer Notes 主题，重点看一看模型压缩，混合精度以及并行训练／推理

* [ ] 调研 三维的 Swiss Roll

* [v] 尝试使用 python + numpy 实现一个感知器函数

* [ ] 调研论文 The Perceptron,a Perceiving and Recognizing Automaton

* [ ] 调研论文 A Logical Calculus of the Ideas Immanent in Nervous Activity

* [ ] 调研`\mathbf`

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

## Mathematics

resources:

1. 《高等数学》同济大学应用数学系

tasks:

* [ ] 调研第八章 多元函数微分法及其应用

* [v] 调研矩阵论

    feedback:

    1. 调研 Amir Beck

    2. cached

        * ML-Prerequests: 机器学习的预备知识（矩阵论，概率论，凸优化等）

            <https://github.com/robinluodh/ADMM_learning_NJU_HBS>

        * ADMM_learning_NJU_HBS: 凸优化、变分法、ADMM 资料学习。来自南大何炳生教授主页。

            <https://github.com/robinluodh/ADMM_learning_NJU_HBS>

* [v] 调研线性代数

## CCL

### cache

* 通过 printf 法，看到`op128.h`文件里主要调用的是`ld_volatile_global()`

    在 print 的 log 中，`in ld_volatile_global()...`与 nccl 的 perf 数据交替出现，数据测试没有问题，说明在传输数据过程中确实用到了`ld_volatile_global()`

    2025/01/23/00: 其实这个只是非宏实现，其他的都是宏定义的 ld volitile global，所以没有 printf 输出。

* `ld_volatile_global()`在两个地方被调用

    1. `Primitives::loadStepValue()`

        用于加载 peer connection 的 info

        * `connStepPtr = conn->head;`, `connStepPtr = conn->tail;`, 看起来`connStepPtr`是 conn 的链表, 这些都在`loadRecvConn()`被调用

        * 有可能 step 是异步处理，所以需要 volatile 加载数据

        * `st_relaxed_sys_global()`由`postPeer()`调用

    2. reduce copy

        用于取 payload 数据。

* nccl 数据传输的调用流程

    run work batch (dev func) -> run work coll -> run ring/tree -> prims -> send

    * `Primitives<> prims`由`RunWorkColl()` -> `runRing()`创建

* `reduceCopyPacks()`是最底层负责干活的函数，每次起一个 warp，warp 里有 32 个线程，每个线程搬运 16 个字节，warp （线程）循环处理 Unroll 组数据，这叫一个 hunk。

    数据可能有多个 src，dst，此时需要做 reduce，把多个 src, dst 合到一处。

* 一个 unroll 中处理无法使用一个完整 warp 处理的数据的方式：

    unroll 为 1 时，因为每个线程是单独计算自己的任务进度，所以可以处理不完整的 warp 的任务

* nccl tmp

    * 多卡之间如何 reduce copy?

    * 如何动态确定多 srcs / dst?

    * cu host alloc 的调用路径

        `ncclAsyncJobMain()` -> `ncclCommInitRankFunc()` -> `initTransportsRank()` -> `devCommSetup()` -> `devCommSetup()` -> `ncclCudaHostCalloc()`

* pthread cond 如果先 signal，再 wait，那么无法正常运行

### tasks

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

    cache:

    * 如果数据横跨两个 cuda device，那么要么开启 p2p，要么使用 host mem 作中转

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

* [ ] 调研 rank 的分配过程

* [ ] 调研 bootstrap 机制

* [ ] 调研 epoll

* [ ] 调研`recvmsg()`, `recvmmsg()`

* [v] 调研 socket 中的`recv()`, `recvfrom()`, `recvmsg()`, `recvmmsg()`有什么区别？ 

* [v] 调研 inet_pton 的返回值

* [ ] 如果未建立连接就 send / recv，或者如果建立了连接后，但是对方没有 send / recv 时就 recv / send，会发生什么？

* [ ] 调研：可以在一个 thread 中打开另一个 thread 吗？

* [ ] 调研常见的基于 poll 的异步事件中心的写法

* [v] 尝试使用 cuda host malloc 实现基于 host 中转的 send / recv

    feedback:

    1. [ ] 在单机上跑通后，需要在两个 node 上跑通。

* [ ] 如果 rdma 中使用的是 va + offset，那么还可以 remote write 吗？此时该如何查表？

* [ ] 如果在 cond wait 的时候 destroy mutex，是否会导致程序出错？

* [ ] 在 for 和 while 中`int a = 0;`，然后再修改`a`的值，`a`第二次循环时是 0 还是修改过的值？

    同理，在循环中`MyClass my_obj;`，是否会多次调用构造函数和析构函数？

* [ ] 调研：可以在一个 thread 中打开另一个 thread 吗？

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

* [v] 调研`cudaStreamCreate()`

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

    2. [ ] 不使用 simple 协议，把 primitives 改成 ll 协议

    3. [v] 调研 cuda `__shared__`

        feedback:

        1. [ ] 调研`extern __shared__ xxx;`申请动态的 shard 数据

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

* [ ] 调研 nccl app

* [v] 调研`MPI_Probe`, <https://mpitutorial.com/tutorials/dynamic-receiving-with-mpi-probe-and-mpi-status/>

    feedback:

    1. 调研 mpi err handler，这个概念是否可以认为是 c 版本的 c++ try catch 错误捕捉机制？

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

* [ ] 使用 cuda 实现矩阵乘法

* [v] 调研 vllm 中 nccl 的用法

    feedback:

    1. 调研 py 调用 C 库函数

    2. vllm 的 pynccl 主要 wrapper 了 all reduce, send, recv 三个函数，all reduce 的算子似乎是直接调用 pytorch 的算子，并没有额外实现一版。

        传入参数中有 cuda stream，但是没有看到 sync 的函数。因此理论上是支持异步的，不知道具体怎么个调用法。

    3. 调研 py 类型提示`group: Union[ProcessGroup, StatelessProcessGroup]`

    4. async 相关的可能在`cuda_wrapper.py`里

        这里面可以重点看下`cudaDeviceSynchronize()`是怎么 wrapper 的，在哪里调用的。

    5. 调研 cuda 函数`cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`

* [ ] 调研 nccl 中的 asm 语句 <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints>

* [v] 调研 nvlink

    13:36 ~ 14:42

    feedback:

    3. [ ] 尝试在 nccl 中把 p2p 传输的代码剥离出来，使用单独的一份代码跑通 nvlink + p2p

    4. [ ] 继续调研 nccl 源码，看是否有 put get 相关的函数

* [v] 调研 linux nvidia kmd 中与 nvlink 相关的部分

    feedback:

    1. 在 nvidia open source kdm 里面有 nvlink, nvswitch 相关的代码

        <https://github.com/NVIDIA/open-gpu-kernel-modules>

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

* [v] 调研 cuda gdb

    17:16 ~ 17:36

    feedback:

    2. 关于`CUDA_VISIBLE_DEVICES`的疑问：假如在启动 cuda-gdb 之前指定这个环境变量，host code 只能看到指定的 device；假如在启动 cuda-gdb 后，改变`CUDA_VISIBLE_DEVICES`，是否只会在指定的 device 上 hit 到断点？

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

* [v] 调研 netlink，ibverbs，fifo，准备面试题

* [ ] 调研 hugging face，看看比 mmdetection 多了什么东西

* [v] 调研如何解决计算，通信，存储资源碎片化的问题

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

* [v] 调研《控制之美》

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

* 调研`dma_addr_t`类型

* 调研`usleep_range()`

* 调研`dev_err()`

* 调研`debugfs_create_dir()`, `debugfs_remove()`

* 调研`rcu_read_lock()`, `rcu_read_unlock()`

* 调研`list_for_each_netry_rcu()`

* 调研`pci_rq_vector()`, `spic_lock_init()`, `INIT_LIST_HEAD()`

* 调研`snprintf()`, `pci_name()`, `request_irq()`

* 调研`pci_enable_msic_range()`

* 调研`pci_iounmap()`, `pci_set_drvdata()`, `pci_enable_device()`, `pci_request_regions()`, `pci_set_master()`, `dma_set_mask_and_coherent()`, `pci_ioremap_bar()`

* 调研`module_pci_driver()`, `MODULE_DEVICE_TABLE()`

* 调研`container_of()`

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

## AI deploy

tasks:

* [v] 使用 linear + activation 拟合一条 sine 曲线

* [v] 使用 cuda 环境拟合一条 sine 曲线

* [ ] 使用 dataset 和 dataloader 在 cpu 环境下拟合 sine 曲线

* [ ] 调研 v100 部署 pytorch 的小模型（CV or NLP）

* [ ] 调研 llama 部署

* [ ] 调研 llama 在 cpu 上的部署

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

* [v] 调研： gpu performance profiling

* [v] 调研 amd gpa: <https://gpuopen.com/gpuperfapi/>

    看看用法，跑一些渲染，ai 应用，尝试性能改进。

* [v] 调研 rgp

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

### 任务列表：

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

## linux driver

cache:

* `list_add()`是在指定 node 后添加 node

* gcc 12 要求所有函数必须有声明，不然会报 warning:

    ```
    make -C /usr/src/linux-headers-6.8.0-40-generic M=/home/hlc/Documents/Projects/linked_list_test modules
    make[1]: Entering directory '/usr/src/linux-headers-6.8.0-40-generic'
    warning: the compiler differs from the one used to build the kernel
      The kernel was built by: x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
      You are using:           gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
      CC [M]  /home/hlc/Documents/Projects/linked_list_test/hello.o
    /home/hlc/Documents/Projects/linked_list_test/hello.c:4:5: warning: no previous prototype for ‘hello_init’ [-Wmissing-prototypes]
        4 | int hello_init(void)
          |     ^~~~~~~~~~
    /home/hlc/Documents/Projects/linked_list_test/hello.c:10:6: warning: no previous prototype for ‘hello_exit’ [-Wmissing-prototypes]
       10 | void hello_exit(void)
          |      ^~~~~~~~~~
      MODPOST /home/hlc/Documents/Projects/linked_list_test/Module.symvers
      LD [M]  /home/hlc/Documents/Projects/linked_list_test/hello.ko
      BTF [M] /home/hlc/Documents/Projects/linked_list_test/hello.ko
    Skipping BTF generation for /home/hlc/Documents/Projects/linked_list_test/hello.ko due to unavailability of vmlinux
    make[1]: Leaving directory '/usr/src/linux-headers-6.8.0-40-generic'
    ```

* linux module 编译不出来，可能是因为`obj-m`写成了`odj-m`

* linux 的 interruptible sleep 是如何实现的？

* 调研这三个头文件

    ```c
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    ```

* 可以在函数声明后就直接构造`struct file_operations`，然后再在其他地方对函数进行定义。

* 不明白`file_operations`中`.owner`有什么用

* `__init`和`__exit`的作用？

* 调研`pr_info`, `pr_err`, `__init`, `MODULE_VERSION`, `MODULE_AUTHOR`, `MODULE_DESCRIPTION`

* 整理一下开发环境的搭建，因为发现只需要安装`build-essential`就可以自动安装 header 文件，那么其实可以简化流程

* [ ] param 被写入 module 中时，module 是如何感知到的？

    2024/05/07/00: 应该修改为，param 被写入 module 中时，是否有机制可以让 module 中的代码感知到变动？

* 调研：
    
    `pci_msix_vec_count`, `pci_find_capability`, `pci_alloc_irq_vectors`,

    `pci_irq_vector`, `request_irq`

* 调研
    
    `kthread_should_stop`, `msleep`, `likely`, `unlikely`, `orderly_poweroff`

* 报错：`insmod: ERROR: could not insert module ./hello.ko: Invalid module format`

    主要是因为编译时候使用的内核版本和当前系统的内核版本不一致。

* 为什么写`MKDEV()`时可以填`MKDEV(255, 0)`, `MKDEV(220, 0)`

    怎么保证 255, 220 这些数字不与其他冲突？

    答：可以用`cat /proc/devices`查看已经注册过的设备

* cached tasks

    调研函数：

    `pci_enable_device`, `dev_set_drvdata`

    `pci_resource_start`, `pci_resource_len`, `pci_ioremap_bar`

    `pci_set_master`, `dma_set_mask`, `pci_ioremap_wc_bar`

    调研：`iommu_domain_alloc`, `iommu_group_get`, `iommu_attach_group`

    `dev_to_node`, `kzalloc_node`, `spin_lock_init`

    `idr_init_base`

    `kfree`, `dev_info`

    调研：

    `writel`, `BUG()`, `readq`, `writeq`, `pci_read_config_dword`, `pci_find_ext_capability`

    `readl`, 

    * `pci_register_driver()`这个也是内核调用，有时间了看下含义

    调研一下`KBUILD_MODNAME`的含义。

    * 需要调研的函数

        * `spin_unlock_irqrestore()`

        * `dev_get_drvdata()`

        * `mdev_register_device()`

    * 调研 linux kernel 函数

        `mdev_get_drvdata()`

        `copy_from_user()`

        `BIT()`

* 如果 linux 系统里安装了 systemd，那么可以使用`journalctl -k`查看历史日志

    如果想把新增的日志写入文件，可以使用`dmesg --follow-new | tee <log_file>`

* 可以使用`dmesg -e`显示消息的大致时间戳

* `module_param_array()`中的数组长度参数只有在 write 数据的时候才会被改变

resources:

* Linux Kernel Development, 3rd Edition

tasks:

* [v] 调研`pci_set_drvdata`

    feedback:

    1. 还是先把 linux driver 开发看完比较好

        先看 qa，再看网站

* [v] linux driver 调研 data exchange between user space and kernel space

    feedback:

    1. 写了内核驱动的代码和用户态代码，成功从用户态向内核写入数据，并从内核读取数据。

        见`ref_11`。

        * user mode 的程序需要使用`sudo ./main`执行，不然没有权限打开 device 文件

        * kernel mode 的`h_write()`的返回值就是 user mode 的`write()`的返回值

            `read()`同理。在写 kernel code 的时候，按照约定俗成，返回写入/读取了多少个字节。

        * 如果使用`fopen()`，`fread()`，`fwrite()`等函数打开文件，那么`dmesg`中会报错。

        * `copy_to_user`, `copy_from_user`返回的是剩余的字节数，与`read()`，`write()`正好相反，需要注意。

    2. kernel 中的内存管理感觉是个问题

        假如希望用户可以无限次乱序 read, write，并且遵循 fifo 的原则，那么可以把 buffer 设计成一个链表，每次调用 read 的时候减少一个节点，调用 write 的时候增加一个节点。

        如果在 read 的时候遇到链表为空，那么就输出 there is nothing to copy。

* [v] 调研 ioctl

    feedback:

    1. 实现了 ioctl 读取与写入数据，见`ref_12`

        output:

        ```
        successfully write data by ioctl
        read value: 123
        ```

    2. 不太明白为什么 ioctl 的 cmd 要靠`#define WR_VALUE _IOW('a','a',int32_t*)`这个构造

* [ ] 调研 sysfs 读写，sync

* [ ] 调研`mutex_lock`, `mutex_unlock`, `mutex_destroy`

* [ ] 调研`kzalloc`, `kfree`

* [v] 调研 workqueue

    feedback:

    1. `device_create()`和`device_add()`有什么区别？

        `device_del()`和`device_destroy()`有什么区别？

* [v] 调研 linked list

* [ ] 调研`select`的用法

* [ ] socket 调研：为什么`accept()`的第三个参数是一个长度指针，它有什么用？

* [ ] 调研：实现一个仅使用 read device 触发的中断程序

* [ ] sync socket programming

## OpenGL

有时间就看看 opengl 各种效果的实现，主要是光影，贴图，动画等。

* cache

    1. 画 cube 不加旋转等于没画

        因为画出来的图形的正面总是一个正方形，无法判断其他几个面画得对不对。

        不将不同面涂成不同的颜色也不行，因为只涂一个颜色无法分辨立体感。

## 计算机图形学 学习 [2]

cache:

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

任务列表：

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

cache:

* 以后再做 c++ 的 qa，要么写一点点语法，要么就是做题

* 找一个 c++ 学习资料，系统学一下比较好

* [ ] 调研 c++ `variant`

* 存储 image 可以有两种方式

	一种是 continuous，即开辟一块连续的内存，每次二维索引都转换成一维。

	另外一种是 separate，即使用`vector<vector<float>>`来存储。这样可以直接索引，其实性能也差不到哪。
	
	这样的话维度其实是固定死的，好在大多数图像都是二维的。如果使用`vector<vector<vector<float>>>`，或者`vector<vector<vector<vector<float>>>>`来存储三维或四维的图片或者数组，会不会损失性能，或者对内存造成浪费？

	默认就以 seperate 形式存储吧，将 2d 图片，3d 图片存储成不同类型的。

	存储图像时，可以使用 int 也可以使用 float。

* c 语言中`#defile`中`##`的用法？

1. 指针的指针，区分指针的指针所指的内容不能被修改，指针所指的内容不能被修改（指针的指针本身不能被修改），指针本身不能被修改

    分别如何 new, malloc

* modern c++

    <https://github.com/federico-busato/Modern-CPP-Programming>

* 学一下 c++ 的 std format

1. 学习一下 c++ make uniquue

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

* [ ] 调研：c++ `lower_bound()`, `upper_bound()`作用
    
    以及如果找不到元素或元素在数组边界时的情况

* [ ] 调研：c++ 迭代器为什么`upper_bound()`的返回值减去`lower_bound()`的返回值等于数组的长度？

* [ ] 调研 c++ 迭代器，increase 相关

## Vim

本项目的目标是学完 vim，可以使用 vim 代替 vscode 进行日常的代码开发和调试，以及文本文档的编辑。

* vim 中的 redo 是哪个？

## 其他

* [v] git stash 恢复

* c++ 版本的 opencl 入门

    <https://objectcomputing.com/resources/publications/sett/july-2011-gpu-computing-with-opencl>

* libeigen

    other linger algebra libraries: <https://stackoverflow.com/questions/1380371/what-are-the-most-widely-used-c-vector-matrix-math-linear-algebra-libraries-a>

    以 libeigen 为主，其他的库对比着学下。

    * numcpp

        矩阵运算库

        <https://github.com/dpilger26/NumCpp>

        <https://zhuanlan.zhihu.com/p/341485401>

