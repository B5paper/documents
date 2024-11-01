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

* 任务完不成应该分两种情况处理，一种是有极大可能完成，只不过时间不够，另一种是还在调研搜集信息阶段，不清楚是否能完。显然这两种情况的处理方式应该是不同的。

* 在准备去执行一项任务时，不应该考虑当前状态是否为最佳，而应该考虑当前状态是否满足最低要求

* 不可能存在静态的平衡，只可能出现动态的平衡

    打乒乓球时要求每打完一拍都要快速“归位”，以做好迎接下一拍的准备，那么生活中是否有一些 routine，可以保证在执行下一项任务前，可以精神饱满，不饿不渴不累，周围没有杂音干扰，注意力可以快速且长期地集中？
    
    比如在学习某篇论文之前，先睡 30 分，再运动 20 分，再吃一点东西，再喝几口水，做完这样的 routine 后，再全身心阅读论文？或者每晚睡够 8 个小时，早上晨跑，吃早饭，10 点喝一次水，中午吃完饭午休，晚上按时睡觉，这样来保证上午 8 点到 11 点一定精力充沛，下午 2 点到 6 点，晚上 7 点到 9 点一定精力充沛。我觉得这样的静态平衡是不可能做到的。

    取而代之的静态方案是，在一项任务的末尾开始考虑与下一项任务的衔接，如果渴了就喝水，如果饿了就吃点零食，如果特别困就去睡觉。在一项任务开始之前，也可以简易地收拾下周围环境，保证接下来 30 ～ 40 分可以注意力集中。这样的动态平衡目前看来是比较好的解决方案。

* 形式与环境不是本质，但是极大程度影响了本质的发现过程

    在 win 系统下，多桌面不方便，所以很难快速地切换任务，也很难给一个任务独立的工作空间，更不可能栈式地递归展开依赖任务，因此在 win 上很难形成像现在一样的任务管理系统。

    虽然说多桌面不是任务管理和时间规划的本质，但是它促进了效率较高的任务管理方式的发现。

    因此改善办公环境，尝试一些新的环境的改变可能会带来一些变化。

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

* cuda 12.1 环境下，编译 nccl 使用 compute_90 编译时，无法跑通 nccl-test

    使用 compute_70 可以跑通。

* cached tabs

    * How to use SSH to run a local shell script on a remote machine?

        <https://stackoverflow.com/questions/305035/how-to-use-ssh-to-run-a-local-shell-script-on-a-remote-machine>

    * 10.6. Launching with SSH

        <https://docs.open-mpi.org/en/v5.0.x/launching-apps/ssh.html>

    * tt-metal

        TT-NN operator library, and TT-Metalium low level kernel programming model. 

        <https://github.com/tenstorrent/tt-metal/>

    * Basic Data Structures and Algorithms

        <https://algo-ds.com/>

    * Build Your Own Text Editor

        <https://viewsourcecode.org/snaptoken/kilo/>

    * lm.rs

        Minimal LLM inference in Rust 

        <https://github.com/samuel-vitorino/lm.rs>

    * vecdb

        Toy vector database written in c99. 

        <https://github.com/montyanderson/vecdb>

    * Rust Design Patterns

        <https://rust-unofficial.github.io/patterns/>

    * ML Code Challenges

        <https://www.deep-ml.com/>

    * posting

        The modern API client that lives in your terminal. 

        <https://github.com/darrenburns/posting>

    * Read the newest State of AI report

        <https://retool.com/blog/state-of-ai-h1-2024>

    * LLMs-from-scratch

        Implementing a ChatGPT-like LLM in PyTorch from scratch, step by step 

        <https://github.com/rasbt/LLMs-from-scratch>

    * barco: Linux containers from scratch in C. 

        <https://github.com/lucavallin/barco>

    * 怎么用apt命令下载内核源码给出步骤

        <https://linuxcpp.0voice.com/?id=39737>

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

* vscode 中，取消了 tab stop 后，还是会有 tab 缩进 2 个空格的现象，这时候还需要取消 Detect Indentation

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

* 虚拟机 120G 磁盘不够用，下次试试 150G

* [ ] 有时间了调研一下`https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.lib_ref/topic/a/asprintf.html`，这好像是个 c api 的文档



* 如何评价 2024 年阿里全球数学竞赛决赛试题？难度怎么样？

    <https://www.zhihu.com/question/659607083>

* 香山 riscv 源代码 repo

    <https://gitee.com/OpenXiangShan/XiangShan/>

* Theseus is a new OS written from scratch in Rust to experiment with novel OS structure

    <https://www.theseus-os.com/Theseus/book/index.html#introduction-to-theseus>

    用 rust 写的操作系统，有时间了看看。

* 如何为一个用户增加指定目录的各种权限？

* [ ] 调研 git reset 查看不同版本的 linux kernel version

* isoinfo 在 genisoimage 包中

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

* 应该对个人系统建立一个项目进行复习

	每次随机选择一个文件，跳转到随机一行，往下读 20 mins

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

* 调研`git bisect`

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

* 其实调研也是一种是 cache，是一种广度优先搜索，它安排了必须要回答的问题，优化了任务列表的结构，整合了小范围内的信息，防止有歧义在前，回答在后的情况发生

    调研不做实验，不使用实验对猜想做验证

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

* [ ] 使用 python ＋ re 写一个英语单词的 parser，每次随机检测指定数量个单词，保存索引，后面每次复习时检测上次抽取的单词 + 融合前几次抽取的单词，时间越久的单词出现的概率越小。

* 使用`{ }`表示 task 是一个长期任务，每次分派任务时，都从这个长期任务里派生出一个短时间任务，直到长期任务被完成为止

    其实这样的任务也可以单独开一个项目来追踪。

* 在随机选择时，必须把权重加上

    权重的不平衡性太大了。

* reorg 应该分三类

    * project pool

    * documents

    * projects

### tasks

* { } reorg: projects

* { } reorg: documents

* { } windows 文件整理

    目前主要整理`D:\Documents\xdx_res`, `D:\shared_folder\ai_resources`, `D:\shared_folder\Downloads`, `D:\Documents\res_processing`这四个文件夹。

* { } 《github入门与实践》

    看到 P7

* { } reorg: documents

* { } reorg: projects

* [ ] 在 virtual box 里安装 150G 的 ubuntu 24.04 虚拟机

* [o] process 1 url  10.03

    <https://www.baeldung.com/linux/single-quote-within-single-quoted-string>

    feedback:

    1. 这个 url 未处理结束，下次继续处理

* [v] reorg: documents 10.14

* [v] 调研 markdown previewer

    要求能显示数学公式

* [v] random select 时，将 projects 文件夹也包含进去

* [v] 调研 python 中不同 path 的变体如何判断是相同 path

    比如`./test`, `/home/hlc/Projects/test`, `test`, `../outside_test/test`，这些应该都等于相同的路径

* [ ] 为 reorg 程序增加指定文件的随机一行的功能

* [ ] 完成程序：遍历索引和目录，找到`ignore.md`中无效的索引和未被收录的目录/文件

* [ ] 调研 python path 判断一个文件夹是否包含另一个文件/文件夹

* [ ] 调研 git ignore 的实现原理

* [ ] 调研 pcie 的中断是否不需要修改中断向量表，这个中断号是否由操作系统提供？

* [ ] 调研 deb 创建安装包

* [ ] 调研`glXQueryVersion()`出自哪个头文件

* [v] reorg: documents

    deps:

    2. 在虚拟机里安装 cpu 版本的 mmdetection，看看能跑通哪些基本功能

    3. 调研 hugging face，看看比 mmdetection 多了什么东西

* [v] reorg: documents

    feedback:

    1. [ ] 增加 cimg note qa，并加入 test collect 里

        1. 增加 qa unit，打开图片，获取指定位置的像素 rgb 值

        2. 保存图片

        可以参考`ref_5`

* [v] reorg: project pool: nccl 调试记录

    feedback:

    1. deps: 在 v100 5.15 系统下安装 docker，并尝试透传 nvidia gpu device

        deps:
        
        * [v] 配置 iptables

    2. deps: gdb remote server

    3. [ ] 增加 docker note qa，并加入 test collec 中

* [v] reorg: project pool  10.29

    14:20 ~ 14:48

* [v] reorg: documents  10.14

## qa

cached:

* 使用 bfs 算法对 qa units 从易到难检测

* 现在主要需要实现 new, prev rand 的功能。等这两个实现后，需要实现 dependency 的功能，根据一个条目，可以查到它的依赖条目，根据依赖条目。

* opengl add qa: 请使用 shader 画一个彩色的 cube，并使之旋转。

* 在做检测时，写出 unit 出自哪里

* 一个 qa 文件里所有的 unit 平分 1

    然后每次遇到熟悉的 unit，可以让这个 unit 的权重减少 n% (暂定 10%)，然后重新分配所有权重，总和仍然是 1

* 需要给每个 unit 设置一个比重，在抽取随机数时按比重抽取

    感觉比较熟悉的，之前重复出现过的 unit，可以把比重设置得低一点

* cmake qa:

    修改`使用 cmake 根据下面的代码编译出`libhmath.so`库，然后编译出可执行文件。`的`[u_0]`。

    增加`mymath.h`和`mymath.cpp`相关内容。

目前的 qa 项目： vulkan, c++, vim, cmake, makefile

目前已经完成随机笔记文件的选择，工程目录在`Documents/Projects/rand_select_py`，但是还有一些缺点，后面迭代改进。

* 感觉 bash, makefile, cmake, sed, awk, vim 这几个都是挺重要的，把 qa 加上去吧，每天练习

* qa 的每个 unit 设计不应过于复杂。如果实际检测时间超过 30 mins，那么需要拆开。

    这里的拆开指的是写成 dependency 的形式，如果 dependency 之间做过，那么直接复杂 dep 的结果，从而减少当前 unit 的 exam 时间。

* 把 vim 加入到每日 qa 中

* opencl 向量相加基本模板

    1. 在两次使用函数得到资源列表时，容易忘写第二次

    2. 总是忘写`clBuildProgram()`

    3. `fseek()`里第二个参数和第三个参数的位置写反了

Tasks:

* 在 vim 中根据正则表达式搜索指定索引所在的位置

* [ ] 为 qa 工具增加`--list`功能

* [ ] 修复 bug:

    `python3 main.py --create-id /home/hlc/Documents/documents/Linux/linux_driver_note_qa.md`

* [v] qa 4 units 05/03

    答对题数：2/4

* [v] 增加 linux driver qa:

    配置 vscode 的内核驱动开发环境

* [ ] 使用`./main --id-to-idx <id> <qa_file>`找到指定哈希值的索引

* [ ] 调研 qa unit 中 dep 功能

* [v] qa 1 unit 10.14

* [v] qa: 2 units 10.28

* [v] qa: 2 units 10.29

    12:42 ~ 13:04

    正确率：1/2

* [v] qa 1 unit  10.14

## cache tabs / process urls

* 需要消化 cached urls

* 处理 url 的时间比想象中要长，每一条大概需要 30 mins

* 可以使用 youtube 学一些英语课，比如 julia，octave 等，这样既锻炼了英语，也学到了东西

### tasks

* [ ] 调研`git revert`的用法

* [ ] 调研`git reflog`

* [o] process 1 url 10/01

    Resetting remote to a certain commit: <https://stackoverflow.com/questions/5816688/resetting-remote-to-a-certain-commit>

    feedback:

    4. 调研`ORIG_HEAD`, `git show ORIG_HEAD`

    5. 调研`git update-ref `

    6. 调研`git log --graph --all --oneline --decorate`

* [v] cache tabs 10.06

* [o] process 1 url 10.06

    <https://www.baeldung.com/linux/last-directory-file-from-file-path>

    feedback:

    1. deps

        1. awk, sed

    2. 没处理完，有时间了接着处理

    3. 这篇博客的思维方式也很好，先处理简单的情况，再处理 corner case，下次学习一下

* [v] sync bash

    feedback:

    1. 调研 bash 的数组

        使用 for 循环打印字符串数组中的所有单词，每个单词一行

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

* [v] cache tabs 10.09

* [v] cache tabs 10.10

* [v] cache tabs 10.11

* [v] cache tabs 10.14

* [v] cache tabs 10.23

    cached tabs:

    * what is load-store communication model in PCIe?
    
        <https://electronics.stackexchange.com/questions/527587/what-is-load-store-communication-model-in-pcie>

    * scale up域的拓扑

        <https://zhuanlan.zhihu.com/p/708991795>

    * 片间互联学习

        <https://zhuanlan.zhihu.com/p/1417863271>

    * scale up/out语义的特点

        <https://zhuanlan.zhihu.com/p/708996966>

    * RISC-V指令集讲解（6）load/store指令

        <https://zhuanlan.zhihu.com/p/394876584>

    * NVidia GPU指令集架构-寄存器

        <https://zhuanlan.zhihu.com/p/688616037>

    * AI System & AI Infra

        <https://github.com/chenzomi12/AISystem>

    * NVSHMEM: OPENSHMEM FOR GPU-CENTRIC COMMUNICATION

        <http://www.openshmem.org/site/sites/default/site_files/SC2017-BOF-NVIDIA.pdf>

    * NVIDIA NVSHMEM

        <https://docs.nvidia.com/nvshmem/index.html>

    * Introduction to Clos Network

        <https://web.stanford.edu/class/ee384y/Handouts/clos_networks.pdf>

    * Can You Really Compare Clos to Chassis when running AI applications? 

        <https://drivenets.com/blog/can-you-really-compare-clos-to-chassis-when-running-ai-applications/>

    * Infinity Fabric (IF) - AMD 

        <https://en.wikichip.org/wiki/amd/infinity_fabric>

    * Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs. 

        <https://github.com/facebookresearch/lingua>

    * How to debug the Linux kernel with GDB and QEMU?

        <https://stackoverflow.com/questions/11408041/how-to-debug-the-linux-kernel-with-gdb-and-qemu>

    * GDB+QEMU调试内核模块(实践篇)

        <https://www.cnblogs.com/powerrailgun/p/12161295.html>

    * qemu debug 输出 qemu gdb调试

        <https://blog.51cto.com/u_16213559/11347864>

    * 在qemu平台使用gdb调试程序

        <https://blog.csdn.net/weixin_42031299/article/details/135028500>

* [o] process 1 url  10.23

    <https://cloud.tencent.com/developer/article/1805119>

    Deps:

    1. 建立 bash note qa

* [v] cache tabs  10.29

    13:39 ~ 13:55

* [v] process 1 urls  10.29

    14:03 ~ 14:19

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

    1. 目前看到 pdf P52 2.2 使用Python实现感知器学习算法

* [ ] 调研 三维的 Swiss Roll

* [v] 调研《Python机器学习》

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

* [v] 调研《Python机器学习》

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

## gpu driver

### cache

* nccl 调试记录

    * 设置环境变量`NCCL_SHM_DISABLE=1`可以禁用 shared host memory，此时会使用 socket 进行通信

    * nccl 调用了`p2pCanConnect()`和`shmCanConnect()`，但是后续会调用`shmSendConnect()`, `shmRecvConnect()`，并未调用 p2p 相关的函数，说明传输数据使用的是 shared host memory，并不是 pcie.

    * 在建立 ring 连接时（`ncclTransportRingConnect()`），调用`ncclTransportP2pSetup()`建立 p2p 连接

        其中，会调用`selectTransport()` -> `transportComm->setup()`，最终调用到`shmRecvSetup()`。

        显然`setup()`函数指针在前面已经被替换成了`shmRecvSetup()`。

        目前看来，应该是用`struct ncclTransport shmTransport;`完成的替换，这个结构体里包含了 proxy 所需要用到的所有 shm 相关的函数。

    * `shmTransport`既包含在`struct ncclTransport* ncclTransports[NTRANSPORTS]`数组中，可以用 transport 索引直接调用到，对应的数组的索引是 1

        `p2pTransport`对应数组的索引是 0，`netTransport`对应 2，`collNetTransport`对应 3。

    * `ncclTransports`在五处地方被使用
    
        1. `proxyConnInit()`未被调用

        2. `proxyFree()`：未调用

        3. `ncclProxyConnect()`：未调用

        4. `selectTransport()`：调用

        5. `ncclTopoComputePaths()`

        说明全程没有用到 proxy。无法简单看代码看出逻辑，可能只要在同一台机器上就不需要创建 proxy。

        猜想：这个可能是在`groupLaunch()` -> `asyncJobLaunch()`阶段就判断出了不需要创建 proxy connect。

    * nccl 中`prims_ll.h`文件里有挺多 load, store 相关的函数，但是整个 nccl 中关于 atomic 的函数并不多。由此推断 nccl 很有可能不包含 load, store, atomic 的通信功能

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


* 在`nvmlwrap.cc:156`这里，当`a = 0, b = 1`时，`ncclNvmlDevicePairs[0][1]`被修改。

    修改它调用的是`nvmlDeviceGetP2PStatus()`函数。

### tasks

* [v] 调研 pci host bridge

* [v] 调研 qemu gdb server

* [ ] 调研 tenstorrent

* [ ] 调研制作 docker image

    1. 透传一个 nvidia device 可以成功跑通 cuda test
    
    2. 透传两个 nvidia gpu，调研是否能跑通 nccl
    
    3. 调研 2 个 gpu 的通信方式

        1. shared host memory

        2. pcie p2p

        3. socket

        4. nvlink

* [v] 调研 load, store, atomic

* [v] 调研 nccl p2p NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED 出现的原因

    feedback:

    2. 实体机上可以跑通 p2p

        两种模式都可以跑通：

        1. P2P/CUMEM/CE

        2. P2P/direct pointer

        跑不通的模式：

        1. SHM/direct/direct

        在调用函数`pfn_nvmlDeviceGetP2PStatus()`时，得到 pcie p2p 不可用的结果。nvml 是 nvidia management library，是 nv 的一个库。显然这个函数是从其他 so 库中加载进来的。

* [v] 调研 nccl p2p

    feedback:

    2. 为什么`p2pCanConnect()`会被执行多次？ 经 cnt 统计一共调用了 16 次。

        nccl 会起两个线程，每个线程独立扫描一遍本机资源，对于本机的两个 gpu，都判断一次 p2p can connect，即 0 - 1, 1 - 0， 因此`p2pCanConnect()`会被调用 4 次。

        1. thread 79, g = 0, p = 1

        2. thread 80, g = 0, p = 1

        3. thread 79, g = 1, p = 0

        4. thread 80, g = 1, p = 0

        5. thread 79, g = 0, p = 1

            这里开始第二次调用`ncclTopoComputePaths()`, recompute paths after triming

        6. thread 80, g = 0, p = 1

        7. thread 79, g = 1, p = 0

        8. thread 80, g = 1, p = 0

        9. thread 36, `ncclAsyncJobMain()` -> `ncclCollPreconnectFunc()` -> `ncclTransportRingConnect()` -> `ncclTransportP2pSetup()` -> `selectTransport()` -> `p2pCanConnect()`, c = 0

        10. thread 37, 

        11. thread 37, c = 1

        12. thread 36, c = 1

        13. thread 36, c = 0

            从这里开始，调用`selectTransport<1>()`

        14. thread 37, c = 0

        15. thread 36, c = 1

        16. thread 37, c = 1

    3. c 为什么会从 0 循环到 1？

        因为`sendMask ＝ 3`，只有低 2 位为 1.

        看不出来 sendMask，recvMask 有什么特别的二进制含义，可能只是为了省内存。

    4. 在 gdb 设置 schedule locking 时，其他线程会被 freeze。

        是否可以让其他线程也运行，但只在当前线程触发断点？

    5. `ncclNvmlDevicePairs[0][1].p2pStatusRead`与`p2pStatusWrite`的值都为`NVML_P2P_STATUS_CHIPSET_NOT_SUPPORTED`

        `ncclNvmlDevicePairInfo ncclNvmlDevicePairs`是一个全局数组，专门记录 p2p 能力的。

* [v] 调研 50 机器上的 nccl 调试

    feedback:

    1. 50 机器物理机上仍需要在`paths.cc`文件中的`ncclTopoCheckP2p()`函数里添加`path->type = PATH_PIX;`，重新编译 nccl，才能使用 pcie p2p，否则只设置 nccl 环境变量无法跑通 p2p.

        同时，不能设置`NCCL_P2P_LEVEL`环境变量。把它设置为`PIX`也跑不通。

## HPC comm

* [o] 调研 nvidia p2p

    feedback:

    1. 在一个虚拟机 node 上透传两个 cuda device，运行 nccl 时，默认情况下走的是 shared memory 传输数据，并没有启用 pcie 的 p2p

    2. 修改环境变量`NCCL_P2P_LEVEL`, `NCCL_P2P_DIRECT_DISABLE`, `NCCL_P2P_DISABLE`都无法启动或禁止 p2p

    3. 设置环境变量`NCCL_SHM_DISABLE=1`可以禁用 shared host memory，此时会使用

* [v] 调研 nccl p2p

    feedback:

    1. nccl 调用了`p2pCanConnect()`和`shmCanConnect()`，但是后续会调用`shmSendConnect()`, `shmRecvConnect()`，并未调用 p2p 相关的函数，说明传输数据使用的是 shared host memory，并不是 pcie。

    2. [ ] 调研 vscode 多线程 debug

    3. 目前看起来是在`ncclTopoCheckP2p()`处失败的

* [v] 调研 pci host bridge

    feedback:

    1. 发现本机资源的几个关键函数：`ncclTopoGetSystem()` -> `ncclTopoComputePaths()` -> `ncclTopoTrimSystem()`

        目前看来是在`ncclTopoComputePaths()`中判断了 pcie p2p 不可用。

        这里的不可用有可能是逻辑判断有问题，也有可能是上一个函数`ncclTopoGetSystem()`在获取资源时，获取的原始数据有误。

    2. 在建立 ring 连接时（`ncclTransportRingConnect()`），调用`ncclTransportP2pSetup()`建立 p2p 连接

        其中，会调用`selectTransport()` -> `transportComm->setup()`，最终调用到`shmRecvSetup()`。

        显然`setup()`函数指针在前面已经被替换成了`shmRecvSetup()`。

        目前看来，应该是用`struct ncclTransport shmTransport;`完成的替换，这个结构体里包含了 proxy 所需要用到的所有 shm 相关的函数。

    3. `shmTransport`既包含在`struct ncclTransport* ncclTransports[NTRANSPORTS]`数组中，可以用 transport 索引直接调用到，对应的数组的索引是 1

        `p2pTransport`对应数组的索引是 0，`netTransport`对应 2，`collNetTransport`对应 3。

    4. `ncclTransports`在五处地方被使用
    
        1. `proxyConnInit()`未被调用

        2. `proxyFree()`：未调用

        3. `ncclProxyConnect()`：未调用

        4. `selectTransport()`：调用

        5. `ncclTopoComputePaths()`

        说明全程没有用到 proxy。无法简单看代码看出逻辑，可能只要在同一台机器上就不需要创建 proxy。

        猜想：这个可能是在`groupLaunch()` -> `asyncJobLaunch()`阶段就判断出了不需要创建 proxy connect。

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

    6. cached tabs

        vscode 多线程调试: <https://zhuanlan.zhihu.com/p/704723451>

    7. 多线程调试时锁定单线程

        GDB scheduler-locking 命令详解

        <https://www.cnblogs.com/pugang/p/7698772.html>

    8. gdb+vscode进行调试12——使用gdb调试多线程 如何实现只对某个线程断点，其他线程正常运行

        <https://blog.csdn.net/xiaoshengsinian/article/details/130151878?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-130151878-blog-140669886.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-130151878-blog-140669886.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=1>

* [o] 调研 HPC 通信 ppt

    feedback:

    1. guideline

        1. optimizing an matrix multiplying task

            * OpenMP

                多线程，不支持多进程，多 node

                不支持 avx512 等 SIMD 指令，不支持 GPU，不支持 MKL，BLAS 等数学运算库

            * OpenMPI

                设计一系列集合通信原语（collective communication primitives），支持更大的集群。

                scatter, gather, reduce, all scatter, all gather, all reduce, broadcast

                Barrier

                重新优化矩阵乘法的代码

                OpenMPI 使用 socket 进行多进程，多 node 通信。

                OpenMPI 支持网络拓扑探测，找到通信效率最高的拓扑，常用拓扑结构为 ring 和 tree（ring 为什么更优？）

            * 集合通信与深度学习

                训练时梯度的依赖（all reduce）, inference 时的计算（send recv）

        2. NCCL 与 chunk pipeline

            nccl 是在 OpenMPI 的基础上提出 chunk pipeline 的方法，使得传输效率更高。

            nccl 的主要拓展：

            * openmpi 只支持 socket 进行跨进程，跨节点通信，nccl 增加了 pcie p2p, shared host memory, infiniband rdma, gpu direct rdma, nvlink, nvswitch 的支持

            * openmpi 调用操作系统的 socket 接口，效率较低，nccl 采用多线程异步 + 主动 poll 事件的方式，不处理中断，使计算任务、事件处理和数据传输三者代码隔离，效率更高，通常会几个 cpu 线程跑满

            * openmpi 的数据传输和计算任务放在一起，nccl 由于做了异步代码隔离，所以可以起多个线程多个 channel 进行通信，使得 cpu 性能不会成为瓶颈


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

* 先重新编译内核 5.19.17，然后再安装 ofed 的驱动（使用`--force-dkms`），然后再 insmod 自己的 ib aux driver，就没有兼容性的问题了

    * 2024/08/15: 如果需要換系统内核，并重新安装 ofed 驱动，那么需要将 ofed 源码从 tar 里重新解压出来。因为在之前编译 dkms 时，在源码目录里生成一些文件，这些文件会导致驱动无法加载成功

* ib core 默认没有把 post send, post recv 和 poll cq 放到 kmd 里，而是交由 umd 处理。

        可以在 ib verbs mask 列表里看到少了这几个 mask。

* 如果需要对不同设备，函数做出不同的行为，一种方法增加一个`enum`类型的函数参数，判断调用者的情况。另一种方法是增加一个编译宏，然后使用`#ifdef xxx`来检测，这样可以在编译时判断调用函数的主体的情况。

    为了只编译一份 lib 就适用多种情况，目前采用的是`enum`方案。

* mpirun 使用 hostname 和 ip addr 的两个注意事项

    * 如果使用 hostname，那么是去`~/.ssh/config`文件中找对应的配置，连接 ssh 

    * 如果使用 ip addr，那么 route 的路由顺序不对可能会导致无法连通

* [ ] 调研`ssh-add`，`ssh-agent`的作用

* [v] 调研`adduser`和`useradd`有什么不同？

* 对于主动采样操作的 log 格式参考

    1. 当成功 poll 到 valid event 时，显示在这之前 poll empty event 多少次，并且清零 empty event 的计数

    2. 没有 poll 到 valid event 时，每 1000 次显示一次结果

* mpi tutorial 的 github repo: <https://github.com/mpitutorial/mpitutorial/tree/gh-pages>

* [ ] ln 是否能创建文件夹的 hard link?

* [ ] 在一个 cq 上申请多个 qp，对于每个 qp 都设置一个 post send 时，需要注意 max cqe 的数量是否够用，这个参数在 create cq 时需要填入。

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

### tasks

* [ ] 调研：`ibv_get_cq_event()`会不会消耗`ibv_poll_cq()`的 wc？

* [ ] 调研为什么 cable 不支持高速率

* [ ] 调研`MPI_Probe`, <https://mpitutorial.com/tutorials/dynamic-receiving-with-mpi-probe-and-mpi-status/>

* [ ] 调研使用`MPI_ERROR`接收未知长度数据

* [ ] 调研 PCI relaxed ordering 

* [ ] 调研`fprintf(stderr," Internal error, existing.\n");`的用法

* [ ] 调研 diff 命令的用法

* [ ] 调研 pynccl 的用法

* [v] 调研 pytorch 调用 nccl wrapper function

* [ ] 调研 docker 中 app 的调试方法

* [ ] 调研 nccl app

* [ ] 调研 makefile 的 submodule

* [ ] 调研 ibv cmd req 中的 driver data

* [ ] 增加 client remote write to server test case

* [ ] 调研 open mpi 的 scatter, gather C 程序

* [ ] 调研 rdma repo 中 pcie driver

* [ ] 调研 iwarp 实现的 ibverbs 

* [ ] 调研 spdx

* [ ] 调研 poc-a pcie 地址映射和寄存器配置流程

* [ ] 调研使用 mmap 维护 cqe

* [] 调研 kgdb

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

### task

* [v] 调研 pci device passthrough in qemu

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

    1. 如果在编译内核时使用 ctrl + C 或者关机强制中断，那么第二次继续编译的时候有可能会编译不通过，此时可以`make clean -f`，再去编译就好了。

    2. 使用自己编译的 6.8.0 内核后，使用`qemu-system-x86-64`启动虚拟机时，使用`-vga std`或`-vga virtio`都可以启动起来，但是会黑屏几分钟。

        看起来像是内核在等什么东西，目前不清楚。

    3. 可以调研下内核源码中 menuconfig 中有什么支持 vga 的选项

## 分布式计算调研

tasks:

* [p] 调研分布式计算，MPI，grpc

	dependency:

	1. [v] 重装系统，安装 virtual box

	1. [v] 创建一些虚拟机

* [v] 调研 nccl

* [ ] 调研 mpi 如何给不同的 node 配置不同的环境变量？

## GPU Perf

cache:

* 调研算子的优化

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

    feedback:

    1. 无论是 windows 还是 ubuntu，rgp 都无法正常启动。

        尝试了各种 app，rgp 都无法 capture profiling。

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

* [v] virt: 调研虚拟机带负载强制关机再重启后的卡死问题

    主要看各个 buffer 是否有残留

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

* [v] 做一道图论题

* [v] 调研算法 resource

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

## 调研 meson, ninja



## 微积分

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

## Vim

本项目的目标是学完 vim，可以使用 vim 代替 vscode 进行日常的代码开发和调试，以及文本文档的编辑。

* vim 中的 redo 是哪个？

## 正则表达式

cache:

* python 里`print()`指定`end=None`仍然会打印换行符，只有指定`end=''`才会不打印换行

* python 里`re`正则表达式匹配的都是字符串，而`^`代表字符串的开头，并不代表一行的开始

    因此使用`^`去匹配每行的开始，其实是有问题的，只能匹配到一次。

* python 的`re`模块不支持非固定长度的 look behind 的匹配

    比如，`(?<+\[.*\]).*`，这个表达式本意是想向前匹配一个`[]`括号，括号中的内容任意，但不能有換行符。

    比如`[hello]this is the world`，想匹配到的内容是`this is the world`。

    但是上面的匹配是不允许的，因为 look behind 时，要匹配的内容是一个非固定长度字符串。

    具体来说可能是因为实现起来太复杂，具体可参考这里：<https://stackoverflow.com/questions/9030305/regular-expression-lookbehind-doesnt-work-with-quantifiers-or>

tasks:

* [v] python regular expression sync



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

