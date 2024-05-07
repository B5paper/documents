# 临时项目管理

这个文件中列出的项目，每天过一遍。

每天应该先启动任务执行系统，如果任务列表为空，那么跳转到这里，收集当天要执行的任务。

将任务管理系统作为一个基本的项目，每周至少整理一次。

每个项目都尽量遵循 复习（随机检测 qa） -> 学习 -> 调研 的步骤，不一定按这个顺序，但这三项起码要有。

有关“学习”：

1. 这里的学习是一个专有概念，其行为包括但不限于，阅读学习材料，做实验验证学习资料上的内容，做实验验证自己的猜想，记笔记，写一个新的 qa，完成任务列表上的任务

目前遇到的最大问题就是，有些问题不清楚能否解决，不清楚靠什么解决，只有在尝试解决的时候才会遇到很多新问题，并不断用已有知识去解决新问题。这样解决问题的思路就像一个无底洞，无法预知解决的时间，只能靠不断往其中投入时间。这类问题目前仍没有很好的解决方案。

不做任务执行系统，就没有顿挫的感觉，就不可能有高效率。只与项目池打交道，会产生无穷无尽的任务，从而无法分辨任务的边界，导致效率越来越低。

* 需要一个 graph 工具，建立不同的东西之间的连接

    stack 工具只适合任务的 trace

* 一个比较好的 explore 的想法是先从 amazon 上搜索书籍，然后在 zlib 或 libgen 上下载

* 每天在使用项目管理系统获得新任务之前，都必须将之前的每日任务列表进度同步到任务管理系统中，并加上`[processed]`标记

    因为任务执行系统会改变项目管理系统中的内容

    项目管理系统最好是当天结束时更新，这样才能不对第二天收集任务清单造成影响

    如果前一天的 task list 不被处理，那么第二天的项目池几乎不可能正常运行

	因为每天临时的变动太大了，远远超过按周处理的频率。

* log 不应该与 task 放在同一个文件里，翻来翻去太费时间

## cached

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

* cached tasK: 调研 u 盘里的资料

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

* 恶性竞争

    假如有两个公司 A 和 B，A 做的是热门泡沫方向，B 做的是扎实工作方向，结果 A 能融到资，B 融不到钱，最终 B 倒闭，A 最后也会因为泡沫破裂而倒闭。

    对融资公司来说，B 可能投资几年都什么起色，但是投资 A 可能有 10% 的机率挣大钱。理性的人可能会按比重分配资产投资，但是不要低估人性的贪婪，为了赌这一点很低的可能性，投资公司可能会加杠杆梭哈。

    假如有三家投资公司 C, D, E，这三家都投了 A 所在的行业，已知这个行业一定有一个胜出者，其实就是 C, D, E 在下注赌。C, D, E 都觉得最幸运的应该是自己，假如自己不跟注，那么挣钱的就一定会是别的两家投资公司。这样很不甘心。

    最终的结局是所有公司都赔了，每个环节的最优选择，导致了全局的最坏结果，非常讽刺。

* connection 非常重要

    必须有一些必做的任务，可以通向以前的问题，以前的缓存，以前的记忆，以前的工程。

    必须有一条路径通向过去。

    目前这些必做的任务就是 reorg 和 qa。

* 如果看到一个名词没有任何头绪，那么就去*调研*它

    比如当天的任务，如果没有头绪，那么就*调研***当天的任务**。

    比如某个项目，如果不知道怎么开展，那么就去*调研*这个项目。

    如果对于某个任务，忘了它是怎么开始的，想不起来和它相关的所有东西，那么就去调研这个任务。

* 串行的任务执行对提高效率非常重要，因为大脑频繁切换任务会降低效率

    可能训练汉语／英语文字阅读也是必要的。

    猜想：大脑的各个模块各个细胞都是通过规律的频率和谐地交互信息，才能达到高度专注的效果。切换任务会切换细胞之间互相配合的模式和频率，导致模式的调整，这个过程会导致注意力无法集中，处理问题的效率降低。

    使任务串行，一个非常大的挑战就是在规定的时间内使用手机，在执行任务时不看。

* 调研一下 zig

* cached task: 将 projects 加入 reorg 中

* 要想对一个东西有熟悉的理解和熟练的运用，必须将它作为更复杂的东西的一部分

* 如果前一天的任务清单上任务没有完成该怎么办？

    目前的方案是全部判为未完成，在第二天重新从项目管理系统中获取新的任务，不再考虑这些任务。

* cached task: 写一个数字图像处理库，暂时不包含深度学习

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

* 应该对个人系统建立一个项目进行复习

	每次随机选择一个文件，跳转到随机一行，往下读 20 mins

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

* 应该建立一个 reference resources 目录，把所有的引用资源都放到这里

    如果每个目录下都是从`ref_0`编号，移动笔记时就会混乱。

    如果统一管理 reference，就不会出现这个问题。

    可以从`ref_0`一直编号到`ref_100`，随着 ref 不断被处理，再删掉`ref_0`重新开始。

    每次删除的时候，可以在 documents 里用`grep`搜索`ref_0`保证没有被引用的地方。

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

* 以后再做 c++ 的 qa，要么写一点点语法，要么就是做题

* 找一个 c++ 学习资料，系统学一下比较好

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

* [v] 思考 qa 的 dependency 机制

    dependency:

    1. [v] 需要先拿到每个 unit 对应的索引号

        1. [v] 需要先学会用 flex 做词法分析

            自己用代码写状态机去解析文件格式太累了，如果用 flex 的话可以大大提高效率。
    
    自动化的 dep 检测或许是不需要的。只要能记录就可以了，然后去手动搜索。

* 如果遇到的知识是条目式的，没有理解难度，但是数量比较大，是否该将其列入到记忆任务中？还是说调研时先跳过？

    * append (2024.03.21)

        如果没有理解难度，那么基本上看一遍就能记住，不需要列到记忆任务中。这个效率其实挺高的，不会占太多时间。

* [思考] 无论是 qa 还是 note，除了添加内容，更需要优化。这个优化该如何进行？

    如果只添加内容，不优化内容，那么东西越来越多，而且组织混乱。

    目前正在进行的一个优化是将按时间顺序写下的笔记，重新整理成按结构存储的数据。

## reorg

* [ ] 建立一个文档索引表，写一个 python 程序，每次随机选择一个文件，跳转到随机一行

	feedback:

	3. 目前没有包含子目录

		比较好的办法是先收集目录，再排除 ignore 目录，再收集文件，再排除 ignore 文件，

		最后直接根据文件的路径随机选择一个。

	4. 在项目管理中加上这个项目，系统的自检查

* [ ] 完成程序：遍历索引和目录，找到`ignore.md`中无效的索引和未被收录的目录/文件

* [ ] python 判断两个路径是否等价，包括相对路径，绝对路径，文件，目录

* [ ] 为 reorg 程序增加指定文件的随机一行的功能

* random select 时，将 projects 文件夹也包含进去

* [v] reorg 30 mins

* [v] reorg 30 mins

* [v] reorg 30 mins

* [v] reorg 30 mins 04/24

* [v] 调研 sync 的含义

* [v] reorg  20 mins 04/25

* [v] reorg 30 mins 04/28

    feedback:

    1. java 的 note 大部分已经空间化了，没有什么需要 reorg 的

* [v] reorg 30 mins 05/03

    feedback:

    1. 清理了 latex 和 opengl 的笔记，但是这两份笔记只有几行，并没有什么需要整理的

        最好还是先让程序扫描一遍目录，然后根据规则排除一些文件，然后再跟上一次使用的数据库做对比，看看新扫描的增加了哪些文件，缺少了哪些文件，手动确认是否更新数据库。

        然后根据更新完的数据库，重新计算每个文件被选中的概率权重，最后根据权重，随机选择一个文件。

        如果被选中的文件没有什么好 reorg 的，那么可以手动调低权重，让它下次出现的概率变低。

        如果一个文件整理完后，发现还有许多要整理的地方，那么就手动升高其权重，增加它出现的概率。

        其实降低别某一个文件概率就是变相地升高其他文件的概率，所以也可以不去设置手动调高概率。

* [v] reorg 30 mins 05/06

    feedback:

    1. 使用 python ＋ re 写一个英语单词的 parser，每次随机检测指定数量个单词，保存索引，后面每次复习时检测上次抽取的单词 + 融合前几次抽取的单词，时间越久的单词出现的概率越小。

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

Tasks:

* 在 vim 中根据正则表达式搜索指定索引所在的位置

* [v] qa 4 units 04/25  20 mins

    feedback:

    1. 3 pass, 1 fail

* [ ] 调研 qa 程序执行`/main.py --list /home/hlc/Documents/documents/Linux/linux_driver_note_qa.md`的 bug

* [v] qa 4 units 20 mins

    feedback:

    1. 修复 bug:

        `python3 main.py --create-id /home/hlc/Documents/documents/Linux/linux_driver_note_qa.md`

* [v] qa 4 units 05/03

    答对题数：2/4

    feedback:

    1. 增加 linux driver qa:

        配置 vscode 的内核驱动开发环境

    2. 误删了 random_exam_py 文件夹，需要重写

        记得用 git 保存到 remote 仓库

* [x] qa 4 units 05/06

    feedback:

    1. 没法 qa 了，先修复 qa 的 python 程序。

* [ ] 使用`./main --id-to-idx <id> <qa_file>`找到指定哈希值的索引

* [ ] 调研 qa unit 中 dep 功能

## cache tabs / process urls

* 需要消化 cached urls

* 处理 url 的时间比想象中要长，每一条大概需要 30 mins

tasks:

* [v] cache tabs  10 mins

* [v] process 1 url  30 mins

* [v] cache tabs  10 mins

* [v] process 1 url  30 mins

* [v] cache tabs  10 mins 04/25

* [v] process 1 url  30 mins 04/25

* [v] cache tabs 10 mins 04/25

* [v] process 1 url 30 mins 04/25

* [v] cache tabs 10 mins 04/28

* [v] process 1 url 30 mins 04/28

* [v] cache tabs 05/06

* [v] process 1 url 05/06

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

tasks:

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

cached:

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

参考资料：

1. pdf 电子书《pdfcoffee.com_opencl-programming-guidepdf-pdf-free》

2. 其他网上的资料

基本的东西会了，剩下的系统地看一看吧，查漏补缺。

重点看一看内置函数。

任务列表：

* [o] 写一个中值滤波算子，用前面学到的内置函数

    dependency:

    1. [v] 调研中值滤波的数学算法

        feedback:

        1. 下载一本数字图像处理的书比较好，网上找的资料太碎片了

        2. 单通道图像求中值的代码可以参考`ref_0`

            下次可以直接试试单张图片。

    2. [o] 使用 cpu 实现中值滤波

        目前只写完了读取图片。见`ref_3`。

	3. 应该 sync 一下 opencl

	4. 已经完成代码，详见`ref_5`。

		同样一张图片执行中值滤波，cpu 使用的 clock 时间为 13163，gpu 使用的时间为 1137。

* [v] 调研第 5 章 current progress: P201 Image Read and Write Functions

	feedback:

	1. 增加每天的英文阅读量

	2. 需要提出一种任务切换需要满足的最小条件，防止任务切换过于频繁

	3. dependency

		需要看 chapter 8: images and samplers

* [v] 改进一下 simple opencl，`add_buf()`同时申请显存和内存。

* [ ] 调研

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

## awk

cache:

* awk 使用`#`标记注释

resources:

1. <https://www.gnu.org/software/gawk/manual/html_node/index.html#SEC_Contents>

## linux driver

cache:

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

    `mutex_lock`, `mutex_unlock`, `mutex_destroy`

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

* [v] linux driver sync

* [v] linux driver: sync 30 mins 05/03

    feedback:

    3. 中间多次未看到实验结果，因为另一个 terminal 没有登录 ssh

        有没有什么办法可以避免这个问题？

* [v] linux driver sync 05/06

    feedback:

    1. what have been done:
    
        1. 整理了 linux module driver 中的 goto 错误处理代码风格
    
        2. 在笔记中切分了设备驱动的 note 和 cache 两部分

        3. 整理了`alloc_chrdev_region()`动态申请设备号的用法，并增加了一个 qa

* [ ] linux driver sync

    主要是看 ioctl 相关的内容，可以尝试整理并测试。

    然后是看`Data exchange between kernel space and user space`

* [ ] 调研`kzalloc`, `kfree`

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

## 分布式计算调研

* [p] 调研分布式计算，MPI，grpc

	dependency:

	1. [v] 重装系统，安装 virtual box

	1. [ ] 创建一些虚拟机

## 微积分

## C/C++

主要任务是学完 modern c++，即 c++23 及之前的内容，找一些开源库看一看。

cache:

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

* [ ] python regular expression sync

## 数学 Mathematics

resources:

1. 《高等数学》同济大学应用数学系

tasks:

* [ ] 调研第八章 多元函数微分法及其应用

## 其他

* c++ 版本的 opencl 入门

    <https://objectcomputing.com/resources/publications/sett/july-2011-gpu-computing-with-opencl>

* libeigen

    other linger algebra libraries: <https://stackoverflow.com/questions/1380371/what-are-the-most-widely-used-c-vector-matrix-math-linear-algebra-libraries-a>

    以 libeigen 为主，其他的库对比着学下。

    * numcpp

        矩阵运算库

        <https://github.com/dpilger26/NumCpp>

        <https://zhuanlan.zhihu.com/p/341485401>

