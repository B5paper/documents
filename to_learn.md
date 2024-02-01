1. git 合并多个 commit，保持 history 整洁：<https://cloud.tencent.com/developer/article/1690638>

1. 有时 local branch 会同时 ahead / behind remote branch，如何查看提前/落后了多少？<https://stackoverflow.com/questions/17719829/check-if-local-git-repo-is-ahead-behind-remote>

1. sync 两个 local branches: <https://stackoverflow.com/questions/20108129/how-to-sync-two-local-branches>

1. 使用 go 语言调用 containerd 的接口：<https://mobyproject.org/blog/2017/08/15/containerd-getting-started/>

1. `unknown service runtime.v1.RuntimeService`: <https://github.com/containerd/containerd/discussions/6616>

1. k8s taint and tolerations: 

    1. <https://www.densify.com/kubernetes-autoscaling/kubernetes-taints/>

    1. <https://kubernetes.io/docs/reference/labels-annotations-taints/>

    1. <https://stackoverflow.com/questions/56614136/how-to-remove-kube-taints-from-worker-nodes-taints-node-kubernetes-io-unreachab>

1. A namespace is stuck in the Terminating state

    1. <https://www.ibm.com/docs/en/cloud-private/3.2.0?topic=console-namespace-is-stuck-in-terminating-state>

    1. <https://stackoverflow.com/questions/55935173/kubernetes-pods-stuck-with-in-terminating-state>

1. git 相关

    Squash: <https://www.internalpointers.com/post/squash-commits-into-one-git>

    reset: <https://www.atlassian.com/git/tutorials/undoing-changes/git-reset>

    rebase: <https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase#:~:text=What%20is%20git%20rebase%3F,of%20a%20feature%20branching%20workflow.>

    branch: <https://www.atlassian.com/git/tutorials/using-branches#:~:text=In%20Git%2C%20branches%20are%20a,branch%20to%20encapsulate%20your%20changes.>

    set upstream branch: <https://devconnected.com/how-to-set-upstream-branch-on-git/>

    设置 upstream，并 fork: <https://www.freecodecamp.org/news/how-to-sync-your-fork-with-the-original-git-repository/>

    Creating and deleting branches within your repository: <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository>

    Pull a certain branch from the remote server: <https://stackoverflow.com/questions/1709177/pull-a-certain-branch-from-the-remote-server>

    How to stop container in containerd/ctr: <https://stackoverflow.com/questions/67171982/how-to-stop-container-in-containerd-ctr>

    Resetting remote to a certain commit: <https://stackoverflow.com/questions/5816688/resetting-remote-to-a-certain-commit>

1. awk

    1. <https://www.gnu.org/software/gawk/manual/html_node/Quoting.html>

    1. <https://www.gnu.org/software/gawk/manual/html_node/Comments.html>

1. bash

    1. <https://www.baeldung.com/linux/single-quote-within-single-quoted-string>

    1. <https://www.baeldung.com/linux/last-directory-file-from-file-path>

    1. <https://stackoverflow.com/questions/11226322/how-to-concatenate-two-strings-to-build-a-complete-path>

    1. <https://linuxhint.com/trim_string_bash/>

    1. Bash Case Statement: <https://linuxize.com/post/bash-case-statement/>

    1. Boolean variables in a shell script?: <https://stackoverflow.com/questions/2953646/how-can-i-declare-and-use-boolean-variables-in-a-shell-script>

    1. <https://unix.stackexchange.com/questions/187651/how-to-echo-single-quote-when-using-single-quote-to-wrap-special-characters-in>

    1. Bash Special Variables: <https://www.baeldung.com/linux/bash-special-variables>

    1. Bash script what is := for?

        <https://stackoverflow.com/questions/1064280/bash-script-what-is-for>

    1. recipe commences before first target. Stop 错误分析

        <https://blog.csdn.net/freege9/article/details/77987536>

    * Bash遍历字符串列表-腾讯云开发者社区-腾讯云

        <https://cloud.tencent.com/developer/article/1805119>

1. sed

    1. <https://stackoverflow.com/questions/46970466/how-to-replace-only-last-match-in-a-line-with-sed>

    1. Multiple commands syntax: <https://www.gnu.org/software/sed/manual/html_node/Multiple-commands-syntax.html>

1. `tar: file changed as we read it`

    tar 在打包目录时，要求目录中的文件不能有改动。如果还没来得及打包的文件被修改，那么就会报错。

    <https://stackoverflow.com/questions/20318852/tar-file-changed-as-we-read-it>

1. k8s 相关：

    Get a Shell to a Running Container: <https://kubernetes.io/docs/tasks/debug/debug-application/get-shell-running-container/>

    How to delete kubernetes stuck CRD deletion: <https://rogulski.it/blog/kubernetes-stuck-resource-action/>

    image pull backoff 可能的原因：<https://www.airplane.dev/blog/kubernetes-imagepullbackoff>

    遇到需要先 login，再 pull image 的情况：<https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/>

* docker relatives

    1. dockerfile ref: <https://docs.docker.com/engine/reference/builder/>

    1. dockerfile 中的 proxy 设置：<https://dev.to/zyfa/setup-the-proxy-for-dockerfile-building--4jc8>

    * docker rename

        <https://docs.docker.com/engine/reference/commandline/rename/>

    * COPY 复制文件

        <https://yeasy.gitbook.io/docker_practice/image/dockerfile/copy>

1. linux 中`cut`的用法：<https://linuxize.com/post/linux-cut-command/>

1. awk

    1. Basic Examples: <https://www.tutorialspoint.com/awk/awk_basic_examples.htm>

    1. How to Read Awk Input from STDIN in Linux: <https://www.tecmint.com/read-awk-input-from-stdin-in-linux/>

    1. 30 Examples for Awk Command in Text Processing: <https://likegeeks.com/awk-command/>

    1. How to get the second column from command output?: <https://stackoverflow.com/questions/16136943/how-to-get-the-second-column-from-command-output>

1. containers policy

    <https://github.com/containers/image/blob/main/docs/containers-policy.json.5.md#policy-requirements>

1. The Cargo Book: <https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html>

    有时间了看看

1. quay.io 上的报错：<https://stackoverflow.com/questions/61436031/error-response-from-daemon-get-https-quay-io-v2-unauthorized-access-to-the>

1. set tutorial: <https://www.grymoire.com/Unix/Sed.html#uh-24>

    Sed Command to Delete a Line: <https://linuxhint.com/sed-command-to-delete-a-line/>

    sed or awk: delete n lines following a pattern: <https://stackoverflow.com/questions/4396974/sed-or-awk-delete-n-lines-following-a-pattern>

1. Regular Expressions: <https://www.grymoire.com/Unix/Regular.html>

1. 学习编程语言

    learning programming through books and examples: <https://riptutorial.com/>

1. 一个个人博客，有时间了看看：<https://feisky.xyz/about/>

1. bash 启动时加载配置文件的过程

    <https://www.cnblogs.com/f-ck-need-u/p/7417651.html>

1. linux 查找包含指定字符串的文件：<https://blog.csdn.net/qq_36344214/article/details/115912107>

1. `?=` in bash: <https://unix.stackexchange.com/questions/367821/what-is-in-bash>

1. Programming-Idioms

    常见的编程任务

    <https://programming-idioms.org/all-idioms>

1. rust 入坑指南：<https://www.bbsmax.com/A/nAJv1Elazr/>

1. Clear explanation of Rust’s module system: <https://www.sheshbabu.com/posts/rust-module-system/>

1. tyk: <https://tyk.io/docs/key-concepts/tcp-proxy/>

1. 英语语法：<https://hzpt-inet-club.github.io/english-note/guide/grammar.html>

1. tiny proxy: <http://tinyproxy.github.io/>

1. 计算机相关的 book 索引：<https://github.com/justjavac/free-programming-books-zh_CN#cc>

1. 密码学书籍：《密码学原理与实践（第三版）》

1. 多层代理

    1. <https://www.jianshu.com/p/65e5dc421efd?u_atoken=9e76061d-c931-4a8b-afcb-7112b4b6226f&u_asession=01BlPNo6p8GAH0xxqYZVzSUNLH89eGueLFuyW5J_SswgUwb9cwrIm_ZiRDVRdOsm75X0KNBwm7Lovlpxjd_P_q4JsKWYrT3W_NKPr8w6oU7K-LBhYCXQLBox-yRMr3PkXjVpSIj1gN2LaYxLWUpsc1qWBkFo3NEHBv0PZUm6pbxQU&u_asig=052J1jEuZGEWMfO-oHpuKvz_yCBJPRdAKaWndk-Ed22acNhPPRxyslEHAzMNGPtip0hdoJDg9GPrHUG7GIMi3lW1Ch1Yxx0QW0Jmn4JfAyMO31yyhovOCxKEusJx9ix7cLQycQSxgoev4UB9YeQuPJ8-rOfoVlSaUKV_qo2uS81w79JS7q8ZD7Xtz2Ly-b0kmuyAKRFSVJkkdwVUnyHAIJzVXSzCAdq94XiTuPldyd9udvHNQQGlUC6smprnUXooZcU4HK4hPy3A7RVNDWi_-Gxu3h9VXwMyh6PgyDIVSG1W9g-3O41MaKlGvHUIo7KbJYTOybdtHlb3igdkDLY_odFSxogSNUdlAp_ZVEZry-KXhFG6Yl5zBlQuFaJfDxJN-bmWspDxyAEEo4kbsryBKb9Q&u_aref=qZ1Fmxk2l%2F%2FlS%2BpX9GAh3qI5uh4%3D>

    1. <https://blog.csdn.net/weixin_43047908/article/details/120684640>

    1. <https://blog.csdn.net/ss810540895/article/details/124987061>

1. 内网穿透 nps: <https://github.com/ehang-io/nps/blob/master/README_zh.md>

1. c语言标准库字符串操作概览：<http://akaedu.github.io/book/ch25s01.html>

1. kvm: <https://zhuanlan.zhihu.com/p/596287657>

1. Bootloader和BIOS、uboot和grub和bootmgr的区别: <https://blog.csdn.net/liao20081228/article/details/81297143>

1. 英语字典：

    1. <https://www.collinsdictionary.com/>

    1. <https://www.oxfordlearnersdictionaries.com/>

1. opencl reduction operation

    基本还是归并的思想，每次处理相邻的两个值，归并后再处理相临的两个值，直到最后只剩一个值。

    有时间了自己实现一个找最大值，最小值，或平均值的 reduction 操作算子。

    Ref:

    1. <https://stackoverflow.com/questions/20613013/opencl-float-sum-reduction>

    1. <https://dournac.org/info/gpu_sum_reduction>

    1. <https://dean-shaff.github.io/blog/c++/opencl/2020/03/29/opencl-reduction-sum.html>

1. 一些全局光照的项目

    1. LumenRenderer, <https://github.com/LumenPT/LumenRenderer>

        一个基于 volumetric bodie 的实时全局光照渲染器。是某个大学的学生课程项目。

        感觉效果应该不会太好，有时间了研究下他们的代码。

    1. <https://github.com/Friduric/voxel-cone-tracing>

        一个基于 voxel cone 的实时全局光照。7 年前的代码，看起来像是学术论文的复现。有空了简单看下。

    1. <https://github.com/Cigg/Voxel-Cone-Tracing>

        又一个基于 voxel cone 的实时全局光照，8 年前的代码。

        看起来像课程毕设。

    1. <https://github.com/djbozkosz/Light-Propagation-Volumes>

        一篇博士论文，基于 volume 的实时全局光。

    1. Strolle，<https://github.com/Patryk27/strolle>

        使用 rust + mesa 驱动开发的实时全局光照。感觉效果一般般，demo 挺阴森的。

    1. github 上搜 global illumination，就能搜到不少的库

    1. <https://github.com/EmbarkStudios/kajiya>

        基于 rust + vulkan 的实时全局光照。看起来像是个比较成熟的项目。

    1. <https://github.com/but0n/Ashes>

        一个使用 js + node 的光追项目

1. C/C++ 语言细节

    1. C 语言中 float 最大值为`FLT_MAX`，但是要使用这个值，需要`#include <float.h>`

        有空了研究下这个库的一些细节知识吧。

        Ref:

        1. <https://www.tutorialspoint.com/c_standard_library/float_h.htm>

        1. <https://stackoverflow.com/questions/4786663/limits-for-floating-point-types>

        1. <https://stackoverflow.com/questions/48630106/what-are-the-actual-min-max-values-for-float-and-double-c>
        
    1. lambda 表达式与函数指针

        <https://www.geeksforgeeks.org/lambda-expressions-vs-function-pointers/>

        假如一个 lambda 表达式 capture 了一些外部变量，那么该怎么写它的函数指针？

    1. Don't inherit from standard types

        <https://quuxplusone.github.io/blog/2018/12/11/dont-inherit-from-std-types/>

        没看过，也不知道讲什么的。自己尝试过继承`array<T, n>`，还挺好用的。

    * <https://en.cppreference.com/w/cpp/language/override>

        c++ override 关键字

    * const rvalue references

        <https://www.sandordargo.com/blog/2021/08/18/const-rvalue-references>

    * `std::remove_cvref`

        <https://en.cppreference.com/w/cpp/types/remove_cvref>

    * const_cast conversion

        <https://en.cppreference.com/w/cpp/language/const_cast>

    * c++ piecewise_construct

        <https://blog.csdn.net/newbeixue/article/details/111185671>

1. 如果一个 obj 文件没有法线信息，那么该怎么办？

    可以通过`facted_normal = normalize( cross( 2-3, 1-3))`估算一下。

    ref: <https://computergraphics.stackexchange.com/questions/10759/how-does-obj-format-determine-vertex-normals>

1. <https://github.com/EmbarkStudios/rust-gpu>

    rust-gpu

    盲猜是个基于 mesa 的 rust gpu 驱动

1. modelgl

    <https://github.com/moderngl/moderngl>

    一个 opengl 的 python binding。看了下 example，确实方便。有空了学学。

    另外一个相关项目： modelgl-window，<https://github.com/moderngl/moderngl-window>。用于提供类似 glfw 那样的窗口管理服务。

1. latex 的效果参考

    1. <https://www.maths.tcd.ie/~dwilkins/LaTeXPrimer/Calculus.html>

        一些常见微积分公式的 latex 写法

    1. <https://artofproblemsolving.com/wiki/index.php/LaTeX:Symbols>

        常用的 latex symbol

    1. <https://www.scijournal.org/articles/parallel-symbol-in-latex>

        latex 中常用的 parallel symbol

        这个网站好像还有不少和 latex 相关的资料，有时间了看看。

    1. <https://www.physicsread.com/latex-absolute-value/>

        绝对值的常见写法。

        同上，这个网站好像也有不少关于 latex 的知识。

1. Bi-Directional Reflectance Distribution Functions

    1. <https://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/chuckm/chuckm_BRDFs_overview.html>

        看起来挺正式的介绍，没仔细看过。

    1. <https://snr.unl.edu/agmet/brdf/brdf-definition.asp>

        同上。

* bandicoot

    基于 gpu 的 c++ 矩阵计算库

    <https://coot.sourceforge.io/docs.html>

    <https://gitlab.com/conradsnicta/bandicoot-code>

* spla

    线性代数库。

    <https://github.com/SparseLinearAlgebra/spla>

* cgal

    解析几何库

    <https://www.cgal.org/> 

* 有关解析几何中，line 的表示的一些资料，有空了看看

    Refs:

    1. <http://jongarvin.com/up/MCV4U/slides/vector_parametric_plane_handout.pdf>

    2. <https://www.britannica.com/science/analytic-geometry/Analytic-geometry-of-three-and-more-dimensions>

    3. <https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_AnalyticGeometry.html>

* 计算机图形学中有关光线追踪，反射，折射，相关的资料

    * 光线在球面的折射（refraction）：

        Refs:

        1. <https://phys.libretexts.org/Courses/University_of_California_Davis/UCD%3A_Physics_9B__Waves_Sound_Optics_Thermodynamics_and_Fluids/04%3A_Geometrical_Optics/4.04%3A_Spherical_Refractors>

        2. <https://www.toppr.com/guides/physics/ray-optics-and-optical-instruments/refraction-at-spherical-surface-and-by-lenses/>

        3. <https://samdriver.xyz/article/refraction-sphere>

    * manuka 渲染器

        <https://www.wetafx.co.nz/research-and-tech/technology/manuka/>

    * GAMES101 相关

        Refs:

        1. 视频：<https://www.bilibili.com/video/BV1X7411F744/?p=14&vd_source=39781faaf2433372c59bdb80774d648e>

        2. ppt 目录：<https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html>

        3. path tracing 相关

            1. <https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_15.pdf>

            1. <https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_16.pdf>

        4. 透明材质渲染

            <https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_17.pdf>

    * 有关 shadow 的资料

        Refs:

        1. <https://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/shadow.pdf>

        2. <https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/ligth-and-shadows.html>

    * 球与直线的相交

        <https://www.cnblogs.com/charlee44/p/13247167.html>

    * about ray tracing

        1. <https://github.com/dannyfritz/awesome-ray-tracing>

        2. <https://github.com/embree/embree>

        3. <https://www.scratchapixel.com/index.html>

        4. <https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html>

        5. <https://github.com/NVIDIAGameWorks/RTXGI>

        6. <https://github.com/GPUOpen-LibrariesAndSDKs/HIPRTSDK>

        7. <https://gpuopen.com/hiprt/>

        8. <https://github.com/Khrylx/DSGPURayTracing>

        9. <https://github.com/blahgeek/hptracing>

* 开源渲染器 mitsuba3

    <https://www.mitsuba-renderer.org/>

    <https://github.com/mitsuba-renderer/mitsuba3> 

* 水面渲染

    都是 github 链接，感觉没啥用，未来可能把这个删了。

    <https://github.com/damienfir/water_rendering>

    <https://github.com/victorpohren/Paraview-to-POVRay-Water-Render>

    <https://github.com/IceLanguage/WaterRenderingLaboratory>

    <https://github.com/ACskyline/Wave-Particles-with-Interactive-Vortices>

    <https://github.com/hehao98/WaterRendering>

    <https://github.com/marcozakaria/URP-LWRP-Shaders>

    <https://community.khronos.org/t/refraction-reflection/35950>

    <https://zhuanlan.zhihu.com/p/486631970>

    opengl water: <https://blog.bonzaisoftware.com/tnp/gl-water-tutorial/>

* tiny object loader

    <https://github.com/tinyobjloader/tinyobjloader>

* opengl tutorial

    <https://zhuanlan.zhihu.com/p/657043402>

* tex commands

    1. <https://docs.mathjax.org/en/latest/input/tex/macros/index.html>

    1. <https://texdoc.org/serve/texbytopic/0>

    1. <https://www.tug.org/utilities/plain/cseq.html>

* amd gpu

    <https://gpuopen.com/>

    <https://cgpress.org/archives/amd-open-sources-firerays-2-0.html>

    <https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK>

* LLaMA: Open and Efficient Foundation Language Models

    <https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/>

    看起来像 facebook 对 llama 的论文简介推荐页。

    目前还没仔细看过这个网页。

    如果有论文介绍的话，以后可能会拜读一下。

* ntroducing Llama 2

    The next generation of our open source large language model

    <https://ai.meta.com/llama/>

    llama2 的欢迎页。不过为什么 url 还是 llama？

    似乎可以直接下载模式，并且网页上有模型参数及性能对比。

    有时间的话玩玩。

* 万字长文：AIGC技术与应用全解析

    <https://zhuanlan.zhihu.com/p/607822576>

    有关 aigc 及大语言模型的常见应用，以及背后用到的技术，以及如果要做成应用，该怎么做。

    看到有挺多应用的，音视频，语言，多模态，视觉，有时间了简单了解一下。

* Stable Diffusion

    <https://github.com/CompVis/stable-diffusion>

    stable diffusion 的 github repo 页面。

    似乎只需要 10G 显存就可以部署起来，有时间了试试吧。

    stable diffusion 是 22 年夏天开源的，距今也只有一年半的时间，但是总以为它是 17 年开源的。

* Stable Diffusion Launch Announcement

    <https://stability.ai/news/stable-diffusion-announcement>

    stable diffusion 的官方运营网站，提供了 api 接口。

* Llama 2

    <https://github.com/facebookresearch/llama>

    llama 的 github repo。

    不知道有啥用。

    因为名气挺大的，所以有机会了拜读下代码。

* Umi-OCR 文字识别工具

    <https://github.com/hiroi-sora/Umi-OCR>

    免费的 ocr 工具。免费的东西谁不喜欢呢？

* torchgpipe

    <https://github.com/kakaobrain/torchgpipe>

    gpipe 的 torch 实现，已经是 4，5 年前的代码了，主流版本的实现已经合并入 torch 中了。

    有时间的话简单看下，这个可能是原始版本的 gpipe 实现。

* 深入理解 Megatron-LM（2）原理介绍

    <https://zhuanlan.zhihu.com/p/650383289>

    megatron 的论文解读，我觉得讲得还不错。这个系列可以看作和原论文相互地看，互相印证。

    <https://zhuanlan.zhihu.com/p/650234985>

    这个是 megatron 系列知乎专栏的第一篇。

* DeepSpeed之ZeRO系列：将显存优化进行到底

    <https://zhuanlan.zhihu.com/p/513571706>

    讲 zero 系列的，我觉得讲得还可以。

* ZeRO-Offload: Democratizing Billion-Scale Model Training

    <https://www.usenix.org/conference/atc21/presentation/ren-jie>

    zero offload 的论文网站

* [译] DeepSpeed：所有人都能用的超大规模模型训练工具

    <https://zhuanlan.zhihu.com/p/343570325>

    系统地讲了下 deepspeed 中用到的各种技术，不仅局限于 zero 系列。

* 图解大模型训练之：流水线并行（Pipeline Parallelism），以Gpipe为例

    <https://zhuanlan.zhihu.com/p/613196255>

    分析了 gpipe 的原理以及时间空间复杂度。讲得还是挺好的。

* 如何评价微软开源的分布式训练框架deepspeed？

    <https://www.zhihu.com/question/371094177/answer/2964829128>

    deepspeed 的解析。一般般吧，没仔细看。

* ColossalAI

    <https://github.com/hpcaitech/ColossalAI>

    colossal ai 是一个分布式训练的工具，实现了大部分的基本功能，比如 pipeline 之类的，而且仓库更新得也很频繁。

* 9个主流的分布式深度学习框架

    <https://zhuanlan.zhihu.com/p/582498905>

    很多都超过两年没更新了，大家主要用的还是 deepspeed, megatron, colossal ai 这三个。

* [细读经典]Megatron论文和代码详细分析(1)

    <https://zhuanlan.zhihu.com/p/366906920>

    megatron 的论文解读。前面已经有类似的论文解读了，这里只是提供另外一种视角看问题。

* 模型并行训练：为什么要用Megatron，DeepSpeed不够用吗？

    <https://zhuanlan.zhihu.com/p/670958880>

    站在了一个更高的视角上，对比 megatron 和 deepspeed 的优缺点，并分析一些遇到的实际问题。

* DeepSpeed ZeRO理论与VLM大模型训练实践

    <https://zhuanlan.zhihu.com/p/675360966>

* Pytorch Distributed

    <https://zhuanlan.zhihu.com/p/348177135>

    有关 pytorch 的分布式训练。没看。

    <https://zhuanlan.zhihu.com/p/615754302>

    同样，没看。

* 论文阅读: PyTorch Distributed: Experiences on Accelerating Data Parallel Training

    <https://zhuanlan.zhihu.com/p/666243122>

* pytorch DistributedDataParallel基本原理及应用

    <https://zhuanlan.zhihu.com/p/420894601>

    没看。

* Pytorch 分布式数据 Distributed Data Parallal

    <https://zhuanlan.zhihu.com/p/460966888>

    没看。

* Pytorch - 分布式通信原语（附源码）

    <https://zhuanlan.zhihu.com/p/478953028>

    没看。

* 分布式机器学习：异步SGD和Hogwild!算法（Pytorch）

    <https://zhuanlan.zhihu.com/p/606063318>

    没看。

* 自动求导系列

    * autograd

    <https://github.com/HIPS/autograd>

        一个比较成熟的自动求慰框架，可以对一些简单的操作求导。

    * <https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/>

        pytorch 的自动求导讲解。没看。

    * <https://medium.com/howsofcoding/pytorch-quick-reference-auto-grad-d615ca7c46e>

        没看。

    * <https://www.youtube.com/watch?v=RxmBukb-Om4&list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs>

        youtube 上现场撸代码的视频。

        <https://github.com/joelgrus/autograd/tree/part01>

        配套 repo。

    * <https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/tutorials/tut4.pdf>

        没看。

    * micrograd

        <https://github.com/karpathy/micrograd>

        个人维护的一个自动求导库。

    * <https://github.com/pranftw/neograd>

        没看。

* LLaMA 超详细解读（paper & code）

    <https://zhuanlan.zhihu.com/p/632102048>

    没看。

* 如何最简单、通俗地理解Transformer？

    <https://www.zhihu.com/question/445556653/answer/3254012065>

    transformer 解释。没看。

* deepspeed

    <https://github.com/microsoft/DeepSpeed>

    deepspeed 的 github repo

* synchronization in vulkan

    <https://www.kdab.com/synchronization-in-vulkan/>

    有关 semaphore 和 fence 的简单介绍。没啥新东西，就当是复习了。

* LLVM IR入门指南

    <https://evian-zhang.github.io/llvm-ir-tutorial/01-LLVM%E6%9E%B6%E6%9E%84%E7%AE%80%E4%BB%8B.html>

    挺好的一份学习资料。有空了看看。

* GPGMM

    <https://github.com/intel/GPGMM>

    intel 开发的一个基于 vulkan 的显存池。

* glslang

    <https://github.com/KhronosGroup/glslang>

    希望能用 glslang 生成一个 ast

* cupy

    NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python

    <https://cupy.dev/>

    用 cuda 作为 backend，用于替代 numpy 和 scipy 的部分功能，提高计算效率。

    不清楚这个库是谁开发的，有空了看看。

* <https://cliutils.gitlab.io/modern-cmake/chapters/testing.html>

    cmake 的一个比较好的教程。这个还是在以前的笔记里找到的。废话比较多，跟着主要思路自己多实践，再查查 api，gpt。

    可以把这个写到任务管理系统里。

* yacc 相关

    * <https://www.geeksforgeeks.org/introduction-to-yacc/>

* compiler

    <https://shperb.github.io/teaching/Compiler_Principles>

    <http://www.craftinginterpreters.com/>

    <https://github.com/aalhour/awesome-compilers>

    <>

* zlibrary

    <https://singlelogin.se>

    ? Just send any letter from your mailbox to our magic email address <blackbox@zlib.se>

* pdf coffee

    <https://pdfcoffee.com/>

* library genesis

    <https://www.libgen.is/>

* cmake 相关

    * CMake: find_package()

        <https://wiki.hanzheteng.com/development/cmake/cmake-find_package>

    * find_library

        <https://cmake.org/cmake/help/latest/command/find_library.html#command:find_library>

    * cmake-modules(7)

        <https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html#manual:cmake-modules(7)>

    * Finding Packages

        <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Finding%20Packages.html>

    * How to Find Packages With CMake: The Basics

        <https://izzys.casa/2020/12/how-to-find-packages-with-cmake-the-basics/>

    * Using Dependencies Guide

        <https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#guide:Using%20Dependencies%20Guide>

    下面这两个主要是调试相关的：

    * Introducing CMake Debugger in VS Code: Debug your CMake Scripts using Open-Source CMake Debugger

        <https://devblogs.microsoft.com/cppblog/introducing-cmake-debugger-in-vs-code-debug-your-cmake-scripts-using-open-source-cmake-debugger/>

    * variable_watch

        <https://cmake.org/cmake/help/latest/command/variable_watch.html>

* 动态漫反射全局光照（Dynamic Diffuse Global Illumination）

    <https://zhuanlan.zhihu.com/p/404520592>

* vulkan resources

    * Vulkan Guide 

        <https://vkguide.dev/>

    * Vulkan-Cookbook 

        <https://github.com/PacktPublishing/Vulkan-Cookbook>

    * Vulkan Tutorial

        <https://vulkan-tutorial.com/Introduction>

        这个就是那个废话很多，条理不清的教程。

    * LunarG: Creator and Curator of the Vulkan SDK

        <https://www.lunarg.com/vulkan-sdk/>

    * How to write a Vulkan driver in 2022

        <https://www.collabora.com/news-and-blog/blog/2022/03/23/how-to-write-vulkan-driver-in-2022/>

    * Getting Started with the Ubuntu Vulkan SDK

        <https://vulkan.lunarg.com/doc/sdk/1.3.268.0/linux/getting_started_ubuntu.html>

    * Vulkan Specification and Proposals

        <https://docs.vulkan.org/spec/latest/appendices/glossary.html#glossary>

        vulkan api reference

    * Vulkan® 1.3.275 - A Specification (with all registered extensions)

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-semaphores-waiting>

    * Mastering-Graphics-Programming-with-Vulkan

        <https://github.com/PacktPublishing/Mastering-Graphics-Programming-with-Vulkan>

    * Synchronization and Cache Control

        <https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-semaphores>

    * Vulkan Timeline Semaphores

        <https://www.khronos.org/blog/vulkan-timeline-semaphores>

    * vulkan 拿 swapchain 的 image 数据

        <https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp>

        这是一个比较老的 vulkan example 库，五六年前就停止维护了，整个 repo 能通过编译，里面有些 example code 在新版的 vulkan example repo 中是没有的，所以还有一些价值。有时间了系统学习一下。

* vulkan copy image from swapchain, and image layout

    * How do I copy any image into Swapchain images?

        <https://www.reddit.com/r/vulkan/comments/upeyel/how_do_i_copy_any_image_into_swapchain_images/>

    * How to get image from swapchain for screenshot. Getting error "Cannot read invalid swapchain image 0x3, please fill the memory before using."

        <https://www.reddit.com/r/vulkan/comments/7wwrs5/how_to_get_image_from_swapchain_for_screenshot/>

    * How to Copy Swap Chain Image to a VkBuffer in Vulkan?

        <https://stackoverflow.com/questions/38985094/how-to-copy-swap-chain-image-to-a-vkbuffer-in-vulkan>

    * C++ (Cpp) vkCmdCopyImageToBuffer Examples

        <https://cpp.hotexamples.com/examples/-/-/vkCmdCopyImageToBuffer/cpp-vkcmdcopyimagetobuffer-function-examples.html>

    * Vulkan从入门到精通38-使用vkCmdCopyImageToBuffer保存VkImage图像

        <https://zhuanlan.zhihu.com/p/469416830>

    * vulkan学习笔记五

        <https://blog.csdn.net/hometoned/article/details/126061556>

    * Tracking image layout transitions

        <https://community.khronos.org/t/tracking-image-layout-transitions/107409>

    * VkImageLayout(3) Manual Page

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageLayout.html>

    * How to deal with the layouts of presentable images?

        <https://stackoverflow.com/questions/37524032/how-to-deal-with-the-layouts-of-presentable-images>

* go 语言相关

    * golangbot

        <https://golangbot.com/>

        看起来是一个发布 go 相关的新闻的网站

    * Learning notes for golang

        <https://github.com/jincheng9/go-tutorial>

    * The Go Handbook – Learn Golang for Beginners

        <https://www.freecodecamp.org/news/go-beginners-handbook/>

    * Golang Tutorial – Learn Go Programming Language

        <https://www.geeksforgeeks.org/golang-tutorial-learn-go-programming-language/>

    * Goroutines – Concurrency in Golang

        <https://www.geeksforgeeks.org/goroutines-concurrency-in-golang/?ref=lbp>

    * Golang Logo Programiz  Learn Go Programming

        <https://www.programiz.com/golang>

    * Go Interface

        <https://www.programiz.com/golang/interface>

    * Go by Example

        <https://gobyexample.com/>

    * Tutorials

        <https://go.dev/doc/tutorial/>

    * Documentation

        <https://go.dev/doc/>

    * Effective Go

        <https://go.dev/doc/effective_go>

    * How to delete an element from a Slice in Golang

        <https://stackoverflow.com/questions/37334119/how-to-delete-an-element-from-a-slice-in-golang>

    * Go Generics 101

        <https://go101.org/generics/101.html>

    * Memory management in Go

        <https://medium.com/@ali.can/memory-optimization-in-go-23a56544ccc0>

* makefile 的一些资源整合

    * How to trim a string in makefile?

        <https://stackoverflow.com/questions/55401729/how-to-trim-a-string-in-makefile>

    * 8.2 Functions for String Substitution and Analysis

        <https://www.gnu.org/software/make/manual/html_node/Text-Functions.html>

    * 6.2.4 Conditional Variable Assignment

        <https://www.gnu.org/software/make/manual/html_node/Conditional-Assignment.html>

    * GNU make

        <https://www.gnu.org/software/make/manual/html_node/index.html#SEC_Contents>

    * 2.6 Another Style of Makefile

        <https://www.gnu.org/software/make/manual/html_node/Combine-By-Prerequisite.html>

    * 6 How to Use Variables

        <https://www.gnu.org/software/make/manual/html_node/Using-Variables.html>

    * How to Use Variables

        <https://ftp.gnu.org/old-gnu/Manuals/make-3.79.1/html_chapter/make_6.html>

    * Force exit from a Makefile target without raising an error

        <https://stackoverflow.com/questions/19773600/force-exit-from-a-makefile-target-without-raising-an-error>

    * Understanding and Using Makefile Variables 

        <https://earthly.dev/blog/makefile-variables/>

    * 4.6 Phony Targets

        <https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html>

    * Is there an easy way to find the source mk file for a specific target?

        <https://stackoverflow.com/questions/58661901/is-there-an-easy-way-to-find-the-source-mk-file-for-a-specific-target>

    * makefile 的 tutorial

        <https://www.gnu.org/software/make/manual/html_node/Using-Variables.html>

        gnu 出的这一套 makefile 教程，我觉得写得还不错，条理挺清晰的，有时间了看看

        <https://stackoverflow.com/questions/2145590/what-is-the-purpose-of-phony-in-a-makefile>

    * What does @: (at symbol colon) mean in a Makefile?

        <https://stackoverflow.com/questions/8610799/what-does-at-symbol-colon-mean-in-a-makefile>

* 直接生成论文格式

    typst, <https://typst.app/>

* 大模型理论基础 

    <https://datawhalechina.github.io/so-large-lm/#/>

    一个 git book，

* How to search a string with spaces and special characters in vi editor

    <https://stackoverflow.com/questions/34036575/how-to-search-a-string-with-spaces-and-special-characters-in-vi-editor>

* mime type 的位置

    <https://askubuntu.com/questions/16580/where-are-file-associations-stored>

* 让 firefox 正确识别并显示 markdown mime type

    <https://superuser.com/questions/696361/how-to-get-the-markdown-viewer-addon-of-firefox-to-work-on-linux>