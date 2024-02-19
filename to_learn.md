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

    * typedef in C++

        <https://www.codesdope.com/cpp-typedef/>

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

    * VkMemoryType(3) Manual Page

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryType.html>

    * vkMapMemory(3) Manual Page

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkMapMemory.html>

    * Vulkan-Samples

        <https://github.com/KhronosGroup/Vulkan-Samples>

    * Vulkan学习资料汇总

        <https://zhuanlan.zhihu.com/p/24798656>

    * Vulkan buffer memory management - When do we need staging buffers?

        <https://stackoverflow.com/questions/44940684/vulkan-buffer-memory-management-when-do-we-need-staging-buffers>

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

    * Makefile中.PHONY的作用

        <https://www.cnblogs.com/idorax/p/9306528.html>

    * Debugging Makefiles

        <https://the-turing-way.netlify.app/reproducible-research/make/make-debugging.html>

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

* opengl

    * OpenGL® and OpenGL® ES Reference Pages

        <https://registry.khronos.org/OpenGL-Refpages/es3/>

    * glfw Window reference

        <https://www.glfw.org/docs/latest/group__window.html#ga15a5a1ee5b3c2ca6b15ca209a12efd14>

    * glRect

        <https://docs.gl/gl3/glRect>

    * glRect — draw a rectangle

        <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glRect.xml>

    * glRect — draw a rectangle

        <https://registry.khronos.org/OpenGL-Refpages/gl2.1/>

    * <https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf>

    * OpenGLES 与 EGL 基础概念

        <https://zhuanlan.zhihu.com/p/74006499>

    * glbinding is a cross-platform C++ binding for the OpenGL API.

        <https://github.com/cginternals/glbinding>

    * opengl es samples

        <https://www.khronos.org/assets/uploads/books/openglr_es_20_programming_guide_sample.pdf>

    * How can I use OpenGL to render to memory without requiring any windowing system library?

        <https://stackoverflow.com/questions/68298698/how-can-i-use-opengl-to-render-to-memory-without-requiring-any-windowing-system>

    * OpenGL基础 - 统一变量Uniform

        <https://zhuanlan.zhihu.com/p/33093968>

    * Introduction to C++ OpenGL Programming
    
        <https://www.cprogramming.com/tutorial/opengl_introduction.html>

    * An Introduction on OpenGL with 2D Graphics

        <https://www3.ntu.edu.sg/home/ehchua/programming/opengl/cg_introduction.html>

    * Confused about z-axis behaviour

        <https://computergraphics.stackexchange.com/questions/2063/confused-about-z-axis-behaviour>

    * glfw Window guide

        <https://www.glfw.org/docs/3.3/window_guide.html#window_close>

    * opengl Tutorial 7 : Model loading

        <http://www.opengl-tutorial.org/beginners-tutorials/tutorial-7-model-loading/>

* 3d models

    * obj文件中v、vt、vn、f这四个参数的含义

        <https://blog.csdn.net/xiongzai2016/article/details/108052800>

    * obj文件(1):obj文件用txt打开并且了解v,f,vn,vt的含义

        <https://blog.csdn.net/weixin_44115959/article/details/123270006>

    * OBJ FILE FORMAT

        <https://www.cs.cmu.edu/~mbz/personal/graphics/obj.html>

    * glGetError — return error information

        <https://registry.khronos.org/OpenGL-Refpages/es2.0/xhtml/glGetError.xml>

    *  原神模型预览器

        <https://github.com/GenshinMatrix/genshin-model-viewer>

    * learn mmd

        <https://learnmmd.com/http:/learnmmd.com/category/weighting-bones/>

    * DanceXR is a versatile character model viewer and motion player

        <https://github.com/alloystorm/dvvr>

    * 数据分析 3d 图形渲染 PMX文件格式解析

        <https://www.kuazhi.com/post/416347.html>

    * bvh+obj File Format Specification

        <https://github.com/rspencer01/bvh-obj>

    * OFF (file format)

        <https://en.wikipedia.org/wiki/OFF_(file_format)>

    * SCN ray-tracing format

        <https://paulbourke.net/dataformats/scn/>

* render farm

    * 3d cat

        <https://www.3dcat.live/news/post-id-120/>

    * rebus farm

        <https://rebusfarm.net/blog/what-is-cloud-rendering>

    * How to build a render farm – 3S Cloud Render Farm

        <https://3sfarm.com/build-a-render-farm/>

    * render bus

        <https://www.renderbus.com/?source=bdsem&keyword=%E8%B5%9B%E8%AF%9A%E6%B8%B2%E6%9F%93&device=pc-cpc&e_matchtype=2&e_creative=58091948671&e_adposition=cr1&bd_vid=4809392668305511569>

    * PMXViewer

        <https://github.com/KentaTheBugMaker/PMXViewer>

* graph visualization

    * rust Visualization libs

        <https://lib.rs/visualization>

    * Graphviz

        <https://www.graphviz.org/>

    * GraphStream

        <https://graphstream-project.org/>

    * networkx

        <https://networkx.org/documentation/stable/>

    * d3.js

        <https://d3-graph-gallery.com/>

    * List of graph visualization libraries

        <https://elise-deux.medium.com/the-list-of-graph-visualization-libraries-7a7b89aab6a6>

* c++

    * Functions with Variable Number of Arguments in C++

        <https://www.scaler.com/topics/cpp/functions-with-variable-number-of-arguments-in-cpp/>

    * C++11特性——右值引用

        <https://blog.csdn.net/gls_nuaa/article/details/126134537>

    * eigen

        <https://gitlab.com/libeigen/eigen>

    * yaLanTingLibs is a collection of modern c++ util libraries

        <https://github.com/alibaba/yalantinglibs>

    * Static functions in C

        <https://www.tutorialspoint.com/static-functions-in-c>

    * Static functions in C

        <https://www.geeksforgeeks.org/what-are-static-functions-in-c/>

    * What is a Static Function in C?

        <https://www.scaler.com/topics/static-function-in-c/>

    * The Library is a small and open-source C++ library for image processing

        <https://cimg.eu/index.html>

        * Combining image channels in CImg

            <https://stackoverflow.com/questions/73779710/combining-image-channels-in-cimg>

    * libvips: A fast image processing library with low memory needs.

        <https://www.libvips.org/>

        <https://cpp.libhunt.com/libvips-alternatives>

    * Type Qualifiers in C++

        <https://prepinsta.com/c-plus-plus/type-qualifiers/>

    * What is the meaning of "qualifier"?

        <https://stackoverflow.com/questions/3785789/what-is-the-meaning-of-qualifier>

    * operator overloading

        <https://en.cppreference.com/w/cpp/language/operators>

    * C++ globally overloaded operator= [duplicate]

        <https://stackoverflow.com/questions/5037156/c-globally-overloaded-operator>

    * Can I call a constructor from another constructor (do constructor chaining) in C++?

        <https://stackoverflow.com/questions/308276/can-i-call-a-constructor-from-another-constructor-do-constructor-chaining-in-c>

    * Calling Constructor with in constructor in same class

        <https://stackoverflow.com/questions/29063703/calling-constructor-with-in-constructor-in-same-class>


* rendering and ray tracing

    * GAMES202-高质量实时渲染

        <https://www.bilibili.com/video/BV1YK4y1T7yY/?spm_id_from=333.999.0.0&vd_source=39781faaf2433372c59bdb80774d648e>

    * 自定义SRP（五）—— 烘焙光照

        <https://zhuanlan.zhihu.com/p/640860572>

    * Baked Light: Light Maps and Probes

        <https://catlikecoding.com/unity/tutorials/custom-srp/baked-light/>

    * Ray Tracing in One Weekend

        <https://raytracing.github.io/books/RayTracingInOneWeekend.html>

    * stb

        <https://github.com/nothings/stb>

    * unreal engine: Lumen Global Illumination and Reflections

        <https://docs.unrealengine.com/5.0/en-US/lumen-global-illumination-and-reflections-in-unreal-engine/>

    * C# and Shader Tutorials: for the Unity Engine

        <https://catlikecoding.com/unity/tutorials/>

    * Radeon ProRender SDK

        <https://radeon-pro.github.io/RadeonProRenderDocs/en/sdk/about.html>

    * Monte Carlo Integration

        <https://www.cnblogs.com/lsy-lsy/p/16560754.html>

    * 光追和路径追踪的区别到底是哪里？

        <https://www.zhihu.com/question/368551323/answer/2793409825>

    * 光线追踪vs路径追踪：游戏内照明的两种实现方式比较

        <https://baijiahao.baidu.com/s?id=1774279027961829017&wfr=spider&for=pc>

    * Basics about path tracing 路径追踪基础

        <https://zhuanlan.zhihu.com/p/588056773>

    * 高数数学咋用二重积分求解菲涅尔积分∫sin(x^2)dx呢，x从零到正无穷大inf？

        <https://www.zhihu.com/question/526300658/answer/2425869091>

    * 这个二重积分求解?

        <https://www.zhihu.com/question/603500493/answer/3048561769>

    * 面试公司的unity特效测试，目测又要凉凉了，哎，自学没有方向感好难

        <https://www.bilibili.com/video/BV1y84y1q72j/?spm_id_from=333.788.recommend_more_video.1&vd_source=39781faaf2433372c59bdb80774d648e>

    * （超详细！）计算机图形学 入门篇 8. 光追I: Recursive(Whitted-Style) Ray Tracing算法与光追的加速结构算法

        <https://zhuanlan.zhihu.com/p/466122358>

    * 【Udemy UE5 最畅销课程】Unreal Engine 5 C++ Developer: Learn C++ & Make Video

        <https://www.bilibili.com/video/BV1M84y1x7Gc/?spm_id_from=333.337.search-card.all.click&vd_source=39781faaf2433372c59bdb80774d648e>

    * 一个实时流体模和流体渲染程序

        <https://www.bilibili.com/video/BV1sj411D7hp/?spm_id_from=333.337.search-card.all.click&vd_source=39781faaf2433372c59bdb80774d648e>

    * [C++/OpenGL] 用可莉酱来测试液体的模拟和渲染吧 (based on Affine Particle-in-cell

        <https://www.bilibili.com/video/BV11m4y1F7c6/?spm_id_from=333.337.search-card.all.click&vd_source=39781faaf2433372c59bdb80774d648e>

    * （深度解析）GAMES 101 作业6：BVH与SAH(Surface Area Heuristic)

        <https://zhuanlan.zhihu.com/p/475966001>

    * raytracing.github.io 

        <https://github.com/RayTracing/raytracing.github.io>

    * Physically Based Rendering: From Theory To Implementation
    
        <https://www.pbr-book.org/>

* opengl

    * Where points OpenGL z-axis?

        <https://stackoverflow.com/questions/3430789/where-points-opengl-z-axis>

    * GPU渲染之路：从图形引擎到内核驱动(三、用户态图形驱动层)

        <https://zhuanlan.zhihu.com/p/651364842>

    * How to get the Graphics Card Model Name in OpenGL or Win32?

        <https://stackoverflow.com/questions/42245870/how-to-get-the-graphics-card-model-name-in-opengl-or-win32>

    * glfw: Context handling

        <https://www.glfw.org/docs/3.0/group__context.html#ga6d4e0cdf151b5e579bd67f13202994ed>

    * VAOs and Element Buffer Objects

        <https://stackoverflow.com/questions/33863426/vaos-and-element-buffer-objects>

    * OpenGL, VAOs and multiple buffers

        <https://stackoverflow.com/questions/14249634/opengl-vaos-and-multiple-buffers>

    * What is the role of glBindVertexArrays vs glBindBuffer and what is their relationship?

        <https://stackoverflow.com/questions/21652546/what-is-the-role-of-glbindvertexarrays-vs-glbindbuffer-and-what-is-their-relatio>

    * Compiling a shader

        <https://subscription.packtpub.com/book/game-development/9781789342253/1/ch01lvl1sec14/compiling-a-shader>

* bvh

    * 4.3 Bounding Volume Hierarchies

        <https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies>

    * GPU-path-tracing-tutorial-3 

        <https://github.com/SimronThapa/GPU-path-tracing-tutorial-3/blob/master/bvh.h>

    * Introduction to Acceleration Structures

        <https://scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/bounding-volume-hierarchy-BVH-part1.html>

    * Introduction to Acceleration Structures

        <https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/bounding-volume-hierarchy-BVH-part2.html>

    * How to build a BVH – Part 1: Basics

        <https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/>

* shader

    * ShaderLab入门

        <https://zhuanlan.zhihu.com/p/533320848>

    * 转载一篇——shader入门经典

        <https://zhuanlan.zhihu.com/p/235461709>

* Unity小白的TA之路-优梦创客

    <https://91maketop.github.io/ta/#/>

* 魔法盒：游戏特效大全

    <https://www.magesbox.com/>

* vulkan

    * GPU驱动开发学习笔记

        <https://www.zhihu.com/column/c_1555603544597549056>

    * Vulkan Render Pass介绍

        <https://zhuanlan.zhihu.com/p/617374058>

    * Vulkan Command Buffers

        <https://zhuanlan.zhihu.com/p/615441145>

    * Vulkan-Docs

        <https://github.com/KhronosGroup/Vulkan-Docs>

    * vulkan dev tools

        <https://vulkan.lunarg.com/sdk/home#linux>

    * Understanding Vulkan Synchronization

        <https://www.khronos.org/blog/understanding-vulkan-synchronization>

    * Compute shader synchronization?

        <https://groups.google.com/g/webgl-dev-list/c/Xdis0MAA4-M>

    * vulkan: Copy Commands

        <http://geekfaner.com/shineengine/blog29_Vulkanv1.2_15.html>

    * Copying Images on the Host in Vulkan

        <https://www.khronos.org/blog/copying-images-on-the-host-in-vulkan>

    * VulkanSamples 

        <https://github.com/LunarG/VulkanSamples/blob/master/BUILD.md>

    * Copy result using vkCmdCopyImage to swap chain image

        <https://www.reddit.com/r/vulkan/comments/47ods9/copy_result_using_vkcmdcopyimage_to_swap_chain/>

    * VK_EXT_external_memory_dma_buf(3) 

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html>

    * VkBufferImageCopy(3) Manual Page

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBufferImageCopy.html>

    * VkImageSubresourceLayers(3) Manual Page

        <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageSubresourceLayers.html>

    * What is the difference between framebuffer and image in Vulkan?

        <https://stackoverflow.com/questions/39557141/what-is-the-difference-between-framebuffer-and-image-in-vulkan>

    * Vulkan有没有类似Opengl的红宝书，蓝宝书那样的书？

        <https://www.zhihu.com/question/527771207>

    * 如何正确的入门Vulkan？

        <https://www.zhihu.com/question/424430509/answer/1632072443>

    * Vulkan Memory and Resources

        <https://www.informit.com/articles/article.aspx?p=2756465&seqNum=3>

    * Vulkan Usage Recommendations

        <https://developer.samsung.com/galaxy-gamedev/resources/articles/usage.html>

    * Getting started with the Vulkan programming model

        <https://subscription.packtpub.com/book/game-development/9781786469809/1/ch01lvl1sec12/getting-started-with-the-vulkan-programming-model>

    * API without Secrets: Introduction to Vulkan* Part 6

        <https://www.intel.com/content/www/us/en/developer/articles/training/api-without-secrets-introduction-to-vulkan-part-6.html>

* How to printf "unsigned long" in C?

    <https://stackoverflow.com/questions/3209909/how-to-printf-unsigned-long-in-c>

* MyGUI is a cross-platform library for creating graphical user interfaces (GUIs) for games and 3D applications.

    <https://github.com/MyGUI/mygui>

* c++

    * C++11 中 std::piecewise_construct 的使用 

        <https://juejin.cn/post/7029372430397210632>

    * An improved emplace() for unique-key maps

        <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4006.html>

    * map::emplace with muliple constructor arguments

        <https://groups.google.com/a/isocpp.org/g/std-discussion/c/izS4U8Ojhss?pli=1>

    * Convert between string, u16string & u32string

        <https://stackoverflow.com/questions/7232710/convert-between-string-u16string-u32string>

    * EA Standard Template Library

        <https://github.com/electronicarts/EASTL>

    * Conan is a package manager for C and C++ developers:

        <https://github.com/conan-io/conan>

    * C Programming/Serialization

        <https://en.wikibooks.org/wiki/C_Programming/Serialization>

    * An Introduction to Object Serialization in C++

        <https://www.codeguru.com/cplusplus/an-introduction-to-object-serialization-in-c/>

    * How to write reflection for C++

        <https://pvs-studio.com/en/blog/posts/cpp/0956/>

    * C - serialization techniques

        <https://stackoverflow.com/questions/6002528/c-serialization-techniques>

* <https://www.ea.com/frostbite>

    一个渲染引擎。

* cairo is a 2D graphics library with support for multiple output devices. 

    <https://www.cairographics.org/>

* rust and alsa

    *  Unable to build / install with cargo: "failed to run custom build command for alsa-sys v0.1.2" #659 

        <https://github.com/Spotifyd/spotifyd/issues/659>

    *  alsa v0.8.1 Thin but safe wrappers for ALSA (Linux sound API) 

        <https://crates.io/crates/alsa>

* Linux Performance

    <https://www.brendangregg.com/linuxperf.html>

* Linux Performance Analysis in 60,000 Milliseconds

    <https://netflixtechblog.com/linux-performance-analysis-in-60-000-milliseconds-accc10403c55>

* What's the difference between ray tracing and path tracing?

    <https://www.quora.com/Whats-the-difference-between-ray-tracing-and-path-tracing>

* Skia: The 2D Graphics Library

    <https://skia.org/>

* How to get a reflection vector?

    <https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector>

* Dot Product

    <https://www.mathsisfun.com/algebra/vectors-dot-product.html>

* glm

    <https://glm.g-truc.net/0.9.2/api/a00277.html>
    
* musl is an implementation of the C standard library built on top of the Linux system call API

    <https://musl.libc.org/>

* pthread_create

    <https://man7.org/linux/man-pages/man3/pthread_create.3.html>

* Unable to display <CL_DEVICE_MAX_WORK_ITEM_SIZE> in OpenCL and C++

    <https://stackoverflow.com/questions/57049160/unable-to-display-cl-device-max-work-item-size-in-opencl-and-c>

* Why does an Ubuntu Server have graphical.target as the default systemd target?

    <https://askubuntu.com/questions/836525/why-does-an-ubuntu-server-have-graphical-target-as-the-default-systemd-target>

    讲 multi-user 的，里面有一些关于 linux 启动过程的东西，有时间可以看一下。

* c++ needs to learn

    * List-initialization (since C++11)

        <https://en.cppreference.com/w/cpp/language/list_initialization>

    * Default Parameters With Default Template Parameters Types

        <https://www.fluentcpp.com/2018/08/10/template-default-arguments-application-smart-iterators/>

    * Thinking in C++ Vol 2 - Practical Programming

        <https://www.linuxtopia.org/online_books/programming_books/c++_practical_programming/c++_practical_programming_107.html>

    * Tricks with Default Template Arguments

        <https://www.foonathan.net/2020/10/tricks-default-template-argument/>

    * Union declaration

        <https://en.cppreference.com/w/cpp/language/union>

    * Resolve "DSO missing from command line" error

        <https://zhangboyi.gitlab.io/post/2020-09-14-resolve-dso-missing-from-command-line-error/>
    
    * libpthread.so.0: error adding symbols: DSO missing from command line

        <https://stackoverflow.com/questions/19901934/libpthread-so-0-error-adding-symbols-dso-missing-from-command-line>

    * How can I clear an input buffer in C?

        <https://stackoverflow.com/questions/7898215/how-can-i-clear-an-input-buffer-in-c>

    * Clearing the buffer when using Getchar (there must be a better way!)

        <https://stackoverflow.com/questions/43954160/clearing-the-buffer-when-using-getchar-there-must-be-a-better-way>

    * Detect Operating System in C

        <https://iq.opengenus.org/detect-operating-system-in-c/>

* c++

    * C++ Syncing threads in most elegant way

        <https://stackoverflow.com/questions/16277840/c-syncing-threads-in-most-elegant-way>

    * Three Simple Ways For C++ Thread Synchronization in C++11 and C++14

        <https://chrizog.com/cpp-thread-synchronization>

    * std::counting_semaphore, std::binary_semaphore

        <https://en.cppreference.com/w/cpp/thread/counting_semaphore>

    * C++ 多线程（七）：信号量 Semaphore 及 C++ 11 实现

        <https://zhuanlan.zhihu.com/p/512969481>

    * C++学习笔记：模板参数

        <https://blog.csdn.net/iuices/article/details/122872720>

    * 类模板三种类模板参数

        <https://blog.csdn.net/u014253011/article/details/80036801>

    * Function template

        <https://en.cppreference.com/w/cpp/language/function_template>

    * Friend Class and Function in C++

        <https://www.geeksforgeeks.org/friend-class-function-cpp/>

    * stringstream Class In C++ – Usage Examples And Applications

        <https://www.softwaretestinghelp.com/stringstream-class-in-cpp/>

    * Stringstream in C++

        <https://www.tutorialspoint.com/stringstream-in-cplusplus>

    * stringstream in C++ and its Applications

        <https://www.geeksforgeeks.org/stringstream-c-applications/>

    * Unnamed and inline namespaces

        <https://www.learncpp.com/cpp-tutorial/unnamed-and-inline-namespaces/>

    * Async.h - asynchronous, stackless subroutines

        <https://github.com/naasking/async.h>

    * Sample code for asynchronous programming in C

        <https://stackoverflow.com/questions/2108961/sample-code-for-asynchronous-programming-in-c>

    * Asynchronous Routines For C

        <https://hackaday.com/2019/09/24/asynchronous-routines-for-c/>

    * typedef in C++

        <https://www.geeksforgeeks.org/typedef-in-cpp/>

    * typedef specifier

        <https://en.cppreference.com/w/cpp/language/typedef>

    * override specifier (since C++11)

        <https://en.cppreference.com/w/cpp/language/override>

    * C++ Delegate模板类的设计

        <https://blog.csdn.net/weixin_50948630/article/details/132110710>

* opengl

    * Tutorial 7 : Model loading
    
        <http://www.opengl-tutorial.org/beginners-tutorials/tutorial-7-model-loading/>

    * The Matrix and Quaternions FAQ

        <http://www.opengl-tutorial.org/assets/faq_quaternions/index.html>

    * Welcome to OpenGL

        <https://learnopengl.com/>

    * opengl读取obj兔子并贴图_OpenGL学习笔记(八)-纹理

        <https://blog.csdn.net/weixin_39925813/article/details/110901035>

    * free 3d model downloading

        <https://free3d.com/>

    *  LearnOpenGL-CN

        <https://learnopengl-cn.readthedocs.io/zh/latest/01%20Getting%20started/01%20OpenGL/>

    * GLSL 语言—矢量和矩阵 [ ] 运算符

        <https://cloud.tencent.com/developer/article/1504841>

    * 欢迎来到OpenGL的世界

        <https://learnopengl-cn.github.io/>

    * glumpy: opengl + numpy

        <https://github.com/glumpy/glumpy>

    * PyGLM: OpenGL Mathematics (GLM) library for Python

        <https://pypi.org/project/PyGLM/>

    * glumpy 1.2.1 

        <https://pypi.org/project/glumpy/>

    * VisPy

        <https://macrocosme.github.io/vispy-and-the-future-of-big-data-visualisation/>

    * 3d graphics

        <http://morpheo.inrialpes.fr/~franco/3dgraphics/index.html>

    * gltut for python

        <https://www.pygame.org/project-gltut+for+python-2797-.html>

    * GLSL Programming/Vector and Matrix Operations

        <https://en.wikibooks.org/wiki/GLSL_Programming/Vector_and_Matrix_Operations>

    * opengl 4.5

        <https://yaakuro.gitbook.io/opengl-4-5/>

    * Buffer Object

        <https://www.khronos.org/opengl/wiki/Buffer_Object>

    * glGetString — return a string describing the current GL connection

        <https://registry.khronos.org/OpenGL-Refpages/gl4/html/glGetString.xhtml>

    * OpenGL® 4.5 Reference Pages 

        <https://registry.khronos.org/OpenGL-Refpages/gl4/>

    * ogl dev: modern opengl tutorials

        <https://ogldev.org/>

    * Download Computer Graphics With Opengl (3rd Edition) 

        <https://vdoc.pub/download/computer-graphics-with-opengl-3rd-edition-6dr3a2ql4vi0>

    * learn opengl

        <https://learnopengl.com/Lighting/Colors>

    * Modern OpenGL

        <https://glumpy.github.io/modern-gl.html>

    * OpenGL Programming/Scientific OpenGL Tutorial 03

        <https://en.wikibooks.org/wiki/OpenGL_Programming/Scientific_OpenGL_Tutorial_03>

    * glm Geometric functions

        <https://glm.g-truc.net/0.9.4/api/a00131.html>

    * awesome-opengl
    
        <https://github.com/eug/awesome-opengl>

    * 浅析OpenGL光照

        <https://www.cnblogs.com/javawebsoa/p/3243737.html>

    * GLSL Tutorial von Lighthouse3D

        <https://cgvr.cs.uni-bremen.de/teaching/cg2_07/literatur/glsl_tutorial/index.html>

    * Getting started with glsl

        <https://riptutorial.com/glsl>

    * OpenGL 101: Textures 

        <https://solarianprogrammer.com/2013/05/17/opengl-101-textures/>

    * How to debug a GLSL shader?

        <https://stackoverflow.com/questions/2508818/how-to-debug-a-glsl-shader>

* learn mmd

    <https://learnmmd.com/http:/learnmmd.com/create-mmd-model-color-morphs-using-pmxe/>

* 记录一次xf86vmode library not found问题

    <https://blog.csdn.net/qq1186351245/article/details/125300966>

* Saba: a mmd player

    <https://github.com/benikabocha/saba>

* compiler

    * glslang

        <https://github.com/KhronosGroup/glslang>

    * Question for pest parser example

        <https://users.rust-lang.org/t/question-for-pest-parser-example/82041>

    * pest

        <https://pest.rs/>

* vulkan-compute 

    <https://github.com/topics/vulkan-compute>

* Getting Started with Vulkan Compute Acceleration

    <https://www.khronos.org/blog/getting-started-with-vulkan-compute-acceleration>

* Arrow Types in LaTeX: A Complete List

    <https://latex-tutorial.com/arrow-latex/>

* sleep()

    * sleep(3) — Linux manual page

        <https://man7.org/linux/man-pages/man3/sleep.3.html>

    * Sleep() Function in C Language

        <https://linuxhint.com/sleep-function-c/>

    * nanosleep(2) — Linux manual page

        <https://man7.org/linux/man-pages/man2/nanosleep.2.html>

* ubuntu 安装显卡驱动的官方教程

    <https://help.ubuntu.com/community/BinaryDriverHowto>

    一种是开源驱动，一种是闭源驱动

* Linux内核API atomic_inc

    <https://deepinout.com/linux-kernel-api/linux-kernel-api-synchronization-mechanism/linux-kernel-api-atomic_inc.html>

* git pull without remotely compressing objects

    <https://stackoverflow.com/questions/7102053/git-pull-without-remotely-compressing-objects>

    这个好像不是 client 端决定的，是 server 端决定的

*  how to disable auto suggestion in Version 8 #13451 

    <https://github.com/ipython/ipython/issues/13451>

    新版本的 ipython 的自动提示非常难用，这个方法可以把自动提示关掉

* Linux depmod command

    <https://www.computerhope.com/unix/depmod.htm>

    depmod 可以找到一个 module 所依赖的模块，之前好像记过

    <https://eng.libretexts.org/Bookshelves/Computer_Science/Operating_Systems/Linux_-_The_Penguin_Marches_On_(McClanahan)/06%3A_Kernel_Module_Management/2.05%3A_Kernel_Module_Management_-_lsmod_Command/2.05.02%3A_Kernel_Module_Management_-_modprobe_Command>

* Visual Studio Code - Convert spaces to tabs

    vscode 有时候在 makefile 里输入的是空格，不是 tab，导致语法错误

* Fix: x11 Connection Rejected Because of Wrong Authentication

    <https://itslinuxfoss.com/x11-connection-rejected-because-of-wrong-authentication/>

    用 vnc 的时候会遇到这个问题，目前还是没弄清楚原理

* gRPC: Transporting massive data with Google’s serialization

    <https://alexandreesl.com/2017/05/02/grpc-transporting-massive-data-with-googles-serialization/>

* 万字长文深入理解 cache，写出高性能代码

    <https://zhuanlan.zhihu.com/p/656338505>

* GRPC C++  1.60.0

    <https://grpc.github.io/grpc/cpp/md_doc_server_reflection_tutorial.html>

* matrix faq

    <http://www.opengl-tutorial.org/assets/faq_quaternions/index.html>

    Identity Matrix: <http://www.c-jump.com/bcc/common/Talk3/Math/GLM/W01_0040_identity_matrix.htm>

* How to flip an image (horizontal and vertical) using CImg?

    <https://github.com/GreycLab/CImg/issues/211>

* How to create a video from images with FFmpeg? [closed]

    <https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg>

* bash: How to read a file into a variable in shell?

    <https://stackoverflow.com/questions/7427262/how-to-read-a-file-into-a-variable-in-shell>

* bash: How to get execution time of a script effectively?

    <https://unix.stackexchange.com/questions/52313/how-to-get-execution-time-of-a-script-effectively>

* git-stash用法小结

    <https://www.cnblogs.com/tocy/p/git-stash-reference.html>

* 有关 web 网页渲染

    * V8 源码分析（二）- JS 引擎和渲染引擎

        <https://miguoer.github.io/blog/web-front/framework/framework-v8-02.html#%E6%B8%B2%E6%9F%93%E5%BC%95%E6%93%8E%E5%8F%8A-webkit-%E4%BD%93%E7%B3%BB%E7%BB%93%E6%9E%84>

    * 从零开始的跨平台渲染引擎（零）——基础架构分析与设计

        <https://zhuanlan.zhihu.com/p/403395505>

    * 深度 | 跨平台Web Canvas渲染引擎架构的设计与思考

        <https://zhuanlan.zhihu.com/p/362690073>

    * Rendering engine

        <https://developer.mozilla.org/en-US/docs/Glossary/Rendering_engine>

    * blink 如何工作

        <https://www.zhihu.com/search?type=content&q=Blink%E5%A6%82%E4%BD%95%E5%B7%A5%E4%BD%9C%20>

* gpu algorithms

    * GPU Sorting Algorithms in OpenCL

        <https://github.com/Gram21/GPUSorting>

    * OpenRC (v. 0.1-beta)

        <https://github.com/macroing/OpenRC>

    * opencl-bitonic-sort

        <https://github.com/icaromagalhaes/opencl-bitonic-sort>

    * ClPy: OpenCL backend for CuPy

        <https://github.com/fixstars/clpy>

    * 并行排序算法原理及其程序实现

        <https://baijiahao.baidu.com/s?id=1761234516988191255&wfr=spider&for=pc>

* opencl 相关

    *  OpenCL C++ Bindings

        <https://github.khronos.org/OpenCL-CLHPP/>

    * a simple usage of OpenCL
    
        <https://www.cs.ubbcluj.ro/~rlupsa/edu/pdp/progs/opencl-sample1.cpp>

    *  OpenCL-examples

        <https://github.com/Dakkers/OpenCL-examples/tree/master>

    * Introduction to OpenCL Programming (C/C++)

        <https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/opencl/>

* C++ 的 traits 技术到底是什么？

    <https://www.zhihu.com/tardis/zm/art/413864991?source_id=1003>

* 你工作中最推荐的 C/C++ 程序库有哪些，为什么？

    <https://www.zhihu.com/question/51134387/answer/574376062>
    
* yasio - 轻量级跨平台socket库

    <https://zhuanlan.zhihu.com/p/69580088>

* 有没有哪些高效的c++ socket框架？

    <https://www.zhihu.com/question/67089512/answer/249255377>

* Asio C++ Library

    <https://think-async.com/Asio/>

* git 相关操作

    * The following untracked working tree files would be overwritten by merge, but I don't care

        <https://stackoverflow.com/questions/17404316/the-following-untracked-working-tree-files-would-be-overwritten-by-merge-but-i>

    * Git Push error: refusing to update checked out branch

        <https://stackoverflow.com/questions/11117823/git-push-error-refusing-to-update-checked-out-branch>

* 分布式计算相关

    * 分布式计算  框架

        <https://www.zhihu.com/search?type=content&q=%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%A1%E7%AE%97%20%20%E6%A1%86%E6%9E%B6>

    * 9个主流的分布式深度学习框架

        <https://zhuanlan.zhihu.com/p/582498905>

    * 分布式图计算引擎

        <https://zhuanlan.zhihu.com/p/385286103>

    * 如何自学分布式深度学习框架？

        <https://www.zhihu.com/question/576308139>
        
    * Ray分布式计算框架详解

        <https://zhuanlan.zhihu.com/p/460600694>

    * 大模型-LLM分布式训练框架总结

        <https://zhuanlan.zhihu.com/p/623746805>

* Socket Programming in C

    <https://www.geeksforgeeks.org/socket-programming-cc/>

    这个其实是 unix 风格的 socket 编译。

* Using multiple namespaces [closed]

    <https://stackoverflow.com/questions/57063459/using-multiple-namespaces>

    之前好像实验过，使用多个 namespace 不影响，只有出现冲突的时候才会报错。

*  OpenCL-path-tracing-tutorial-3-Part-2

    <https://github.com/straaljager/OpenCL-path-tracing-tutorial-3-Part-2>

* 浅析Linux内核中的链表

    <https://blog.csdn.net/m0_74282605/article/details/128037229>

* c++11 yield函数的使用

    <https://blog.csdn.net/c_base_jin/article/details/79246211>

* glTexImage2D — specify a two-dimensional texture image

    <https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml>

* 贝塞尔曲线与插值

    * cxx-spline

        <https://github.com/snsinfu/cxx-spline>

    * Interpolation (scipy.interpolate)

        <https://docs.scipy.org/doc/scipy/reference/interpolate.html>

    * Bezier Curve in Modern C++

        <https://stackoverflow.com/questions/53219377/bezier-curve-in-modern-c>

    * How to make OpenGL camera and Ray-tracer camera show the same image?

        <https://stackoverflow.com/questions/37682152/how-to-make-opengl-camera-and-ray-tracer-camera-show-the-same-image>

    * OpenGL rotating a camera around a point

        <https://stackoverflow.com/questions/287655/opengl-rotating-a-camera-around-a-point>

    * 「科普扫盲」贝塞尔曲线

        <https://baijiahao.baidu.com/s?id=1646828274206074616&wfr=spider&for=pc>

    * 图形学 · 简谈 Bézier curve

        <https://zhuanlan.zhihu.com/p/470453595?utm_id=0>

    * 【数学与算法】贝塞尔(Bézier)曲线

        <https://blog.csdn.net/u011754972/article/details/123494165>

* How to return value from an asynchronous callback function? [duplicate]

    <https://stackoverflow.com/questions/6847697/how-to-return-value-from-an-asynchronous-callback-function>

