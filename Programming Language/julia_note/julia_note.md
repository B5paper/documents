# Julia Note

## cache

* julia

    Julia 是一款高性能、动态类型的编程语言，专为科学计算、数值分析和数据科学而设计。其核心特点是像 Python 一样易写，像 C 一样快。

    以下是 Julia 主要用法的简述：

    1. 核心特点与设计理念

        * 即时编译（JIT）：使用 LLVM 编译器，首次运行函数时编译，后续调用速度极快。

        * 多重分派：函数行为根据所有参数类型动态决定，便于编写通用且高效的代码。

        * 动态类型：语法简洁，交互体验类似 Python/MATLAB。

        * 高性能：无需向量化即可获得高性能，循环速度快（区别于 MATLAB/R/Python+NumPy）。

        * 互操作性：轻松调用 C、Fortran、Python（PyCall）、R 等库。

    2. 基本语法

        变量赋值：无需声明类型。

        ```julia
        x = 10
        name = "Julia"
        ```

        数组索引从 1 开始（可配置为 0）。

        ```julia
        arr = [1, 2, 3]
        println(arr[1])  # 输出 1
        ```

        循环与条件

        ```julia
        for i in 1:5
            if i % 2 == 0
                println("$i 是偶数")
            end
        end
        ```

        函数定义

        ```julia
        function add(a, b)
            return a + b
        end
        # 简写
        add(a, b) = a + b
        ```

    3. 类型系统与多重分派

        类型可选，但用于声明时可优化性能。

        ```julia
        function add(a::Int, b::Int)::Int
            return a + b
        end
        ```

        多重分派示例：

        ```julia
        greet(x::String) = println("Hello, $x")
        greet(x::Int) = println("You entered number $x")
        greet("World")  # 调用第一个
        greet(42)       # 调用第二个
        ```

    4. 高性能编程技巧

        避免全局变量：将代码包装在函数中。

        使用类型稳定性：函数内变量类型尽量不变。

        预分配数组：避免循环中动态扩容。

        ```julia
        function slow(n)
            a = 0
            for i in 1:n
                a += i
            end
            return a
        end
        ```

    5. 包管理与生态系统

        内置包管理器：

        ```julia
        using Pkg
        Pkg.add("包名")
        ```

        常用包：

        * DataFrames：数据处理（类似 pandas/R）。

        * Plots / Makie：绘图。

        * Flux / MLJ：机器学习。

        * DifferentialEquations：求解微分方程。

        * JuMP：优化建模。

    6. 并行与分布式计算

        原生支持多核、分布式计算。

        ```julia
        using Distributed
        addprocs(4)  # 添加4个进程
        @distributed for i in 1:10
            # 并行任务
        end
        ```

    7. 工作流程示例

        交互环境：在 REPL、Jupyter Notebook 或 VS Code 中编写代码。

        性能分析：使用 `@time`、`@profiler` 或第三方工具。

        调试：内置调试器 `Debugger.jl`。

    8. 与 Python 的对比与协作

        优势：性能更高，语法更一致，专为数值计算设计。

        劣势：社区规模较小，某些库不够成熟。

        混合使用：通过 PyCall 调用 Python 库（如 NumPy、PyTorch）。

    9. 典型应用场景

        数值模拟与高性能计算

        数据科学与可视化

        机器学习与深度学习

        优化与运筹学

        金融建模与量化分析

    简单示例：计算 π 的蒙特卡洛方法

    ```julia
    function estimate_pi(n)
        inside = 0
        for _ in 1:n
            x, y = rand(), rand()
            if x^2 + y^2 <= 1
                inside += 1
            end
        end
        return 4 * inside / n
    end

    println(estimate_pi(10^6))
    ```

    学习资源

    * 官方文档：docs.julialang.org

    * 中文社区：JuliaCN

    Julia 适合需要高性能但又不愿牺牲开发效率的场景，尤其适合科学计算和原型开发。

* julia 安装

    按官网提示，<https://julialang.org/downloads/>，执行

    ```bash
    curl -fsSL https://install.julialang.org | sh
    ```

* julia 学习资料

    * 官网 learn

        <https://julialang.org/learning/>

    * get started

        <https://docs.julialang.org/en/v1/manual/getting-started/>

    * doc

        <https://docs.julialang.org/en/v1/>

    * julia hub

        <https://juliahub.com/ui/Home>

    * 中文社区

        <https://discourse.juliacn.com/>

    * julia data science (datawhale 出品)

        <https://juliadatascience.io/>

    * 知乎专栏

        <https://zhuanlan.zhihu.com/julia-language?author=mrlaomang>

    * Julia语言入门 pku

        <https://www.math.pku.edu.cn/teachers/lidf/docs/Julia/html/_book/basics.html>

    * julia academy

        <https://juliaacademy.com/>

    * QuantEcon

        Open source code for quantitative economic modeling

        计量经济学代码。

        <https://github.com/quantecon>

    * Think Julia:

        <https://benlauwens.github.io/ThinkJulia.jl/latest/book.html>

    * Julia By Example

        <https://juliabyexample.helpmanual.io/>

    * JuliaBoxTutorials

        <https://github.com/JuliaAcademy/JuliaTutorials>

    * 100 Julia exercises with solutions

        <https://discourse.julialang.org/t/100-julia-exercises-with-solutions/78580>

    * JuliaDynamics 

        <https://github.com/JuliaDynamics>

        <https://juliadynamics.github.io/JuliaDynamics/>

    * JuliaML

        <https://github.com/JuliaML>

    * JuliaAstro

        <https://github.com/JuliaAstro>

    * Quantitative Finance in Julia 

        <https://github.com/JuliaQuant>

    * julia 的 vscode 插件

        <https://www.julia-vscode.org/>

    * Pluto.jl

        响应式笔记本（类似 Observable）

        <https://plutojl.org/>

    * Revise.jl

        开发必备！实时更新代码

        <https://github.com/timholy/Revise.jl>

    可视化与绘图

    * Plots.jl - 统一绘图接口（推荐新手）

        <https://docs.juliaplots.org/latest/>

    * Makie.jl - 高性能交互式绘图

        <https://docs.makie.org/stable/>

    * VegaLite.jl - 声明式统计可视化

        <https://www.queryverse.org/VegaLite.jl/stable/>

    数据科学栈

    ```julia
    # 完整的数据科学工作流
    DataFrames.jl      # 数据处理（类似pandas）
    CSV.jl             # 读写CSV
    StatsBase.jl       # 基础统计
    GLM.jl             # 回归模型
    MLJ.jl             # 机器学习框架
    ScikitLearn.jl     # scikit-learn接口
    ```

    科学计算

    * DifferentialEquations.jl - 微分方程求解（世界顶级）

        <https://docs.sciml.ai/DiffEqDocs/stable/>

    * Optim.jl - 优化工具

        <https://julianlsolvers.github.io/Optim.jl/stable/>

    * JuMP.jl - 数学优化建模

        <https://jump.dev/>

    机器学习/深度学习

    * Flux.jl - 灵活的深度学习框架

        <https://fluxml.ai/>

    * MLJ.jl - 机器学习统一接口

        <https://juliaai.github.io/MLJ.jl/stable/>

    * Knet.jl - 深度学习框架

        <https://github.com/denizyuret/Knet.jl>

    🏆 竞赛与社区

    * JuliaCon - 年度会议（视频全部在线）

        <https://juliacon.org/2026/>

    * Advent of Code - 用Julia解决编程挑战

        <https://github.com/topics/advent-of-code-julia>

    * Google Summer of Code - 参与开源项目

        <https://julialang.org/jsoc/>

    📱 社交与社区

    * Discourse论坛：英文主站 <discourse.julialang.org>

    * Slack：#beginners频道适合提问

    * 中文社区：

        微信公众号：JuliaCN

        QQ群：搜索"Julia语言"

        知乎专栏：Julia语言

    明星项目推荐学习

    * Julia数值计算示例 - 数值分析实现

        <https://github.com/search?q=Julia-Numerical-Analysis&type=repositories>

        <https://github.com/ivanslapnicar/NumericalMathematics>

    * 用Julia写游戏 - 游戏开发

        <https://github.com/JuliaGaming>

* 退出 julia

    * CTRL-D

    * `exit()`

* 如果表达式后有`;`那么 julia 不回显。`ans`只绑定最后一次回显的值。

* julia `include()`本质是执行文件中的所有代码

    `include("file.jl")`

* 运行 julia 脚本：`$ julia script.jl`

* julia 帮助模式

    命令行下按`?`进入帮助模式。输入任何关键字后按回车，可显示相关的帮助文档。

## topics