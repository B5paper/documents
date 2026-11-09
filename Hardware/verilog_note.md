# verilog note

## cache

* 所以说，如果没有时钟，就没法用 gtkwave 了吧？

    不是的，**没有时钟照样可以用 GTKWave 看波形**，这是个常见的误解。GTKWave 只是打开 VCD 文件、按时间轴画信号变化的工具，它不关心"有没有时钟"，只关心 VCD 里有没有记录信号值随时间的变化。

    ### 组合逻辑为什么也能有波形

    关键在于：**波形的时间轴由你在 C++ 测试台里传给 `dump()` 的时间戳决定，而不是由时钟决定**。哪怕模块里完全没有 `clk`，你只要在每次给输入、调用 `eval()` 之后调用 `tfp->dump(sim_time++)`，VCD 里就会记录下每个时刻的 `a`、`b`、`y` 值，GTKWave 打开后就能画出阶梯状跳变。

    回看你上一轮的异或测试台，它其实**已经能出波形**了——只要编译时加了 `--trace`、代码里调用了 `tfp->dump()`，`xor_gate.vcd` 就会被生成，`gtkwave xor_gate.vcd` 就能看到 `a`、`b`、`y` 在 4 个时间步上的跳变。整个过程没有任何时钟信号参与。

    ### 组合逻辑波形和时序逻辑波形的区别

    | | 时序逻辑（带时钟） | 组合逻辑（无时钟） |
    |---|---|---|
    | 时间轴的"节拍" | 由你手动翻转 `clk` 推动 | 由你给输入、递增 `sim_time` 推动 |
    | 波形上看到什么 | 时钟边沿 + 沿后输出更新 | 每次输入变化后输出立即跳变 |
    | `sim_time` 的物理意义 | 通常对应半个/一个时钟周期 | 只是一个递增的序号，没有真实时间含义 |

    所以组合逻辑的波形时间轴往往看起来"很抽象"——比如 0、1、2、3 四个时刻分别对应四种输入组合，GTKWave 里横轴单位是仿真时间步，不是纳秒。但这不影响你看清"输入变了，输出跟着变"的因果关系。

    ### 一个容易混淆的点

    有人觉得"没有时钟就没法看波形"，可能是因为在真实硬件或某些仿真器里，波形采样是按时钟边沿触发的。但 Verilator + VCD 的机制是**事件驱动的记录**：`dump()` 被调用一次，就记一帧当前所有信号值。你什么时候调 `dump()` 完全由 C++ 代码决定，跟设计里有没有时钟无关。

    ### 想让组合逻辑波形更"像样"的小技巧

    如果你希望波形横轴看起来更像真实时间，可以在每次输入变化后多 `dump()` 几步、或者让 `sim_time` 跳跃式递增，制造出"输入保持一段时间再变"的效果。例如：

    ```cpp
    // 每种输入组合保持 10 个时间单位
    for (int i = 0; i < 4; i++) {
        top->a = inputs[i][0];
        top->b = inputs[i][1];
        top->eval();
        for (int t = 0; t < 10; t++) {
            tfp->dump(sim_time++);
        }
    }
    ```

    这样 GTKWave 里每个输入组合会占一段宽度，跳变位置更清晰。但这纯粹是显示上的处理，组合逻辑本身并不需要"保持时间"才能正确工作。

    总结一句：**波形记录和时钟无关，只和 `dump()` 调用有关**。没有时钟的组合逻辑，只要测试台里调了 `dump()`，GTKWave 就能看。

* verilator usage introduction

    git repo: <https://github.com/verilator/verilator.git>

    * install

        ```bash
        apt install verilator
        apt install gtkwave
        ```

        check:

        ```bash
        verilator --version
        ```

        output:

        ```
        Verilator 4.038 2020-07-11 rev v4.036-114-g0cd4a57ad
        ```

        注：

        1. verilator 最新版本是 5.x，使用 apt 安装的版本是 4.x，落后一些

    * hello world

        * verilator 4 版本

            `our.v`:

            ```verilog
            module our;
               initial begin
                  $display("Hello World");
                  $finish;
               end
            endmodule
            ```

            `sim_main.cpp`:

            ```cpp
            #include "Vour.h"
            #include "verilated.h"

            int main(int argc, char** argv) {
                Verilated::commandArgs(argc, argv);
                Vour* top = new Vour;
                while (!Verilated::gotFinish()) {
                    top->eval();
                }
                delete top;
                return 0;
            }
            ```

            compile:

            ```bash
            verilator --cc --exe --build -j -Wall sim_main.cpp our.v
            ```

            参数解释：

            --cc：生成 C++ 模型（而非 SystemC 或仅做 lint）。

            --exe：告诉 Verilator 用户提供了 C++ 测试台，需要生成可执行文件。

            --build：让 Verilator 自动调用 make 完成编译链接。

            output:

            ```
            make: Entering directory '/home/test/projs/test_our/obj_dir'
            g++  -I.  -MMD -I/usr/share/verilator/include -I/usr/share/verilator/include/vltstd -DVM_COVERAGE=0 -DVM_SC=0 -DVM_TRACE=0 -faligned-new -fcf-protection=none -Wno-bool-operation -Wno-sign-compare -Wno-uninitialized -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable -Wno-shadow      -Os -c -o sim_main.o ../sim_main.cpp
            ar -cr Vour__ALL.a Vour__ALL.o
            ranlib Vour__ALL.a
            g++    sim_main.o verilated.o Vour__ALL.a      -o Vour
            make: Leaving directory '/home/test/projs/test_our/obj_dir'
            ```

            run:

            ```bash
            ./obj_dir/Vour
            ```

            output:

            ```
            Hello World
            - our.v:4: Verilog $finish
            ```

    * 反相器与波型

        要用 Verilator 跑一个带时钟激励的反相器并查看波形，关键是在 C++ 测试台里手动翻转时钟，并调用波形记录接口。Verilator 本身不生成时钟，也不自带波形查看器，时钟和波形都需要你在 C++ 侧主动驱动和记录。

        下面是一个可以直接跑通的完整例子。

        ### 第一步：写反相器 Verilog

        创建 `inv.v`：

        ```verilog
        module inv (
            input  wire clk,
            input  wire din,
            output reg  dout
        );

            always @(posedge clk) begin
                dout <= ~din;
            end

        endmodule
        ```

        这里用了一个时钟驱动的寄存器，方便你在波形里清楚看到“时钟沿到来时输出更新”的效果。

        ### 第二步：写 C++ 测试台（生成时钟 + 记录波形）

        创建 `sim_main.cpp`。关键点：**在每个仿真时间步，翻转 `clk` 并调用 `eval()`，然后调用 `dump()` 把当前信号值写入 VCD 文件**。

        ```cpp
        #include "Vinv.h"
        #include "verilated.h"
        #include "verilated_vcd_c.h"

        int main(int argc, char** argv) {
            Verilated::commandArgs(argc, argv);
            Verilated::traceEverOn(true);          // 全局开启波形追踪

            Vinv* top = new Vinv;

            VerilatedVcdC* tfp = new VerilatedVcdC;
            top->trace(tfp, 99);                   // 记录所有层级
            tfp->open("inv.vcd");                  // 输出波形文件名

            vluint64_t sim_time = 0;

            top->clk = 0;
            top->din = 0;

            // 跑 10 个时钟周期，每个周期 2 个时间单位（半周期翻转一次）
            for (int cycle = 0; cycle < 10; cycle++) {
                // 低电平半周期
                top->clk = 0;
                top->eval();
                tfp->dump(sim_time++);

                // 高电平半周期（上升沿在这里）
                top->clk = 1;
                top->eval();
                tfp->dump(sim_time++);

                // 每个周期改变一次输入，方便观察反相效果
                top->din = !top->din;
            }

            tfp->close();
            delete top;
            delete tfp;
            return 0;
        }
        ```

        几个容易踩坑的地方：

        - `Verilated::traceEverOn(true)` 必须调用，否则波形不会被记录。
        - `tfp->dump()` 必须在每次 `eval()` 之后调用，且传入递增的时间戳，否则波形时间轴会错乱。
        - `tfp->close()` 不能漏，否则 VCD 文件可能不完整。

        * `top->din = !top->din;`这里`!`表示逻辑取反，得到的值只有 0 或 1，如果使用`~`取反，那么会按位取反，得到的值可能是 0 和 0xff，导致输出为乱码。 

        ### 第三步：编译

        ```bash
        verilator --cc --trace --exe --build -j -Wall inv.v sim_main.cpp
        ```

        和 hello world 的区别在于多了 `--trace` 选项，它让 Verilator 生成的模型支持波形 dump。

        ### 第四步：运行并查看波形

        ```bash
        ./obj_dir/Vinv
        gtkwave inv.vcd
        ```

        如果系统没有 GTKWave，Ubuntu 下可以 `sudo apt install gtkwave`。打开后，把 `clk`、`din`、`dout` 拖进波形窗口，你应该能看到：每个时钟上升沿到来后，`dout` 变成与 `din` 相反的值。

        ### 关于时钟激励的一个提醒

        Verilator 的 C++ 测试台里，时钟**不是自动跑的**。你在 `for` 循环里手动把 `clk` 从 0 翻到 1、再翻回 0，这就是时钟。每个“0→1→0”构成一个周期。上面的例子半周期占 1 个时间单位，所以波形上时钟周期是 2 个时间单位。如果你想要更真实的时间感（比如 10ns 周期），可以在翻转后给 `sim_time` 增加更大的步长，或者在 `dump()` 时使用更细的时间分辨率，但 Verilator 本身是周期精确仿真器，仿真时间由你传入 `dump()` 的数值决定。

    * 组合逻辑的 example: 那如果我想测试一个组合逻辑呢，比如 y = ab' + a'b

        组合逻辑的测试和时序逻辑有两个关键区别：**没有时钟**，输出只取决于当前输入，所以激励方式是"给一组输入 → eval → 观察输出"，而不是按时钟节拍推进；另外，组合逻辑可以**穷举真值表**来验证，这对 2 输入逻辑非常方便。

        下面用 `y = ab' + a'b`（即异或）为例，给出两个版本：手写激励版和穷举真值表版。

        ### Verilog 模块

        创建 `xor_gate.v`：

        ```verilog
        module xor_gate (
            input  wire a,
            input  wire b,
            output wire y
        );

            assign y = (a & ~b) | (~a & b);

        endmodule
        ```

        注意用 `assign` 连续赋值，这是组合逻辑的典型写法，没有 `always @(posedge clk)`。

        ### 版本一：手写激励 + 波形

        `sim_main.cpp`：

        ```cpp
        #include "Vxor_gate.h"
        #include "verilated.h"
        #include "verilated_vcd_c.h"

        int main(int argc, char** argv) {
            Verilated::commandArgs(argc, argv);
            Verilated::traceEverOn(true);

            Vxor_gate* top = new Vxor_gate;

            VerilatedVcdC* tfp = new VerilatedVcdC;
            top->trace(tfp, 99);
            tfp->open("xor_gate.vcd");

            vluint64_t sim_time = 0;

            // 穷举 2 输入的全部 4 种组合
            int inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};

            for (int i = 0; i < 4; i++) {
                top->a = inputs[i][0];
                top->b = inputs[i][1];
                top->eval();                     // 组合逻辑：eval 后输出立即稳定
                tfp->dump(sim_time++);

                printf("a=%d b=%d -> y=%d\n", top->a, top->b, top->y);
            }

            tfp->close();
            delete top;
            delete tfp;
            return 0;
        }
        ```

        编译命令和之前一样，只需换文件名：

        ```bash
        verilator --cc --trace --exe --build -j -Wall xor_gate.v sim_main.cpp
        ./obj_dir/Vxor_gate
        ```

        预期输出：

        ```
        a=0 b=0 -> y=0
        a=0 b=1 -> y=1
        a=1 b=0 -> y=1
        a=1 b=1 -> y=0
        ```

        这正是异或的真值表，说明逻辑正确。用 `gtkwave xor_gate.vcd` 也能看到 4 段输入组合下 `y` 的跳变。

        ### 版本二：自检式测试台（推荐做法）

        实际验证中更有价值的是让测试台**自己判断对错**，而不是靠人眼看 printf。组合逻辑没有时钟，所以循环里直接遍历输入、比对期望值即可：

        ```cpp
        #include "Vxor_gate.h"
        #include "verilated.h"
        #include <cstdio>

        int main(int argc, char** argv) {
            Verilated::commandArgs(argc, argv);
            Vxor_gate* top = new Vxor_gate;

            int errors = 0;
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    top->a = a;
                    top->b = b;
                    top->eval();

                    int expected = a ^ b;         // 参考模型
                    if (top->y != expected) {
                        printf("FAIL: a=%d b=%d y=%d expected=%d\n",
                               a, b, top->y, expected);
                        errors++;
                    }
                }
            }

            if (errors == 0)
                printf("PASS: all 4 cases correct\n");
            else
                printf("FAILED: %d errors\n", errors);

            delete top;
            return errors ? 1 : 0;
        }
        ```

        编译时不需要 `--trace`（不记录波形可以更快）：

        ```bash
        verilator --cc --exe --build -j 0 -Wall xor_gate.v sim_main.cpp
        ./obj_dir/Vxor_gate
        ```

        这里用 `a ^ b` 作为参考模型（reference model），把 DUT 输出和它比对。这种"激励 + 期望值 + 自动比对"的结构就是最简单的自检测试台（self-checking testbench），实际项目中会把它扩展成随机激励、覆盖率统计等。

        ### 组合逻辑测试的要点

        | 要点 | 说明 |
        |------|------|
        | 不需要时钟 | 给输入后直接 `eval()`，输出立即更新，没有"等下一个沿"的概念 |
        | 穷举输入空间 | n 输入组合逻辑最多 2^n 种输入，小规模时全遍历最彻底 |
        | 关注 `eval()` 时机 | 如果模块有内部组合链，一次 `eval()` 通常已收敛；但存在多级反馈或复杂结构时，Verilator 会在单次 `eval()` 内迭代到稳定 |
        | 波形用途 | 组合逻辑波形主要看输入变化后输出是否按预期跳变，时间轴可以只按"输入序号"递增，不必模拟真实时间 |
        | 容易漏测的输入 | 如果模块有多个输入端口，穷举时注意别只测部分端口组合；对 2 输入没问题，输入多了要考虑用随机激励 |

        如果你的组合逻辑规模变大（比如 8 输入），就不适合再手写穷举数组了，可以改用双重循环或随机数生成激励，并用参考模型自动比对——结构和版本二完全一样，只是激励来源换成循环/随机函数。

* 三位加法器

    ```verilog
    module adder(count, sum, a, b, cin);
        input [2:0] a, b;
        input cin;
        output count;
        output [2:0] sum;
        assign {count, sum} = a + b + cin;
    endmodule
    ```

    其中，count 表示向上进位。

    注：

    1. assign {count, sum} = a + b + cin; 讲讲这行 verilog 代码的含义

        # Verilog 代码 `assign {count, sum} = a + b + cin;` 详解

        ## 一、整体功能

        这是一个**带进位输出的加法器**的核心语句。它将 `a + b + cin` 的运算结果拆成两部分：

        - `sum`：低位的和
        - `count`：高位的进位（carry out）

        本质上等价于一个**全加器**（Full Adder）的行为。

        ## 二、逐部分拆解

        ### 1. `assign` —— 连续赋值

        `assign` 用于对 `wire`（线网）类型变量进行**连续赋值**，属于组合逻辑。只要右边表达式的操作数发生变化，左边就立即更新（行为上相当于持续驱动的组合电路）。

        ### 2. `{count, sum}` —— 拼接（Concatenation）

        花括号 `{}` 是 Verilog 的**位拼接运算符**，把多个信号按**高位在左、低位在右**的顺序拼成一个更宽的向量。

        假设：
        - `a`、`b` 是 `N` 位宽
        - `cin` 是 1 位
        - `sum` 是 `N` 位宽
        - `count` 是 1 位（进位）

        那么 `{count, sum}` 就是一个 **N+1 位**的向量，其中：
        - 最高位 = `count`
        - 低 N 位 = `sum`

        ### 3. `a + b + cin` —— 算术加法

        Verilog 中加法结果的位宽由**操作数中最大位宽**决定，并考虑进位。这里加法会产生一个 N+1 位的结果（含进位）。

        ### 4. 赋值的方向

        `assign {count, sum} = a + b + cin;`

        右边 N+1 位的加法结果，**整体赋给**左边 N+1 位的拼接向量。于是：
        - 加法结果的**最高位**自动落到 `count`（进位输出）
        - 加法结果的**低 N 位**自动落到 `sum`（和输出）

        这就巧妙地实现了「和 + 进位」的分离。

        ## 三、举个具体例子

        设 `a = 4'b1111`，`b = 4'b0001`，`cin = 1'b0`：

        ```
          a + b + cin = 1111 + 0001 + 0 = 1_0000  (5 位)
        ```

        赋值时：

        ```
        {count, sum} = 5'b1_0000
             ↑     ↑
           count=1  sum=0000
        ```

        结果：`sum = 4'b0000`，`count = 1'b1`，正确表示 15 + 1 = 16（4 位和溢出，进位为 1）。

        ## 四、与等价写法的对比

        ```verilog
        // 写法 A：拼接法（本题）
        assign {count, sum} = a + b + cin;

        // 写法 B：分开写
        wire [N:0] tmp = a + b + cin;
        assign sum   = tmp[N-1:0];
        assign count = tmp[N];
        ```

        写法 A 更简洁，是 Verilog 中处理**进位**的惯用技巧，常见于：
        - 全加器 / 行波进位加法器
        - 多位加法器的进位链
        - 计数器、累加器中的溢出检测

        ## 五、注意事项

        1. **位宽要匹配**：左边拼接总位宽应等于右边加法结果位宽，否则会发生截断或补零。
        2. **`count` 的命名**：这里 `count` 实际是 carry-out，命名上有点容易误解（通常叫 `cout`），阅读时要结合上下文确认它是进位而非计数器。
        3. **`sum` 位宽**：`sum` 必须与 `a`、`b` 同宽（N 位），`count` 为 1 位，才能正确捕获进位。
        4. **无符号运算**：Verilog 默认按无符号处理，若 `a`、`b` 声明为 `signed`，需注意符号扩展的影响。

        ## 六、一句话总结

        > 这行代码把 `a + b + cin` 的 N+1 位加法结果，通过**位拼接**一次性拆成「低 N 位的和 `sum`」和「最高位的进位 `count`」，是 Verilog 里实现带进位加法器的经典写法。

* 二选一多路选择器的几种写法

    * 行为描述

        ```verilog
        module muxtwo (out, a, b, sl);
            input a, b, sl;
            output out;
            reg out

            always @ (sl or a or b)
                if (! sl) out = a;
                else out = b;
        endmodule
        ```

        当控制信息 sl 为低电平时，输出 out 与输入 a 相同，否则与 b 相同。

        注：

        1. always 语句块里没有 {}，verilog 是如何区分 always 的作用域的？难道 always 的作用是 module 内全局的？

    * 布尔表达式

        ```verilog
        module muxtwo (out, a, b, sl);
            input a, b, sl;
            output out;

            wire nsl, sela, selb;

            assign nsl = ~sl;
            assign sela = a & nsl;
            assign selb = b & sl;
            assign out = sela | selb;
        endmodule
        ```

        注：

        1. 二选一多路选择器的逻辑表达式怎么写？

    * 模块描述

        ```verilog
        module muxtwo (out, a, b, sl);
            input a, b, sl;
            output out;

            not u1(nsl, sl);
            and #1 u2(sela, a, nsl);
            and #1 u3(selb, b, sl);
            or #1 u4((out, sela, selb);
        endmodule
        ```

        注：

        1. # 和 1 之间有没有空格？

* 将行为描述或逻辑表达式描述的 verilog 代码转换为模块描述的 verilog 的过程叫综合（synthesis）

* always @ (sl or a or b)

    表示只要 sl 或 a 或 b，其中有一个变化时，就执行下面的语句。

* 电路的几种级别的抽象

    1. 系统级（system-level）

        用语言提供的高级结构能够实现待设计模块的外部性能的模型

    2. 算法级（algorithm-level）

        用语言提供的高级结构能够实现算法运行的模型。

    3. RTL 级（register transfer level）

        描述数据在寄存器之间的流动和如何处理、控制这些数据流动的模型

    4. 门级（gate-level）

        描述逻辑门及逻辑门之间连接的模型

    5. 开关级（switch-level）

        描述器件中和储存节点以及它们之间连接的模型。

* 逻辑网表：edif 文件

* 自顶向下的设计（Top_Down 设计）

    从系统级开始，把系统划分为基本单元，然后再把每个基本单元划分为下一层次的基本单元，一起这样做下去，直到可以直接用 EDA 元件库中的基本元件来实现为止。

* 软核（soft core）: 把功能经过验证的、可综合的、实现后电路结构总门数在 5000 门以上的 verilog hdl 模型称之为软核

* 固核（firm core）: 把在某一种现场可编程门阵列（FPGA）器件上实现的、经验证是正确的、总门数在 5000 门以上电路结构编码文件称为固核

* 硬核（hard core）: 把在某一种专用集成电路工艺的（asic）器件上实现的、经验证是正确的、总门数在 5000 门以上的电路结构版图掩膜称为硬核

* verilog 的基本设计单元是 模块（block）

    ```verilog
    module block(a, b, c, d);
        input a, b;
        output c, d;

        assign c = a | b;
        assign d = a & b;
    endmodule
    ```

* verilog 模块例化端口映射方式

	**Verilog 模块例化：命名关联（`.端口名`） vs 位置关联**

	先说结论：

	**Verilog 支持两种端口映射方式：命名关联（`.xxx(信号)`）、位置关联（按顺序写，类似C传参），不是必须用 `.xxx`。但工业界强烈推荐用命名关联，不推荐位置关联。**

	> 注意：C的函数参数是**值传递**；Verilog模块例化是**硬件连线**，本质完全不一样，只是语法看着有点像。

	1. 两种写法对比

        子模块定义：

        ```verilog
        module sub(
            input  a,
            input  b,
            output c
        );
        // ...
        endmodule
        ```

        **✅ 命名关联（推荐，`.端口名`）**

        ```verilog
        sub u_sub(
            .a(top_a),
            .b(top_b),
            .c(top_c)
        );
        ```

        - `.a` 指**子模块的端口名a**，括号里是**顶层的信号**
        - 顺序随便调换，不影响
        - 要求：`.后面的名字`必须和子模块端口名字**完全一致**

        **✅ 位置关联（可以用，类似C按位置匹配）**
        ```verilog
        sub u_sub( top_a, top_b, top_c );
        ```
        - 按照端口定义顺序一一对应，**不需要写端口名**
        - 第一个信号绑定子模块第一个端口`a`，第二个绑定`b`，第三个绑定`c`
        - 顺序不能乱，不能跳过端口

	2. 为什么工程上几乎都只用 `.xxx` 命名关联？
        1. 硬件端口增减非常容易踩坑（位置关联最大痛点）

            C函数新增参数，编译器会直接报参数数量不匹配。

            但Verilog位置例化：**子模块中间增加/删除端口，不会报语法错误，只会错位连线！**

            例子：

            原来子模块端口：`a,b,c`

            后来改成：`a,new_sig,b,c`

            如果你用位置例化：

            ```verilog
            sub u_sub(top_a, top_b, top_c);
            ```

            现在映射变成：

            `top_a → a`

            `top_b → new_sig`

            `top_c → b`

            **c端口悬空！**

            工具不会报错，直接生成错误硬件，仿真/综合才发现，极难排查。

            > 命名关联就不会：新增端口不会打乱已有连线，多余端口可以不接，不会错位。

        2.  端口顺序可读性差

            模块端口一多（十几个、几十个），靠肉眼数顺序非常容易数错。

            命名关联一眼就能看到：哪个顶层信号连到子模块哪个端口，代码自解释。

        3. 支持部分端口不连接

            命名关联可以只接部分端口，不用写全部；

            位置关联必须按顺序把所有端口都列出来，不能跳过。

            ```verilog
            sub u_sub(
                .a(top_a),
                .c(top_c)  // 不接b
            );
            ```

        4. 支持端口重排

            子模块端口列表顺序调整，命名关联代码**完全不用改**。

	3. 澄清：为什么 `.xxx` 的名字必须和子module端口一致？

        `.port_name(signal)` 的语义：

        > `.port_name` = **子模块内部的端口名称**
        > 括号内 = **顶层模块的信号名**

        所以 `.`后面的标识符，就是用来**索引子模块端口**的，自然必须和子模块定义的端口名字一模一样。

        括号里的顶层信号名字**不需要和子模块端口同名**，这是很多新手混淆的点。

        ```verilog
        sub u_sub(
            .a( my_input_signal ), // .a 是子模块端口；my_input_signal是顶层信号，名字可以不一样
            .b( enable ),
            .c( result_out )
        );
        ```

	4. 和C语言本质区别（重点）

        - C：函数调用，**软件，传递参数值**，参数顺序是编译器约定；函数参数是临时变量。
        - Verilog模块例化：**实例化硬件电路**，不是函数调用。
          相当于把一块硬件电路放到顶层，**把一根根导线连到这块电路的引脚上**。
          - 命名关联：**按引脚名字接线**（最直观，像电路板上按丝印引脚名接线）
          - 位置关联：**按引脚排列顺序接线**（像DIP芯片引脚从1开始数）

	5. 简单总结

        1. Verilog**允许位置例化**，可以不用`.xxx`，类似C按顺序传；
        2. 工程不推荐位置例化，因为子模块端口改动极易造成**静默连线错位，无编译报错**；
        3. `.port()` 语法里，`.port` 必须等于子模块端口名，因为这是指定要连接**子模块的哪个引脚**；括号内顶层信号名可以任意；
        4. 模块例化不是函数调用，是**硬件实例 + 连线**，和C语义完全不同。

	如果你想，我可以举一个**位置例化踩坑的仿真例子**，直观看到错位连线的bug。

* 详细讲讲 verilog 中的 parameter

	`parameter` 是 Verilog 中用于定义**常量**和实现**模块参数化**的核心关键字。它的核心价值在于：让你写一次模块，就能通过传入不同的参数值，在多次例化时得到不同规格的电路（如不同位宽的加法器、不同深度的 FIFO），从而极大提高代码的复用性。

	### 基本语法与声明位置

	`parameter` 可以在模块内部声明，也可以在 Verilog-2001 引入的模块头部参数端口列表（`#()`）中声明。后者是现代设计中最常用、最推荐的方式，因为它让模块的“可配置接口”一目了然。

	**推荐写法（ANSI 风格，Verilog-2001 及以后）**：
	```verilog
	module counter #(
		parameter WIDTH = 4,        // 位宽，默认 4
		parameter INIT  = 0         // 初值，默认 0
	) (
		input clk,
		input rst,
		output reg [WIDTH-1:0] q
	);
		// 模块内部逻辑
	endmodule
	```
	这种写法将参数直接放在模块名后的 `#()` 中，例化时传参非常直观。

	**传统写法（Verilog-1995 风格）**：
	```verilog
	module counter (clk, rst, q);
		parameter WIDTH = 4;
		parameter INIT  = 0;
		// ...
	endmodule
	```
	这种写法把参数声明放在模块内部，虽然也能用，但可读性和传参便利性不如头部声明。

	### 如何修改参数值（参数覆盖）

	在例化模块时，你有两种方式修改默认参数值：

	**1. 例化时传参（推荐）**
	这是最常用、最清晰的方式，支持**命名关联**（`.参数名(值)`）和**位置关联**（按声明顺序给值）。命名关联不易出错，尤其参数多的时候。
	```verilog
	// 命名关联（推荐）
	counter #(
		.WIDTH(8),
		.INIT(10)
	) u_counter (
		.clk(clk),
		.rst(rst),
		.q(q)
	);

	// 位置关联（需严格按顺序）
	counter #(8, 10) u_counter (...);
	```


	**2. 使用 `defparam` 语句**
	这种方式通过层次化路径直接修改已例化模块的参数值。虽然能用，但在现代 RTL 设计中**不推荐**，因为它破坏了模块的封装性，且综合工具对其支持不一，不利于代码维护。
	```verilog
	counter u_counter (...);
	defparam u_counter.WIDTH = 8;  // 不推荐
	```

	### 关键区分：`parameter` vs. `localparam`

	这是面试和实际工程中经常遇到的一个点。两者都定义常量，但**可变性**不同：

	| 特性 | `parameter` | `localparam` |
	| :--- | :--- | :--- |
	| **能否被外部覆盖** | **可以**。父模块例化时能修改。 | **不可以**。值被锁定在模块内部。 |
	| **主要用途** | 模块的可配置接口（位宽、深度等）。 | 模块内部的固定常量（状态机编码、派生常量）。 |
	| **引入标准** | Verilog-1995 | Verilog-2001 |

	**为什么需要 `localparam`？**
	一个典型场景是：某个内部常量是由外部可配置的 `parameter` 计算得来的。比如 RAM 的**总容量**由**字宽**和**地址位宽**决定。字宽和地址位宽需要开放给用户配置（用 `parameter`），但总容量不希望用户再去改动（否则可能算错），此时就应该用 `localparam`：
	```verilog
	module myram #(
		parameter WORD_SIZE = 8,
		parameter ADDR_SIZE = 4
	) (...);
		localparam MEM_SIZE = WORD_SIZE * (1 << ADDR_SIZE);  // 派生常量，禁止外部覆盖
		// ...
	endmodule
	```
	这样，用户例化时只能改 `WORD_SIZE` 和 `ADDR_SIZE`，`MEM_SIZE` 会自动跟着变，避免了不一致的风险。

	### 常见用途与最佳实践

	*   **定义位宽与深度**：这是最经典的用法。用 `parameter` 定义数据位宽、地址位宽、FIFO 深度，让模块像“模板”一样可缩放。
	*   **定义延迟与时序**：在门级建模或仿真模型中，用 `parameter` 定义门延迟，方便在不同工艺库或仿真场景间切换。
	*   **状态机编码**：状态机的状态值通常用 `localparam` 定义，因为它们是模块内部实现细节，不应被外部篡改。
	*   **命名规范**：建议 `parameter` 和 `localparam` 的名称**全部大写**（如 `DATA_WIDTH`、`IDLE`），以便在代码中一眼区分出它们是常量，而非普通信号或变量。

	---
	**如果你想进一步了解 `parameter` 与 `` `define `` 宏定义在作用域和调试上的具体差异，或者想看一个参数化 FIFO 的完整代码示例，可以告诉我。**

* 简述二进制加法器的设计思路与 verilog 代码实现

    **二进制加法器设计思路**

    1. 半加器（Half Adder）

        - 两个1位二进制数相加
        - 输出：和(S)与进位(C)
        - 逻辑表达式：
          - S = A ⊕ B
          - C = A & B

    2. 全加器（Full Adder）

        - 三个1位二进制数相加（两个加数+进位输入）
        - 输出：和(S)与进位输出(Cout)
        - 逻辑表达式：
          - S = A ⊕ B ⊕ Cin
          - Cout = (A & B) | (A & Cin) | (B & Cin)

    3. 多位加法器设计方法

        - **行波进位加法器**：全加器级联，低位进位连接到高位进位输入
        - **超前进位加法器**：并行计算进位，减少延迟

    **Verilog代码实现**

    1. 半加器

        ```verilog
        module half_adder (
            input  a,      // 加数1
            input  b,      // 加数2
            output s,      // 和
            output c       // 进位
        );

        assign s = a ^ b;  // 异或得到和
        assign c = a & b;  // 与得到进位

        endmodule
        ```

    2. 全加器

        ```verilog
        module full_adder (
            input  a,      // 加数1
            input  b,      // 加数2
            input  cin,    // 进位输入
            output s,      // 和
            output cout    // 进位输出
        );

        assign s = a ^ b ^ cin;
        assign cout = (a & b) | (a & cin) | (b & cin);

        endmodule
        ```

    3. 4位行波进位加法器

        ```verilog
        module adder_4bit_ripple (
            input  [3:0] a,      // 4位加数1
            input  [3:0] b,      // 4位加数2
            input  cin,          // 进位输入
            output [3:0] s,      // 4位和
            output cout          // 进位输出
        );

        wire c1, c2, c3;         // 内部进位信号

        // 实例化4个全加器
        full_adder fa0 (.a(a[0]), .b(b[0]), .cin(cin),   .s(s[0]), .cout(c1));
        full_adder fa1 (.a(a[1]), .b(b[1]), .cin(c1),    .s(s[1]), .cout(c2));
        full_adder fa2 (.a(a[2]), .b(b[2]), .cin(c2),    .s(s[2]), .cout(c3));
        full_adder fa3 (.a(a[3]), .b(b[3]), .cin(c3),    .s(s[3]), .cout(cout));

        endmodule
        ```

    4. 4位超前进位加法器

        ```verilog
        module adder_4bit_lookahead (
            input  [3:0] a,
            input  [3:0] b,
            input  cin,
            output [3:0] s,
            output cout
        );

        wire [3:0] g, p;        // 生成信号和传播信号
        wire [4:0] c;           // 进位信号

        assign c[0] = cin;

        // 计算生成信号和传播信号
        assign g = a & b;       // 生成信号：当a和b都为1时产生进位
        assign p = a ^ b;       // 传播信号：当a或b为1时传播进位

        // 超前进位逻辑
        assign c[1] = g[0] | (p[0] & c[0]);
        assign c[2] = g[1] | (p[1] & g[0]) | (p[1] & p[0] & c[0]);
        assign c[3] = g[2] | (p[2] & g[1]) | (p[2] & p[1] & g[0]) | (p[2] & p[1] & p[0] & c[0]);
        assign c[4] = g[3] | (p[3] & g[2]) | (p[3] & p[2] & g[1]) | (p[3] & p[2] & p[1] & g[0]) | (p[3] & p[2] & p[1] & p[0] & c[0]);

        // 计算和
        assign s = p ^ c[3:0];

        assign cout = c[4];

        endmodule
        ```

    5. 使用Verilog运算符的简单实现

        ```verilog
        module adder_simple (
            input  [3:0] a,
            input  [3:0] b,
            input  cin,
            output [3:0] s,
            output cout
        );

        assign {cout, s} = a + b + cin;

        endmodule
        ```

    总结

    - **行波进位加法器**：结构简单，但延迟较大（随位数增加而增加）
    - **超前进位加法器**：并行计算进位，速度快但硬件复杂度高
    - **实际应用**：可根据需求选择合适的设计，或使用Verilog内置的"+"运算符

## topics

### verilog 语法

* 整数

    * 二进制：b, B, 十进制：d, D，十六进制：h, H, 八进制：o, O

        example:

        * `8'b10101100`

        * `8'ha2`

    * x 代表不定值，z 和 ? 代表高阻值。

    * 负数的负号必须写在表达式最前面

        example: `-8'd5`

    * 下划线只能加在数字之间，不能加在进制和数字之间

        `16'b1010_1011_1111_1010`

    * 常量默认是 32 位

    * 字符串中，每个字符用 8 位 ascii 码表示

* 参数（parameter）

    example:

    ```verilog
    parameter mbs = 7;
    parameter e = 25, f = 29;
    parameter r = 5.7;
    parameter byte_size = 8, byte_msb = byte_size - 1;
    parameter average_delay = (r + 1) / 2;
    ```

* 反标注（back-annotate）

    在一个模块中改变另一个模块的参数时，需要使用 defparam 命令。在做布线后仿真时，就是利用这种方法把布线延迟通过布线工具生成的延迟参数文件反标注（back-annotate）到门级 verilog 网表上。

    ```verilog
    include "Top.v"
    include "Block.v"
    include "Annotate.v"
    
    module Test;
        wire W;
        Tpp T();
    endmodule

    module Top;
        wire W;
        Block B1();
        Block B2();
    endmodule

    module Block;
        Parameter P = 0;
    endmodule

    module Annotate;
        defparam
            Test.T.B1.P = 2,
            Test.T.B2.P = 3;
    endmodule
    ```

* 网络数据类型

    * wire

        由单个门或 assign 赋值的变量

        module 中输入、输出信号的默认类型就是 wire 型

    * tri

        由多个驱动源驱动

    如果没有门或 assign 赋值，那么 wire / tri 的值为高阻值 z。
    
    真值表：

    | wire / tri | 0 | 1 | x | z |
    | - | - | - | - | - |
    | 0 | 0 | x | x | 0 |
    | 1 | x | 1 | x | 1 |
    | x | x | x | x | x |
    | z | 0 | 1 | x | z |

* reg

    example:

    ```verilog
    reg rega;  // 一位寄存器
    reg [3:0] regb;
    reg [4:1] regc, regd;
    ```

    reg 的默认初始值为不定值 x

    reg 型数据常用来表示 always 模块内的指定信号，常代表触发器。在 always 模块内赋值的每一个信号都必须定义成 reg 型。

* memory 型

    创建：

    ```verilog
    reg[n-1:0] mema[255:0];  // 定义一个 memory 型变量 mema，其有 256 个 n 位 reg 变量


    parameter wordsize = 16,
              memsize = 256;
    reg[wordsize - 1 : 0] mem[memsize - 1 : 0], writereg, readreg;  // 一行内同时定义 mem 和 reg 变量

    
    reg[n-1:0] rega;  // 一个 n 位的寄存器
    reg mema [n-1:0];  // 一个由 n 个 1 位寄存器构成的 mem 变量
    ```

    索引与赋值：

    ```verilog
    mema[3] = 0;  // 使用下标索引到具体的寄存器
    ```

* 运算符及表达式

    1. 算术运算符：+, -, *, /, %

        关于取模运算 %，如果两个数中有一个是负数，那么先按正数计算，结果的符号取第一个数的符号：

        ```verilog
        -10 % 3  // -1
        11 % -3  // 2
        ```

        在进行算术运算操作时，如果某一个操作数有不确定的值，那么表达式的结果也为不定值 x

    2. 赋值运算符：=, <=

    3. 关系运算符：>, <, >=, <=

    4. 逻辑运算符：&&, ||, !

    5. 条件运算符：?:

    6. 位运算符：~, |, ^, &, ^~

        ^~ 表示按位同或（异或非）

    7. 移位运算符：<<, >>

        使用 0 来填充空位。

    8. 拼接（concatation）运算符：{}

        example:

        ```verilog
        {a, b[3:0], w, 3'b101}
        {a, b[3], b[2], b[1], b[0], w, 1'b1, 1'b0, 1'b1}
        {4{w}}  // 等同于 {w, w, w, w}
        {b, {3{a, b}}}  // 等同于 {b, a, b, a, b, a, b}
        ```

    9. 等式运算符：==, !=, ===, !==

        == 和 != 只作逻辑比较，如果操作数中有 x, z，那么结果为 x. (x 也算作 0 吗？)

        === 和 !== 会按位比较，如果操作数中有 x, z，那么也纳入比较。常用于 case 语句。

    10. 缩减运算符（reduction operator）

        ```verilog
        reg[3:0] b = 4'b1011;
        reg c;
        c = &b;  // 1 & 0 & 1 & 1 = 0
        ```

* 赋值语句和块语句

    * 赋值语句

        * 非阻塞（non blocking）赋值

            `b <= a;`

            1. 在语句块中，上面语句所赋的变量值不能立即为下面语句所用

            2. 块结束后才能完成这次赋值操作，而所赋的变量值是上一次赋值得到的

            3. 在编写可综合时序逻辑模块时，这是最常用的赋值方法

        * 阻塞（blocking）赋值

            `b = a;`

            1. 赋值语句执行完后，块才结束

            2. b 的值在赋值语句执行完后立即改变

            3. 在时序逻辑中使用时，可能会产生意想不到的结果

    * 块语句

        * 顺序块，begin end

            1. 块内的语句按顺序执行

            2. 每条语句的延迟时间是相对于前一条语句的仿真时间而言的

            3. 直到最后一条语句执行完，程序流程控制才跳出该语句块

            ```verilog
            begin
                statement_1;
                statement_2;
                ...
            end

            begin: block_name
                块内声明语句;
                statement_1;
                ...
            end
            ```

            块内声明语句可以是 parameter, reg, integer, real

            example:

            ```verilog
            begin
                areg = breg;
                creg = areg;
            end


            begin
                areg = breg;
                # 10 creg = areg;  // 在两条赋值语句之间延迟 10 个时间单位
            end


            parameter d = 50;
            reg [7:0] r;
            begin
                # d  r = 'h35;
                # d  r = 'hE2;
                # d  r = 'h00;
                # d  r = 'hF7;
                # d  -> end_wave;  // 触发事件 end_wave
            end
            ```

        * 并行块，fork join

            1. 块内语句是同时执行的

            2. 块内每条语句的延迟时间是相对于程序流程控制进入到块内的仿真时间

            3. 延迟时间是用来给赋值语句提供执行时序的

            4. 当按时间时序排序在最后的语句执行完后或一个 disable 语句执行时，程序流程控制跳出该程序块

            ```verilog
            fork
                statement_1;
                statement_2;
                ...
            join


            fork: block_name
            块内声明语句
                statement_1;
                ...
            join
            ```

            块内说明语句可以是 parameter, reg, integer, real, time, event

            example:

            ```verilog
            fork
                # 50    r = 'h35;
                # 100   r = 'hE2;
                # 150   r = 'h00;
                # 200   r = 'hF7;
                # 250   ->  end_wave;  // 触发事件 end_wave
            ```

* 块名

    块名的好处：

    1. 可以在块内定义局部变量，只在块内使用

    2. 可以允许块被其他语句调用，如 disable 语句

    3. verilog 中所有变量都是静态全局的，只有唯一值

* if else 语句

    ```verilog
    if (cond)
        statement;

    if (cond)
        statement_1;
    else
        statement_2;

    if (cond_1)
        statement_1;
    else if (cond_2)
        statement_2;
    eles if (cond_3)
        ...
    else
        statement;
    ```

    条件语句必须在过程块中使用，即 initial 和 always

    一些说明：

    1. if 条件中，0, x, z 都按 false 处理，1 按 true 处理

    2. if / else 下面跟的语句可以使用 begin ... end ... 扩展成语句块

    3. 缩写

        ```verilog
        if (expr)  // if (expr == 1)
        if (! expr)  // if (expr != 1)
        ```

        注：

        1. 所有非 1 值都是 false 吗？

* io 口

	输入：`input [信号位宽 - 1:0] 端口名;`

	输出：`output [信号位宽 - 1:0] 端口名;`

	输入输出：`inout [信号位宽-1:0] 端口名;`

	i/o 说明也可以写在端口声明语句里：

	```verilog
	module module_name(input port1, input port2, ...
						output port1, output port2, ...);
	```

* wire 和 reg 类型变量的声明

	```verilog
	reg [width-1:0] rvar_1, rvar2, ...;
	wire [width-1:0] wvar_1, wvar_2, ...;
	```

* 功能定义

	* `assign a = b & c;`

	* `and # 2 ul(q, a, b);`

		实例化一个与门`ul`，其输入端为 a, b, 输出为 q, 输出延迟为 2 个单位时间

	* always

        ```verilog
        always @(posedge clk or posedge clr);

		begin
			if (clr) q <= 0;
			else if (en) q <= d;
		end
		```

		always 块内的语句按顺序执行，always 块又称为过程块。多个 always 块是同时执行的。

* 只有连续赋值语句 assign 和实例引用语句可以独立于过程块而存在于模块的功能定义部分。

* verilog 模块中，initial 块，always 块，连续赋值语句，实例引用，都是并行的


### rtl and tuto

* 什么是 rtl

    **RTL** 全称 **Register Transfer Level**，中文叫**寄存器传输级**。它是数字芯片设计中最核心的抽象层次，也是你写 Verilog 代码时最常打交道的那个层面。

    ### 🎯 一句话理解

    RTL 描述的是：**数据在寄存器之间怎么流动、每个时钟周期做什么运算。**

    它不关心晶体管怎么排布，也不关心门电路怎么连，只关心：
    - 有哪些**寄存器**（存储状态）
    - 寄存器之间的**组合逻辑**做什么运算
    - 每个**时钟沿**，数据怎么从一级寄存器传到下一级

    ### 🧱 一个最简单的例子

    ```verilog
    always @(posedge clk) begin
        if (rst)
            counter <= 0;
        else
            counter <= counter + 1;
    end
    ```

    这段代码就是 RTL。它说的是：

    - 有一个寄存器 `counter`
    - 每个时钟上升沿，如果复位就清零，否则加 1
    - `counter + 1` 是组合逻辑，结果在下一个时钟沿锁进寄存器

    它没有说用几个门、什么工艺、多少延迟——那些是综合工具的事。

    ### 🔗 RTL 在芯片设计流程中的位置

    芯片设计从抽象到物理，大致是这么一层层往下走的：

    | 层次 | 描述什么 | 谁来做 |
    |---|---|---|
    | **系统级** | 算法、架构 | 架构师 |
    | **RTL** | 寄存器 + 组合逻辑 | **你写的 Verilog** |
    | **门级** | 与门、或门、触发器 | 综合工具 |
    | **晶体管级** | MOS 管、W/L | 布局布线 + 定制 |
    | **物理版图** | 几何图形、金属层 | 后端工程师 |

    RTL 是**人写的最后一层**。再往下，就是工具自动生成的了。所以 RTL 代码的质量，直接决定芯片的功能和性能上限。

    ### 📦 RTL 里有什么

    RTL 代码主要由这几类东西构成：

    **1. 寄存器（时序逻辑）**
    ```verilog
    always @(posedge clk) q <= d;
    ```
    用时钟沿驱动，存储状态。

    **2. 组合逻辑**
    ```verilog
    assign sum = a + b;
    always @(*) y = sel ? x : z;
    ```
    没有时钟，输入变输出就变。

    **3. 有限状态机（FSM）**
    ```verilog
    always @(posedge clk) begin
        case (state)
            IDLE: if (start) state <= RUN;
            RUN:  if (done)  state <= IDLE;
        endcase
    end
    ```
    控制数据流动的"大脑"。

    **4. 模块层次**
    ```verilog
    module cpu_top;
        alu   u_alu (...);
        regfile u_regfile (...);
    endmodule
    ```
    大设计拆成小模块，层层例化。

    ### ⚖️ RTL 和其他层次的区别

    这是最容易混淆的地方，用你之前问过的概念对比：

    | 层次 | 代码长什么样 | 能仿真什么 | 用什么仿真 |
    |---|---|---|---|
    | **RTL** | `always @(posedge clk) q <= d;` | 功能对不对 | VCS / Verilator |
    | **门级网表** | `DFFQ u_dff (.CK(clk), .D(d), .Q(q));` | 功能 + 时序 | VCS + SDF |
    | **晶体管级** | `M1 drain gate source bulk nmos W=... L=...` | 模拟精度 | SPICE |

    RTL 仿真**零延迟**，只看逻辑；门级后仿加**真实延迟**，看时序；SPICE 看**电压电流波形**，看模拟性能。

    ### 🚀 回到你跑 RISC-V 核的场景

    你之前问的"跑一个 RISC-V 核"，那个核就是用 RTL 写的。比如：

    - **Rocket**：Chisel 写的，但生成的是 FIRRTL/RTL
    - **BOOM**：同样是 Chisel
    - **香山**：Chisel
    - **PicoRV32**：纯 Verilog RTL
    - **Ibex**：SystemVerilog RTL

    当你用 VCS 编译这些代码时，VCS 读的就是 RTL。仿真时，每个时钟上升沿，RTL 里的 `always` 块被唤醒，模拟 CPU 取指、译码、执行。这就是 RTL 仿真的本质——**用软件模拟 RTL 描述的逻辑行为**。

    ### 📌 一句话总结

    RTL 是**寄存器传输级**，是数字芯片设计中最核心的抽象层。它用 Verilog/SystemVerilog 描述"数据在寄存器之间怎么流动、每个时钟周期做什么运算"。它是人写的最后一层，往下就交给综合工具变成门级网表。你跑 RISC-V 核时，VCS 仿真的就是 RTL。

* 我想入门 RTL，应该从哪入门

    1. 选语言和教材

        * 《Verilog HDL 数字设计与综合》：经典入门书，涵盖基础和组合/时序逻辑设计，很多高校用它做教材。

        * 《Verilog 数字设计原理》：2025 年出的新书，内容比较新，从 Verilog 基础讲到高级话题和逻辑综合，示例比较全。

        * 《CMOS 数字集成电路设计与实践》：把 RTL 设计和底层的 CMOS 原理结合起来讲，适合你想了解“门级以下”是怎么回事的场景。

        **SystemVerilog** 是 Verilog 的超集，验证和设计都能用。如果你想一步到位，也可以直接看 **《SystemVerilog入门指南》**，但它内容偏全，新手可能会觉得有点厚。

    2. 理解 RTL 的核心概念

        * 模块（Module）：Verilog 的基本单元，像一个“积木块”，有输入输出端口。

        * wire 和 reg：`wire` 是导线，用来连接；`reg` 是存储单元，用在 `always` 块里。**但注意**，`reg` 类型不一定会综合成物理寄存器，它只是 Verilog 里的一种数据类型。

        * 组合逻辑 vs 时序逻辑：没有时钟、输出只跟当前输入有关的，是组合逻辑（用 `assign` 或 `always @(*)`）。跟时钟沿绑定的，是时序逻辑（用 `always @(posedge clk)`）。

        * 阻塞（=）和非阻塞（<=）赋值：这是最容易出错的地方。简单记：组合逻辑用 `=`，时序逻辑用 `<=`。

    3.  第三步：动手写，用工具跑

        光看书没用，必须自己写代码、跑仿真。

        1.  **从“小”开始**：先写个 **2 选 1 多路选择器**、**D 触发器**或者 **4 位计数器**。这些代码十几行，能让你立刻理解“描述”是什么意思。

        2.  **用开源工具跑通**：你之前问过 VCS，但 VCS 收费。入门阶段，**Icarus Verilog（iverilog）** 是更好的选择，免费、安装简单，命令行编译运行，能快速看到结果。

        3.  **看波形**：用 **GTKWave**（配合 iverilog）打开仿真生成的 `.vcd` 文件，亲眼看看你写的计数器在每个时钟沿是怎么变化的。这一步非常关键，能把你脑子里的“代码”和实际的“电路行为”对应起来。

    4. 第四步：循序渐进的路线

        你可以按这个顺序慢慢来：

        **语法基础 → 简单组合逻辑（MUX、译码器）→ 简单时序逻辑（计数器、移位寄存器）→ 状态机（FSM）→ 用开源核练手。**

    等你把这些基础吃透了，再去 GitHub 上找一些用 Verilog 写的 RISC-V 核（比如 **PicoRV32**），读一读别人的代码是怎么组织的，那时候你就能真正看懂“核”是怎么回事了。

* 分享一些 RTL 入门项目的代码

    > 说明：全部 Verilog‑2001，可综合 DUT + Testbench，**VCS可以直接编译仿真**，也可以放到 Vivado 综合上板。
    > VCS 编译示例：`vcs -sverilog dut.v tb.v -o simv`，`./simv -gui`，Verdi 打开波形。
    > ⚠️ DUT（设计）**不能出现 initial、#延时**；Testbench（测试）只能用来仿真，不可综合。

    ---

    ## 项目1：带溢出4位ALU（组合逻辑，入门第一个）
    ### dut_alu.v
    ```verilog
    module dut_alu(
        input      [3:0] a,
        input      [3:0] b,
        input      [1:0] op,  //00 add,01 sub,10 and,11 or
        output reg [3:0] res,
        output reg       overflow
    );

    always @(*) begin
        res = 4'd0;
        overflow = 1'b0;
        case(op)
            2'b00: begin {overflow, res} = a + b; end
            2'b01: begin {overflow, res} = a - b; end
            2'b10: begin res = a & b; end
            2'b11: begin res = a | b; end
        endcase
    end
    endmodule
    ```
    ### tb_alu.v
    ```verilog
    module tb_alu;
    reg [3:0] a,b;
    reg [1:0] op;
    wire [3:0] res;
    wire overflow;

    dut_alu u_dut(.*);

    initial begin
        $fsdbDumpfile("alu.fsdb");
        $fsdbDumpvars(0,tb_alu);
        a=0; b=0; op=0;
        #20;
        a=4'd12; b=4'd5; op=2'b00; #20;
        a=4'd3;  b=4'd8; op=2'b01; #20;
        a=4'b1100;b=4'b1010;op=2'b10;#20;
        a=4'b1100;b=4'b1010;op=2'b11;#20;
        #40;
        $display("ALU test done");
        $finish;
    end
    endmodule
    ```

    ---

    ## 项目2：带使能、同步复位8bit计数器（时序逻辑）
    ### dut_cnt8.v
    ```verilog
    module dut_cnt8(
        input               clk,
        input               rst_n, //同步低复位
        input               en,    //计数使能
        output reg [7:0]    cnt,
        output              ov
    );
    assign ov = (cnt == 8'hff);

    always @(posedge clk) begin
        if(!rst_n) begin
            cnt <= 8'd0;
        end
        else if(en) begin
            cnt <= cnt + 1'b1;
        end
    end
    endmodule
    ```
    ### tb_cnt8.v
    ```verilog
    module tb_cnt8;
    reg clk;
    reg rst_n;
    reg en;
    wire [7:0] cnt;
    wire ov;

    dut_cnt8 u_dut(.*);

    initial begin
        $fsdbDumpfile("cnt8.fsdb");
        $fsdbDumpvars(0,tb_cnt8);
        clk =0;
        forever #5 clk=~clk;
    end

    initial begin
        rst_n = 1'b0;
        en = 1'b0;
        #12;
        rst_n =1'b1;
        #20;
        en =1'b1;
        #600;
        en=1'b0;
        #100;
        $finish;
    end
    endmodule
    ```

    ---

    ## 项目3：序列检测器 检测 1011，两段式FSM（FSM入门，面试高频）
    > 输入串行bit流，检测连续序列`1‑0‑1‑1`，命中拉高flag一个周期。
    ### dut_seq_det.v
    ```verilog
    module dut_seq_det(
        input        clk,
        input        rst_n,
        input        din,
        output reg   flag
    );

    localparam S0=3'b001;
    localparam S1=3'b010;
    localparam S2=3'b100;
    localparam S3=3'b011;
    localparam S4=3'b101;

    reg [2:0] curr_state;
    reg [2:0] next_state;

    //第一段：时序，状态寄存器
    always @(posedge clk) begin
        if(!rst_n)
            curr_state <= S0;
        else
            curr_state <= next_state;
    end

    //第二段：组合，状态跳转
    always @(*) begin
        next_state = curr_state;
        case(curr_state)
            S0: next_state = din ? S1 : S0;
            S1: next_state = din ? S1 : S2;
            S2: next_state = din ? S3 : S0;
            S3: next_state = din ? S4 : S2;
            S4: next_state = din ? S1 : S2;
        endcase
    end

    //输出逻辑（Moore）
    always @(*) begin
        flag = (curr_state == S4);
    end
    endmodule
    ```
    ### tb_seq_det.v
    ```verilog
    module tb_seq_det;
    reg clk;
    reg rst_n;
    reg din;
    wire flag;

    dut_seq_det u_dut(.*);

    initial begin
        $fsdbDumpfile("seq.fsdb");
        $fsdbDumpvars(0,tb_seq_det);
        clk=0;
        forever #5 clk=~clk;
    end

    initial begin
        rst_n=0; din=0;
        #12 rst_n=1;
        //输入序列 1 0 1 1
        din=1;#10;
        din=0;#10;
        din=1;#10;
        din=1;#10;
        #100;
        $finish;
    end
    endmodule
    ```

    ---

    ## 项目4：同步FIFO，深度8，位宽8bit（中等难度，模块+指针逻辑）
    > 读写同一个时钟，wr_en写，rd_en读，empty/full标志。
    ### dut_sync_fifo.v
    ```verilog
    module dut_sync_fifo(
        input               clk,
        input               rst_n,
        input               wr_en,
        input      [7:0]    wdata,
        input               rd_en,
        output reg [7:0]    rdata,
        output              full,
        output              empty
    );
    parameter DEPTH = 8;
    parameter AW = 3;

    reg [7:0] mem [0:DEPTH-1];
    reg [AW-1:0] wr_ptr;
    reg [AW-1:0] rd_ptr;
    reg [AW:0] cnt;

    assign full  = (cnt == DEPTH);
    assign empty = (cnt == 0);

    always @(posedge clk) begin
        if(!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            cnt <=0;
        end
        else begin
            //写
            if(wr_en && !full) begin
                mem[wr_ptr] <= wdata;
                wr_ptr <= wr_ptr + 1'b1;
            end
            //读
            if(rd_en && !empty) begin
                rdata <= mem[rd_ptr];
                rd_ptr <= rd_ptr +1'b1;
            end
            //计数
            case({wr_en&&!full, rd_en&&!empty})
                2'b10: cnt <= cnt +1'b1;
                2'b01: cnt <= cnt -1'b1;
                default: cnt <= cnt;
            endcase
        end
    end
    endmodule
    ```
    ### tb_sync_fifo.v
    ```verilog
    module tb_sync_fifo;
    reg clk;
    reg rst_n;
    reg wr_en;
    reg [7:0] wdata;
    reg rd_en;
    wire [7:0] rdata;
    wire full,empty;

    dut_sync_fifo #(.DEPTH(8)) u_dut(.*);

    initial begin
        $fsdbDumpfile("fifo.fsdb");
        $fsdbDumpvars(0,tb_sync_fifo);
        clk=0;
        forever #5 clk=~clk;
    end

    initial begin
        rst_n=0;
        wr_en=0; wdata=0; rd_en=0;
        #12 rst_n=1;
        //写3笔
        wr_en=1; wdata=8'h11;#10;
        wdata=8'h22;#10;
        wdata=8'h33;#10;
        wr_en=0;#20;
        //读出
        rd_en=1;#30;
        rd_en=0;#20;
        $finish;
    end
    endmodule
    ```

    ---

    ## 项目5：UART TX（波特率9600，50M系统时钟）
    > 50M，9600波特，分频计数：50_000_000 /9600 ≈ 5208
    ### dut_uart_tx.v
    ```verilog
    module dut_uart_tx(
        input               clk,
        input               rst_n,
        input               tx_valid,
        input      [7:0]    tx_data,
        output reg          tx,
        output              tx_busy
    );
    parameter BAUD_DIV = 16'd5208;

    reg [15:0] cnt;
    reg [3:0]  bit_cnt;
    reg [9:0]  shift_reg;

    assign tx_busy = (bit_cnt != 4'd0);

    always @(posedge clk) begin
        if(!rst_n) begin
            cnt <=0;
            bit_cnt <=0;
            tx <=1'b1;
            shift_reg <=10'b0;
        end
        else begin
            if(!tx_busy && tx_valid) begin
                shift_reg <= {1'b1,tx_data,1'b0}; //停止位，8数据，起始位
                bit_cnt <=4'd10;
                cnt <= 16'd0;
            end
            else if(tx_busy) begin
                if(cnt == BAUD_DIV -1) begin
                    cnt <= 0;
                    tx <= shift_reg[0];
                    shift_reg <= shift_reg >>1;
                    bit_cnt <= bit_cnt -1'b1;
                end else begin
                    cnt <= cnt +1'b1;
                end
            end
        end
    end
    endmodule
    ```

    ---

    # VCS Makefile模板（新建Makefile）
    ```makefile
    # Makefile
    DUT = dut_alu.v
    TB  = tb_alu.v
    TOP = tb_alu

    sim:
        vcs -sverilog -full64 $(DUT) $(TB) -o simv -debug_access+all
        ./simv

    gui:
        ./simv -gui &

    clean:
        rm -rf simv *.fsdb csrc ucli.key DVEfiles *.vpd
    ```
    运行：`make sim` 启动仿真，`make gui` 唤起Verdi/DVE。

    ---

    如果你接下来需要：
    1. UART_RX接收，完成UART回环整套工程
    2. APB3‑GPIO 从机代码（IC求职简历项目）
    3. 异步FIFO（格雷码CDC）
    4. RV32I单周期CPU
    可以告诉我，我可以分批发整套代码+tb。

    > 练习建议：不要直接复制，**手敲一遍，故意写几个bug，然后用Verdi波形debug，这个才是RTL入门最重要的训练**。

* RTL入门，在fpga或vcs上有哪些入门项目

    # RTL入门项目（FPGA板 + VCS仿真两条路线）

    > 前置基础：Verilog基础语法，理解module、always、assign、时序/组合逻辑，会写testbench。
    > 两条路径：**VCS纯仿真（先练逻辑，不需要开发板）**；**FPGA板（综合、布局布线、上板调试，看硬件时序）**。
    > 学习顺序：组合逻辑 → 时序逻辑 → 有限状态机FSM → 接口协议 → 小型SOC子模块。

    ## 第一阶段：基础（只做VCS仿真，可选FPGA上板）

    目标：掌握时序、复位、计数器、FSM，学会写testbench，看懂VCS编译、仿真波形。

    1. **4位加法器、带进位ALU**

        - 功能：加减、与或非，溢出标志。
        - VCS：写 tb 激励，随机数测试，对比预期结果。
        - FPGA拓展：按键输入，数码管显示结果。
        - 知识点：组合逻辑，避免latch，位宽截断，溢出。

    2. **通用计数器（可预置、使能、溢出标志）**

        - 8bit/16bit计数器，同步复位、异步复位二选一实现。
        - VCS tb：测试清零、使能开关、重载初值、溢出。
        - FPGA拓展：分频，LED秒闪烁。
        - 知识点：时序逻辑，复位策略，建立保持时间基础概念。

    3. **分频器（偶数分频、奇数分频、占比50%）**

        - 50MHz → 1Hz，奇数分频不能简单计数，需要双边沿。
        - VCS仿真看输出时钟波形。
        - FPGA：板载晶振输入，分频输出驱动LED。
        - 知识点：时钟生成，不要在内部逻辑做门控时钟（入门坑）。

    4. **移位寄存器，串入并出SIPO、并入串出PISO**

        - 8bit移位，用于串行数据收发基础。
        - VCS tb发送串行bit流，观察并行输出。
        - FPGA拓展：按键输入移位，LED并行输出。
        - 知识点：时序采样，bit流同步。

    5. **简单FSM（1‑2‑3节拍脉冲发生器，序列检测器）**
        - 序列检测器：检测串行输入是否出现`1011`。两种写法：一段式、两段式FSM。
        - VCS tb灌入数据流，检查命中标志。
        - FPGA：按键作为输入，LED标记检测成功。
        - 知识点：状态机编码（二进制、格雷码、独热码），两段式推荐工业风格。

    > ✅ 阶段练习要求：**全部用VCS跑仿真，DUT + TB，生成vpd/fsdb波形，DVE打开看波形debug，不要一开始就上FPGA。**
    > VCS基础命令（入门）
    ```bash
    vcs -sverilog dut.v tb.v -o simv
    ./simv -gui
    ```

    ## 第二阶段：中等项目（VCS仿真优先，之后FPGA上板验证）
    > 目标：理解模块划分，顶层例化，跨模块信号，握手逻辑，简单总线。
    1. **同步FIFO（深度8，宽度8bit）**
        - wr_en, rd_en，空empty、满full标志，计数指针，**同步时钟读写**。
        - VCS tb：写满，读空，边写边读，边界case压力测试。
        - FPGA拓展：按键写数据，数码管读出FIFO内容。
        - 知识点：FIFO指针，空满判断；**先做同步FIFO，不要一上来异步FIFO**。

    2. **UART串口（8N1，波特率9600）**
        - 分成两个模块：UART_TX发送、UART_RX接收。
        - TX：并转串，起始位+8数据+停止位。
        - RX：采样，串转并，错误标志（帧错）。
        - VCS：tb自己例化TX→RX回环测试。
        - FPGA：FPGA‑USB转串口，电脑串口助手收发字符。
        - 知识点：波特率分频，过采样，跨时钟域基础（本项目收发同时钟则不需要）。

    3. **SPI主机（简单模式CPOL=0 CPHA=0）**
        - SPI主机，输出SCLK、CS、MOSI，输入MISO。
        - VCS：tb模拟SPI从机，回环测试读写。
        - FPGA拓展：驱动SPI OLED屏幕，打印数字字符。
        - 知识点：串行同步协议，片选，时序图对照规格书。

    4. **简单RAM（寄存器堆或者block ram）**
        - 同步读写RAM，宽度8深度32。
        - VCS读写测试，写后读。
        - FPGA：推断FPGA片上BRAM，不要用寄存器堆做大RAM（资源爆炸）。
        - 知识点：RAM建模，FPGA综合推断BRAM，不要写组合逻辑RAM。

    5. **定时器+中断发生器**
        - 可编程计数初值，超时后产生脉冲中断标志。
        - VCS测试不同重载值，中断产生时序。
        - FPGA：定时器1秒中断，翻转LED。
        - 知识点：中断标志寄存，标志清零逻辑。

    ## 第三阶段：进阶项目（工业RTL风格，适合简历项目，VCS必做，FPGA可选）
    > 适合准备数字IC/FPGA求职，严格遵守模块划分，可综合编码，完整testbench，覆盖率收集。
    1. **APB3 Slave 外设（GPIO或者定时器）**
        - APB从机，寄存器映射，读写寄存器，输出GPIO。
        - VCS TB：写APB驱动，发起APB读写，验证寄存器。
        - 知识点：AMBA APB协议，寄存器建模，地址译码。
        > 这是从FPGA过渡到数字IC非常重要的项目。

    2. **异步FIFO（不同读写时钟域）**
        - 写域wclk，读域rclk，格雷码指针跨时钟域，两级打拍同步。
        - VCS：两个独立时钟，随机读写压力测试。
        - 知识点：CDC跨时钟域，格雷码，亚稳态概念，是数字IC面试高频考点。

    3. **简易RISC‑V 单周期CPU（RV32I最小子集）**
        - 取指、译码、执行、寄存器堆、ALU，支持add,sub,lw,sw,beq。
        - VCS：加载指令mem，仿真跑汇编小程序。
        - FPGA：外接BRAM作为指令存储器，七段数码管看结果。
        - 知识点：流水线前序，CPU架构，顶层模块集成。

    4. **AXI‑Stream数据流水线，简单DMA**
        - AXI‑Stream主机发送，FIFO做缓冲，从机接收。
        - VCS做数据流压测，反压测试。
        - 知识点：valid/ready握手，数据流背压，现代总线思想。

    # 路线选择建议
    ## 路线A：数字IC方向（想做RTL验证、前端设计）
    > **优先VCS，少花时间FPGA。**
    顺序：基础组合/时序 → FSM → 同步FIFO → APB外设 → 异步FIFO → RV单周期CPU。
    产出：每个项目 DUT + TB + Makefile编译脚本，VCS跑仿真，收集code coverage，波形debug。FPGA**可选，不是必须**。

    ## 路线B：FPGA开发方向
    > VCS仿真先验证逻辑，**再综合上板**。
    顺序：基础逻辑 → 分频、UART、SPI → FIFO → OLED/LCD显示，简单图像处理。

    # RTL入门避坑
    1. 不要把仿真和综合混为一谈：`initial`只能tb，**DUT可综合模块禁止initial**。
    2. 不要用门控时钟，入门用时钟使能。
    3. testbench和设计代码严格分开，DUT只写可综合代码。
    4. VCS一定要看波形，**不要只看打印信息**，bug大多藏在波形时序。

    # 工具环境搭建
    - VCS：需要synopsys虚拟机/服务器环境，写Makefile封装vcs编译命令，一键跑仿真，生成fsdb，Verdi/DVE看波形。
    - FPGA：Xilinx Artix‑7（XC7A35T，正点原子），工具Vivado；Intel：Cyclone IV，Quartus。

    如果你需要，我可以：
    1. 给第一个项目：4位ALU DUT + VCS Testbench完整代码+Makefile。
    2. 给一套循序渐进练习列表，每个项目写清楚需求、验收标准。
    3. VCS入门脚本，Makefile模板，从零搭建仿真环境。



