# verilog note

## cache

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

* block

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

* io 口

	输入：`input [信号位宽 - 1:0] 端口名;`

	输出：`output [信号位宽 - 1:0] 端口名;`

	输入输出：`inout [信号位宽-1:0] 端口名;`

	I/O 说明也可以写在端口声明语句里：

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

	* `always @(posedge clk or posedge clr);`

		```verilog
		begin
			if (clr) q <= 0;
			else if (en) q <= d;
		end
		```

		always 块内的语句按顺序执行，always 块又称为过程块。多个 always 块是同时执行的。

* 只有连续赋值语句 assign 和实例引用语句可以独立于过程块而存在于模块的功能定义部分。

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

* reg 型数据常用来表示 always 模块内的指定信号，常代表触发器。在 always 模块内赋值的每一个信号都必须定义成 reg 型。

* memory 型：`reg[n-1:0] mema[255:0];`
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

