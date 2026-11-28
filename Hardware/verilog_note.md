# verilog note

## cache

* 目前看到 123 页第 17 题，因为忘了触发器的定义，所以暂时不看了。

* synth

    ```verilog
    always @(posedge clock)
        if (A)
            C = B;
    ```

    这段 Verilog 代码虽然只有短短三行，但它涉及综合中的几个重要概念：**时序逻辑推断、阻塞赋值的风险、以及锁存器/触发器推断**。下面详细展开。

    ---

    ## 1. 代码本身分析

    ```verilog
    always @(posedge clock)
        if (A)
            C = B;
    ```

    这是一个**边沿触发的 always 块**，敏感列表只有 `posedge clock`。块内有一个条件语句，条件是 `A`，条件成立时执行 `C = B`（**阻塞赋值**）。条件不成立时，`C` 没有任何赋值。

    ---

    ## 2. 综合工具会推断出什么？

    ### 2.1 关键问题：`if (A)` 没有 else 分支

    在边沿触发的 always 块中，当 `A` 为假时，`C` 没有被赋值。对于时序逻辑，综合工具必须**保持 C 的旧值**——这等价于一个带使能端的寄存器：

    ```verilog
    always @(posedge clock)
        if (A)
            C <= B;   // 综合结果等价于：
    // C 在 A=0 时保持不变
    ```

    综合结果是一个 **D 触发器，带时钟使能（clock enable）**：

    - `clock` → 触发器的时钟端
    - `A` → 使能端（enable）
    - `B` → 数据输入端（D）
    - `C` → 输出端（Q）

    **重要区别**：这里**不会**推断出锁存器（latch）。锁存器只会在**电平敏感**的 always 块（如 `always @(*)` 或 `always @(A or B)`）中因条件不完整而产生。边沿触发块中条件不完整，推断的是**带使能的触发器**，这是完全合法且常见的综合结果。

    ### 2.2 映射到 FPGA / ASIC

    - **FPGA**：会映射到 slice 中的 FDRE（带 CE 的 D 触发器），`A` 接到 CE 引脚，资源利用很自然。
    - **ASIC**：标准单元库中有带使能的 DFF（如 `DFFE`），或者综合工具会用 mux + 普通 DFF 实现（D 端前加一个 `A ? B : Q` 的反馈 mux）。

    ---

    ## 3. 阻塞赋值 `=` 在时序块中的问题

    这是这段代码**最需要警惕**的地方。在 `always @(posedge clock)` 中使用阻塞赋值 `=` 而非非阻塞赋值 `<=`，会带来以下风险：

    ### 3.1 仿真与综合的语义差异

    - **综合**：工具通常仍能正确推断出触发器（因为是在时钟边沿块中），但这是"碰巧"能工作，依赖于工具对阻塞赋值的宽容处理。
    - **仿真**：阻塞赋值会立即生效，若同一块内后续语句读取 `C`，会读到新值；而非阻塞赋值读到的是旧值。这会导致**仿真行为与综合后网表行为不一致**（仿真-综合不匹配，simulation-synthesis mismatch）。

    ### 3.2 多语句时的竞争风险

    ```verilog
    always @(posedge clock) begin
        if (A) C = B;
        D = C;          // 阻塞：D 拿到的是本拍更新后的 C
    end
    ```

    而非阻塞版本：

    ```verilog
    always @(posedge clock) begin
        if (A) C <= B;
        D <= C;         // D 拿到的是上一拍的 C
    end
    ```

    两者综合出的硬件**不同**。阻塞赋值在时序块中容易让设计者误判流水线级数。

    ### 3.3 规范建议

    **时序逻辑一律使用非阻塞赋值 `<=`**。这是 Verilog 编码规范（如 Clifford Cummings 的经典论文）的硬性建议。应改写为：

    ```verilog
    always @(posedge clock)
        if (A)
            C <= B;
    ```

    ---

    ## 4. 位宽与类型问题

    - `C` 和 `B` 的位宽必须匹配，否则综合工具会做隐式截断/扩展并给出 warning。
    - 若 `C` 声明为 `reg`（Verilog-2001 中 always 块内被赋值的变量必须是 `reg` 或 `logic`），综合工具根据上下文推断为触发器；若声明为 `wire` 则编译报错。
    - 若 `C` 是多位向量，综合出的是一组并联的带使能触发器。

    ---

    ## 5. 复位问题

    这段代码**没有复位**。综合结果是一个**无复位触发器**：

    - FPGA 上电后触发器初值不确定（除非使用 `initial` 或寄存器初值属性，FPGA 支持但 ASIC 通常不支持）。
    - ASIC 中无复位触发器上电状态未知，可能带来 X 传播问题。
    - 工程上通常建议加入同步或异步复位：

    ```verilog
    always @(posedge clock or posedge reset)
        if (reset) C <= 1'b0;
        else if (A) C <= B;
    ```

    ---

    ## 6. 综合结果总结

    | 方面 | 结论 |
    |------|------|
    | 推断硬件 | 带时钟使能的 D 触发器（DFFE），**不是锁存器** |
    | 使能信号 | `A` |
    | 数据输入 | `B` |
    | 时钟 | `clock` 上升沿 |
    | 复位 | 无 |
    | 赋值方式风险 | 阻塞赋值在时序块中不规范，仿真/综合可能不一致 |
    | 建议改写 | `if (A) C <= B;`，并考虑加复位 |

    ---

    ## 7. 一句话总结

    这段代码综合出一个**以 A 为使能、B 为数据输入、clock 上升沿触发的 D 触发器**；`if` 缺少 `else` 在边沿块中不会产生锁存器而是产生使能逻辑；真正需要修正的是**把阻塞赋值改为非阻塞赋值**，并视需求补充复位。

    注：

    1. 会综合出多路选择器吗？

* verilog 中，wire [7:0] A; wire [3:0] B; 我想让 B 赋值给 A 的高 4 位，怎么办？

    在 Verilog 中，有几种方式可以把 `B` 赋值给 `A` 的高 4 位：

    ## 方法 1：位选 + 拼接（最常用）

    ```verilog
    assign A[7:4] = B;
    ```

    低 4 位可以根据需要赋其他值，比如：

    ```verilog
    assign A = {B, 4'b0000};   // 高4位=B，低4位=0
    assign A = {B, A[3:0]};    // 高4位=B，低4位保持原值（注意不能这样写，会组合环）
    ```

    ⚠️ 注意：`assign A = {B, A[3:0]};` 这种写法会形成组合逻辑环，**不行**。如果低 4 位要保留，应该拆成不同驱动或者用寄存器：

    ```verilog
    // 如果 A 是 wire，低 4 位必须由其他地方驱动
    wire [3:0] low_part;
    assign A = {B, low_part};
    ```

    ## 方法 2：直接在 always 块中赋值（A 需要是 reg）

    ```verilog
    reg [7:0] A;
    wire [3:0] B;

    always @(*) begin
        A[7:4] = B;
        // A[3:0] 保持或赋其他值
    end
    ```

    ## 方法 3：拼接赋值

    ```verilog
    assign A = {B, A_low};  // A_low 是低 4 位来源
    ```

    ## 关键点

    - `wire` 类型**不能**在 `always` 块里赋值，只能用 `assign` 连续赋值。
    - 如果 A 是 `wire` 且要部分赋值（只赋高 4 位），低 4 位必须有其他驱动源，否则是悬空（z）。
    - 如果低 4 位想保持"上一拍的值"，A 必须是 `reg` 并在 `always` 块里只更新 `A[7:4]`：

    ```verilog
    reg [7:0] A;
    always @(posedge clk) begin
        A[7:4] <= B;   // 低4位保持不变
    end
    ```

    **最推荐**：如果只是单纯把 B 放到 A 高 4 位，用 `assign A[7:4] = B;` 最简洁清晰。

* 详细讲讲 wire, reg, integer, real, parameter 的区别

    在 Verilog 中，这五种数据类型虽然写法相似，但用途和底层含义完全不同。**`wire` 和 `reg` 是设计硬件时最核心的两种信号类型；`integer` 和 `real` 主要用于仿真或辅助计算；`parameter` 则是用于提高代码可读性与可重用性的常量。**

    ### wire：代表物理连线
    `wire` 表示硬件电路中的物理连接（导线），本身不能存储数值，必须由外部驱动源持续驱动。
    *   **赋值方式**：只能在 `assign` 连续赋值语句或模块端口连接中被驱动。
    *   **默认值**：如果没有驱动源，其值为高阻态 `z`。
    *   **硬件对应**：综合后就是一根导线，常用于组合逻辑的中间信号或模块间的连接。

    ### reg：代表存储单元（抽象）
    `reg` 是寄存器类型的抽象，它在被赋新值之前会保持原来的值。**注意：`reg` 并不一定对应硬件触发器**，它的最终硬件形态取决于赋值方式。
    *   **赋值方式**：只能在 `always` 或 `initial` 过程块中被赋值。
    *   **硬件对应**：
        *   如果在带时钟边沿的 `always` 块中赋值，综合为**触发器**。
        *   如果在组合逻辑的 `always` 块（敏感列表为电平）中赋值，综合为**硬件连线**（与 `wire` 类似）。
    *   **默认值**：未赋值时通常为不定态 `x`。

    ### integer：32位有符号整数
    `integer` 本质上是一个 32 位的**有符号**寄存器类型（等价于 `reg signed [31:0]`）。
    *   **主要用途**：通常用于**仿真测试**（如循环变量、文件操作）或辅助计算，**不建议**用作需要综合成硬件的信号端口或寄存器输出。
    *   **注意**：如果将其用于位宽为 32 位的硬件信号，需警惕有符号数到无符号数的隐式转换风险。

    ### real：双精度浮点数
    `real` 用于存储**双精度浮点数**（通常为 64 位）。
    *   **主要用途**：仅用于**仿真**中的实数运算（如模拟物理量）。
    *   **限制**：**不可综合**。它不能用于模块端口，也不能按位访问，不能直接通过原语端口。

    ### parameter：运行时常量
    `parameter` 用于定义**符号常量**，其主要目的是提高代码的可读性和可维护性（如定义位宽、状态机状态、延迟）。
    *   **核心特性**：它是**可重写**的。在模块例化时，可以通过参数传递（`#(.PARAM_NAME(value))`）来改变模块内部的行为，这使得代码具有极强的**可重用性**（例如构建参数化的加法器）。
    *   **补充**：如果某个常量**绝对不允许**被外部修改，应使用 `localparam` 来替代 `parameter`。

    ### 总结对比
    | 数据类型 | 核心角色 | 赋值场景 | 硬件含义 |
    | :--- | :--- | :--- | :--- |
    | **wire** | 连线 | `assign` / 端口 | 物理导线 |
    | **reg** | 存储（抽象） | `always` / `initial` | 取决于块：触发器或组合线 |
    | **integer** | 仿真整数 | 过程块 | 通常不直接对应硬件 |
    | **real** | 仿真浮点 | 过程块 | 无（不可综合） |
    | **parameter** | 常量 | 声明时 / 例化时 | 固化在电路中的常数 |

* Verilog 中通常不支持给参数加类型

    你只能写成 parameter a = 3;

    input 和 inout 的类型也不能写成 integer, 但是 integer 可以用于 output 端口。

    如果 input 必须要写符号，那么可以写成`input signed [31:0] a`或`input wire signed [31:0] a`

* @(A) count = count + 1; 详细讲讲这种写法。这个是只作用于一行吗？可以写 begin end  吗？

    # 详解 `@(A) count = count + 1;` 这种写法

    ## 一、语法本质：`@(事件控制)` 是一个"语句"

    `@(A)` 本身**不是**一个独立语句，而是一个**事件控制前缀**，它必须修饰**后面的一条语句**。Verilog 的语法是：

    ```
    事件控制 ::= @ ( 事件表达式 ) 语句
    ```

    所以 `@(A) count = count + 1;` 的完整含义是：

    > **等待 A 发生跳变，然后执行 `count = count + 1;` 这一条语句。**

    关键点：**`@(...)` 作用的是紧随其后的"一条语句"**，而不是"一行"。语句和行没有必然关系（Verilog 不是按行解析的，分号 `;` 才是语句结束）。

    ## 二、只作用于一行/一条语句吗？

    **是的，只作用于它后面的那一条语句。**

    ```verilog
    @(A) count = count + 1;      // 只等 A，然后执行这一条赋值
    @(B) count = count + 1;      // 只等 B，然后执行这一条赋值
    ```

    上面第二行的 `@(B)` 是**另一个独立的事件控制**，和第一行无关。两个 `@` 是两次独立的"等待"。

    如果后面跟的是多条语句而不用 `begin/end` 包起来，就**只有第一条**受 `@` 控制，后面的会立刻执行：

    ```verilog
    @(A) x = 1; y = 2;   // 只有 x=1 等 A；y=2 不受 @(A) 控制，紧跟 x=1 之后立刻执行
    ```

    这往往不是想要的结果，所以要作用多条语句时必须用 `begin/end`。

    ## 三、可以写 `begin ... end` 吗？

    **完全可以，而且这是标准写法。** 语法上 `@(...)` 后面可以跟任意一条语句，而 `begin...end` 本身就是一个"复合语句"，所以合法：

    ```verilog
    @(A) begin
        count = count + 1;
        flag  = 1;
    end
    ```

    含义：**等待 A 跳变，然后依次执行 begin/end 块里的所有语句。**

    这一整块 `@(A) begin ... end` 在语法上等价于"一条语句"，作为一个整体被事件控制。

    ### 对比：单条 vs begin/end

    ```verilog
    // 写法1：只等 A 执行一条
    @(A) count = count + 1;

    // 写法2：等 A 执行多条（等价于把多条打包成一条复合语句）
    @(A) begin
        count = count + 1;
        other = other + 1;
    end
    ```

    两者都是合法的，区别只在于"等待之后要执行几条语句"。

    ## 四、和 `always @(...)` 的区别（容易混淆）

    很多人会把 `@(A) stmt;` 和 `always @(A) stmt;` 搞混，二者完全不同：

    | 写法 | 含义 |
    |------|------|
    | `@(A) stmt;` | **一次性**等待：执行到这里暂停，A 跳变一次、执行 stmt 一次，然后继续往下走 |
    | `always @(A) stmt;` | **永久**敏感：每当 A 跳变就执行 stmt，反复进行，永不退出 |

    在你原代码里：

    ```verilog
    always begin
        @(A) count = count + 1;   // 等一次 A
        @(B) count = count + 1;   // 再等一次 B
    end                            // 然后回到 begin 重新等 A……
    ```

    外层 `always` 提供了"无限循环"，内层的 `@(A)`/`@(B)` 负责"每次循环里等待一次事件"。所以整体效果是**交替等 A、等 B，各加一次**。如果去掉外层 `always`，那就只会等一次 A、等一次 B，然后就结束了。

    ## 五、几个语法要点与坑

    1. **`@` 后面必须是"一条语句"**，可以是：
       - 单条语句：`@(A) x = 1;`
       - 复合语句：`@(A) begin ... end`
       - 也可以是 `if/for/case` 等（它们本身也是语句）：`@(A) if (en) x = 1;`

    2. **`begin/end` 里可以再嵌套事件控制**，这正是你代码的结构：
       ```verilog
       always begin
           @(A) begin
               count = count + 1;
               // 还可以在这里再等别的事件
           end
           @(B) count = count + 1;
       end
       ```

    3. **不要漏分号**：`@(A) begin ... end` 后面**不需要**分号（`end` 自身就是语句终结）；而 `@(A) x = 1;` 需要分号。

    4. **`@(A)` 是双边沿**（任意跳变）。只想上升沿写 `@(posedge A)`。

    5. **可综合性**：`@(A) ...` 这种"中途等待"的写法一般只用于仿真/testbench，综合工具通常不支持。

    ## 六、小结

    - `@(A) count = count + 1;` = "等 A 跳变一次，然后执行这一条赋值"，**`@` 只作用于紧随其后的那一条语句**。
    - 要让它作用于多条语句，用 `@(A) begin ... end` 包起来，完全合法。
    - 它和 `always @(A)` 不同：前者是**一次性等待**，后者是**永久敏感**；你代码里靠外层 `always` 把它变成循环交替等待。
    - "一行"和"一条语句"不是一回事，判定边界看分号和 `begin/end`，不看换行。

* state

    ```verilog
    reg [7:0] count;
    initial
        count = 0;
    always begin
        @(A) count = count + 1;
        @(B) count = count + 1;
    end
    ```

    # 这段 Verilog 代码详解

    ## 一、逐行解析

    **`reg [7:0] count;`**
    - 声明一个 8 位寄存器变量 `count`，取值范围 0~255，溢出后回绕到 0。

    **`initial count = 0;`**
    - 在仿真 0 时刻把 `count` 初始化为 0。注意这是**仿真专用**的初始化方式，不可综合为硬件。

    **`always begin ... end`**
    - 一个无限循环的过程块，内部没有敏感列表（用的是 `@` 事件控制），所以它会在时间 0 就开始执行，并一直循环下去。

    **`@(A) count = count + 1;`**
    - **阻塞等待**：执行到这里时，进程挂起，直到信号 `A` 发生**任意变化**（上升沿或下降沿，即 `posedge` 或 `negedge` 都算）才继续。然后执行 `count = count + 1`。

    **`@(B) count = count + 1;`**
    - 同理，等待 `B` 发生任意变化，然后 `count` 加 1。

    ## 二、整体行为（核心逻辑）

    这是一个**事件驱动的计数器**，但计数节奏由 A、B 两个信号"交替"触发：

    ```
    时间0: count = 0
          └─ 等待 A 变化
    A 变化: count = 1
          └─ 等待 B 变化
    B 变化: count = 2
          └─ 循环回到 begin，等待 A 变化
    A 变化: count = 3
          └─ 等待 B 变化
    B 变化: count = 4
          ...
    ```

    **关键点：A 和 B 必须"交替"出现变化，count 才会持续递增。** 在等待 `@(A)` 期间 B 的变化被忽略；在等待 `@(B)` 期间 A 的变化被忽略。

    ## 三、几个容易误解的地方

    1. **`@(A)` 不是"等待 A 为高"**，而是"等待 A 有跳变"。要只响应上升沿应写 `@(posedge A)`。

    2. **A、B 的跳变必须交替**。如果 A 连续跳变两次而 B 不动，第二次 A 的变化会被"漏掉"，因为此时进程正阻塞在 `@(B)` 上。

    3. **没有敏感列表的 `always` 会无限循环**，靠 `@` 语句阻塞来"暂停"，这是合法且常见的建模手法（类似 Verilog 中的"事件等待"风格）。

    ## 四、仿真行为示例

    ```verilog
    // 激励
    initial begin
        A = 0; B = 0;
        #10 A = 1;   // count -> 1
        #10 B = 1;   // count -> 2
        #10 A = 0;   // count -> 3
        #10 B = 0;   // count -> 4
        #10 A = 1;   // count -> 5
    end
    ```

    每个时刻只有一个信号变化且严格交替，所以 count 按 0,1,2,3,4,5 递增。

    若改成：
    ```verilog
    #10 A = 1;  // count->1
    #10 A = 0;  // 此时进程阻塞在 @(B)，A 的这次变化被忽略
    #10 B = 1;  // count->2
    ```
    可以看到 A 的第二次变化被"吞掉"了。

    ## 五、可综合性

    - `initial` 块 + 这种"事件交替等待"的 `always` 结构**通常不可综合**（综合工具一般不支持无敏感列表的 always 加事件等待的建模方式）。
    - 它主要用于**仿真测试平台（testbench）**中建模某种握手/交替事件计数，或教学演示事件控制语句。
    - 若要做可综合的 A/B 边沿计数器，应写成带 `posedge` 敏感列表的同步逻辑，例如：

    ```verilog
    always @(posedge clk) begin
        if (A_rise || B_rise) count <= count + 1;
    end
    ```

    ## 六、小结

    | 项目 | 说明 |
    |------|------|
    | 功能 | 每检测到 A、B 交替的一次跳变，count 加 1 |
    | 触发条件 | `@(A)`/`@(B)` 为**任意跳变**（双边沿） |
    | 计数节奏 | A、B 必须严格交替变化 |
    | 溢出 | 8 位回绕（255→0） |
    | 用途 | 仿真建模/教学，一般不可综合 |

    本质上，这段代码用"阻塞在事件上"的方式，实现了一个**由两个信号交替驱动的脉冲计数器**。

* case 语句会只执行第一个匹配的语句，然后自动退出 case

* case 语句可以一行写多个条件：

    ```verilog
    A = 0;
    B = 0;
    casex(ADDRESS)
        5'b00???: A = 1;
        5'b01???: B = 1;
        5'b10?00, 5'b11?00:
            begin
                A = 1;
                B = 1;
            end
    endcase
    ```

* 详细讲讲 verilog 中 case, casex, casez 的区别

    In Verilog, the core difference between `case`, `casez`, and `casex` lies in **how they treat the uncertainty values `x` and `z` during matching**.

    Simply put: `case` requires exact matching; `casez` ignores `z` (and `?`); `casex` ignores both `x` and `z`. In synthesizable design, **`casex` is strongly discouraged** because it can mask real errors in the circuit, leading to inconsistencies between simulation and synthesis.

    ### Detailed Comparison

    | Statement | Matching Rules | Common Use Cases | Synthesis/Design Recommendations |
    | :--- | :--- | :--- | :--- |
    | **`case`** | **Exact matching**. The expression and case items must match bit-by-bit, including `x` and `z`. | Used for precise decoding, state machine transitions, and other scenarios requiring strict matching. | **Recommended**. This is the safest and most standard approach. |
    | **`casez`** | Treats `z` in the expression or case items as **"don't care"**. In numeric literals, `?` is typically used instead of `z` to represent "don't care" bits. | Suitable for scenarios where priority or partial bits need to be ignored, such as instruction decoding. | **Use with caution**. It is synthesizable, but one should be aware that it performs a "symmetric" comparison (see below for details). It is recommended to use `?` only in case items. |
    | **`casex`** | Treats both `x` and `z` as **"don't care"**. | Almost only used in simulation testbenches to handle uncertain states. | **Avoid using in synthesizable RTL**. Because it may cause `x` states in the circuit to be erroneously matched, hiding bugs. |

    ### Key Details and Pitfalls

    #### 1. What is "Symmetric Comparison"?
    This is the most dangerous aspect of `casex` (and `casez`). The so-called "don't care" is **bidirectional**: not only does `x`/`z` in the case item count as "don't care," but `x`/`z` in the expression being tested is also treated as "don't care" .
    *   **Consequence**: If the signal being tested (e.g., `sel`) experiences an unknown state `x` due to a fault (such as an uninitialized register), `casex` will treat this `x` as "don't care" and may match the first branch, causing the circuit to output a seemingly correct but actually incorrect value, masking the real bug .
    *   `casez` is relatively safer because signals typically do not spontaneously become high-impedance `z` during normal operation, but one should still be vigilant .

    #### 2. Synthesis Tool Behavior
    Although the Verilog standard defines the simulation behavior of `casex`/`casez`, synthesis tools (such as Quartus, Vivado) typically **treat `x` and `z` as don't-care conditions** when processing them. This means that the netlist generated by synthesis may be logically optimized, and its behavior may differ from the RTL simulation results (especially when `x` states occur) .

    #### 3. Overlapping Case Items
    Using `casez` or `casex` can easily lead to overlapping ranges between case items. For example, if the first item is `2'b0?` and the second is `2'b00`, when the input is `00`, both items match, but Verilog only executes the first one. This may cause synthesis tools to generate additional priority logic, increase circuit complexity, or produce unexpected behavior .

    ### Modern Alternative: SystemVerilog `case inside`
    If your toolchain supports SystemVerilog, it is recommended to use `case () inside` instead of `casez`/`casex`.
    *   **Advantage**: It uses **asymmetric comparison**. "Don't care" bits (wildcards) can only be used in case items; if the tested expression itself contains `x` or `z`, it does **not** count as a match. This retains the convenience of wildcards while correctly propagating uncertainty in the circuit, avoiding bug masking .

    ### Summary
    1.  **Prioritize `case`**: Use standard `case` for synthesizable code.
    2.  **Use `casez` cautiously**: If you must ignore certain bits, use `casez` with `?`, but ensure the tested signal does not produce unexpected `z`.
    3.  **Disable `casex`**: In RTL design, `casex` can hide critical `x`-state errors and should be avoided.
    4.  **Upgrade to `case inside`**: If conditions permit, migrate to SystemVerilog's `case inside`.

* `8'bZ0 = 8'bZZZZ_ZZZ0`

* `input P[3:0], Q[3:0], R[3:0];`这样写是可以的。

* 编译预处理

    * `define

        ```verilog
        `define 标识符（宏名） 字符串（宏内容）
        
        `define signal string
        ```

        使用 signal 来代替 string 这个字符串

        example:

        ```verilog
        `define WORDSIZE 8
        module
            reg[1:`WORDSIZE] data;  // 相当于 reg[1:8] data;
        // 宏名前要加上符号 ` 
        ```

        宏名和宏内容必须在同一行中进行声明。如果在宏内容中包含有注释行，注释行不会作为被转换的内容。

        宏定义不能被下面的表达式分隔开：

        注释行，数字，字符串，确认符，关键词，双目和三目字符运算符

        ```verilog
        `define first_half "start of string  // 非法
        $display(`first_half end of string");
        ```

    * `include

        将另外一个源文件的全部内容包含进来。

        ```verilog
        `include "文件名"
        ```

        文件路径名可以是相对的，也可以是绝对的。

        可以将多个 include 命令写在一行：

        ```verilog
        `include "fileB"  `include "fileC"  // including fileB and fileC
        ```

    * `timescale

        说明跟在该命令后的模块的时间单位和时间精度

        ```verilog
        `timescale <时间单位>/<时间精度>
        ```

        时间单位用于定位模块中仿真时间和延迟时间的基准单位，时间精度用于声明该模块的仿真时间的精确程度，该参数用于对延迟时间值进行取整操作（仿真前），因此又被称作取整精度。

        时间单位和时间精度必须是整数，有效数字为 1，10，100，单位为秒（s），毫秒（ms），微秒（us），纳秒（ns），皮秒（ps），飞秒（fs）。

        example:

        ```verilog
        `timescale 10 ns/1 ns

        #1.55 set = 0;
        ```

        首先计算 1.55 * 10ns = 15.5ns

        然后看到精度为 1ns，所以把 15.5ns 四合五入到 16ns。

        所以真实的延迟时间是 16 ns.

        $printtimescale 可以输出一个模块的时间单位和时间精度。

        $time 和 $realtime 及 %t 格式声明来输出显示 eda 工具记录的时间信息。

    * `ifdef, `else, `endif

        ```verilog
        `ifdef 宏名
            程序段 1
        `else
            程序段 2
        `endif


        `ifdef 宏名
            程序段 1
        `endif
        ```

        verilog 也支持 `ifndef, `elsif

    * $test, $plusargs

        ```verilog
        module test;
            reg a, b, c;
            initial begin
                a = 1'b1; b = 1'b0; c = 1'b1; 
                if ($test $plusargs ("DISPLAY_VAR"))
                    $display("display = %b", {a, b, c});
                else
                    $display("no display");
            end
        endmodule
        ```

        仅当运行时设置了标志 DISPLAY_VAR 时才显示变量。可以指定 +DDISPLAY_VAR 选项在程序运行时设置标志。

        可以使用 $value$plusargs 进一步控制条件执行。

        ```verilog
        module test;
            reg [8 * 128 - 1 : 0] test_string;
            integer clk_period;
            ...
            ...
            initial begin
                if ($value $plusargs ("test name = %s", test_string))
                    $readmemh(test_string, vectors);
                else
                    $display("test name option not specified");

                if ($value $plusargs("clk_t = %d", clk_period))
                    forever #(clk_period / 2)  clk = ~clk;
                else
                    $display("Clock period option name not specified");
            end
        endmodule
        ```

        若要启动上述选项，需要 +testname=test1.vec +clk_t=10 来启动仿真器。

        注：

        1. $value $plusargs 有点像 module parameter, 或者环境变量

* $random

    返回一个 32 位的随机数，带符号。

    通常用法：

    ```verilog
    $random % b  // b > 0
    // 输出的范围为 (-b + 1):(b - 1)
    ```

    example:

    ```verilog
    reg[23:0] rand;
    rand = $random % 60;  // -59 ~ 59

    reg[23:0] rand;
    rand = {$random} % 60;  // 0 ~ 59
    ```

    example 2: 产生宽度随机的随机脉冲序列作为测试信号源

    ```verilog
    `timescale 1ns/1ns
    module random_pulse(dout);
        output [9:0] dout;
        reg [9:0] dout;

        integer delay1, delay2, k;
        initial begin
            # 10 dout = 0;
            for (k = 0; k < 100; k = k + 1) begin
                delay1 = 20 * ({$random} % 6);
                // delay1 在 0 ~ 100 ns 间变化
                delay2 = 20 * (1 + {$random} % 3);
                // delay2 在 20 ~ 60 ns 间变化
                #delay1 dout = 1 << ({$random} % 10);
                // dout 的 0 ~ 9 位中随机出现 1, 并出现的时间在 0 ~ 100 ns 间变化
                #delay2 dout = 0;
                // 脉冲的宽度在 20 ~ 60 ns 间变化
            end
        end
    endmodule
    ```

* $readmemb, $readmemh

    从文件中读取数据到存储器中。

    * $readmemb("<data_file_name>", <memory_name>);

    * $readmemb("<data_file_name>", <memory_name>, <start_addr>);

    * $readmemb("<data_file_name>", <memory_name>, <start_addr>, <end_addr>); 

    * $readmemh("<data_file_name>", <memory_name>);

    * $readmemh("<data_file_name>", <memory_name>, <start_addr>);

    * $readmemh("<data_file_name>", <memory_name>, <start_addr>, <end_addr>);

    其中 b 表示二进制，h 表示十六进制。

    example:

    ```verilog
    module test;
        reg [7:0] memory[0:7];
        integer i;

        initial begin
            // 读取数据
            $readmemb("init.dat", memory);
            // 显示初始化后的存储器的内容
            for (i = 0; i < 8; i = i + 1) begin
                $display("memory[%d] = %b", i, memory[i]);
            end
        end
    endmodule
    ```

    `init.dat`:

    ```
    @002
    11111111 01010101
    00000000 10101010

    @006
    1111zzzz 00001111
    ```

    其中`@`开头的数字表示地址。数据文件可以使用`/* xxx */`和`//`进行注释。

    output:

    ```
    Memory[0] = xxxxxxxx
    Memory[1] = xxxxxxxx
    Memory[2] = 11111111
    Memory[3] = 01010101
    Memory[4] = 00000000
    Memory[5] = 10101010
    Memory[6] = 1111zzzz
    Memory[7] = 00001111
    ```

* $monitor

    格式：

    ```verilog
    $monitor(p1, p2, ..., pn);
    $monitor;
    $monitoron;
    $monitoroff;
    ```

    监控和输出参数列表中的表达式或变量值。

    其参数列表中输出控制格式字符串和输出表列的规则和 $display 中的一样。

    example:

    ```verilog
    $monitor($time, , "rxd = %b  txd = %b", rxd, txd);
    ```

    使用 $time 后，参数列表中变量或表达式的值同时发生变化的时刻可以通过标明同一时刻的多行输出来显示。（没看明白）

    在 $display 中也可以这样使用。

    上面的语句中，`, ,`代表一个空参数，空参数在输出时显示为空格。

    * $monitoron, $monitoroff，打开或关闭 monitor

        当调用 $monitoron 时，总是立刻输出显示当前时刻参数列表中的值, 通常用于输出初始值

        默认情况下，monitor 是打开的。

        $monitor 通常在 initial 块中调用。

* $time

    返回一个 64 位的整数来表示当前仿真时刻值

    example:

    ```verilog
    `timescale 10 ns/1 ns
    module test;
        reg set;
        parameter p=1.6;
        initial begin
            $monitor($time, ,"set = ", set);
            # p set = 0;
            # p set = 1;
        end
    endmodule
    ```

    output:

    ```
    0 set = x
    2 set = 0
    3 set = 1
    ```

    可以看出 time 先取 1.6 和 3.2，然后再取整。

* $realtime

    返回实数时间。

    ```verilog
    `timescale 10 ns/1 ns
    module test;
        reg set;
        parameter p=1.55;
        initial begin
            $monitor($realtime, , "set = ", set);
            # p set = 0;
            # p set = 1;
        end
    endmodule
    ```

    output:

    ```
    0 set=x
    1.6 set=0
    3.2 set=1
    ```

* $finish

    syntax:

    ```verilgo
    $finish;
    $finish(n);
    ```

    退出仿真器，结束仿真过程。

    n 取值：

    * 0: 不输出任何信息

    * 1: 输出当前仿真时刻和位置

    * 2: 输出当前仿真时刻、位置和在仿真过程中所用 memory 以及 cpu 的时间统计。

    默认为 1.

* $stop

    syntax:

    ```verilog
    $stop;
    $stop(n);
    ```

    把 eda 工具置成暂停模式，在仿真环境下给出一个交互式命令提示符, 将控制权交给用户。

* 详细讲讲 verilog 中的分频器与倍频器

    在Verilog与FPGA设计中，**分频器**通常指用逻辑资源（计数器、触发器）将输入时钟频率降低；而**倍频器**的情况比较特殊：纯Verilog逻辑几乎无法稳定地生成比输入时钟更快的时钟，实际工程中倍频主要依赖FPGA内部的**锁相环（PLL）或混合模式时钟管理器（MMCM）** 硬件资源，Verilog代码仅负责例化IP核。

    ### 分频器：用计数器“数”出低频时钟

    分频的核心是计数器。根据分频系数的奇偶性，实现方式略有不同。

    **偶数倍分频**最简单，能天然获得50%占空比。思路是用计数器数到 `N/2 - 1` 时翻转输出时钟，并清零计数器，循环往复。例如10分频，计数器数到4就翻转输出，输出高低电平各持续5个输入时钟周期。

    **奇数倍分频**如果对占空比没要求，一个计数器也能完成，只需在特定计数值翻转即可。但如果要求**50%占空比**，单靠上升沿计数做不到（因为奇数周期无法均分），需要借助**双边沿**：分别用上升沿和下降沿各产生一个占空比非50%的奇数分频时钟，两者相位差半个时钟周期，最后做逻辑“或”或“与”运算，拼出50%占空比。以三分频为例，上升沿逻辑产生高1周期、低2周期的波形，下降沿逻辑产生错开半周期的同形波形，两者相或即得50%占空比的三分频。

    此外还有**半整数分频**（如2.5分频）和**小数分频**。半整数分频的核心是在计数到某个值时翻转触发时钟的极性，使计数器多“卡”半个周期。小数分频则通常用多个整数分频器交替工作来逼近目标平均频率，但输出抖动较大，实际使用较少。

    需要留意：用Verilog逻辑分频产生的时钟，如果直接驱动大量触发器，可能存在毛刺风险。可靠的做法是将分频输出作为**时钟使能信号**使用，而不是直接当作时钟布线，或者确保分频输出是寄存器输出、无组合逻辑毛刺。

    ### 倍频器：Verilog逻辑的“禁区”与PLL的正道

    严格来说，**用纯Verilog逻辑对单一时钟做稳定的整数倍频，在工程上是不现实的**。你无法用“数”的方式无中生有地造出比输入时钟更快的边沿——逻辑门和触发器的延迟决定了你无法可靠地产生输入时钟上升沿之间的额外边沿。因此，搜索到的所谓“Verilog倍频器”代码，其仿真模型在真实硬件中往往无法稳定工作，或者仅适用于极低频率、对抖动毫无要求的场景。

    工程上的倍频，一律走**硬件专用资源**这条路。Xilinx 7系列及以后的FPGA内部有**时钟管理单元（CMT）**，包含PLL和MMCM。你在Vivado中配置PLL IP核，设定输入频率（如板载50MHz）和期望输出频率（如100MHz、200MHz），工具会自动计算倍频/分频参数。Verilog代码只需要例化这个IP核，把输入时钟接进去，输出时钟接出来即可。PLL的优势是输出时钟质量高、抖动小、相位可调，且不占用通用逻辑资源。

    所以，一个实用的设计原则是：**降频用计数器，升频用PLL**。只有在极少数需要节省PLL资源、且对时钟质量要求不高的场合，才会考虑用逻辑分频；而倍频则几乎始终依赖PLL/MMCM。

* top 用 localparam WIDTH 作为端口宽度 —— 合法但不灵活

    ```verilog
    module top#(
        localparam WIDTH = 16
    )(
    ```

    localparam 在参数端口列表里是合法的，但它不能被实例化覆盖。如果 top 是顶层或你希望上层能改位宽，应改用 parameter：

    ```verilog
    module top#(
        parameter WIDTH = 16
    )(
    ```

    如果 top 确实是固定顶层且不想被覆盖，localparam 可以保留，但要清楚这个限制。

* parameter 有默认值与无默认值的写法

    module full_adder#(
        parameter WIDTH
    )(
    和
    module full_adder#(
        parameter WIDTH ＝ 16
    )(
    这两种写法有什么区别

    这两种写法的区别在于 **parameter 是否有默认值**：

    ## 1. 无默认值写法
    ```verilog
    module full_adder#(
        parameter WIDTH
    )(
        ...
    );
    ```

    - `WIDTH` **没有默认值**，属于"必须被覆盖"的参数。
    - 在实例化时，**必须显式传递** `WIDTH`，否则：
      - 某些工具会报错（推荐行为）；
      - 某些工具会将其置为默认的 `'x`/0 或给出警告，行为不确定。
    - 适合**强制使用者指定参数**的场景。

    实例化时必须写：
    ```verilog
    full_adder #(.WIDTH(8)) u0 (...);
    ```

    ## 2. 有默认值写法
    ```verilog
    module full_adder#(
        parameter WIDTH = 16
    )(
        ...
    );
    ```

    - `WIDTH` **有默认值 16**。
    - 实例化时可以**省略**参数，此时使用默认值 16：
    ```verilog
    full_adder u0 (...);              // WIDTH = 16
    full_adder #(.WIDTH(8)) u1 (...); // WIDTH = 8
    ```
    - 适合**大多数情况用默认值、特殊时覆盖**的场景，接口更友好。

    ## 小结对比

    | 项目 | `parameter WIDTH` | `parameter WIDTH = 16` |
    |------|-------------------|------------------------|
    | 默认值 | 无 | 16 |
    | 实例化省略参数 | 不合法/不确定 | 合法，用 16 |
    | 是否必须传参 | 是 | 否 |
    | 推荐用途 | 强制指定 | 通用、带默认配置 |

    > 补充：你第二个写法里用的是全角等号 `＝`，在 Verilog 中**不合法**，必须是半角 `=`。实际代码要写成 `parameter WIDTH = 16`。

    另外，`parameter` 若放在模块参数端口列表 `#( )` 中，实例化时用 `#(.WIDTH(...))` 或 `#(值)` 覆盖；若放在模块体内（`parameter WIDTH = 16;`），则不能被实例化覆盖（除非用 `defparam`，不推荐）。

* verilog 中对 module 的形参设置位宽，但是我想把这个位宽参数化，方便后面可以动态调整，怎么办

    在 Verilog 中，模块的端口位宽可以通过 `parameter` 参数化，这样在例化时可以动态调整。以下是几种常见做法：

    ## 1. 基本参数化端口位宽

    ```verilog
    module my_module #(
        parameter WIDTH = 8,
        parameter DEPTH = 16
    )(
        input  wire                  clk,
        input  wire                  rst_n,
        input  wire [WIDTH-1:0]      din,
        output wire [WIDTH-1:0]      dout,
        input  wire [DEPTH-1:0]      valid
    );
        // 内部逻辑
        reg [WIDTH-1:0] data_reg;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                data_reg <= {WIDTH{1'b0}};
            else
                data_reg <= din;
        end
        assign dout = data_reg;
    endmodule
    ```

    ## 2. 例化时覆盖参数

    ```verilog
    // 方式一：按位置覆盖
    my_module #(16, 32) u_inst (
        .clk  (clk),
        .rst_n(rst_n),
        .din  (din16),
        .dout (dout16),
        .valid(valid32)
    );

    // 方式二：按名字覆盖（推荐）
    my_module #(
        .WIDTH(16),
        .DEPTH(32)
    ) u_inst (
        .clk  (clk),
        .rst_n(rst_n),
        .din  (din16),
        .dout (dout16),
        .valid(valid32)
    );
    ```

    ## 3. 多层传递参数（子模块参数化）

    ```verilog
    module top #(
        parameter DATA_W = 8
    )(
        input wire [DATA_W-1:0] din,
        output wire [DATA_W-1:0] dout
    );
        // 把参数继续传给子模块
        sub_module #(
            .WIDTH(DATA_W)
        ) u_sub (
            .din (din),
            .dout(dout)
        );
    endmodule
    ```

    ## 4. 使用 `localparam` 派生参数

    如果位宽是其他参数的函数，建议用 `localparam`：

    ```verilog
    module fifo #(
        parameter DATA_W = 8,
        parameter DEPTH  = 16
    )(
        input  wire [DATA_W-1:0] din,
        output wire [DATA_W-1:0] dout
    );
        // 派生参数
        localparam ADDR_W = $clog2(DEPTH);  // 根据深度自动计算地址位宽

        reg [DATA_W-1:0] mem [0:DEPTH-1];
        reg [ADDR_W-1:0] wr_ptr;

        // ...
    endmodule
    ```

    ## 5. 注意事项

    | 要点 | 说明 |
    |------|------|
    | **参数名用大写** | 如 `WIDTH`、`DATA_W`，区分于端口名 |
    | **例化推荐按名覆盖** | `#(.WIDTH(16))`，避免顺序错误 |
    | **`$clog2` 是系统函数** | 用于计算地址位宽，需工具支持（Verilog-2005+） |
    | **避免在端口声明里用 `localparam`** | 端口只能用 `parameter`，`localparam` 只能用于内部 |
    | **SystemVerilog 更灵活** | 可用 `parameter type`、`$bits()` 等 |
    | **仿真与综合一致** | 参数在综合时是常量，必须为确定值 |

    ## 6. SystemVerilog 增强写法（可选）

    ```systemverilog
    module my_module #(
        parameter int WIDTH = 8,
        parameter type DATA_T = logic [WIDTH-1:0]
    )(
        input  DATA_T din,
        output DATA_T dout
    );
        // ...
    endmodule
    ```

    这样位宽就完全参数化了，例化时只需指定 `WIDTH` 即可动态调整，无需修改模块内部代码。

    * 那如果既在端口里有 parameter，又在 module 里写了 parameter，那么在例化的时候，my_module #(16, 32) u_inst (  这种按位置定义参数该怎么写？

        你描述的这种写法在标准 Verilog 中**存在语义冲突**，直接按位置覆盖会出问题。

        ### 核心结论：不要混用两种参数声明方式

        如果你的模块**同时**在 `#( ... )` 参数端口列表中声明了参数，又在模块内部（`module` 与 `endmodule` 之间、端口声明之外）写了 `parameter`，那么**模块内部的那个 `parameter` 会被工具当作 `localparam` 处理**，即变成局部常量，**无法在例化时被覆盖**。

        因此，你无法通过 `my_module #(16, 32)` 来同时覆盖这两类参数。按位置覆盖的语法，只对 `#( ... )` 参数端口列表中的参数有效。

        ### 正确的做法：统一放在参数端口列表中

        如果你想按位置覆盖，就把所有希望被外部调整的参数都放进 `#( ... )` 里：

        ```verilog
        module my_module #(
            parameter WIDTH = 8,   // 第1个位置
            parameter DEPTH = 16   // 第2个位置
        )(
            input  wire [WIDTH-1:0] din,
            output wire [WIDTH-1:0] dout
        );
            // 内部逻辑
        endmodule
        ```

        例化时按位置写：

        ```verilog
        my_module #(16, 32) u_inst (
            .din (din16),
            .dout(dout16)
        );
        // 此时 WIDTH=16, DEPTH=32
        ```

        ### 如果确实有“内部参数”的需求

        如果你有一些参数**不打算让外部修改**，只是想基于 `WIDTH` 做内部计算，应该显式使用 `localparam`，这样语义清晰，也不会引起混淆：

        ```verilog
        module my_module #(
            parameter WIDTH = 8
        )(
            // 端口
        );
            localparam INTERNAL_WIDTH = WIDTH * 2;  // 明确这是内部派生参数
            // ...
        endmodule
        ```

        ### 强烈建议：按名字覆盖

        虽然你问的是按位置覆盖，但 Verilog 社区和工程实践都**强烈推荐按名字覆盖**，因为这样可以避免参数顺序调整或数量变化带来的隐蔽错误：

        ```verilog
        my_module #(
            .WIDTH(16),
            .DEPTH(32)
        ) u_inst ( ... );
        ```

        这样即使将来你在 `#( ... )` 中增加、删除或重排参数，例化代码也不会出错。

* genvar 不能在 generate 外赋值

* 不能在 generate 块内对 genvar 赋值

    比如

    ```verilog
    genvar i;
    generate
        i = 0;  // error
        for (xxx) begin
            ...
        end
    endgenerate
    ```

* genvar 是编译期常量，只能用于 generate 的索引，不能被赋值或在运行时修改。

* generate 里的 if/else if/else 是静态展开，不是运行时分支

    generate if 在编译期根据条件是否恒真来决定是否生成代码。i 是 genvar，条件 i==0、i<=2 不会"依次执行"，而是被当作常量表达式。

* verilog 里定义多个变量比较麻烦，有没有定义数组的方式
    wire cout0, cout1, cout2;

    在 Verilog 中，有几种方式可以定义一组相关的信号，避免一个个声明：

    ## 1. 使用 wire/reg 数组（向量）

    ```verilog
    wire [3:0] cout;   // cout[0], cout[1], cout[2], cout[3]
    ```
    注意这是**位向量**，`cout[0]` 是 1 bit。这其实就是最常用的方式，等价于声明了 4 根独立的 wire，只是打包在一起。

    ```verilog
    wire [2:0] cout;   // cout[0], cout[1], cout[2]
    ```

    ## 2. 多维数组（Verilog-2001 支持）

    ```verilog
    wire [7:0] cout [0:3];   // 4 个 8-bit 信号：cout[0]...cout[3]，每个是 8 bit
    ```

    ## 3. 用 generate 批量例化

    如果每根 wire 接的是同一个模块的不同实例：

    ```verilog
    genvar i;
    generate
      for (i = 0; i < 3; i = i + 1) begin : gen_cout
        wire cout;
        // 例化子模块
        my_module u (.cout(cout));
      end
    endgenerate
    ```
    这样每个 `cout` 在各自 generate block 里独立命名，互不冲突。

    ## 4. 用 SystemVerilog（推荐）

    ```verilog
    wire cout [3];          // unpacked array
    wire logic [3:0] cout;  // packed
    ```

    ## 实际建议

    你原来写：
    ```verilog
    wire cout0, cout1, cout2;
    ```
    直接改成：
    ```verilog
    wire [2:0] cout;
    // 使用 cout[0], cout[1], cout[2]
    ```

    这是最简洁、最标准的做法，综合后完全等价，只是代码更清爽。唯一要注意的是：**位向量不能跨模块端口单独传递**（端口只能传整个向量或单独 bit，这没问题），以及在 `always` 里赋值要用 `reg` 类型（或 SV 的 `logic`）。

    * wire [2:0] cout; 这样的写法，会不会在综合时，优先使用并排线，而不是独立线？

        不会。`wire [2:0] cout;` 这种写法在综合时**不会**被强制要求使用“并排线”或某种特殊的物理走线方式。它在功能上完全等价于三根独立的 1-bit wire，综合工具会根据实际的电路连接和时序需求来自由决定物理实现。

        ### 为什么不会影响物理走线

        1.  **语法本质是“分组”而非“物理捆绑”**：在 Verilog 中，向量（vector）的声明仅仅是为了在代码层面方便地用一个名字来操作一组相关的信号。它描述的是逻辑上的关系，而不是物理上的布局约束。
        2.  **综合工具的视角**：综合工具在解析代码后，会将其转化为底层的逻辑门和连线。对于 `cout[2:0]`，工具看到的是三个独立的逻辑节点（net），它们之间除了被命名在同一个向量里之外，并没有任何强制性的物理关联。工具会根据你的逻辑设计（比如每个 bit 分别连接到哪里、时序要求如何）来决定是让它们靠在一起走线，还是为了布通率而将它们分开。
        3.  **物理实现由后端决定**：真正的“并排走线”与否，是由布局布线（P&R）工具根据芯片或 FPGA 的物理资源、拥塞情况以及时序约束来决定的。RTL 代码里的向量声明本身并不包含“并排布局”的指令。

        ### 唯一相关的特殊情况

        虽然向量本身不强制物理并排，但综合工具可能会因为**逻辑功能的相似性**而做一些优化。例如，如果你用向量定义了一组结构完全相同的寄存器或选择器，一些高级综合工具（如 Design Compiler）可能会将它们推断为“多比特（Multibit）”库单元。这种库单元在物理上可能确实是紧密排列的，目的是为了共享面积、降低功耗和改善时钟偏移。但这属于工具自动推断的优化结果，而不是 `wire` 向量声明本身的强制要求，且主要针对寄存器或三态单元，而非纯组合逻辑的普通 `wire`。

        所以，放心使用 `wire [2:0] cout;` 即可，它是最简洁且标准的写法，不会对你的物理设计施加额外的限制。

* verilog 中，位宽是修饰`input / output`的，不是修饰变量的

    ```verilog
    input [31:0] in_1, in_2;  // OK, 可以一行定义多个变量
    input [31:0] in_1, [15:0] in_2;  // Error, 位宽不是用来修饰变量的
    input wire in_1, in_2;  // OK, input 和 wire 可以写在一起
    input wire in_1, wire in_2;  // Error, 与位宽的情况类似，wire 不是用来修饰变量的
    ```

    ```verilog
    input wire in_1, in_2;  // OK

    input in_1, in_2;
    wire in_1, in_2;  // OK, wire / reg 可以与 innput / output 写在不同行
    ```

* 既然 RCA 使用的是把 c_out 接到上一位的 c_in，那么理论上，如果都使用 wire 的话，是没有延迟的吧？还是说，一定需要一个 reg 寄存器来保存？

    这是一个很好的问题，触及了硬件设计中一个根本性的误解。答案是：**组合逻辑的 wire 连接一样有延迟，而且延迟正是 RCA 的核心问题；完全不需要 reg 来"保存"进位。**

    ## 一、wire 连接本身不产生"零延迟"

    wire 在 Verilog 里只是**一根连线的抽象**，它表示"这两个点电气相连"。但物理上，信号从 `c_out` 传到下一级的 `c_in`，要经过：

    1. **门电路的传播延迟**：`c_out` 不是凭空出现的，它是由与门、或门算出来的。`c_out = g | (p & c_in)` 这个逻辑本身要经过门的开关时间（工艺上几皮秒到几百皮秒不等）。
    2. **连线延迟**：金属互连有电阻和电容，信号沿导线传播需要时间（RC 延迟），虽然短，但非零。
    3. **扇出负载**：`c_out` 驱动下一级多个门输入，负载越大，翻转越慢。

    所以"用 wire 直接连"只是描述**拓扑上的连接**，不改变**物理上的延迟**。`c_out` 接到下一级 `c_in`，下一级的门必须等 `c_out` 真正稳定后才能算出结果——这个等待就是延迟。

    ## 二、RCA 的延迟从哪里来

    关键在于：**每一级的 `c_out` 依赖上一级的 `c_in`，而每一级的计算都要花门延迟。**

    ```
    第 0 级：算 c1，耗时 t
    第 1 级：必须等 c1 稳定，再算 c2，耗时再 +t
    第 2 级：必须等 c2 稳定，再算 c3，耗时再 +t
    ...
    ```

    这是一个**因果链**：第 i 级的输出在物理上不可能早于第 i-1 级的输出稳定。即使全部用 wire 连接、没有任何寄存器，这条链的**累计延迟**就是 N × t。这不是"存储"问题，而是"计算必须按顺序发生"的问题。

    ## 三、为什么不需要 reg

    reg 在 Verilog 中描述的是**时序元件（触发器/锁存器）**，它在时钟边沿采样并保持值。RCA 是**纯组合逻辑**，它的进位是"输入一变，经过若干门延迟后输出跟着变"的**连续响应**，不需要任何时钟或存储。

    - **不需要 reg 保存进位**：进位是一个组合信号，只要输入 `a`、`b`、`c_in` 保持稳定，进位就会稳定在正确值。它没有"必须被时钟锁存"的需求。
    - **加了 reg 反而错**：如果你用寄存器把每级进位打一拍，那就变成了**流水线加法器**——确实能提高吞吐（每拍处理一级），但**单次运算的延迟反而增加**（N 个时钟周期才能出结果），而且需要时钟、面积和功耗代价。这不是 RCA 的本意。

    ## 四、一个关键区分：拓扑延迟 vs 时序延迟

    | 概念 | 含义 | RCA 中的体现 |
    |------|------|-------------|
    | **组合延迟**（传播延迟） | 输入变化到输出稳定的物理时间 | 进位链 N × 门延迟，这是 RCA 的瓶颈 |
    | **时序延迟**（时钟周期） | 需要几个时钟沿才出结果 | 纯组合 RCA 是 0 个时钟周期（同一周期内出结果），但时钟周期必须 ≥ 组合延迟 |

    用 wire 连的 RCA，**在一个时钟周期内就能算出结果**（如果时钟周期足够长，容纳 N×t 的延迟）。它不需要 reg，但它**限制了时钟频率**——时钟周期不能短于最坏路径延迟。这正是为什么高位宽 RCA 在高频设计中不可用，要换 CLA。

    ## 五、类比

    把 RCA 想象成**一排人传水桶**：

    - 每个人（一级全加器）接到上一个人的桶（进位）后，才能决定自己往下一人递什么。
    - 中间不需要"仓库"（reg）暂存，水桶是连续传的。
    - 但第 N 个人拿到桶的时间，必然是"第 1 个人处理时间 + 第 2 个人 + ... + 第 N 个人"，这个累计就是延迟。
    - 你不可能因为"没用仓库"就让第 N 个人提前拿到桶——因果顺序决定了必须等。

    ## 六、那 CLA 为什么快

    CLA 不是消除门延迟，而是**改变依赖结构**：它用 `g`、`p` 直接展开出 `c_i` 的表达式，使得 `c_i` 不再依赖 `c_{i-1}` 的计算结果，而是所有 `c_i` 可以**并行**从 `g`、`p`、`c0` 算出。虽然每个 `c_i` 的表达式更复杂（门更多、扇入更大），但它们的计算**不互相等待**，所以总延迟从 O(N) 降到 O(1)（理想）或 O(log N)（分层）。这同样不需要 reg，纯粹是组合逻辑的**结构优化**。

    ## 总结

    - **wire 连接 ≠ 零延迟**：wire 是拓扑抽象，物理延迟来自门和连线，无法消除。
    - **RCA 不需要 reg**：进位是组合信号，连续传播即可；加 reg 会变成流水线，改变的是吞吐而非单次延迟。
    - **RCA 慢的本质**：进位链的**因果依赖**导致延迟线性累积，这是结构问题，不是存储问题。解决它要靠 CLA 这类**改变依赖结构**的方法，而不是加寄存器。

    * 那么如何判断 RCA 在某个时间点是否稳定？比如我输入 a, b 后，经过多少时间，可以认为 64 位全加器的 sum 和 c_out 是可用的？

        在仿真环境（如 Verilator）里，你**无法直接“等待一个物理时间”然后读取结果**。Verilator 的 `eval()` 是一个功能仿真调用，它会在内部迭代直到所有组合逻辑稳定（收敛），然后返回。调用 `eval()` 之后读到的 `sum` 和 `c_out`，在**逻辑功能上**就是可用的。

        但如果你问的是**真实芯片**里需要等多久，答案取决于具体的工艺和电路实现，不存在一个通用的“多少纳秒”的固定数值。不过，对于 64 位行波进位加法器（RCA），有一个经典的理论延迟估算可以参考。

        ### 64 位 RCA 的理论门延迟

        根据数字逻辑教材中的经典模型，RCA 的延迟与位数成正比。对于第 i 位（从 0 开始计数）：
        *   第 i 位的和 `Si` 大约在第 `2i + 1` 个门延迟后稳定。
        *   第 i 位的进位 `Ci` 大约在第 `2i` 个门延迟后稳定。

        因此，对于一个 64 位加法器：
        *   最高位进位 `C64`（即 `c_out`）的延迟约为 **128 个门延迟**（`2 × 64`）。
        *   最高位和 `S63` 的延迟约为 **127 个门延迟**（`2 × 63 + 1`）。

        **关键点**：这里的“门延迟”是一个抽象单位，**不是**真实的皮秒或纳秒。在真实芯片中，一个门延迟可能是几十皮秒，也可能是几百皮秒，取决于晶体管工艺（如 7nm 还是 180nm）、供电电压、温度以及具体的逻辑门类型。因此，128 个门延迟对应的真实时间，必须通过**静态时序分析（STA）** 工具，结合具体的工艺库文件来精确计算，无法从理论公式直接换算。

        ### 在 Verilator 仿真中的实际操作

        你不需要（也无法）在 C++ 测试平台里插入一个 `#delay` 来等待 RCA “稳定”。Verilator 的 `eval()` 调用本身就承担了“让组合逻辑收敛”的职责。

        你之前写的代码逻辑是正确的：
        ```cpp
        dut->a = ...; dut->b = ...; dut->c_in = ...;
        dut->eval();          // 内部迭代直至所有组合逻辑稳定
        // 此时读取 dut->s 和 dut->c_out 在功能上就是正确的
        ```
        Verilator 内部有一个“settle region”（稳定区域），会反复迭代组合逻辑，直到没有信号再发生变化为止，然后才返回。所以，只要调用了一次 `eval()`，你就应该认为输出是“可用”的。

        **总结**：在 Verilator 仿真里，`eval()` 返回即代表逻辑稳定，无需额外等待。在真实硬件中，64 位 RCA 的 `c_out` 延迟约为 128 个理论门延迟，但精确时间必须由特定工艺的 STA 工具给出。

    * 那我如果是写 rtl 的工程师呢，我如何在代码里判断这个全加器的输出是有效的？

        作为 RTL 工程师，你在代码里**无法直接“判断”一个纯组合逻辑全加器的输出在某个物理时刻是否有效**。RTL 代码描述的是逻辑功能，不包含物理延迟信息。判断输出是否“可用”，是**时序分析工具（STA）** 的工作，而不是 RTL 代码本身能完成的。

        ### 为什么 RTL 代码里判断不了

        你写的 Verilog 描述的是“输入变了，输出逻辑上应该变成什么”，而不是“输出需要多少皮秒才能变过来”。真实的门延迟、线延迟、工艺角（PVT）信息，在 RTL 阶段都是未知的。

        Verilog 仿真器（包括 Verilator）在 `eval()` 时会让组合逻辑在功能上收敛，但这只是**逻辑稳定**，不代表**时序满足**。一个 64 位 RCA 在功能仿真里瞬间出结果，但在真实芯片里可能因为延迟太大而无法在目标时钟周期内稳定。

        ### RTL 工程师真正该做的事：写约束，交给 STA

        你的职责是**定义时序约束**，然后由综合工具和 STA 工具去验证。对于全加器这种组合逻辑，关键在于它被夹在哪些寄存器之间。

        **核心判断标准是 Setup 时间约束**：

        数据从上一级触发器发出，经过组合逻辑（你的加法器），必须在下一个时钟沿到来之前，提前 `T_setup` 时间稳定在下一级触发器的输入端。

        用公式表达就是：
        ```
        时钟周期 ≥ 触发器输出延迟 (T_cq) + 组合逻辑延迟 (T_comb) + 建立时间 (T_setup) + 时钟不确定性
        ```

        你的 RTL 代码不需要、也无法计算 `T_comb`。你要做的是：

        1.  **定义时钟**：在综合脚本（如 SDC）中，用 `create_clock` 声明时钟频率。这告诉工具“这个加法器必须在多少纳秒内算完”。
        2.  **运行综合/STA**：工具会根据目标工艺库，计算每条路径的实际延迟。
        3.  **看时序报告**：如果加法器的路径被报告为 **Setup Violation（建立时间违例）**，说明在你的目标频率下，64 位 RCA 太慢了，输出在时钟沿到来时还没稳定，不可用。

        ### 如果时序不满足，RTL 层面能做什么

        如果 STA 告诉你这个 RCA 是关键路径（Critical Path），导致时序违例，作为 RTL 工程师，你**不应该**试图在代码里“等一等”或者“判断一下”。你应该**改变微架构**：

        *   **换成超前进位加法器（CLA）或前缀加法器**：用逻辑结构换延迟，这是最直接的 RTL 优化。
        *   **插入流水线寄存器**：把 64 位加法拆成两级（例如两个 32 位），中间加一级寄存器。这样每个时钟周期只需要算一半，代价是输出延迟一拍（Latency +1）。这是 RTL 工程师解决长组合路径最常用的手段。

        ### 一句话总结

        **RTL 代码定义逻辑，STA 定义时序。** 你不需要在 Verilog 里判断加法器何时稳定，你需要写对时序约束，看 STA 报告是否通过。如果不过，就改架构（CLA/流水线），而不是在代码里等。

    * 所以作为 rtl 工程师，我先默认组合逻辑是可以瞬间完成的，直接拿结果来用。后面做 sta 时，我需要声明这个输出要在多少拍内完成？

        你的直觉大体对，但有一个关键术语需要纠正：**STA 约束的不是“多少拍内完成”，而是“一个时钟周期内必须完成”（对于单周期组合路径）。**

        ### RTL 阶段：默认组合逻辑“逻辑上瞬间完成”

        是的。你写 RTL 时，心态就是：`assign s = a + b;` 之后，同一个时钟周期内 `s` 就是可用的。RTL 描述的是**逻辑关系**，不关心物理延迟。你不需要在代码里写“等 5ns 再用 `s`”。这是正确的抽象层次。

        ### STA 阶段：你约束的是“时钟周期”，不是“拍数”

        对于一条**纯组合逻辑路径**（从触发器 A 出发，经过你的加法器，到触发器 B 的输入），它必须在**一个时钟周期内**完成。你不需要声明“几拍”，因为默认它就被当作单周期路径。你要做的是：

        1. **定义时钟频率**：`create_clock -period 10 [get_ports clk]`，这等于告诉工具“每拍 10ns”。
        2. **工具自动分析**：STA 会检查所有从 A 到 B 的路径，计算 `T_cq + T_comb + T_setup` 是否 ≤ 10ns。你的加法器延迟 `T_comb` 就包含在内。
        3. **看报告**：如果违例，说明这个组合逻辑在一个周期内算不完。

        所以准确说法是：**你声明时钟周期，工具验证组合逻辑能否在一拍内完成**。你不需要（也不能）单独声明“这个加法器要在几拍内完成”——“几拍”是由你的微架构决定的，不是约束能改变的。

        ### 如果你想让它“多拍完成”，那是改架构，不是改约束

        如果 64 位 RCA 一拍算不完，STA 会报违例。这时你有两个 RTL 层面的选择：

        - **换更快的结构**（CLA/前缀加法器），让它仍在一拍内完成。
        - **插入流水线寄存器**，把一条长组合路径切成两段，每段一拍。这**改变了 RTL 结构**，此时它变成了**多周期路径**（准确说是流水线，Latency 增加），你需要相应调整上下游逻辑的握手/对齐。

        注意：还有一类叫 **Multicycle Path** 的 STA 约束（`set_multicycle_path`），用于“逻辑上允许两拍完成、但中间没有寄存器”的特殊情况。但这**不是**给普通加法器用的——它适用于那些被使能信号控制、确实可以慢几拍采样的路径。对普通全加器，默认就是单周期，不要滥用 multicycle 约束来“掩盖”违例，那会导致真实芯片出错。

        ### 一句话总结

        - **RTL**：默认组合逻辑同周期可用，直接拿结果。
        - **STA**：你声明**时钟周期**，工具验证组合路径能否在**一拍内**满足 setup。不能单独声明“几拍完成”。
        - **想多拍**：改 RTL 加流水线寄存器，而不是写约束。`set_multicycle_path` 只用于特定合法场景，不是给普通组合逻辑“放水”的工具。

* 详细讲讲行波进位和超前进位的设计思想

    行波进位（Ripple Carry）和超前进位（Carry Lookahead）是加法器设计的两种基本思想，核心差异在于**如何处理进位链的延迟**。下面从设计动机、结构、延迟分析和权衡四个维度详细展开。

    1. 一、问题的起点：进位为什么是瓶颈

        一个 1 位全加器的输出为：

        ```
        s   = a ^ b ^ c_in
        c_out = (a & b) | (a & c_in) | (b & c_in)
        ```

        把 `c_out` 改写成更利于分析的形式，引入两个中间信号：

        - **生成信号（Generate）**：`g = a & b`

            无论当前位的 1-bit 加法器的进位输入（c_in）是什么，如果 g = 1，那么当前位的 1-bit 加法器的 c_out 一定为 1。

        - **传播信号（Propagate）**：`p = a ^ b`

            这一位决定是否把进位输入"传"到输出（用异或而非或，是为了让 `s = p ^ c_in` 同时成立）。

            即如果 p = 1，那么说明 a + b 的和为 1，且没有产生进位。此时 c_out 为多少，主要看 c_in 的值。如果 c_in 为 1，结合 a + b = 1，会产生进位，即 c_out = 1。如果 c_in 为 0，那么结合 a + b = 1，不产生进位，即 c_out = 0.

            也就是说，c_out 一直“跟随” c_in 的值，和 c_in 保持一致。因此，p 的意义就变成了 c_in 是否传播给 c_out.

            如果 p = 0，那么 a + b 要么为 0，要么为 2，c_out 不再跟随 c_in，即“不再传播”。

        注：

        1. 这两个信号的概念是如何分析出来的？似乎可以通过公式看出来。后面有空了再研究下。

            或者可以问问 AI，查查这两个概念的发展历史。

        于是：

        ```verilog
        c_out = g | (p & c_in)
        // 如果 g = 1，那么由当前两个加数已经可以决定必定进位了，不需要再考虑 c_in
        // 如果 g = 0，那么继续考虑 c_in，需要 p 决定 c_out 是否跟随 c_in

        s     = p ^ c_in
        // 这里和前面的“用异或而非或，是为了让 `s = p ^ c_in` 同时成立”，有什么关系？
        ```

        这个形式揭示了关键：**每一位的进位输出依赖于前一位的进位输入**。N 位加法器的第 N 位进位必须等前面所有位的进位依次算完，这就是延迟的根源。

    2. 二、行波进位（Ripple Carry Adder, RCA）

        **设计思想**

        最直接的做法：把 N 个 1 位全加器**串联**，第 i 位的 `c_out` 直接接到第 i+1 位的 `c_in`。

        ```
        c0 = c_in
        c1 = g0 | (p0 & c0)
        c2 = g1 | (p1 & c1)
        c3 = g2 | (p2 & c2)
        ...
        cN = g(N-1) | (p(N-1) & c(N-1))
        ```

        **延迟分析**

        每一位的进位必须等前一位算完，形成一条**串行进位链**。假设一级逻辑（与/或门）延迟为 τ：

        - `c1` 需要 1 级（在 `c0` 已知后）
        - `c2` 需要 2 级
        - `cN` 需要 N 级

        所以**最坏路径延迟与位宽 N 成正比：O(N)**。对于 32 位或 64 位加法器，进位链会成为关键路径，严重限制时钟频率。

        **优点**

        - **面积最小**：每位的逻辑几乎不重复，硬件开销线性增长。
        - **结构简单**：易于理解、易于综合、布线规整。
        - **功耗低**：门数少，翻转活动相对集中。

        **缺点**

        - **延迟随位宽线性增长**，高位宽时性能差。

        **适用场景**

        位宽较小（如 ≤8 位）、对面积/功耗敏感而对速度要求不高的场合；或作为更大设计中的非关键路径组件。

    3. 三、超前进位（Carry Lookahead Adder, CLA）

        **设计思想**

        RCA 的问题在于"进位要一位一位传"。CLA 的核心洞察是：**把进位表达式递归展开，用 `g` 和 `p` 直接表示每一位的进位，从而并行计算所有进位**。

        从 `c1 = g0 | (p0 & c0)` 出发，逐级代入：

        ```
        c1 = g0 | (p0 & c0)
        c2 = g1 | (p1 & c1)
           = g1 | (p1 & g0) | (p1 & p0 & c0)
        c3 = g2 | (p2 & c2)
           = g2 | (p2 & g1) | (p2 & p1 & g0) | (p2 & p1 & p0 & c0)
        c4 = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0) | (p3 & p2 & p1 & p0 & c0)
        ```

        观察规律：**每一位的进位只依赖于 `g`、`p` 和最初的 `c0`，不再依赖前一级的进位输出**。因此所有 `c_i` 可以**同时**计算，延迟从 O(N) 降到 O(1)（在理想的多输入门模型下）。

        **延迟分析**

        以 4 位 CLA 为例：
        - `g`、`p` 在 1 级门延迟内并行算出；
        - 每个 `c_i` 是一个"与-或"表达式，若用与或门实现约 2 级；
        - `s_i = p_i ^ c_i` 再加 1 级。

        总延迟约为 **常数级（与 N 无关，理想情况下 3~4 级）**，远优于 RCA 的 O(N)。

        **代价：扇入爆炸**

        看 `c4` 的表达式：

        ```
        c4 = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0) | (p3 & p2 & p1 & p0 & c0)
        ```

        最后一项是 **5 输入与门**，整个表达式是 **5 输入或门**。位宽越大，与项越长、或项越多，门的**扇入（fan-in）** 急剧增长。实际工艺中门扇入有限（通常 ≤4），所以：

        - **纯 CLA 只适合小位宽（如 4 位一组）**；
        - 大位宽需要**分组级联**，即"块超前进位"（Block CLA / 分层 CLA）。

        **分层超前进位（实际工程做法）**

        把 N 位分成若干 4 位块，每块内部用 CLA 并行算进位，块之间再用一级"块进位逻辑"连接。块内引入**块生成 `G`** 和**块传播 `P`**：

        ```
        G = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0)   // 本块是否产生进位
        P = p3 & p2 & p1 & p0                                        // 本块是否传播进位
        C_out = G | (P & C_in)
        ```

        这样每块内部 O(1)，块间再用一层 CLA，整体延迟为 **O(log N)** 级别，同时扇入可控。这是真实 CPU 加法器（如 Kogge-Stone、Brent-Kung、Han-Carlson 等并行前缀加法器）的基础思想——它们都是 CLA 的推广，用不同的树形结构平衡延迟、面积和布线。

    4. 四、两种思想的本质对比

        | 维度 | 行波进位 RCA | 超前进位 CLA |
        |------|-------------|-------------|
        | 进位计算方式 | 串行，逐位传递 | 并行，用 g/p 直接展开 |
        | 最坏延迟 | O(N) | O(1)（小位宽）/ O(log N)（分层） |
        | 面积 | 最小，线性 | 较大，门数和布线复杂度高 |
        | 扇入 | 每级固定（小） | 随位宽增长，需分组控制 |
        | 功耗 | 较低 | 较高（并行翻转多） |
        | 设计复杂度 | 极简 | 较高（尤其分层/前缀树） |
        | 典型用途 | 小位宽、非关键路径 | 高性能 CPU、大位宽关键路径 |

    5. 五、设计思想的提炼

        1. **RCA 的思想是"复用"**：用同一个 1 位全加器复制 N 次，以面积换简单，代价是延迟线性累积。它体现了"模块化 + 串行组合"的朴素设计。

        2. **CLA 的思想是"展开与并行化"**：识别出进位递推关系 `c_{i+1} = g_i | (p_i & c_i)` 是一个**线性递推**，通过代数展开把它变成只依赖初始量的并行表达式，从而打破串行依赖。这与软件中"把循环展开成并行计算"是同一类思想。

        3. **工程上的折中**：纯 RCA 太慢，纯 CLA 扇入爆炸，实际设计走中间路线——**分组 + 分层**（块内 CLA、块间再 CLA），或用**并行前缀树**（Kogge-Stone 延迟最优但布线密，Brent-Kung 面积优但延迟稍大，Han-Carlson 折中）。这体现了硬件设计中**延迟、面积、功耗、布线**四者的持续权衡。

        4. **g/p 信号的抽象价值**：把"生成"和"传播"分离出来，是理解所有现代加法器（乃至进位保存、前缀加法器）的钥匙。RCA 只用到了 `c_out` 的原始形式，而 CLA 及其衍生结构都建立在对 `g`、`p` 的组合运算之上。

    6. 六、和之前代码的呼应

        - 你之前那段 `c_out = a & b | a & c_in | a & c_in` 之所以错，正是因为漏了 `b & c_in`（即 `p & c_in` 中的一部分），本质上是没把 `g`、`p` 的关系理清。
        - 结构化写法里 4 个 `full_adder1` 串联，就是标准的 **RCA**；若想改成 CLA，需要把 4 位的 `g`、`p` 都提出来，按上面的展开式直接算 `c1~c4`，而不是逐级接 `c_out`。
        - 行为级 `{c_out, s} = a + b + c_in` 则由综合工具自行选择实现（通常综合成 CLA 或前缀加法器），这也是为什么推荐用它——把微架构决策交给工具。

    简言之：**RCA 是"串行传递"的暴力解法，CLA 是"代数展开 + 并行计算"的聪明解法，而真实芯片用的是二者思想融合的分层/前缀结构。**

* 那你给我写个四位全加器的example代码吧

    下面给你一个**4 位全加器**的完整示例，包含三种常见写法（行为级、结构化、门级），以及对应的 C++ 测试平台，方便你用 Verilator 直接跑起来。

    1. 行为级写法（最推荐，简洁且不易出错）

        ```verilog
        // full_adder4.v
        module full_adder4 (
            input  [3:0] a,
            input  [3:0] b,
            input        c_in,
            output [3:0] s,
            output       c_out
        );
            // 把进位和和拼接在一起，一次赋值
            assign {c_out, s} = a + b + c_in;
        endmodule
        ```

        这是最干净的写法。`{c_out, s}` 是 5 位，右边 `a + b + c_in` 在 Verilog 中会自动扩展到足够的位宽（`a`、`b` 各 4 位，`c_in` 1 位，结果需要 5 位），高位自然就是进位。

    2. 结构化写法（用 1 位全加器例化 4 次）

        如果你想展示"行波进位"的结构：

        ```verilog
        // full_adder1.v —— 1 位全加器
        module full_adder1 (
            input  a,
            input  b,
            input  c_in,
            output s,
            output c_out
        );
            assign s     = a ^ b ^ c_in;
            assign c_out = (a & b) | (a & c_in) | (b & c_in);
        endmodule
        ```

        ```verilog
        // full_adder4_structural.v
        module full_adder4_structural (
            input  [3:0] a,
            input  [3:0] b,
            input        c_in,
            output [3:0] s,
            output       c_out
        );
            wire c1, c2, c3;

            full_adder1 fa0 (.a(a[0]), .b(b[0]), .c_in(c_in), .s(s[0]), .c_out(c1));
            full_adder1 fa1 (.a(a[1]), .b(b[1]), .c_in(c1),   .s(s[1]), .c_out(c2));
            full_adder1 fa2 (.a(a[2]), .b(b[2]), .c_in(c2),   .s(s[2]), .c_out(c3));
            full_adder1 fa3 (.a(a[3]), .b(b[3]), .c_in(c3),   .s(s[3]), .c_out(c_out));
        endmodule
        ```

        注意这里 `c_out` 的表达式已经修正为 `(a & b) | (a & c_in) | (b & c_in)`，不再是之前那个重复 `a & c_in` 的错误版本。

    3. 门级写法（仅用于教学，实际不推荐手写）

        ```verilog
        module full_adder4_gate (
            input  [3:0] a,
            input  [3:0] b,
            input        c_in,
            output [3:0] s,
            output       c_out
        );
            wire [4:0] c;
            assign c[0] = c_in;

            genvar i;
            generate
                for (i = 0; i < 4; i = i + 1) begin : fa
                    xor  g1 (s[i], a[i], b[i], c[i]);
                    and  g2 (w1, a[i], b[i]);
                    and  g3 (w2, a[i], c[i]);
                    and  g4 (w3, b[i], c[i]);
                    or   g5 (c[i+1], w1, w2, w3);
                end
            endgenerate

            assign c_out = c[4];
        endmodule
        ```

        （上面的 `w1`/`w2`/`w3` 需要额外声明为 wire，这里为简洁省略了声明，实际写时要补上。）

    4. C++ 测试平台（Verilator 用）

        以行为级模块 `full_adder4` 为例，写一个穷举/随机测试的 `sim_main.cpp`：

        ```cpp
        #include "Vfull_adder4.h"
        #include "verilated.h"
        #include <cstdio>
        #include <cstdlib>

        int main(int argc, char** argv) {
            Verilated::commandArgs(argc, argv);
            Vfull_adder4* dut = new Vfull_adder4;

            int errors = 0;
            // 穷举所有输入组合：a,b 各 16 种，c_in 2 种，共 512 种
            for (int a = 0; a < 16; ++a) {
                for (int b = 0; b < 16; ++b) {
                    for (int cin = 0; cin < 2; ++cin) {
                        dut->a    = a;
                        dut->b    = b;
                        dut->c_in = cin;
                        dut->eval();               // 组合逻辑，eval 一次即可

                        int expected_sum  = a + b + cin;
                        int expected_s    = expected_sum & 0xF;
                        int expected_cout = (expected_sum >> 4) & 0x1;

                        if (dut->s != expected_s || dut->c_out != expected_cout) {
                            printf("FAIL: a=%d b=%d cin=%d -> s=%d cout=%d, expected s=%d cout=%d\n",
                                   a, b, cin, dut->s, dut->c_out, expected_s, expected_cout);
                            ++errors;
                        }
                    }
                }
            }

            if (errors == 0)
                printf("PASS: all 512 cases passed.\n");
            else
                printf("FAILED: %d errors.\n", errors);

            delete dut;
            return errors ? 1 : 0;
        }
        ```

        编译运行（假设文件为 `full_adder4.v` 和 `sim_main.cpp`）：

        ```bash
        verilator --cc full_adder4.v --exe sim_main.cpp --build
        ./obj_dir/Vfull_adder4
        ```

    **关键点回顾**

    1. **行为级 `{c_out, s} = a + b + c_in;`** 是最不容易写错的方式，工具会自动处理位宽扩展。
    2. 如果手写进位逻辑，正确公式是 `c_out = (a & b) | (a & c_in) | (b & c_in)`，**三项缺一不可**，且不要重复。
    3. 纯组合逻辑的 C++ 测试平台**不需要 `while (!Verilated::gotFinish())` 循环**，直接对每组输入 `eval()` 一次并检查输出即可。
    4. 用穷举 512 种输入做自检，是组合逻辑验证最简单可靠的方式。

* 端口声明风格问题（非致命，但不好）

    ```verilog
    module top(a, b, c_in, s, c_out);
        input[3:0] a, b;
        input c_in;
        output[3:0] s;
        output c_out;
    ```

    * 使用了老式端口列表（模块头只列名字，方向在位宽在内部声明）。这种写法在 Verilog-2001 之后已被 ANSI 风格取代，可读性差、易出错。推荐：

        ```verilog
        module top(
            input  [3:0] a, b,
            input        c_in,
            output [3:0] s,
            output       c_out
        );
        ```

    * input[3:0] 中 input 和 [ 之间缺空格，虽然合法但不规范。

* verilog 如何表示只有 1 位的 wire

    在 Verilog 中，表示只有 1 位的 wire 有几种写法，最常用的是直接省略位宽声明。

    ### 1. 省略位宽（最常用）

    ```verilog
    wire a;        // 默认就是 1 位
    wire b;        // 1 位
    ```

    当 `wire` 声明中不写 `[msb:lsb]` 范围时，**默认位宽为 1**。这是最简洁、最惯用的写法，绝大多数代码都这么写。

    ### 2. 显式写成 `[0:0]`

    ```verilog
    wire [0:0] a;  // 显式声明为 1 位
    ```

    这种写法在语法上完全合法，位宽就是 1 位（从 0 到 0）。但它**不推荐**用于普通单比特信号，因为：

    - 冗余，读起来反而让人以为是有意为之的多位向量；
    - 在端口连接、拼接等场景下容易引起误解。

    它通常只在需要强调“这是一个 1 位向量”或做参数化生成时才偶尔出现。

    ### 3. 在端口列表中

    ```verilog
    module m (
        input  wire       a,   // 1 位输入
        output wire       y,   // 1 位输出
        input  wire [7:0] d    // 8 位输入
    );
    ```

    端口声明中同样遵循“省略位宽即 1 位”的规则。

    ### 对比总结

    | 写法 | 位宽 | 推荐度 | 说明 |
    |------|------|--------|------|
    | `wire a;` | 1 | ★★★★★ | 标准写法，默认 1 位 |
    | `wire [0:0] a;` | 1 | ★★ | 合法但冗余，易误解 |
    | `wire [0:0] a` 在参数化中 | 1 | 视情况 | 仅当位宽由参数决定时合理 |

    ### 一个常见误区

    注意不要和 `wire [1:0] a;` 混淆——那是 **2 位**（bit 1 和 bit 0），不是 1 位。`[0:0]` 才是 1 位。

    所以，**表示 1 位 wire 的标准做法就是直接写 `wire a;`，不写位宽范围。**

* $finish 与 $finish() 有什么不同？

    在 Verilog/SystemVerilog 的语境下，`$finish` 和 `$finish()` **本质上是同一个系统任务**，没有功能上的区别。它们的差异仅仅在于**语法形式**，而不是行为。

    ### 语法形式上的区别

    *   **`$finish;`**：这是最传统、最常见的写法。它不传递任何参数，调用时系统会使用默认行为（通常等价于 `$finish(1)`）。
    *   **`$finish();`**：这在语法上是一个带空参数列表的函数调用形式。在 SystemVerilog 中，括号是可选的，所以这种写法与 `$finish;` 完全等价，同样表示不传递任何参数。

    所以，你之前看到的 `$finish` 和 `$finish()`，只是同一种系统任务的不同书写习惯而已。

    ### 真正有区别的是带参数的 `$finish(n)`

    真正会影响仿真行为的是带参数的版本 `$finish(n)`，其中 `n` 控制仿真结束时打印信息的详细程度：

    *   **`$finish(0)`**：结束仿真时**不打印**任何额外信息。
    *   **`$finish(1)`**：打印当前仿真时间和 `$finish` 语句所在的位置（这是**默认**行为，等同于不带参数的 `$finish`）。
    *   **`$finish(2)`**：除了打印时间和位置，还会输出内存和 CPU 时间的使用统计。

    ### 结合你之前的 Verilator 场景

    需要特别注意的是，你之前讨论的 `while (!Verilated::gotFinish())` 循环，其行为与 Verilator 如何处理 `$finish` 有关。在 Verilator 中，`$finish` 的执行并不会立刻让 C++ 程序退出，而是设置一个内部标志（`gotFinish()` 变为真），以便让 `final` 块等清理代码有机会运行。这也是为什么在 C++ 测试平台中，你需要主动检查 `Verilated::gotFinish()` 来决定是否退出循环。

    对于纯组合逻辑，正如之前讨论的，你通常不需要依赖 Verilog 里的 `$finish` 来控制 C++ 流程，因此也无需纠结是写 `$finish` 还是 `$finish()`。

* 一个 sim main cpp 的 example

    ```verilog
    #include "Vdut.h"          // 生成的 DUT 头文件
    #include "verilated.h"
    #include "verilated_vcd_c.h" // 如果需要波形

    int main(int argc, char** argv) {
        Verilated::commandArgs(argc, argv);
        Verilated::traceEverOn(true);
        
        Vdut* top = new Vdut;  // 实例化
        VerilatedVcdC* tfp = new VerilatedVcdC;
        top->trace(tfp, 99);
        tfp->open("wave.vcd");
        
        // 初始化
        top->clk = 0;
        top->reset = 1;
        
        // 仿真循环
        while (!Verilated::gotFinish()) {
            // 生成时钟、驱动输入、eval、dump波形、检查输出
            top->eval(); 
            tfp->dump(main_time);
            // ... 你的激励逻辑 ...
        }
        
        top->final();
        tfp->close();
        delete top;
        return 0;
    }
    ```

* verilator 的 examle 里，make_tracing_c 有逐行注释，讲得比较详细

* 一些可能有用的开源项目

    * OpenTitan

    * ibex、rocket-chip

    * universal-verification-methodology/learn_digital_verilog

* c++ 需要使用 val(), eval_step(), eval_end_step()

* Verilated::commandArgs()

    使用这个函数给 verilator 传参数。

    常见参数：

    * +verilator+debug

        Enable simulation runtime debugging. Equivalent to +verilator+debugi+4.

    * +verilator+debugi+value

    * +verilator+error+limit+value

        Set number of non-fatal errors (e.g. assertion failures) before exiting simulation runtime. Also a␛ects
number of $stop calls needed before exit. Defaults to 1.

    等等。

* 编译 verilator 的 doc 目录时，需要安装的工具

    ```bash
    pip install --user sphinx sphinx-rtd-theme
    sudo apt-get install latexmk texlive-latex-base texlive-latex-extra
    sudo apt install libpod-latex-perl
    ```

* bear 可以识别 verilator 的编译命令，并生成 compile_commands.json

    也可以识别 make 里的 verilator 的编译命令。

    example:

    ```makefile
    all:
        verilator --cc --exe --build -j -Wall sim_main.cpp our.v
    ```

    ```bash
    bear -- make
    ```

    生成的 compile_commands.json 的部分内容：

    ```json
    [
      {
        "arguments": [
          "/usr/bin/g++",
          "-I.",
          "-I/usr/share/verilator/include",
          "-I/usr/share/verilator/include/vltstd",
          "-DVM_COVERAGE=0",
          "-DVM_SC=0",
          "-DVM_TRACE=0",
          "-faligned-new",
          "-fcf-protection=none",
    ...
    ```

* 其他常用的任务

    $bitstoreal, $rtoi, $setup, $finish, $skew, $hold,

    $setuphold, $itor, $period, $time, $printtimescale,

    $timefoemat, $realtime, $width, $realtobits, $recovery

* 值变转储文件

    值变转储文件（VCD）是一个 ascii 文件，它包含仿真时间、范围与信号的定义以及仿真运行过程中信号值的变化等信息。

    转储信号：$dumpvars

    选择 vcd 文件的名称：$dumpfile

    选择转储过程的起点和终点：$dumpon, $dumpoff

    选择生成检测点：$dumpall

    ```verilog
    // 指定 vcd 文件名，若不指定 vcd 文件，则由仿真器指定一默认文件名
    initial
        $dumpfile("myfile.dmp");  // 仿真信息转储到 myfile dmp 文件

    // 转储模块中的信号
    initial
        $dumpvars;  // 没有指定变量范围，把设计中全部信号都转储
    initial
        $dumpvars(1, top);  // 转储模块实例 top 中的信号
        // 1 表示层次的等级，只转储 top 下第一层信号。即转储 top 模块中的变量，而不转储 top 中调用模块中的变量

    initial
        $dumpvars(2, top.m1);  // 转储 top.m1 模块下两层的信号

    initial
        $dumpvars(0, top.m1);  // 0 表示转储 top.m1 模块下面各个层的所有信号

    // 启动和停止转储过程
    initial begin
        $dumpon;
        #100000 $dumpoff;  // 经过 100 000 个仿真时间单位后，停止转储过程
    end

    // 生成一个检查点，转储所有 vcd 变量的现行值
    initial
        $dumpall;
    ```

    $dumpfile 和 $dumpvars 任务通常在仿真开始时指定，$dumpon, $dumpoff 和 $dumpall 任务在仿真过程中控制转储过程。

* 选通显示, strobing, $strobe

    在同一时钟沿内，等其他赋值语句执行完毕后，再显示数据

    ```verilog
    always @(posedge clock)
    begin
        a = b;
        c = d;
    end

    always @(posedge clock)
        $stroke("displaying a = %b, c = %b", a, b);
    ```

    在这个例子中，如果使用 $display，那么 $display 可能在 a = b, c = d 之前执行。

* 文件输出

    * 打开文件

        ```verilog
        <file_handler> = $fopen("<file_name>");
        ```

        $fopen 返回一个多通道描述符（multichannel descriptor）的 32 位值

        多通道描述符中只有一位被设置成 1。

        标准输出的多通道描述符，其最低位被置成 1。标准输出也称为通道 0。标准输出默认是常开的。

        $fopen 的每一次调用打开一个新的通道，并且每次设置新的一位，直到第 30 位。第31 位是保留位。

        example:

        ```verilog
        // 多通道描述符
        integer handle1, handle2, handle3;  // 整型数为 32 位
        // 标准输出是打开的：descriptor = 32'h0000_0001
        initial begin
            handle1 = $fopen("file1.out");  // handle1 = 32'h0000_0002 (bit 1 set 1)
            handle2 = $fopen("file2.out");  // handle2 = 32'h0000_0004 (bit 2 set 1)
            handle3 = $fopen("file3.out");  // handle3 = 32'h0000_0008 (bit 3 set 1)
        end
        ```

    * 写文件

        ```verilog
        $fdisplay(<file_descriptor>, p1, p2, ..., pn);
        $fmonitor(<file_descriptor>, p1, p2, ..., pn);
        $fwrite
        $fstrobe
        ```

        * p1, p2, ..., pn 可以是变量、信号名或带引号的字符串

        * 文件描述符是一个多通道描述符，它可以是一个文件句柄或者多个文件句柄按位的组合

        example:

        ```verilog
        // 所有的句柄已经在前面定义
        
        integer desc1, desc2, desc3;
        initial begin
            desc1 = handle1 | 1;  // desc1 = 32'h0000_0003
            $fdisplay(desc1, "display 1");  // 写到文件 file1.out 和标准输出 stdout

            desc2 = handle2 | handle1;  // desc2 = 32'h0000_0006
            $fdisplay(desc2, "display 2");  // 写到文件 file1.out 和 file2.out

            desc3 = handle3;  // desc3 = 32'h0000_0008
            $display(desc3, "display 3");
        end
        ```

    * 关闭文件

        ```verilog
        $fclose(<文件描述符>);
        ```

    * %m 可以显示当前在哪个模块。不需要参数

        ```verilog
        module M;
            initial
                $display("displaying in %m");
        endmodule
        ```

        ```verilog
        module top;
            M m1();
            M m2();
            M m3();
        endmodule
        ```

        output:

        ```
        Displaying in top.m1
        Displaying in top.m2
        Displaying in top.m3
        ```

* 字符串输出的格式控制

    * 使用 %0 可以使用最少的位数显示表达式的值

        ```verilog
        module printval
            reg[11:0] rl;
            initial begin
                rl = 10;
                $display("printing with maximum size=%d=%h, rl, rl");
                $display("printing with minimum size=%0d=%0h", rl, rl);
            end
        endmodule
        ```

        output:

        ```
        Printing with maximum size=10=00a;
        Printing with minimum size=10=a;
        ```

    * 如果输出中表达式的值包含 x 或 z，那么规则如下：

        * 在输出格式为十进制的情况下

            1. 如果表达式值的所有位均为不定值，则输出结果为小写的 x

            2. 如果表达式值的所有位均为高阻值，则输出结果为小写的 z

            3. 如果表达式值的部分位为不定值，则输出结果为大写的 X

            4. 如果表达式值的部分位为高阻值，则输出结果为大写的 Z

        * 在输出格式为十六进制和八进制的情况下

            1. 每 4 位二进制数代表 1 位十六进制数, 每 3 位二进制数代表 1 位八进制数

            2. 如果表达式值相对应的某进制数的所有位均为不定值，则该位进制数的输出的结果为小写的 x

            3. 同理，如果所有位都是高阻值，则结果为小宝的 z

            4. 部分为不定值，则为大写的 X

            5. 部分为高阻值，则为大写的 Z

        * 对于二进制，输出的第一位都为 0, 1, x, z

        examples:

        ```verilog
        $display("%d", 1'bx);
        $display("%h", 14'bx0_1010);
        $display("%h %o", 12'b001x_xx10_1x01, 12'b001_xxx_101_x01);
        ```

* 常用的系统任务

    * $display, $write

        ```verilog
        $display (p1, p2, ..., pn);
        $write(p1, p2, ..., pn);
        ```

        输出信息。p1 是 format.

        $display 自动地在输出后进行换行。$write 则可以在一行里输出多个信息。

        format:

        | 输出格式 | 说明 |
        | - | - |
        | %h 或 %H | 以十六进制数的形式输出 |
        | %d 或 %D | 以十进制数的形式输出 |
        | %o 或 %O | 以八进制的形式输出 |
        | %b 或 %B | 以二进制数的形式输出 |
        | %c 或 %C | 以 ascii 码字符的形式输出 |
        | %v 或 %V | 输出网络型数据信号强度 |
        | %m 或 %M | 输出等级层次的名字 |
        | %s 或 %S | 以字符串的形式输出 |
        | %t 或 %T | 以当前的时间格式输出 |
        | %e 或 %E | 以指数的形式输出实型数 |
        | %f 或 %F | 以十进制数的形式输出实型数 |
        | %g 或 %G | 以指数或十进制数的形式输出实型数。无论何种格式都以较短的结果输出 |

        转码序列：

        | 转码序列 | 功能 |
        | - | - |
        | \n | 换行 |
        | \t | 横向跳格（即跳到下一个输出区） |
        | \\ | 反斜杠字符 \ |
        | \" | 双引号字符 " |
        | \0 | 1 到 3 位八进制数代表的字符 |
        | %% | 百分号 % |

        * example 1

            ```verilog
            module disp;
                reg[31:0] rval;
                pulldown(pd);
                initial begin
                    rval = 101;
                    $display("rval = %h hex %d decimal", rval, rval);
                    $display("rval = %o otal %b binary", rval, rval);
                    $display("rval has %c ascii character value", rval);
                    $display("pd strength value is %v", pd);
                    $display("current scope is %m");
                    $display("%s is ascii value for 101", 101);
                    $display("simulation time is %t", $time);
                end
            endmodule
            ```

            在位前面补 0：

            ```verilog
            $display("d = %0h a = %0h", data, addr);
            ```

        * example 2

            ```verilog
            module printval;
                reg[11:0] r1;
                initial begin
                    r1 = 10;
                    $display("Printing with maximum size=%d=%h", 41, r1);
                    $display("Printing with minimum size=%0d%0h", r1, r1);
                end
            endmodule
            ```

* 带符号函数

    带符号函数的返回值可以作为带符号数进行运算。

    ```verilog
    module top;
        ...
        function signed[63:0] compute_signed(input[63:0] vector);
            ...
        endfunction

        if (compute_signed(vector) < -3) begin
            ...
        end
    endmodule
    ```

* 常量函数

    常量函数用于计算一些复杂的表达式，可以用于代替常量。

    ```verilog
    module ram(...);
        parameter RAM_DEPTH = 256;
        input [clog2(RAM_DEPTH) - 1 : 0] addr_bus;
        ...
        function integer clogb2(input integer depth);
        begin
            for (clogb2 = 0; depth > 0; clogb2 = clogb2 + 1)
                depth = depth >> 1;
            end
        endfunction
        ...
    endmodule
    ```

* 自动（递归）函数

    如果使用了关键字 automatic，那么仿真器每一次函数调用动态地分配新的地址空间，每一个函数调用对各自的地址空间进行操作。

    自动函数中声明的局部变量不能通过层次名进行访问。

    ```verilog
    module top;
        ...
        function automatic integer factorial;
        input [31:0]oper;
        integer i;
        begin
            if (operand >= 2)
                factorial = factorial(oper - 1) * oper;  // 递归调用
            else
                factorial = 1;
        end
    endfunction


    // 调用该函数
    integer result;
    initial begin
        result = factorial(4);  // 调用 4 的阶乘
        $display("factorial of 4 is % 0d ", result);  // 显示 24
    end
    ...
    endmodule
    ```

* 与函数相关的 example

    * 奇偶检验位计算

        ```verilog
        module parity;
            reg[31:0] addr;
            reg parity;

            initial begin
                addr = 32'h3456_789a;
                # 10 addr = 32'hc4c6_78ff;
                # 10 addr = 32'hff56_ff9a;
                # 10 addr = 32'h3faa_aaaa;
            end

            always @(addr) begin
                parity = calc_parity(addr);
                $display("Parity calculated = %b", calc_parity(addr));
            end

            // 定义偶校验计算函数
            function calc_parity;
                input[31:0] address;
                begin
                    calc_parity = ^address;  // 返回所有地址位的异或值
                end
            endfunction
        endmodule
        ```

    * 使用 C 风格进行变量声明的函数定义

        ```verilog
        function calc_parity(input[31:0] address);
        begin
            calc_parity = ^address;
        end
        endfunction
        ```

    * 左 / 右移位寄存器

        ```verilog
        module shifter;

            'define LEFT_SHIFT 1'b0
            'define RIGHT_SHIFT 1'b1
            reg[31:0] addr, left_addr, right_addr;
            reg control;

            // 每当新地址出现时就计算右移位和左移位的值
            always @(addr) begin
                left_addr = shift(addr, 'LEFT_SHIFT);
                right_addr = shift(addr, 'RIGHT_SHIFT);
            end

            // 定义移位函数，其输出是一个 32 位的值
            function[31:0] shift;
                input[31:0] address;
                input control;
                begin
                    // 根据控制信号适当地设置输出值
                    shift = (control == 'LEFT_SHIFT) ? (address << 1) : (address >> 1);
                end
            endfunction

        endmodule
        ```

* function

    syntax:

    ```verilog
    function <return_val_type_or_range> (function_name);
        <port_description_statements>
        <var_type_descrip_statements>
            begin
                <statement>
                ...
            end
    endfunction
    ```

    如果不写返回值类型，那么默认返回值为一位寄存器类型:

    ```verilog
    function[7:0] getbyte;
        input[15:0] address;
        begin
            <statements>  // 从地址字节中提取低字节的程序
            getbyte = result_expression;  // 赋值返回值
        end
    endfunction
    ```

    函数的使用规则：

    1. 函数的定义不能包含任何的时间控制语句，即任何用 #, @, 或 wait 来标识的语句

    2. 函数不能启动任务

    3. 定义函数时至少要有一个输入参量

    4. 在函数的定义中必须有一条赋值语句给函数中的一个内部变量赋以函数的结果值，该内部变量具有和函数名相同的名字

    * example: 阶乘函数的定义和调用

        ```verilog
        module tryfact;
            function[31:0] factorial;
                input[3:0] operand;
                reg[3:0] index;
                begin
                    factorial = 1;  // 0 的阶乘为 1，1 的阶乘为 1
                    for (index = 2; index <= operand; index = index + 1)
                        factorial = index * factorial;
                end
            endfunction

            // 函数的测试
            reg[31:0] result;
            reg[3:0] n;
            
            initial begin
                result = 1;
                for (n = 2; n <= 9; n = n + 1) begin
                    $display("Partial result n = %d result = %d", n, result);
                    result = n * factorial(n) / ((n*2) + 1);
                end
                $display("Finalresult = %d", result);
            end
        endmodule
        ```

* task

    task definition:

    ```verilog
    task <task_name>:
        <port, data type declaration>
        <statements;>
    endtask
    ```

    task invoke:

    ```verilog
    <task_name>(port_1, port_2, ..., port_n);
    ```

    example:

    ```verilog
    task my_task;
        input a, b;
        inout c;
        output d, e;
        ...
        <statement>  // 执行任务工作相应的语句
        ...
        c = foo1;  // 赋初始值
        d = foo2;  // 对任务的输出变量赋值
        e = foo3;
    endtask

    my_task(v, w, x, y, z);
    ```

    * example 2: 描述红绿黄交通灯行为的 verilog 模块

        ```verilog
        module traffic_lights;
            reg clock, red, amber, green;
            parameter on = 1, off = 0, red_tics = 350,
                amber_tics = 30, green_tics = 200;
            // 交通灯初始化
            initial red = off;
            initial amber = off;
            initial green = off;

            // 交通灯控制时序
            always begin
                red = on;  // 开红灯
                light(red, red_tics);  // 调用等待任务
                green = on;  // 开绿灯
                light(green, green_tics);  // 等待
                amber = on;  // 开黄灯
                light(amber, amber_tics);  // 等待
            end

            // 定义交通灯开启时间的任务
            task light;
                output color;
                input[31:0] tics;
                
                begin
                    repeat(tics)
                        @(posedge clock)  // 等待 tics 个时钟的上升沿
                    color = off;  // 关灯
                end
            endtask

            // 产生时钟脉冲的 always 块
            always begin
                # 100 clock = 0;
                # 100 clock = 1;
            end
        endmodule
        ```

        该模块只是一个行为模块，不能综合成电路网表。

* task 与 function 的不同

    1. 函数只能与主模块共用同一个仿真时间单位，而任务可以定义自己的仿真时间单位

    2. 函数不能启动任务，而任务能启动其他任务和函数

    3. 函数至少要有一个输入变量，而任务可以没有或有多个任务类型的变量

    4. 函数返回一个值，而任务不返回值

    函数的目的是通过返回一个值来响应输入信号的值，任务则可以支持多种目的，能计算多个结果值，这些结果值只能通过被调用的任务的输出或总线端口送出。

* always or

    由关键词 or 连接的多个事件名或者信号名组成的列表称为敏感列表。or 被用来表示这种关系，或者使用 , 来代替。

    1. example 1:

        有异步复位的电平敏感锁存器。

        ```verilog
        always @ (reset or clock or d)
        begin
            if (reset)  // 若 reset 信号为高，把 q 置零
                q = 1'b0;
            else if (clock)  // 若 clock 信号为高，锁存输入信号 d
                q = d;
        end
        ```

    2. example 2:

        使用逗号的敏感列表，等价于 or

        ```verilog
        always @ (reset, clock, d)
        begin
            if (reset)
                q = 1'b0;
            else if (clock)
                q = d;
        end
        ```

        用 reset 异步下降沿复位，clock 正跳变沿触发的 D 寄存器。

        ```verilog
        always @ (posedge clk, negedge reset)
        if (!reset)
            q <= 0;
        else
            q <=d;
        ```

    * `always @#`, `always @(*)` 表示对其后面语句块中所有输入变量的变化都是敏感的

        ```verilog
        always @ (a or b or c or d or e or f or g or h or p or m)
        begin
            out1 = a ? b + c : d + e;
            out2 = f ? g + h : p + m;
        end


        always @ (*)
        begin
            out1 = a ? b + c : d + e;
            out2 = f ? g + h : p + m;
        end
        ```

    * 电平敏感时序控制

        后面的语句和语句块需要等待某个条件为真才能执行。

        ```verilog
        always
            wait (count_enable)  # 20 count = count + 1;
        ```

        若`count_enable`为 0，则不执行后面的语句。否则在 20 个时间单位后执行`count = count + 1;`。如果`count_enable`始终为 1，那么`count`每过 20 个时间单位加 1

* 结构说明语句

    一个模块中可以有多个 initial 和 always 块。每个 initial 和 always 语句在仿真的一开始同时立即开始执行。

    * initial

        ```verilog
        initial
        begin
            statement_1;
            statement_2;
            ...
            statement_n;
        end
        ```

        initial 语句不占用仿真时间，即在 0 ns 内，完成初始化工作。

        example:

        ```verilog
        initial
        begin
            inputs = 'b000000;  // 初始时刻为 0
            # 10 inputs = 'b011001;
            # 10 inputs = 'b011011;
            # 10 inputs = 'b011000;
            # 10 inputs = 'b001000;
        end
        ```

    * always

        ```verilog
        always  # half_period areg = ~areg;
        ```

        生成一个周期为 peroid (= 2 * half_period) 的无限延续的信号波形。常用这种方法来描述时钟信号，并作为激励信号来测试所设计的电路。

        ```verilog
        reg[7:0] counter;
        reg tick;
        always @(posedge areg)
        begin
            tick = ~tick;
            counter = counter + 1;
        end
        ```

        ```verilog
        always @(posedge clock or posedge reset)
        begin
            ...
        end

        always @(a or b or c)  // 电平触发，只要 a, b, c 发生变化，就执行 always 块
        begin
            ...
        end
        ```

* examples

    * 四选一多路选择器

        ```verilog
        module mux4_to_1(out, i0, i1, i2, i3, s1, s0);
        output out;
        input i0, i1, i2, i3;
        input s1, s0;
        reg out;

        always @(s1 or s0 or i0 or i1 or i2 or i3)
        begin
            case({s1, s0})
            2'b00: out = i0;
            2'b01: out = i1;
            2'b10: out = i2;
            2'b11: out = i3;
            default: out = 1'bx;
            endcase
        end
        endmodule
        ```

    * 四位计数器

        四位脉动进位计数器。

        ```verilog
        module counter(Q, clock, clear);

        output[3:0] Q;
        input clock, clear;
        reg[3:0] Q;

        always @ (posedge clear or negedge clock)
        begin
            if (clear)
                Q <= 4'd0;  // 为了生成诸如触发器一类的时序逻辑，建议使用非阻塞赋值
            else
                Q <= Q + 1;  // Q 是一个四位的寄存器，计数超过 15 后又会归零，因此模 16 没有必要
        end
        endmodule
        ```

* 详细讲讲 verilog 中关于参数和参数覆盖的语法

    你之前看到的 `# ( N )` 是 Verilog 参数覆盖机制中的一种写法。Verilog 的参数系统主要涉及两个关键字：`parameter` 和 `localparam`，以及几种在实例化时修改参数值的方法。

    **核心概念：parameter 与 localparam 的区别**

    在定义模块时，你需要根据用途选择使用 `parameter` 还是 `localparam`。

    *   **`parameter`**：用于定义**可配置**的常量。它的主要目的是作为模块对外的一个“接口”，允许在实例化该模块时，从外部修改这个值，从而让同一个模块设计能适配不同的位宽、深度或延迟需求。例如，你之前代码中的 `parameter N = 4;` 就是一个典型的可覆盖参数。
    *   **`localparam`**：用于定义**模块内部**的常量。它**不能**在模块实例化时被覆盖。`localparam` 通常用于定义那些由 `parameter` 计算得出的派生值，或者纯粹的内部状态编码，以保护这些值不被外部意外修改。例如，在一个 RAM 模块中，如果 `WORD_SIZE` 和 `ADDR_SIZE` 是参数，那么 `MEM_SIZE` 就应该用 `localparam` 定义，因为它是由前两者算出来的，不应该让用户单独指定一个可能矛盾的值。

    ### 参数覆盖的三种方式

    在实例化一个带有 `parameter` 的模块时，你有三种方式来覆盖它的默认值。**推荐使用第二种（按名称覆盖）**。

    #### 1. 按位置覆盖
    这种方式要求你严格按照子模块内部定义参数的**顺序**来传值。

    *   **语法**：`模块名 #(值1, 值2, ...) 实例名 (端口连接);`
    *   **示例**：假设 `adder_cla` 模块内部定义参数的顺序是 `N`，那么 `adder_cla #(N) adder3(...)` 就是把外部 `N` 的值传给 `adder_cla` 的第一个参数。
    *   **风险**：如果子模块的设计者后来调整了参数的声明顺序，或者增加了一个新参数，你原有的实例化代码就会在不知不觉中把值传错位置，导致功能错误。

    #### 2. 按名称覆盖（推荐）
    这种方式通过指定参数名来传值，清晰且安全。

    *   **语法**：`模块名 #(.参数名1(值1), .参数名2(值2)) 实例名 (端口连接);`
    *   **示例**：`adder_cla #(.N(N)) adder3(...)`。这明确告诉编译器：把 `adder_cla` 模块里名为 `N` 的参数，覆盖为上层模块当前的 `N` 值。
    *   **优势**：代码可读性极强，且完全不受子模块参数声明顺序变化的影响。即使子模块新增了参数，你未指定的参数会继续使用其默认值，不会出错。

    #### 3. 使用 `defparam` 语句（已过时，不推荐）
    这是一种在模块外部通过层次化路径直接修改参数值的旧方法。

    *   **语法**：`defparam 实例名.参数名 = 新值;`
    *   **状态**：该语法在 Verilog-2001 中被引入 `#()` 方式后，已被标记为**不推荐使用（deprecated）**，在新的 IEEE 标准（如 SystemVerilog）中甚至可能被移除。
    *   **缺点**：它会让参数的修改位置与实例化位置分离，代码难以阅读和调试。此外，部分综合工具对它的支持也不够好。

    ### 总结与建议

    对于你之前看到的代码，将 `adder_cla # ( N )` 改为 **`adder_cla # (.N(N))`** 会是更稳健的工程实践。它能明确表达你的意图，并且在未来维护时更不容易出错。

* 条件生成语句

    * example: 根据条件生成参数化乘法器

        ```verilog
        module multiplier (product, a0, a1);
        parameter a0_width = 8;
        parameter a1_width = 8;

        localparam product_width = a0_width + a1_width;

        output[product_width - 1 : 0] product;
        input[a0_width - 1 : 0] a0;
        input[a1_width - 1 : 0] a1;

        generate
            if (a0_width < 8 || a1_width < 8)
                cal_multiplier  # (a0_width, a1_width) m0 (product, a0, a1);
            else
                tree_multiplier  # (a0_width, a1_width) m0 (product, a0, a1);
        endgenerate
        endmodule
        ```

    * 使用 case 生成语句实现 N 位加法器

        ```verilog
        module adder(co, sum, a0, a1, ci);
        parameter N = 4;

        output[N-1:0] sum;
        output co;
        input[N-1:0] a0, a1;
        input ci;

        generate
            case(N)
                1: adder_1bit adder1(co, sum, a0, a1, ci);  // 1 位加法器
                2: adder_2bit adder2(co, sum, a0, a1, ci);  // 2 位加法器
                default: adder_cla  # ( N ) adder3(co, sum, a0, a1, ci);  // 默认选用位宽为 N 的超前进位加法器
            endcase
        endgenerate
        endmodule
        ```

* 生成块

    1. example: 生成两条 N 位总结变量进行按位异或

        ```verilog
        module bitwise_xor(out, i0, i1);
        parameter N = 32;

        output [N-1:0] out;
        input [N-1:0] i0, i1;

        // 临时变量，只能用于生成块
        genvar j;

        // 生成许多门级模块
        generate
            for (j = 0; j < N; j = j + 1)
            begin: xor_loop
                xor gl (out[j], i0[j], i1[j]);
            end
        endgenerate
        ```

        注：

        1. 本质是等价于实现：

            `assign out = i0 ^ i1;`

        2. 如果需要访问具体的异或门实例，可以这样：`xor_loop[j].gl`

        3. always 等价实现

            ```verilog
            reg[N-1:0] out;
            generate
                for (j = 0; j < N; j = j + 1)
                begin: bit
                    always @ (i0[j] or i1[j])
                        out[j] = i0[j] ^ i1[j];
                end
            endgenerate
            ```

    2. example 2: 脉动加法器

        ```verilog
        module ripple_adder(co, sum, a0, a1, ci);
        parameter N = 4;

        output[N-1:0] sum;
        output co;
        input[N-1:0] a0, a1;
        input ci;

        wire[N-1:0] carry;
        assign carry[0] = ci;

        genvar i;

        generate
            for (i = 0; i < N; i = i + 1) begin : r_loop
                wire t1, t2, t3;
                xor g1 (t1, a0[i], a1[i]);
                xor g2 (sum[i], t1, carry[i]);
                and g3 (t2, a0[i], a1[i]);
                and g4 (t3, t1, carry[i]);
                or g5 (carry[i+1], t2, t3);
            end
        endgenerate

        assign co = carry[N];
        endmodule
        ```

* 命令块 named block

    特点：

    1. 命令块可以声明局部变量

    2. 命名块是设计层次的一部分，命名块中声明的变量可以通过层次名引用进行访问

    3. 命令块可以被禁用，例如停止其执行

    example:

    1. 命名块的层次名引用

        ```verilog
        module top;
        initial
        begin: block1
            integer i;  // 静态局部变量，top.block1.i
            ...
        end

        initial
        fork: block2
            reg i;  // top.block2.i
            ...
        join
        ```

    2. disable 中止命名块执行

        和 C 语言的 break 比较像。

        example:

        ```verilog
        reg [15:0] flag;
        integer i;

        initial
        begin
            flag = 16'b0010_0000_0000_0000;
            i = 0;
            begin: block1
                while (i < 16)
                begin
                    if (flag[i])
                    begin
                        $display("encountered a true bit at element number %d", i);
                        disable block1;  // 发现 1 后，提前中止
                    end
                    i = i + 1;
                end
            end
        end
        ```

* case

    ```verilog
    case(expr)
        expr_1: statement_1;
        expr_2: statement_2;
        ...
        default: statement_n;
    endcase
    ```

    casex, casez 是一样的用法。

    case 语句的所有表达式值的位宽必须相等。

    真值表：

    * case

        | case | 0 | 1 | x | z |
        | - | - | - | - | - |
        | 0 | 1 | 0 | 0 | 0 |
        | 1 | 0 | 1 | 0 | 0 |
        | x | 0 | 0 | 1 | 0 |
        | z | 0 | 0 | 0 | 1 |

    * casez

        | casez | 0 | 1 | x | z |
        | - | - | - | - | - |
        | 0 | 1 | 0 | 0 | 1 |
        | 1 | 0 | 1 | 0 | 1 |
        | x | 0 | 0 | 1 | 1 |
        | z | 1 | 1 | 1 | 1 |

    * casex

        | casex | 0 | 1 | x | z |
        | - | - | - | - | - |
        | 0 | 1 | 0 | 1 | 1 |
        | 1 | 0 | 1 | 1 | 1 |
        | x | 1 | 1 | 1 | 1 |
        | z | 1 | 1 | 1 | 1 |

* 在 always 块内，如果 if / case 下某个变量没有赋值，那么这个变量将保持原值。此时会生成一个锁存器。

    case 语句中缺少 default 也通常会生成锁存器。

* example: 使用 case 语句实现四选一多路选择器

    ```verilog
    module mux4_to_1 (out, i0, i1, i2, i3, s1, s0);
        output out;
        input i0, i1, i2, i3;
        input s1, s0;

        reg out;

        always @(s1 or s0 or i0 or i1 or i2 or i3)
        begin
            case ({s1, s0})
                2'b00: out = i0;
                2'b01: out = i1;
                2'b10: out = i2;
                2'b11: out = i3;
                default: out = 1'bx;
            endcase
        end
    endmodule
    ```

* 循环语句

    * forever

        连续执行语句。

        ```verilog
        forever statement;

        forever
        begin
            statements;
        end
        ```

        forever 语句常用于产生周期性的波形，用来作为仿真测试信号。

        forever 语句必须写在 initial 块中。

    * repeat

        连续执行一条语句 n 次。

        ```verilog
        repeat (expr) statement;
        repeat (expr)
        begin
            statements;
        end
        ```

        example 乘法器：

        ```verilog
        parameter size=8, longsize=16;
        reg [size:1] opa, opb;
        reg [longsize:1] result;

        begin: mult
            reg [longsize:1] shift_opa, shift_opb;
            shift_opa = opa;
            shift_opb = opb;
            result = 0;
            repeat(size)
            begin
                if (shift_opb[1])
                    result = result + shift_opa;
                shift_opa = shift_opa << 1;
                shift_opb = shift_opb >> 1;
            end
        end
        ```

    * while

        ```verilog
        while (expr) statement;
        ```

        执行一条语句直到某个条件不满足。

        example: 使用 while 语句对 rega　这个八位二进制数中值为　１　的位进行计数。

        ```verilog
        begin: conut1s
            reg [7:0] tempreg;
            count = 0;
            tempreg = rega;
            while (tempreg)
                begin
                    if (tempreg[0]) count = count + 1;
                    tempreg = tempreg >> 1;
                end
            end
            ...
        end
        ```

    * for

        ```verilog
        for (expr_1; expr_2; expr_3) statement;
        ```

        * example 1: 初始化 memory

            ```verilog
            begin: init_mem
                reg[7:0] tempi;
                    for (tempi = 0; tempi < memsize; tempi = tempi + 1)
                        memory[tempi] = 0;
            end
            ```

        * example 2: 用 for 语句实现乘法器

            ```verilog
            parameter size = 8, longsize = 16;
            reg [size:1] opa, opb;
            reg [longsize:1] result;

            begin: mult
                integer bindex;
                result = 0;
                for (bindex = 1; bindex <= size; bindex = bindex + 1)
                    if (opb[bindex])
                        result = result + (opa << (bindex - 1));
            end
            ```

        * example 3: 统计八位寄存器中 1 的个数

            ```verilog
            begin: count1s
                reg[7:0] tempreg;
                count = 0;
                for (tempreg = rega, tempreg; tempreg = tempreg >> 1)
                    if (tempreg[0])
                        count = count + 1;
            end
            ```

* initial 语句

    * example 1

        使用 begin ... end ...，无延迟顺序执行。

        ```verilog
        reg x, y;
        reg [1:0] z, w;

        initial
        begin
            x = 1'b0;
            y = 1'b1;
            z = {x, y};
            w = {y, x};
        end
        ```

    * example 2

        使用 begin ... end ...，带延迟顺序执行

        ```verilog
        reg x, y;
        reg [1:0] z, w;
        initial
        begin
            x = 1'b0;
            # 5 y = 1'b1;
            # 10 z = {x, y};
            # 20 w = {y, z};
        end
        ```

    * example 3
    
        使用 fork ... join ...，带延迟并行执行。

        ```verilog
        reg x, y;
        reg [1:0] z, w;

        initial
        fork
            x = 1'b0;
            # 5 y = 1'b1;
            # 10 z = {x, y};
            # 20 w = {y, x};
        join
        ```

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

        - 两个 1 位二进制数相加
        - 输出：和(S)与进位(C)
        - 逻辑表达式：
          - S = A ⊕ B  // 异或
          - C = A & B

    2. 全加器（Full Adder）

        - 三个 1 位二进制数相加（两个加数 + 进位输入）
        - 输出：和 (S) 与进位输出 (Cout)
        - 逻辑表达式：
          - S = A ⊕  B ⊕ Cin
          - Cout = (A & B) | (A & Cin) | (B & Cin)

    3. 多位加法器设计方法

        - 行波进位加法器：全加器级联，低位进位连接到高位进位输入
        - 超前进位加法器：并行计算进位，减少延迟

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

    4. 4 位超前进位加法器

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

### FIFO

### adder 加法器

* 报错：Signal unoptimizable: Feedback to clock or circular logic: 'top.c_outs'

    这个报错的意思是：Verilator 在 c_outs 信号上检测到了组合逻辑环路（或反馈），导致该信号无法被优化。Verilator 默认把这个警告当作错误处理，所以编译直接失败。

    多维数组或 generate 块：信号经过数组或 generate 块传递时，Verilator 可能无法正确区分不同位或不同实例，误报环路。

    * 用 --report-unoptflat：带这个选项跑 Verilator，它会给出详细的依赖图和可能的信号拆分建议，帮你定位问题。

    * -Wno-UNOPTFLAT

        因为只是警告，所以可以使用这个编译选项强行继续编译

    * wire c_outs[WIDTH] /* verilator split_var */;

    * 正常的写法 (ripple-carry)

        ```verilog
        wire carry [0:WIDTH];

        assign carry[0] = c_in;
        assign c_out = carry[WIDTH];

        generate
            for (genvar i = 0; i < WIDTH; i = i + 1) begin : gen_full_adder
                adder_1_bit u_adder (
                    .a    (a[i]),
                    .b    (b[i]),
                    .c_in (carry[i]),
                    .s    (s[i]),
                    .c_out(carry[i+1])
                );
            end
        endgenerate
        ```

    注：

    1. 忽略警告后，可以继续编译运行，可以正常跑通。说明 verilog 允许 generate for 中写 if

        前面的警告可能是因为 verilator 无法识别 generate 中的 i 的展开，所以以为使用了同样的 wire 变量，从而造成循环依赖

    2. 无论是使用`wire [3:0] c_outs;`还是`wire c_outs [3];`，都会报循环依赖的警告

        写成分立变量，则不会再报循环依赖：

        ```verilog
        wire c_out_0, c_out_1, c_out_2; 

        genvar i;
        generate
            for (i = 0; i < 4; i = i + 1) begin : gen_addr
                if (i == 0) begin
                    adder_1_bit ad_0(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_in),
                        .s(s[i]),
                        .c_out(c_out_0)
                    );
                end
                else if (i == 1) begin
                    adder_1_bit ad_1(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_out_0),
                        .s(s[i]),
                        .c_out(c_out_1)
                    );
                end
                else if (i == 2) begin
                    adder_1_bit ad(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_out_1),
                        .s(s[i]),
                        .c_out(c_out_2)
                    );
                end
                else begin
                    adder_1_bit ad_3(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_out_2),
                        .s(s[i]),
                        .c_out(c_out)
                    );
                end
            end
        endgenerate
        ```

        说明 verilator 4.038 是根据变量名进行推断是否循环依赖，并没有建立真正的依赖图。这是 verilator 的推测能力不够，不是 verilog的 语法限制，也不是 generate for 里不能写 if.

    3. 破解循环依赖，可以写成`begin ... loop ... end`三个块的形式

        ```verillog
        wire c_outs[WIDTH];

        adder_1_bit ad_0(
            .a(a[0]),
            .b(b[0]),
            .c_in(c_in),
            .s(s[0]),
            .c_out(c_outs[0])
        );

        genvar i;
        generate
            for (i = 1; i < WIDTH - 1; i = i + 1) begin : gen_addr
                adder_1_bit ad(
                    .a(a[i]),
                    .b(b[i]),
                    .c_in(c_outs[i-1]),
                    .s(s[i]),
                    .c_out(c_outs[i])
                );
            end
        endgenerate

        adder_1_bit ad_3(
            .a(a[WIDTH - 1]),
            .b(b[WIDTH - 1]),
            .c_in(c_outs[WIDTH - 2]),
            .s(s[WIDTH - 1]),
            .c_out(c_out)
        );
        ```

        即把初始化过程和结尾过程单独处理，中间的循环则使用 generate for 的形式

    4. generate for 里的变量也可以写成这种形式

        ```verilog
        for (genvar i = 1; i < WIDTH 
        ```

    5. 下面这种写法仍有循环依赖, 但是输出结果正常。说明 for 并不是一次性展开的，而是对 i 进行遍历后，进入不同的 if 分支

        ```verilog
        wire [2:0] c_outs;

        genvar i;
        generate
            for (i = 0; i < 4; i = i + 1) begin : gen_addr
                if (i == 0) begin
                    adder_1_bit ad_0(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_in),
                        .s(s[i]),
                        .c_out(c_outs[i])
                    );
                end
                else if (i <= 2) begin
                    adder_1_bit ad(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_outs[i-1]),
                        .s(s[i]),
                        .c_out(c_outs[i])
                    );
                end
                else begin
                    adder_1_bit ad_3(
                        .a(a[i]),
                        .b(b[i]),
                        .c_in(c_outs[i-1]),
                        .s(s[i]),
                        .c_out(c_out)
                    );
                end
            end
        endgenerate
        ```

* 使用 generate for 实现多位全加器

    ```verilog
    full_adder1 fa0 (.a(a[0]), .b(b[0]), .c_in(c_in), .s(s[0]), .c_out(c1));
    full_adder1 fa1 (.a(a[1]), .b(b[1]), .c_in(c1),   .s(s[1]), .c_out(c2));
    full_adder1 fa2 (.a(a[2]), .b(b[2]), .c_in(c2),   .s(s[2]), .c_out(c3));
    full_adder1 fa3 (.a(a[3]), .b(b[3]), .c_in(c3),   .s(s[3]), .c_out(c_out));
    ```

    可以用 **generate for** 来消除这种重复写法。注意：模块实例化不能用普通的 `for` 循环（那是过程块里用的），要用 `generate` 的 `for`。

    ## 写法一：generate for（推荐）

    ```verilog
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_fa
            full_adder1 fa (
                .a     (a[i]),
                .b     (b[i]),
                .c_in  (i == 0 ? c_in : carry[i-1]),
                .s     (s[i]),
                .c_out (carry[i])
            );
        end
    endgenerate

    wire [3:0] carry;  // 内部进位
    assign c_out = carry[3];
    ```

    要点：
    - `genvar i;` 必须声明，且通常放在 `generate` 外面。
    - `begin : gen_fa` 给生成块起名字，方便层次化引用（如 `gen_fa[0].fa`）。
    - `i == 0 ? c_in : carry[i-1]` 这种索引在 genvar 常量下是合法的，综合工具能展开成静态连接。

    ## 写法二：把进位链也用数组统一处理

    如果不想在端口里写三目运算，可以先把 `c_in` 并入一个进位向量：

    ```verilog
    wire [4:0] c;          // c[0]=c_in, c[4]=c_out
    assign c[0] = c_in;

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_fa
            full_adder1 fa (
                .a     (a[i]),
                .b     (b[i]),
                .c_in  (c[i]),
                .s     (s[i]),
                .c_out (c[i+1])
            );
        end
    endgenerate

    assign c_out = c[4];
    ```

    这种写法最干净：`c[i]` 作为第 i 位的输入进位，`c[i+1]` 作为输出进位，自动串成行波进位链。

    ## 补充说明

    - 如果只是想减少代码量、且位数固定很小，也可以直接写 `full_adder1 fa[3:0](...)` 这种**实例数组**，但端口连接会变成向量形式，对进位这种"逐位不同"的连接反而不方便，所以进位链场景还是 generate for 最合适。
    - 综合工具（Vivado/Quartus/Design Compiler）都能正确展开 generate for，展开后和你手写的 4 个实例等价。
    - 注意 `genvar` 循环变量在循环体内是常量，所以 `a[i]`、`carry[i-1]` 都是合法的静态位选。

### verilator

* verilator 不是一个传统的仿真器，而是把 verilog / systemverilog 转换成 c++ 代码，从而可以直接执行。因此 verilator 是一个编译器，而不是一个完全的仿真器。

    编译的 .cpp, .h 的结果叫做 Verilating，编译的过程称为 to Verilate, 输出是一个 Verilated model.

    启动仿真的 c++ 文件称为 wrapper

* verilator usage introduction

    home page: <https://www.veripool.org>

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

            * --cc：生成 C++ 模型（而非 SystemC 或仅做 lint）。

            * --exe：告诉 Verilator 用户提供了 C++ 测试台，需要生成可执行文件。

            * --build：让 Verilator 自动调用 make 完成编译链接。

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

### verilog 语法

* verilog 模块（module）写端口的方式

    * ANSI 风格

        ```verilog
        module my_mod(input a, input b, output c);  // 默认情况下都是 wire 类型
            assign c = a & b;
        endmodule

        
        module my_mod_2(input a, input b, output reg c);  // 如果需要，可以指定 output 为 reg 类型
            always @(*) c = a & b;
        endmodule
        ```

        注：

        1. input 不能是 reg 类型，即`input reg a`会报错。

    * 非 ANSI 风格

        ```verilog
        module top(a, b, c);
            input a, b;
            output c;
            wire a, b, c;   // 这种风格下,端口名和信号声明分开写是允许的
            assign c = a & b;
        endmodule


        module top(a, b, c);
            input wire a, b;
            output c;  // 默认是 wire 类型
            assign c = a & b;
        endmodule
        ```

        注：

        1. `module top(a, b, c);`中的`(a, b, c)`不能省略，否则会报错。

* verilog 的基本设计单元是 模块（block）

    ```verilog
    module block(a, b, c, d);
        input a, b;
        output c, d;

        assign c = a | b;
        assign d = a & b;
    endmodule
    ```

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



