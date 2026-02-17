# verilog note

## cache

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
