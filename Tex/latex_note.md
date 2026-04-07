# LaTex Note

## cache

* 详细讲讲 latex 中 \begin{align} 与 \begin{aligned} 有什么不同

    在 LaTeX 中，`align` 和 `aligned` 名字虽然像，但它们的**角色**和**脾气**完全不同。简单来说，一个是“独立大户”，一个是“随和插件”。

    以下是它们的详细区别：

    ---

    1. 核心定义与归属

        两者都属于 `amsmath` 宏包（使用前需声明 `\usepackage{amsmath}`）。

        * **`align` (环境)**：
            * 是一个**独立的环境**。
            * 它占据整行的宽度，并自动为其中的每一行方程式生成**编号**（除非使用 `align*`）。

        * **`aligned` (对齐积木)**：
            * 是一个**子环境（内联环境）**。
            * 它不能独立存在，必须嵌套在其他数学环境（如 `equation`、`displaymath` 或 `[ ... ]`）内部。它将多行公式打包成一个整体。

    ---

    2. 编号行为 (Numbering)

        这是两者最直观的区别：

        | 环境 | 编号特征 |
        | :--- | :--- |
        | **`align`** | **逐行编号**。每一行末尾都会有一个独立的序号（(1), (2), (3)...）。 |
        | **`aligned`** | **整体编号**。因为它被包含在 `equation` 中，所以整个公式组通常只在中间显示**一个编号**。 |

        **示例对比：**

        **使用 `align`：**
        ```latex
        \begin{align}
            x &= a + b \\
            y &= c + d
        \end{align}
        ```
        *结果：第一行标 (1)，第二行标 (2)。*

        **使用 `aligned`：**
        ```latex
        \begin{equation}
        \begin{aligned}
            x &= a + b \\
            y &= c + d
        \end{aligned}
        \end{equation}
        ```
        *结果：两行公式右侧正中间只有一个编号 (1)。*

    ---

    3. 布局与位置 (Placement)

        * **`align`** 总是尝试占据全行宽度。如果你想在段落中间插入一个简单的等式，`align` 会强制换行并居中。
        * **`aligned`** 像一个“字符”一样灵活。它可以放在文字中间（虽然不常见），或者放在括号里。

        > **典型用法：带大括号的方程组**
        > 如果你想给一组对齐的公式左边加一个大括号，你**必须**使用 `aligned`：
        > ```latex
        > \[
        > \left\{
        > \begin{aligned}
        >     2x + y &= 5 \\
        >     x - 3y &= 0
        > \end{aligned}
        > \right.
        > \]
        > ```

    ---

    4. 垂直对齐方式

        `aligned` 提供了一个可选参数来控制它相对于周围内容的垂直对齐点：
        * `\begin{aligned}[t]`：顶部对齐 (top)
        * `\begin{aligned}[b]`：底部对齐 (bottom)
        * 默认是垂直居中对齐。

        而 `align` 作为顶级环境，不支持这种对齐微调。

    ---

    **总结：该选哪一个？**

    | 你的需求 | 推荐方案 |
    | :--- | :--- |
    | 需要多行公式，且**每行都要单独编号** | `\begin{align}` |
    | 多行公式，但**不想要任何编号** | `\begin{align*}` |
    | 多行公式，但只需要**一个统一的编号** | `equation` 嵌套 `aligned` |
    | 在公式外部有**大括号**或作为其他复杂结构的一部分 | `aligned` |
    | 在一行内并列排放两组公式 | 在 `equation` 中放两个 `aligned` |

    **一句话总结：** `align` 是为了排版整个公式段落，而 `aligned` 是为了在更大的数学公式里构造一个局部的多行对齐块。

* ⊕ 是按位与的意思吗？如何用 latex 输入？

    ⊕ 的含义

    **⊕** 不是按位与，而是**异或（XOR）**运算。

    逻辑运算符对照表

    | 运算符 | 含义 | Verilog符号 | 逻辑表达式 |
    |-------|------|------------|-----------|
    | ⊕ | 异或（XOR） | `^` | A ⊕ B = (A & ~B) \| (~A & B) |
    | & | 与（AND） | `&` | A & B |
    | \| | 或（OR） | `\|` | A \| B |
    | ~ | 非（NOT） | `~` | ~A |

    异或运算真值表

    | A | B | A ⊕ B |
    |---|---|-------|
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 0 |

    LaTeX输入方法

    1. 异或符号（⊕）

        ```latex
        % 方法1：直接使用 \oplus
        $A \oplus B$

        % 方法2：使用异或运算符
        $A \veebar B$  % 另一种异或符号
        ```

    2. 其他逻辑运算符的LaTeX表示

        ```latex
        % 与运算（AND）
        $A \& B$        % 使用\&
        $A \cdot B$      % 点乘
        $A \land B$      % 逻辑与符号

        % 或运算（OR）
        $A | B$          % 直接输入|
        $A \lor B$       % 逻辑或符号

        % 非运算（NOT）
        $\lnot A$        % 逻辑非
        $\bar{A}$        % 上划线
        $A'$             % 撇号表示

        % 复合表达式示例
        $S = A \oplus B \oplus C_{in}$
        $C_{out} = (A \& B) | (A \& C_{in}) | (B \& C_{in})$
        ```

    3. 完整示例

        ```latex
        \documentclass{article}
        \usepackage{amsmath}  % 使用amsmath包获得更好的数学符号支持

        \begin{document}

        半加器逻辑表达式：
        \begin{align}
        S &= A \oplus B \\
        C &= A \cdot B
        \end{align}

        全加器逻辑表达式：
        \begin{align}
        S &= A \oplus B \oplus C_{in} \\
        C_{out} &= (A \cdot B) + (A \cdot C_{in}) + (B \cdot C_{in})
        \end{align}

        \end{document}
        ```

    4. 常用LaTeX逻辑符号速查

        | 符号 | LaTeX命令 | 含义 |
        |-----|-----------|------|
        | ⊕ | `\oplus` | 异或 |
        | ∧ | `\land` 或 `\wedge` | 与 |
        | ∨ | `\lor` 或 `\vee` | 或 |
        | ¬ | `\lnot` 或 `\neg` | 非 |
        | ⊻ | `\veebar` | 异或（另一种形式） |
        | ⊗ | `\otimes` | 异或（类似符号） |

    总结

    - **⊕** = 异或（XOR）
    - **&** = 与（AND）
    - **|** = 或（OR）
    - **~** = 非（NOT）

    在LaTeX中，异或最常用的命令是 `\oplus`

* latex 中，使用 \nu 表示希腊字母 ν（nu）

* LaTeX 中尖括号和角度符号写法

    在 LaTeX 中，你说的类似尖括号 < > 的符号有两种常见用法和表示方法：

    1. 作为尖括号本身

        直接输入 < 和 >：

        ```latex
        $\langle t \rangle$
        ```

        效果：$\langle t \rangle$

    2. 作为角度符号

        通常用 \angle 命令：

        ```latex
        $\angle ABC$  % 角度 ABC
        $\angle 90^\circ$  % 90度角
        ```

        效果：$\angle ABC$

    3. 物理中的狄拉克符号（尖括号）

        ```latex
        $\langle \psi | \phi \rangle$  % 内积
        $\langle x \rangle$  % 期望值
        ```

    4. 比较两种写法

        ```latex
        $<t>$  % 普通小于大于号，间距较小
        $\langle t \rangle$  % 真正的尖括号，间距合适
        ```

    建议：

    * 数学模式中用 \langle 和 \rangle，不要直接输入 < >

    * 角度符号用 \angle

    * 普通文本中可以直接输入 <t>，但在数学公式中建议使用专门的命令

    这些符号在数学、物理、计算机科学中都很常见，正确的 LaTeX 写法能让文档更专业美观。

* "正比于"符号用 latex 如何表示

    在 LaTeX 中，“正比于”符号有以下几种常用表示方法：

    1. 最常用的符号

        ```latex
        \propto
        ```

        效果：$\propto$

    2. 其他可用的表示

        ```latex
        \varpropto  % 变体形式
        ```

        效果：$\varpropto$

        ```latex
        \sim  % 波浪号，有时也用于表示比例关系
        ```

        效果：$\sim$

    3. 用法示例

        ```latex
        当 $x$ 增大时，$y \propto x$。
        ```

        效果：当 $x$ 增大时，$ y \propto x $。

        ```latex
        力 $F$ 与加速度 $a$ 成正比：$F \propto a$。
        ```

        效果：力 $F$ 与加速度 $a$ 成正比：$F \propto a$。

    4. 在公式环境中使用

        ```latex
        \begin{equation}
        y \propto x^n
        \end{equation}
        ```

        效果：

        $$ y \propto x^n$$

        推荐使用 \propto，这是最标准和最清晰的表示“正比于”的符号。

* 在 LaTeX 中输入 % 符号时，由于 % 是注释字符，需要转义才能正常显示。

    在 latex 数学公式中，用`\%`表示。

* `\mathbf`将数学符号设置为正体粗体

    * 对希腊字母不生效

    * 希腊字母可以使用`\bm`设置为斜体粗体，比如`\bm{\alpha}`

    * 与文本模式中的 `\textbf` 不同，`\mathbf` 专用于数学模式。

    效果：

    $\mathbf{ABC}$, $\mathbf{abc}$, $\mathbf{123}$, $\mathbf{\alpha \beta \gamma}$, $\mathbf{+-*/}$, $\mathbf{你好}$

* `\exp`后通常要加上括号，比如`\exp \left( x + y \right)`，不能只写成`\exp{x + y}`，更不能写成`\exp x + y`。

* latex 输入花括号

    ```latex
    % 单独显示花括号
    \{ \} 

    % 在数学环境中使用
    $ A = \{ x \in \mathbb{R} \mid x > 0 \} $

    % 集合表示
    $$ S = \{ 1, 2, 3, \dots, n \} $$

    % 在文本中使用
    这是左花括号：\{，这是右花括号：\}
    ```

* 倒三角符号的 latex 是$\nabla$

* 希腊字母（Greek Letters）

    | 符号名称 | 小写 LaTeX | 大写 LaTeX |
    | - | - | - |
    | Alpha | `\alpha` $\alpha$ | `A` $A$ |
    | Beta | `\beta` $\beta$ | `B` $B$ |
    | Gamma | `\gamma` $\gamma$ | `\Gamma` $\Gamma$ |
    | Delta	| `\delta` $\delta$ | `\Delta` $\Delta$ |
    | Epsilon | `\epsilon` $\epsilon$ |	`E` $E$ |
    | Zeta | `\zeta` $\zeta$ | `Z` $Z$ |
    | Eta | `\eta` $\eta$ | `H` $H$ |
    | Theta | `\theta` $\theta$ | `\Theta` $\Theta$ |
    | Lambda | `\lambda` $\lambda$ | `\Lambda` $\Lambda$ |
    | Mu | `\mu` $\mu$ | `M` $M$ |
    | Nu | `\nu` $\nu$ | `N` $N$ |
    | Xi | `\xi` $\xi$ | `\Xi` $\Xi$ |
    | Pi | `\pi` $\pi$ | \Pi $\Pi$ |
    | Rho | `\rho` $\rho$ | `P` $P$ |
    | Sigma | `\sigma` $\sigma$ | `\Sigma` $\Sigma$ |
    | Tau | `\tau` $\tau$ | `T` $T$ |
    | Phi | `\phi` $\phi$ | `\Phi` $\Phi$ |
    | Psi | `\psi` $\psi$ | `\Psi` $\Psi$ |
    | Omega | `\omega` $\omega$ | `\Omega` $\Omega$ |

* latex 有关集合包含的符号

    $\subseteq$, $\subsetneq$, $\subsetneqq$

* 转置的写法：`x^\intercal`，效果：$x^\intercal$

## note

* 下括号

    $\underbrace{1+2+\cdots+100}_{5050}$

* 差集：`\setminus`，效果：$\setminus$
