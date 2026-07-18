# LaTex Note

## cache

* latex 如何换行？

	好的，在 LaTeX 中换行有多种方式，适用于不同的场景。下面我将从最常用到最不常用的方式为您详细介绍。

	### 1. 简单的换行与换段

	这是最基础、最常用的两种方式。

	*   **换段：** 连续两次按 **`Enter`** 键（即空一行）
		*   **效果：** 开始一个新的段落，并且会自动首行缩进。
		*   **示例：**
			```latex
			\documentclass{article}
			\begin{document}
			
			这是第一段的内容。这是一个演示文本，用来展示如何开始一个新的段落。
			
			这是第二段。你可以看到，在编译后的PDF中，这一段的开头有明显的缩进。
			
			\end{document}
			```

	*   **换行：** 使用 **`\\`** 或 **`\newline`** 命令
		*   **效果：** 开始新的一行，但**不会**缩进。通常在诗歌、地址或者表格中使用。
		*   **注意：** `\\` 比 `\newline` 更常用，但在普通段落中应谨慎使用，因为这不是 LaTeX 推荐的分段方式。
		*   **示例：**
			```latex
			\documentclass{article}
			\begin{document}
			
			这是第一行。\\
			这是使用双反斜杠强制换行后的第二行。\\ % 推荐使用这个
			这是使用 \verb|\newline| 命令换行后的第三行。\newline
			这是第四行。
			
			\end{document}
			```

	---

	### 2. 保持上下文连贯的换行（不缩进）

	当你想开始一个“新段落”，但又不想让它缩进时（例如在文档开头或 after a section title），可以使用 `\noindent` 命令。

	*   **示例：**
		```latex
		\documentclass{article}
		\begin{document}
		
		\section{简介}
		这是简介部分的第一段，通常不会缩进。
		
		\noindent 这是简介部分的第二段。因为我们使用了 \verb|\noindent| 命令，所以这一行也没有缩进。
		
		这是简介部分的第三段，这是一个新的常规段落，所以它会正常缩进。
		
		\end{document}
		```

	---

	### 3. 用于特定环境的换行

	在某些环境下，换行规则会有所不同。

	*   **居中环境 `center`：** 使用 `\\` 换行。
		```latex
		\begin{center}
			第一行，居中显示。\\
			第二行，也居中显示。\\
			第三行。
		\end{center}
		```

	---

	### 总结与对比

	| 命令或操作 | 效果 | 使用场景 |
	| :--- | :--- | :--- |
	| **空一行** (按两次Enter) | **换段**，并自动**首行缩进** | **最常用**，用于分隔两个逻辑上的段落。 |
	| **`\\`** 或 **`\newline`** | **换行**，但**不缩进** | 用于诗歌、地址、表格内部等需要强制换行的地方。在普通段落中慎用。 |
	| **`\noindent`** | 开始一个**不缩进**的段落 | 当新段落不需要首行缩进时，例如紧跟在标题、图片之后。 |
	| **`\par`** | 等同于空一行，用于换段 | 在宏定义或复杂代码中比空一行更可靠。 |

	### 最佳实践建议

	对于普通的文本段落，**强烈建议使用“空一行”的方式来换段**。这是 LaTeX 设计哲学的一部分，让你专注于内容与结构，而不是具体的版式。让 LaTeX 来处理缩进、行距和分页等细节问题。只有在非常特殊和必要的情况下，才使用 `\\` 来强制换行。

* 给出常用 latex 符号对照表

	好的，这是一份常用的 LaTeX 符号对照表，特别是那些与常规键盘符号差异较大或需要特殊命令的符号。

	### LaTeX 常用符号对照表

	#### 1. 希腊字母（Greek Letters）

	| 符号名称         | 小写 LaTeX        | 大写 LaTeX        |
	| :--------------- | :---------------- | :---------------- |
	| Alpha            | `\alpha` $\alpha$     | `A` $A$               |
	| Beta             | `\beta` $\beta$      | `B` $B$               |
	| Gamma            | `\gamma` $\gamma$    | `\Gamma` $\Gamma$     |
	| Delta            | `\delta` $\delta$    | `\Delta` $\Delta$     |
	| Epsilon          | `\epsilon` $\epsilon$| `E` $E$               |
	| Zeta             | `\zeta` $\zeta$     | `Z` $Z$               |
	| Eta              | `\eta` $\eta$      | `H` $H$               |
	| Theta            | `\theta` $\theta$   | `\Theta` $\Theta$     |
	| Lambda           | `\lambda` $\lambda$ | `\Lambda` $\Lambda$   |
	| Mu               | `\mu` $\mu$       | `M` $M$               |
	| Nu               | `\nu` $\nu$       | `N` $N$               |
	| Xi               | `\xi` $\xi$       | `\Xi` $\Xi$           |
	| Pi               | `\pi` $\pi$       | `\Pi` $\Pi$           |
	| Rho              | `\rho` $\rho$      | `P` $P$               |
	| Sigma            | `\sigma` $\sigma$   | `\Sigma` $\Sigma$     |
	| Tau              | `\tau` $\tau$      | `T` $T$               |
	| Phi              | `\phi` $\phi$      | `\Phi` $\Phi$         |
	| Psi              | `\psi` $\psi$      | `\Psi` $\Psi$         |
	| Omega            | `\omega` $\omega$   | `\Omega` $\Omega$     |

	#### 2. 关系符号（Relational Symbols）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 约等于           | `\approx`             | $\approx$           |
	| 不等于           | `\neq`                | $\neq$              |
	| 恒等于           | `\equiv`              | $\equiv$            |
	| 大于等于         | `\geq`                | $\geq$              |
	| 小于等于         | `\leq`                | $\leq$              |
	| 远大于           | `\gg`                 | $\gg$               |
	| 远小于           | `\ll`                 | $\ll$               |
	| 正比于           | `\propto`             | $\propto$           |
	| 相似于（波浪）   | `\sim`                | $\sim$              |
	| 平行             | `\parallel`           | $\parallel$         |
	| 垂直             | `\perp`               | $\perp$             |

	#### 3. 运算符（Operators）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 加减             | `\pm`                 | $\pm$               |
	| 减加             | `\mp`                 | $\mp$               |
	| 乘（点）         | `\cdot`               | $\cdot$             |
	| 乘（叉）         | `\times`              | $\times$            |
	| 除（分式）       | `\div`                | $\div$              |
	| 无限             | `\infty`              | $\infty$            |
	| 偏微分           | `\partial`            | $\partial$          |
	| 梯度（Nabla）    | `\nabla`              | $\nabla$            |
	| 积分             | `\int`                | $\int$              |
	| 二重积分         | `\iint`               | $\iint$             |
	| 三重积分         | `\iiint`              | $\iiint$            |
	| 求和             | `\sum`                | $\sum$              |
	| 连乘             | `\prod`               | $\prod$             |
	| 平方根           | `\sqrt{x}`            | $\sqrt{x}$          |
	| n次方根          | `\sqrt[n]{x}`         | $\sqrt[n]{x}$       |

	#### 4. 箭头（Arrows）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 左箭头           | `\leftarrow`          | $\leftarrow$        |
	| 右箭头           | `\rightarrow`         | $\rightarrow$       |
	| 双向箭头         | `\leftrightarrow`     | $\leftrightarrow$   |
	| 向上箭头         | `\uparrow`            | $\uparrow$          |
	| 向下箭头         | `\downarrow`          | $\downarrow$        |
	| 长右箭头（推导） | `\longrightarrow`     | $\longrightarrow$   |
	| 映射             | `\mapsto`             | $\mapsto$           |
	| 推出             | `\implies`            | $\implies$          |

	#### 5. 集合论（Set Theory）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 属于             | `\in`                 | $\in$               |
	| 不属于           | `\notin`              | $\notin$            |
	| 子集             | `\subset`             | $\subset$           |
	| 真子集           | `\subsetneq`          | $\subsetneq$        |
	| 并集             | `\cup`                | $\cup$              |
	| 交集             | `\cap`                | $\cap$              |
	| 空集             | `\emptyset`           | $\emptyset$         |
	| 全体实数集       | `\mathbb{R}`          | $\mathbb{R}$        |
	| 全体整数集       | `\mathbb{Z}`          | $\mathbb{Z}$        |
	| forall           | `\forall`             | $\forall$           |
	| exists           | `\exists`             | $\exists$           |

	#### 6. 几何与逻辑（Geometry & Logic）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 角               | `\angle`              | $\angle$            |
	| 三角形           | `\triangle`           | $\triangle$         |
	| 平行四边形       | `\parallelogram`      | $\parallelogram$    |
	| 度               | `\degree`             | $\degree$           |
	| 因为             | `\because`            | $\because$          |
	| 所以             | `\therefore`          | $\therefore$        |
	| 逻辑与           | `\land`               | $\land$             |
	| 逻辑或           | `\lor`                | $\lor$              |
	| 逻辑非           | `\lnot`               | $\lnot$             |

	#### 7. 括号与定界符（Brackets & Delimiters）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 绝对值           | `\| x \|`             | $\| x \|$           |
	| 范数             | `\Vert x \Vert`       | $\Vert x \Vert$     |
	| 向上取整         | `\lceil x \rceil`     | $\lceil x \rceil$   |
	| 向下取整         | `\lfloor x \rfloor`   | $\lfloor x \rfloor$ |
	| **可伸缩括号**   |                       |                     |
	| 普通括号         | `( )`                 | $( )$               |
	| 可伸缩圆括号     | `\left( ... \right)`  | $\left( \frac{a}{b} \right)$ |
	| 可伸缩方括号     | `\left[ ... \right]`  | $\left[ \frac{a}{b} \right]$ |
	| 可伸缩花括号     | `\left\{ ... \right\}`| $\left\{ \frac{a}{b} \right\}$ |

	#### 8. 装饰与标记（Accents & Decorations）

	| 符号名称         | LaTeX                 | 效果                |
	| :--------------- | :-------------------- | :------------------ |
	| 向量箭头         | `\vec{v}`             | $\vec{v}$           |
	| 向量粗体         | `\mathbf{v}`          | $\mathbf{v}$        |
	| 点在上           | `\dot{x}`             | $\dot{x}$           |
	| 两点在上         | `\ddot{x}`            | $\ddot{x}$          |
	| 横线在上（平均） | `\bar{x}`             | $\bar{x}$           |
	| 尖角在上（估计） | `\hat{x}`             | $\hat{x}$           |
	| 波浪在上         | `\tilde{x}`           | $\tilde{x}$         |

	使用提示：

	1.  **空格**：LaTeX 会忽略大多数空格。要添加空格，使用 `\,` (小空格), `\ ` (反斜杠+空格，中等空格), `\quad` (大空格), `\qquad` (更大空格)。
	
    2.  **转义字符**：要输入本身是命令的字符（如 `#`, `$`, `%`, `&`, `_`, `{`, `}`, `\`），需要在它们前面加上反斜杠 `\`，例如 `\$` 输出 `$`。
	
    3.  **公式环境**：这些符号通常用在数学环境中：

		* **行内公式**：使用 `$ ... $`，例如 `$\infty$` 在行内显示为 $\infty$。
		* **显示公式**：使用 `\[ ... \]` 或 `$$ ... $$`，公式会单独居中显示在一行。

* latex 中小于等于号

	* `\le` 或 `\leq`

    * 更扁一些：`\leqslant`

* 空集用 latex 怎么写

	`\varnothing`

	这是最常用、最标准的空集符号，是一个带斜线的圆圈。

	方法二：使用 `\emptyset`（不推荐）

	这是 LaTeX 内置的命令，不需要额外宏包，但它显示的符号更像数字 0 加上斜线，在数学排版中不如 `\varnothing` 美观。

* 去心邻域使用 latex 怎么表示，比如 U 上面有个小圆圈

	好的，在 LaTeX 中表示去心邻域（即在字母 U 上加一个小圆圈）有几种方法，最常用和推荐的是使用 `amsmath` 或 `amssymb` 宏包提供的 `\mathring` 命令。

	这里为您详细说明：

	方法一：使用 `\mathring{}` 命令（最标准）

	这是数学模式下最标准和语义化的表示方法。

	\mathring{U}`：

    ```latex
    $\mathring{U}(x_0, \delta)$
    ```

	方法二：使用 `\overset{\circ}{U}`

	这种方法使用 `\overset` 命令将小圆圈放在 U 的上方。它也需要 `amsmath` 宏包。

	```latex
	$\overset{\circ}{U}$
	```

	这个方法的好处是语义非常清晰，一看就知道是在上面加一个圈。

* latex 中各种双竖线怎么写？

    在 LaTeX 里，“双竖线”可能指不同的符号，下面把常见几类全部列出来，给你对应的写法和渲染效果。

    1. 逻辑/集合常用的双竖线（‖）

        (2) 数学模式的双竖线（范数符号）

        ```
        \lVert x \rVert
        ```

        效果：( $\lVert x \rVert$ )

        也可以用

        ```
        \Vert x \Vert
        ```

        效果：( $\Vert x \Vert$ )

    2. 表示“并行”的双竖线（∥）

        数学环境：

        ```
        \parallel
        ```

        效果：( $A \parallel B$ )

    3. 表示“满足”条件的双竖线（| 或 ||）

        **单竖线：**

        ```
        \mid        % 数学中的中竖线
        \vert       % 同上
        ```

        **双竖线：**

        ```
        \mid\mid
        \vert\vert
        ```

        但更推荐使用 `\parallel` 或 `\Vert`（取决于语义）。

    4. 表格里的竖线（||）

        在 `tabular` 里：

        ```
        \begin{tabular}{||c|c||}
        a & b \\
        c & d \\
        \end{tabular}
        ```

        这里的 `||` 就是加双竖线分隔线，不需要额外转义。

    5. 逻辑“或”符号（∨）以及双竖线“或”符号（||）

        C++ / 编程风格的“||”想原样显示：

        ```
        \verb!||!
        ```

        或

        ```
        \texttt{||}
        ```
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
