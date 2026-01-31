# LaTex Note

## cache

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
