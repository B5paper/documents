# LaTex Note

## cache

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
