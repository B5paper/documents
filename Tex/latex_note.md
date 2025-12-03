# LaTex Note

## cache

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
