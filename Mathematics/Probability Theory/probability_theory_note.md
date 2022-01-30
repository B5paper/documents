# 概率论与数理统计

## 概率论的基本概念

**随机试验**

具有下面三个特点的试验称为随机试验：

1. 可以在相同的条件下重复地进行

1. 每次试验的可能结果不止一个，并且能事先明确试验的所有可能结果

1. 进行一次试验之前不能确定哪一个结果会出现

**样本空间**

随机试验$E$的所有可能结果组成的集合称为$E$的样本空间，记为$S$。

**样本点**

样本空间的元素（即随机试验可能的结果）称为样本点。

**事件**

试验$E$的样本空间$S$的子集为$E$的随机事件，简称事件。

**事件发生**

在每次试验中，当且仅当事件中的一个样本点出现时，称这一事件发生。

**基本事件**

由一个样本点组成的单点集，称为基本事件。

**必然事件**

样本空间$S$称为必然事件。

**不可能事件**

空集$\varnothing$称为不可能事件。

事件间的关系：

设试验$E$的样本空间为$S$，而$A$，$B$，$A_k$（$k = 1, 2, \cdots$）是$S$的子集。

1. 若$A \subset B$，则称事件$B$包含事件$A$。这里指的是事件$A$发生必导致事件$B$发生。

1. 若$A \subset B$且$B \subset A$，则称事件$A$与事件$B$相等。

1. 事件$A \cup B = \{ x \, \vert \, x \in A 或 x \in B \}$称为事件$A$与事件$B$的和事件。当且仅当$A$，$B$中至少有一个发生时，事件$A \cup B$发生。

    类似地，称$\bigcup\limits_{k=1}^n A_k$为$n$个事件$A_1$，$A_2$，$\dots$，$A_n$的和事件。

1. 事件$A \cap B = \{ x \, \vert \, x \in A 且 x \in B \}$称为事件$A$与事件$B$的积事件。当$A$，$B$同时发生时，事件$A \cap B$发生。$A \cap B$也记作$AB$。

    类似地，称$\bigcap\limits_{k=1}^n A_k$为$n$个事件$A_1$，$A_2$，$\dots$，$A_n$的积事件。

1. 事件$A - B = \{ x \, \vert \, x \in A 且 x \not \in B \}$称为事件$A$与事件$B$的差事件。当且仅当$A$发生，$B$不发生时，事件$A - B$发生。

1. 若$A \cap B = \varnothing$，则称$A$与$B$是不相容的，或互斥的。事件$A$与事件$B$不能同时发生。基本事件两两互不相容。

1. 若$A \cup B = S$且$A \cap B = \varnothing$，则称事件$A$与事件$B$互为逆事件或对立事件。对于每次试验，事件$A$，$B$有且仅有一个发生。$A$的对立事件$\bar A = S - A$。

事件运算的定律：

1. 交换律

    $A \cup B = B \cup A$

    $A \cap B = B \cap A$

1. 结合律

    $A \cup (B \cup C) = (A \cup B) \cup C$

    $A \cap (B \cap C) = (A \cap B) \cap C$

1. 分配律

    $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$

    $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$

1. 德摩根律

    $\overline{A \cup B} = \overline A \cap \overline B$

    $\overline{A \cap B} = \overline A \cup \overline B$

**频率**

在相同条件下进行$n$次试验，在这$n$次试验中，事件$A$发生的次数$n_A$称为事件$A$发生的频数。比值$n_A / n$称为事件$A$发生的频率，并记为$f_n(A)$。

频率的基本性质：

1. $0 \leq f_n(A) \leq 1$

1. $f_n(S) = 1$

1. 若$A_1$，$A_2$，$\dots$，$A_k$是两两互不相容的事件，则

    $f_n(A_1 \cup A_2 \cup \cdots \cup A_k) = f_n(A_1) + f_n(A_2) + \cdots + f_n(A_k)$

**概率**

设$E$是随机试验，$S$是它的样本空间。对于$E$的每一事件$A$赋予一个实数，记为$P(A)$，称为事件$A$的概率，如果集合函数$P(\cdot)$满足下列条件：

1. 非负性：对于每一个事件$A$，有$P(A) \geq 0$

1. 规范性：对于必然事件$S$，有$P(S) = 1$

1. 可列可加性：设$A_1$，$A_2$，$\dots$是两两互不相容的事件，即对于$A_i A_j = \varnothing$，$i \neq j$，$i, j = 1, 2, \dots$，有$P(A_1 \cup A_2 \cup \cdots) = P(A_1) + P(A_2) + \cdots$

概率的性质：

1. $P(\varnothing) = 0$

1. 有限可加性

    若$A_1$，$A_2$，$\dots$，$A_n$是两两互不相容的事件，则有

    $$P(A_1 \cup A_2 \cup \cdots \cup A_n) = P(A_1) + P(A_2) + \cdots + P(A_n)$$

1. 设$A$，$B$是两个事件，若$A \subset B$，则有

    $$P(B - A) = P(B) - P(A)$$

    $$P(B) \geq P(A)$$
    
1. 对于任一事件$A$，有$P(A) \leq 1$

1. 逆事件的概率

    对于任一事件$A$，有$P(\overline A) = 1 - P(A)$。

1. 加法公式

    对于任意两事件$A$，$B$，有

    $$P(A \cup B) = P(A) + P(B) - P(AB)$$

    对于三个事件$A_1$，$A_2$，$A_3$，有

    $$P(A_1 \cup A_2 \cup A_3) = P(A_1) + P(A_2) + P(A_3) - P(A_1 A_2) - P(A_1 A_3) - P(A_2 A_3) + P(A_1 A_2 A_3)$$

    一般地，对于任意$n$个事件$A_1$，$A_2$，$\dots$，$A_n$，有

    $$\begin{aligned} P(A_1 \cup A_2 \cup \cdots \cup A_n) = \sum\limits_{i=1}^n P(A_i) - \sum\limits_{1 \leq i \lt j \leq n} P(A_i A_j) \\ + \sum\limits_{1 \leq i \lt j \lt k \leq n} P(A_i A_j A_k) + \cdots + (-1)^{n-1} P(A_1 A_2 \cdots A_n) \end{aligned}$$

**等可能概型**

若某个试验$E$具有下面两个特点：

1. 试验的样本空间只包含有限个元素

1. 试验中每个基本事件发生的可能性相同

那么这个试验被称为等可能概型或古典概型。

等可能概型中事件$A$的计算公式：

$$P(A) = \sum\limits_{j=1}^k P(\{ e_{i_j} \}) = \dfrac k n = \dfrac{A 包含的基本事件数}{S 中基本事件的总数}$$

**实际推断原理**：概率很小的事在一次试验中实际上几乎是不发生的。若概率很小的事件在一次试验中竟然发生了，那么就有理由怀疑假设的正确性。

**条件概率**：设$A$，$B$是两个事件，且$P(A) \gt 0$，称$P(B \, \vert \, A) = \dfrac{P(AB)}{P(A)}$为在事件$A$发生的条件下事件$B$发生的条件概率。

概率的一些性质也适用于条件概率，比如$P(B_1 \cup B_2 \, \vert \, A) = P(B_1 \, \vert \, A) + P(B_2 \, \vert \, A) - P(B_1 B_2 \, \vert \, A)$。

**乘法定理**：设$P(A) \gt 0$，则有$P(AB) = P(B \, \vert \, A) P(A)$。

对于三事件$A$，$B$，$C$，若有$P(AB) \gt 0$（可推得$P(A) \geq P(AB) \gt 0$），则有$P(ABC) = P(C \, \vert \, AB) P(B \, \vert \, A) P(A)$。

一般地，设$A_1$，$A_2$，$\dots$，$A_n$为$n$个事件，$n \geq 2$，且$P(A_1 A_2 \cdots A_{n-1}) \gt 0$，则有$P(A_1 A_2 \cdots A_n) = P(A_n \, \vert \, A_1 A_2 \cdots A_{n-1}) P(A_{n-1} \, \vert \, A_1 A_2 \cdots A_{n-2}) \cdots P(A_2 \, \vert \, A_1) P(A_1)$

**划分**：设$S$为试验$E$的样本空间，$B_1$，$B_2$，$\dots$，$B_n$为$E$的一组事件。若

1. $B_i B_j = \varnothing$，$i \neq j$，$i, j = 1, 2, \dots, n$

1. $B_1 \cup B_2 \cup \cdots B_n = S$

则称$B_1$，$B_2$，$\cdots$，$B_n$为样本空间$S$的一个划分。

**全概率公式**：设试验$E$的样本空间为$S$，$A$为$E$的事件，$B_1$，$B_2$，$\dots$，$B_n$为$S$的一个划分，且$P(B_i) \gt 0$（$i = 1, 2, \dots, n$），则

$$P(A) = P(A \, \vert \, B_1) P(B_1) + P(A \, \vert \, B_2) P(B_2) + \cdots + P(A \, \vert \, B_n) P(B_n)$$

**贝叶斯（Bayes）公式**：设试验$E$的样本空间为$S$。$A$为$E$的事件，$B_1$，$B_2$，$\dots$，$B_n$为$S$的一个划分，且$P(A) \gt 0$，$P(B_i) \gt 0$（$i = 1, 2, \dots, n$），则

$$P(B_i \, \vert \, A) = \dfrac{P(A \, \vert \, B_i) P(B_i)}{\sum\limits_{j=1}^n P(A \, \vert \, B_j) P(B_j)}, \quad i = 1, 2, \dots, n$$

**独立**：设$A$，$B$是两事件，如果满足等式$P(AB) = P(A) P(B)$，则称事件$A$，$B$独立。

对于三个事件的情况：设$A$，$B$，$C$是三个事件，如果满足等式

$$\begin{cases} P(AB) = P(A) P(B) \\[3px] P(BC) = P(B) P(C) \\[3px] P(AC) = P(A) P(C) \\[3px] P(ABC) = P(A) P(B) P(C) \end{cases}$$

则称事件$A$，$B$，$C$相互独立。

一般地，设$A_1$，$A_2$，$\dots$，$A_n$是$n$（$n \geq 2$）个事件，如果对于其中任意 2 个，任意 3 个，$dots$，任意$n$个事件的积事件的概率，都等于各事件概率之积，则称事件$A_1$，$A_2$，$\dots$，$A_n$相互独立。

推论：

1. 若事件$A_1$，$A_2$，$\dots$，$A_n$（$n \geq 2$）相互独立，则其中任意$k$（$2 \leq k \leq n$）个事件也是相互独立的。

1. 若$n$个事件$A_1$，$A_2$，$\dots$，$A_n$（$n \geq 2$）相互独立，则将$A_1$，$A_2$，$\dots$，$A_n$中任意多个事件换成它们各自的对立事件，所得的$n$个事件仍相互独立。

若$P(A) \gt 0$，$P(B) \gt 0$，则$A$，$B$相互独立与$A$，$B$不相容不能同时成立。（因为若$A$，$B$不相容，则$P(AB) = 0$）

两事件相互独立的含义是它们中一个已发生，不影响另一个发生的概率。在实际应用中，对于事件的独立性常常是根据事件的实际意义去判断。

**定理**：设$A$，$B$是两事件，且$P(A) \gt 0$。若$A$，$B$相互独立，则$P(B \, \vert \, A) = P(B)$。反之亦然。

**定理**：若事件$A$与$B$相互独立，则下列各对事件也相互独立：$A$与$\overline B$，$\overline A$与$B$，$\overline A$与$\overline B$。

## 随机变量及其分布

**随机变量**：设随机试验的样本空间为$S = \{ e \}$。$X = X(e)$是定义在样本空间$S$上实值单值函数。称$X = X(e)$为随机变量。

感想：

1. 随机变量即用实数刻画事件。

随机变量的取值随试验的结果而定，在试验之前不能预知它取什么值，且它的取值有一定的概率。这些性质显示了随机变量与普通函数有着本质的差异。

**分布函数**：设$X$是一个随机变量，$x$是任意实数，函数$F(x) = P\{ X \leq x \}$，$-\infty \lt x \lt \infty$，称为$X$的分布函数。

对于任意实数$x_1$，$x_2$（$x_1 \lt x_2$），有$P\{ x_1 \lt X \leq x_2 \} = P\{ X \leq x_2 \} - P\{ X \leq x_1 \} = F(x_2) - F(x_1)$。

分布函数完整地描述了随机变量的统计规律性。

设离散型随机变量$X$的分布律为$P\{ X = x_k \} = p_k$，$k = 1, 2, \dots$，由概率的可加性得$X$的分布函数为$F(x) = P\{ X \leq x \} = \sum\limits_{x_k \leq x} P\{ X = x_k \} = \sum\limits_{x_k \leq x} p_k$

**连续型随机变量**：如果对于随机变量$X$的分布函数$F(x)$，存在非负可积函数$f(x)$，使对于任意实数$x$有$F(x) = \displaystyle\int_{-\infty}^x f(t) \mathrm d t$，则称$X$为连续型随机变量，$f(x)$称为$X$的概率密度函数，简称概率密度。

常见的连续型随机变量：

1. 均匀分布

    若连续型随机变量$X$具有概率密度$f(x) = \begin{cases} \dfrac{1}{b - a}, \quad a \lt x \lt b \\ 0, \quad 其他 \end{cases}$，则称$X$在区间$(a, b)$上服从均匀分布。记为$X \sim U(a, b)$。

1. 指数分布

    若连续型随机变量$X$的概率密度为$f(x) = \begin{cases} \dfrac 1 \theta e^{-x / \theta}, \quad x \gt 0 \\ 0, \quad 其他 \end{cases}$，其中$\theta \gt 0$为常数，则称$X$服从参数为$\theta$的指数分布。

    随机变量$X$的分布函数为$F(x) = \begin{cases} 1 - e^{-x / \theta}, \quad x \gt 0 \\ 0, \quad 其他 \end{cases}$

    指数分布的性质：

    1. 对于任意$s, t \gt 0$，有$P\{ X \gt s + t \, \vert \, X \gt s \} = P\{ X \gt t \}$

1. 正态分布

    若连续型随机变量$X$的概率密度为$f(x) = \dfrac{1}{\sqrt{2\pi} \sigma} e^{-\dfrac{(x - \mu)^2}{2 \sigma^2}}$，$-\infty \lt x \lt \infty$，其中$\mu$，$\sigma$（$\sigma \gt 0$）为常数，则称$X$服从参数为$\mu$，$\sigma$的正态分布或高斯（Gauss）分布，记为$X \sim N(\mu, \sigma^2)$。

    $X$的分布函数为$F(x) = \dfrac{1}{\sqrt{2\pi} \sigma} \displaystyle\int_{-\infty}^x e^{-\dfrac{(t - \mu)^2}{2 \sigma^2}} \mathrm d t$

    当$\mu = 0$，$\sigma = 1$时称随机变量$X$服从标准正态分布。其概率密度$\varphi(x)$和分布函数$\Phi(x)$分布为：

    $$\varphi(x) = \dfrac{1}{\sqrt{2\pi}} e^{-x^2 / 2}$$

    $$\Phi(x) = \dfrac{1}{\sqrt{2\pi}} \displaystyle\int_{-\infty}^x e^{-t^2 / 2} \mathrm d t$$

    性质：$\Phi (-x) = 1 - \Phi(x)$

    正态分布与标准正态分布之间的关系：若$X \sim N(\mu, \sigma^2)$，则$Z = \dfrac{X - \mu}{\sigma} \sim N(0, 1)$

随机变量的函数的分布：

由已知的随机变量$X$的概率分布去求得它的函数$Y = g(X)$（$g(\cdot)$是已知的连续函数）的概率分布。这里$Y$是这样的随机变量，当$X$取值$x$时，$Y$取值$g(x)$。

定理：设随机变量$X$具有概率密度$f_X(x)$，$-\infty \gt x \gt \infty$，又设函数$g(x)$处处可导且恒有$g'(x) \gt 0$（或恒有$g'(x) \gt 0$），则$Y = g(X)$是连续型随机变量，其概率密度为

$$f_Y(y) = \begin{cases} f_X[h(y)]\lvert h'(y) \rvert, \quad \alpha \lt y \lt \beta, \\ 0, \quad 其他 \end{cases}$$

其中$\alpha = \min\{ g(-\infty), g(\infty) \}$，$\beta = \max\{ g(-\infty), g(\infty) \}$，$h(y)$是$g(x)$的反函数。

例题：

1. 设随机变量$X$具有以下的分布律，试求$Y = (X - 1)^2$的分布律

    |$X$|-1|0|1|2|
    |-|-|-|-|-|
    |$p_k$|0.2|0.3|0.1|0.4|

    解：$Y$所有可能的取值为 0，1，4，

    $P\{ Y = 0 \} = P\{ (X-1)^2 = 0 \} = P\{ X = 1 \} = 0.1$

    $P\{ Y = 1 \} = P\{ X = 0 \} + P\{ X = 2 \} = 0.7$

    $P\{ Y = 4 \} = P\{ X = -1 \} = 0.2$

    因此$Y$的分布律为

    |$Y$|0|1|4|
    |-|-|-|-|
    |$p_k$|0.1|0.7|0.2|

1. 设随机变量$X$具有概率密度

    $f_X(x) = \begin{cases} \dfrac x 8, \quad 0 \lt x \lt 4 \\ 0, \quad 其他 \end{cases}$

    求随机变量$Y = 2X + 8$的概率密度

## 多维随机变量及其分布

二维随机变量：设$(X, Y)$是二维随机变量，对于任意实数$x$，$y$，二元函数：$F(x, y) = P\{ (X \leq x) \cap (Y \leq y) \}$称为二维随机变量$(X, Y)$的分布函数，或称为随机变量$X$和$Y$的联合分布函数。记作$P\{ X \leq x, Y \leq y \}$

对于二维随机变量$(X, Y)$的分布函数$F(x, y)$，如果存在非负可积函数$f(x, y)$使对于任意$x$，$y$有

$$F(x, y) = \int_{-\infty}^y \int_{-\infty}^x f(u, v) \mathrm d u \mathrm d v$$

则称$(X, Y)$是连续型的二维随机变量，函数$f(x, y)$称为二维随机变量$(X, Y)$的概率密度，或称为随机变量$X$和$Y$的联合概率密度。

边缘分布函数：设二维随机变量$(X, Y)$的分布函数为$F(x, y)$，则$X$和$Y$的分布函数$F_X(x)$和$F_Y(y)$依次称为二维随机变量$(X, Y)$关于$X$和$Y$的边缘分布函数。

$F_X(x) = P\{ X \leq x \} = P\{ P \leq x, Y \lt \infty \} = F(x, \infty)$

同理，$F_Y(y) = F(\infty, y)$

边缘分布律：对于离散型随机变量，有$F_X(x) = F(x, \infty) = \sum\limits_{x_i \leq x} \sum\limits_{j = 1}^\infty p_{ij}$，因此$p_{i \cdot} = \sum\limits_{j=1}^\infty p_{ij} = P\{ X = x_i \}$，$i = 1, 2, \cdots$，$p_{\cdot j} = \sum\limits_{i=1}^\infty p_{ij} = P\{ Y = y_j \}$，$j = 1, 2, \cdots$

条件分布：设$(X, Y)$是二维离散型随机变量，对于固定的$j$，若$P\{ Y = y_i \} \gt 0$，则称$P\{ X = x_i \, \vert \, Y = y_i \} = \dfrac{P\{ X = x_i, Y = y_i \}}{P\{ Y = y_i \}} = \dfrac{p_{ij}}{p_{\cdot j}}$，$i = 1, 2, \dots$，为在$Y = y_j$条件下随机变量$X$的条件分布律。

同样，对于固定的$i$，若$P\{ X = x_i \} \gt 0$，则称$P\{ Y = y_i \, \vert \, X = x_i \} = \dfrac{P\{ X = x_i, Y = y_j \}}{P\{ X = x_i \}} = \dfrac{p_{ij}}{p_{i\cdot}}$，$j = 1, 2, \dots$