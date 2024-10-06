# Matrix Theory 矩阵论

矩阵论里感觉可能比较有用的几个地方：

1. 线性变换，基底

2. 矩阵分解

3. 广义逆，最小二乘法求线性方程组的拟合近似解

4. 谱分析与盖尔圆

5. 矩阵微积分

6. 矩阵的特征值（有什么意义？）

## cache

* 数环

    设$Z$为非空数集且其中任何两个相同或互异的数之和、差与积仍属于$Z$（即数集关于加、减、乘法运算封闭），则称$Z$是一个数环。

    说明：

    1. 可以把数环定义为：对于每个数$x$，可以定义它的相反数$-x$，它的相反数也在数环中，且数环对加法和乘法封闭。

        减法本质上和加法是一样的，这样就不用单独考虑减法了。

    2. 乘法算是加法的简便记法吗？似乎不能。比如$x \times 2$可以认为是$x + x$，但是$x \times 1.5$没法写成加法的形式。

* 全体实数组成一个数域，叫做实数域，记为$\mathbb R$。

    全体复数组成一个数域，叫做复数域，记为$\mathbb C$。

    说明：

    1. 什么是数域？

    2. 数域有大数域，小数域的概念吗？比如实数是复数的一部分，所以实数数域比复数数域更小吗？

* 线性空间

    设$V$是一个非空集合，$P$是一个数域，如果$V$满足如下两个条件：

    1. 在$V$中定义一个封闭的加法运算，即当$\boldsymbol x, \boldsymbol y \in V$时，有惟一的和$\boldsymbol x + \boldsymbol y \in V$，并且加法运算满足 4 条性质：

        1. $\boldsymbol x + \boldsymbol y = \boldsymbol y + \boldsymbol x$ （交换律）

        2. $\boldsymbol x + (\boldsymbol y + \boldsymbol z) = (\boldsymbol x + \boldsymbol y) + z$ （结合律）

        3. 存在零元素$\boldsymbol 0 \in V$，对于$V$中任何一个元素$\boldsymbol x$都有$\boldsymbol x + \boldsymbol 0 = \boldsymbol x$

        4. 存在负元素，即对任一元素$\boldsymbol x \in V$，存在有一元素$\boldsymbol y \in V$，使$\boldsymbol x + \boldsymbol y = \boldsymbol 0$

    2. 在$V$中定义一个封闭的数乘运算（数与元素的乘法），即当$\boldsymbol x \in V$, $\lambda \in P$时，有惟一的$\lambda \boldsymbol x \in V$，且数乘运算满足 4 条性质：

        1. $(\lambda + \mu) \boldsymbol x = \lambda \boldsymbol x + \mu \boldsymbol x$ （分配律）

        2. $\lambda (\boldsymbol x + \boldsymbol y) = \lambda \boldsymbol x + \lambda \boldsymbol y$ （数因子分配律）

        3. $\lambda (\mu \boldsymbol x) = (\lambda \mu) \boldsymbol x$ （结合律）

        4. $1 \boldsymbol x = \boldsymbol x$

    其中$\boldsymbol x$，$\boldsymbol y$，$\boldsymbol z$表示$V$中的任意元素，$\lambda$，$\mu$是数域$P$中任意数，$1$是数域$P$中的单位数。

    这时，我们说$V$是数域$P$上的线性空间。

    不管$V$的元素如何，当$P$为实数域$\mathbb R$时，则称$V$为实线性空间，当$P$为复数域$\mathbb C$时，就称$V$为复线性空间。

    说明：

    1. 零元素和数域$P$上的单位数$1$的存在很重要吗？为什么？如果删去这两条性质，会有什么后果？

    2. 线性运算主要就是相加和数乘，不包括乘法、除法、指数对数运算等等。

* 线性空间例子

    * 以实数为系数，次数不超过$n$的一元多项式的全体（包括 0），记作

        $$P[x]_n = \{ a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0 \ \vert \  a_n, a_{n-1}, \cdots, a_1, a_0 \in \mathbb R \}$$

        按照多项式的加法和数乘，构成一个线性空间。

* 线性相关与线性无关

    如果式$\boldsymbol x = k_1 \boldsymbol x_1 + k_2 \boldsymbol x_2 + \cdots + k_r \boldsymbol x_r$中的$k_1$，$k_2$，$\cdots$，$k_r$不全为零，且使

    $$k_1 \boldsymbol x_1 + k_2 \boldsymbol x_2 + \cdots + k_r \boldsymbol x_r = \boldsymbol 0$$

    则称向量组$\boldsymbol x_1$，$\boldsymbol x_2$，$\cdots$，$\boldsymbol x_r$线性相关，否则就称其为线性无关。

    说明：

    1. 如果一组向量是线性相关的，那么其中一个向量就能被其他向量通过线性运算表示出来。
    
        比如 xOy 平面中任何一个向量都可以被$\vec{\boldsymbol x_0} = (1, 0)$和$\vec{\boldsymbol y_0} = (0, 1)$通过线性运算表示出来，那么$\vec{\boldsymbol x_0}$, $\vec{\boldsymbol y_0}$与 xOy 平面中其他任何一个向量组成的向量组，都是线性相关的。

        如果一个向量组中，任何一个向量都无法被其它向量线性表示，那么这组向量就是线性无关的。

## note

(empty)