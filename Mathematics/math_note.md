# Math Note

## cache

* Beta 分布

    Beta 分布简介

    Beta 分布是定义在区间 $[0, 1]$ 上的连续概率分布，由两个正参数 $\alpha$ (alpha) 和 $\beta$ (beta) 控制。

    1. 概率密度函数（PDF）

        $$f(x; \alpha, \beta) = \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)} , 0 \leq x \leq 1$$

        其中 $B(\alpha, \beta)$ 是 Beta 函数：

        $$B(\alpha, \beta) = \int_0^1 t^{\alpha - 1} (1 - t)^{\beta - 1} dt = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}$$

        $\Gamma$ 是伽马函数。

    2. 统计特征

        均值：

        $$E[X] = \frac{\alpha}{\alpha + \beta}$$

        方差：

        $$Var(X) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

        众数（当 $\alpha \gt 1$,$\beta \gt 1$ 时）：

        $$Mode = \frac{\alpha - 1}{\alpha + \beta - 2}$$

    3. 形状变化

        通过改变 $\alpha$ 和 $\beta$ 可以产生多种形状：

        * $\alpha = \beta = 1$：均匀分布 $U(0,1)$

        * $\alpha \gt 1$, $\beta \gt 1$：单峰，在 (0,1) 内

        * $\alpha < 1$, $\beta < 1$：U 形（两端高中间低）

        * $\alpha > 1$, $\beta = 1$：在 1 处为 0，在 0 处为有限值，右偏

        * $\alpha = 1$, $\beta > 1$：在 0 处为 0，在 1 处为有限值，左偏

        * $\alpha < 1$, $\beta > 1$：反 J 形（集中在 0 附近）

        * $\alpha > 1$, $\beta < 1$：J 形（集中在 1 附近）

    4. 与二项分布的关系

    如果先验分布是 Beta 分布，观测到二项分布数据后，后验分布也是 Beta 分布（共轭先验）：

    先验：$Beta(\alpha, \beta)$, 数据：$k$次成功（总试验 $n$）

    后验：$Beta(\alpha + k,\beta + n - k)$

    5. 应用场景

        * 比例或概率的建模：例如点击率、转化率等（因为范围在 0~1）

        * 贝叶斯统计：作为二项分布的成功概率 p 的先验分布

        * 顺序统计量：均匀分布样本的第 k 个顺序统计量服从 Beta(k, n-k+1)

        * 项目管理：PERT 中用于任务完成时间的建模（经过缩放平移）

        * 机器学习：作为分类器输出的校准分布，或变分推断中的变分分布

    6. 与其他分布的关系

        * Gamma 分布：如果 $X \sim \mathrm{Gamma}(\alpha, 1)$, $Y \sim \mathrm{Gamma}(\beta, 1)$,则

            $$\frac{X}{X + Y} \sim \mathrm{Beta}(\alpha, \beta)$$

        * F 分布：有转换关系

        * 二项分布/伯努利分布：共轭先验

* $\Gamma$ 函数（Gamma function）

    $\Gamma$ 函数（Gamma function） 是阶乘在实数和复数域的推广，计算它主要有以下几种方法：

    1. 定义

        * 1.1 积分定义（$Re(z) > 0$）

            $$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt$$

            对于正整数 $n$：

            $$\Gamma(n) = (n-1)!$$

            特殊值：

            $$\Gamma(1) = 1,\ \Gamma(1/2) = \sqrt{\pi}$$

        * 1.2 递推关系

            $$\Gamma(z+1) = z \Gamma(z)$$

            这个性质让我们可以将任意复平面上的 $\Gamma$ 函数计算转化到主区间（如 1 ≤ Re(z) < 2）。

    2. 计算方法

        * 2.1 直接数值积分

            * 对积分定义式进行数值积分（如 Gauss-Laguerre 积分）

            * 适用于 Re(z) > 0

        * 2.2 Lanczos 近似（最常用）

            这是计算 $\Gamma$ 函数最流行的高精度方法。核心公式：

            $$\Gamma(z+1) = \sqrt{2 \pi} \left( z + g + \frac 1 2 \right)^{z + \frac 1 2} e^{-(z + g + \frac 1 2)} A_g(z)$$

            其中 $g$ 是常数（常用 $g = 607 / 128$，$A_g(z)$ 是一组预先计算的有理函数近似。

            特点：

            * 在整个复平面（除负整数点）都有高精度

            * 计算速度快

            * 被多数数学库采用

    3. 特殊情况的简便计算
    
        * 3.1 正整数

            $$\Gamma(n) = (n-1)! = 1 \times 2 \times \cdots \times (n-1)$$

        * 3.2 半整数

            利用递推和 $\Gamma (1/2) = \sqrt{\pi}$:

            $$\Gamma(n + \frac 1 2) = \frac{(2n)!}{4^n n!} \sqrt{\pi}$$

            例如：

            $$\Gamma(3/2) = \frac 1 2 \sqrt{\pi},\ \Gamma(5/2) = \frac 3 4 \sqrt{\pi}$$

        * 3.3 负数的 $\Gamma$ 函数

            利用反射公式：

            $$\Gamma(1-z) \Gamma(z) = \frac{\pi}{\sin (\pi z)}$$

            例如：

            $$\Gamma(-1/2) = \frac{\Gamma(1/2)}{(-1 / 2)} = -2 \sqrt{\pi}$$

            注意：负整数处是极点（发散）。

    4. 常用近似公式

        * 4.1 Stirling 公式（大 z）

            $$\Gamma(z) \sim \sqrt{\frac{2 \pi}{z}}(\frac{z}{e})^z (1 + \frac{1}{12 z} + \frac{1}{288 z^2} - \cdots)$$
            
            适用于 $\lvert z \rvert$ 较大时。

        * 4.2 小 z 近似

            用递推转到 z>1 再计算。

    5. 实际应用中的计算

        编程实现：

        * Python: math.gamma()（实数）, scipy.special.gamma()（复数）

        * C++: std::tgamma()（C++11）

        * MATLAB: gamma()

        * R: gamma()

        Python 示例：

        ```python
        import math
        print(math.gamma(5))      # 24.0
        print(math.gamma(2.5))    # 1.329340388179137
        ```

        注意事项：

        * 大参数可能溢出（如 $\Gamma(172)$ 超过双精度浮点）

        * 负整数参数返回 NaN 或报错

        * 可用对数 Gamma 函数避免溢出：

            $$\ln \Gamma(z) = \ln(\Gamma(z))$$

            函数：math.lgamma()。

    6. 与 Beta 函数的关系

        Beta 函数通过 Γ 函数计算：

        $$\Beta(x, y) = \frac{\Gamma(x) \Gamma(y)}{\Gamma(x + y)}$$

        这就是 Beta 分布的归一化常数的来源。

    7. 历史与发展

        * 由欧拉（1729）和勒让德（命名）引入

        * 推广了阶乘到非整数

        * 在概率统计、数论、物理中广泛应用

    总结计算建议：

    * 日常使用：直接调用数学库函数

    * 需要理解：记住递推公式 $\Gamma(z + 1) = z \Gamma(z)$ 和 $\Gamma(1 / 2) = \sqrt{\pi}$

    * 数值稳定性：大参数时使用对数 Gamma 函数

    * 实现原理：Lanczos 近似是工业标准方法

* 初等数学的英文

    elementary mathematics

    primary mathematics （小学阶段的数学）

    fundamental mathematics：强调数学基础原理

## note
