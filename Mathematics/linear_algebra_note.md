# Linear Algebra Note

## cache

* 矩阵乘法

    设$\boldsymbol A = (a_{ij})$是一个$m \times s$矩阵，$\boldsymbol B = (b_{ij})$是一个$s \times n$矩阵，那么规定矩阵$\boldsymbol A$与矩阵$\boldsymbol B$的乘积是一个$m \times n$矩阵$\boldsymbol C = (c_{ij})$，其中

    $$c_{ij} = \sum_{k=1}^s a_{ik} b_{kj}$$

    $$(i = 1, 2, \cdots, m; j = 1, 2, \cdots, n)$$

    并把此乘积记作$\boldsymbol C = \boldsymbol A \boldsymbol B$.

    说明：

    * 看书上的意思，矩阵乘法是由线性变换得到的。是否有根据其他的运算得到某种算子的例子呢？

    * 矩阵的乘法真的是由线性变换启发而得到的吗，还是根据其他的例子启发而得到的？

* 矩阵的运算满足结合律和分配律，但不满足交换律：

    1. $(\boldsymbol{AB}) \boldsymbol C = \boldsymbol A (\boldsymbol{BC})$

    2. $\lambda (\boldsymbol{AB}) = (\lambda \boldsymbol A) \boldsymbol B = \boldsymbol A (\lambda \boldsymbol B)$

    3. $\boldsymbol A (\boldsymbol B + \boldsymbol C) = \boldsymbol{AB} + \boldsymbol{AC}$

        $(\boldsymbol B + \boldsymbol C) \boldsymbol A = \boldsymbol{BA} + \boldsymbol{CA}$

    说明：

    * 为什么要分这么多情况讨论？能否仅使用$(\boldsymbol{AB}) \boldsymbol C = \boldsymbol A (\boldsymbol{BC})$和$\boldsymbol A (\boldsymbol B + \boldsymbol C) = \boldsymbol{AB} + \boldsymbol{AC}$就推导出其他所有的运算律？如果不能，为什么？

* 如果矩阵$\boldsymbol A$是一个方阵，那么就可以定义矩阵的幂：

    $$\boldsymbol A^k \boldsymbol A^l = \boldsymbol A^{k + l}$$

    $$(\boldsymbol A^k)^l = \boldsymbol A^{kl}$$

* 由于$(\boldsymbol{AB})^2 = \boldsymbol{ABAB}$，而$\boldsymbol A^2 \boldsymbol B^2 = \boldsymbol{AABB}$

    由于矩阵乘法不满足交换律，所以没办法把$\boldsymbol{ABAB}$中间的$\boldsymbol{BA}$变成$\boldsymbol{AB}$，因此$\boldsymbol{ABAB} \neq \boldsymbol{AABB}$。

    进而我们可以得出$(\boldsymbol{AB})^2 \neq \boldsymbol A^2 \boldsymbol B^2$，并且可以推广到$(\boldsymbol{AB})^k \neq \boldsymbol A^k \boldsymbol B^k$。

* 矩阵的转置满足的运算律

    1. $(\boldsymbol A^\intercal)^\intercal = \boldsymbol A$

    2. $(\boldsymbol A + \boldsymbol B)^\intercal = \boldsymbol A^\intercal + \boldsymbol B^\intercal$

    3. $(\lambda \boldsymbol A)^\intercal = \lambda \boldsymbol A^\intercal$

    4. $(\boldsymbol{A B})^\intercal = \boldsymbol B^\intercal \boldsymbol A^\intercal$

## note
