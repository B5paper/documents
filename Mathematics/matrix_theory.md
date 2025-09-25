# Matrix Theory 矩阵论

矩阵论里感觉可能比较有用的几个地方：

1. 线性变换，基底

2. 矩阵分解

3. 广义逆，最小二乘法求线性方程组的拟合近似解

4. 谱分析与盖尔圆

5. 矩阵微积分

6. 矩阵的特征值（有什么意义？）

## cache

* 谱分析

    谱分析是矩阵论从“静态”描述走向“动态”分析的关键桥梁，它研究的核心是矩阵的特征值集合（称为“谱”）及其相关的特征向量如何决定和揭示矩阵（以及它所代表的线性算子或系统）的深层性质和行为。

    核心概念:

    * 谱：一个矩阵的所有特征值构成的集合，记作 $\sigma(A)$。

    * 谱半径：谱中特征值绝对值的最大值，记作 $\rho(A) = \max \{ \lvert \lambda \rvert : \lambda \in \sigma(A) \}$。它决定了与矩阵相关的迭代过程（如求解线性方程组）的收敛速度。

    * 谱定理：这是谱分析的基石。它指出，厄米特矩阵（在实数域中即对称矩阵 $A^T = A$）和酉矩阵（在实数域中即正交矩阵 $A^T A=I$）都可以被单位正交基对角化。这意味着：

        * 它们有完整的正交特征向量集。

        * 矩阵的 action 可以分解为在相互垂直的特征方向上进行简单的伸缩变换。

    **谱分析的意义与应用**:

    * 系统稳定性分析：

        * 在线性微分方程组 $\frac{dx}{dt} = Ax$ 中，系统的长期行为完全由 $A$ 的谱决定。

        * 如果所有特征值的实部都小于零，系统是稳定的（解会衰减到零）。

        * 如果存在特征值的实部大于零，系统是不稳定的（解会指数增长）。

        * 示例：在结构力学中，特征值代表结构的固有振动频率，特征向量代表相应的振型。

    * 矩阵函数的计算：

        * 如果矩阵 $A$ 可对角化，即 $A = PDP^{−1}$，其中 $D$ 是对角矩阵（对角线为特征值），那么矩阵函数（如指数函数 $e^A$）的计算变得非常简单：$f(A) = P f(D) P^{−1}$。

        * 示例：$e^A = P e^D P^{−1}$，而 $e^D$ 就是对角线上每个元素取指数，这极大地简化了计算。

    * 数据降维与主成分分析（PCA）：

        * PCA 的本质就是数据的协方差矩阵的谱分解。

        * 协方差矩阵是实对称矩阵，其特征向量（主成分）指向数据方差最大的方向，对应的特征值表示该方向上的方差大小。

        * 通过保留对应最大特征值的几个特征向量，就能实现数据降维，同时保留最主要的信息。

    * 图论与网络分析：

        * 图的拉普拉斯矩阵的谱（谱图理论）揭示了图的许多重要性质，如连通性、分割方式、 robustness 等。

        * 例如，拉普拉斯矩阵的第二小特征值（称为代数连通度）的大小反映了图的连通强度。

    * 量子力学：

        * 系统的哈密顿算符对应的矩阵的特征值代表系统可能存在的能级。

    **代码**：

    谱分析的核心工具就是特征值分解和奇异值分解（SVD）。SVD 可以看作是任意矩阵的谱分析的推广。

    * numpy

        ```py
        import numpy as np
        import matplotlib.pyplot as plt

        # 生成一个对称矩阵（其谱是实数）
        A = np.random.randn(5, 5)
        A = A + A.T # 使其对称

        # 1. 特征值分解 (Spectural Decomposition)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # eigenvalues 就是谱
        # eigenvectors 的每一列是对应的特征向量

        print("谱（特征值）:", eigenvalues)
        print("谱半径:", np.max(np.abs(eigenvalues)))

        # 验证谱定理：对于对称矩阵，特征向量是正交的
        print("特征向量矩阵是否正交（近似单位矩阵）:")
        print(eigenvectors.T @ eigenvectors) # 应近似于单位矩阵

        # 2. 奇异值分解 (SVD) - 适用于任何矩阵
        B = np.random.randn(4, 6)
        U, S, Vt = np.linalg.svd(B)
        # S 是奇异值向量（非负），奇异值的平方就是 B^T B 或 B B^T 的特征值。
        print("奇异值谱:", S)
        ```

    * SciPy

        ```py
        from scipy.linalg import eigh, schur

        # eigh 是专门用于厄米特/对称矩阵的，更高效且能确保特征值按序返回
        eigvals_symmetric, eigvecs_symmetric = eigh(A)
        print("按升序排列的特征值:", eigvals_symmetric)

        # Schur 分解可以得到（拟）三角矩阵，对角线上的元素就是特征值，数值上比直接求特征值更稳定。
        T, Z = schur(A)
        print("Schur 形式矩阵 T（上三角），对角线是特征值:")
        print(np.diag(T))
        ```

    谱分析是矩阵论的精华所在，它将抽象的矩阵运算转化为对特征值（谱）和特征方向（特征向量）的直观理解。通过“谱”这个透镜，我们可以洞察线性系统的稳定性、数据的内在结构、网络的拓扑特性等深远问题。它无疑是连接矩阵理论与众多科学工程应用的核心桥梁。

* 矩阵论

    矩阵论可以理解为线性代数的深化和扩展。如果说本科的线性代数主要研究有限维向量空间和线性映射的基本性质（如行列式、矩阵运算、特征值、二次型等），那么矩阵论则在此基础上，向更深、更广、更实用的方向拓展。

    核心内容:

    * 矩阵分解：这是矩阵论的核心内容。将复杂的矩阵分解成几个结构简单、性质清晰的矩阵的乘积，以便于分析和计算。

        LU分解：用于求解线性方程组。

        QR分解：用于求解最小二乘问题和特征值计算。

        特征值分解：将方阵对角化，用于分析系统的稳定性和振动模式。

        奇异值分解（SVD）：极其重要，适用于任意矩阵。是数据降维（如PCA）、推荐系统、图像压缩、自然语言处理等领域的数学基础。

    * 矩阵范数：衡量矩阵的“大小”，类似于向量的模。用于分析线性方程组的误差稳定性、数值计算的收敛性等。

    * 矩阵函数：如何定义和计算矩阵的函数，如 $e^A$（矩阵指数，用于求解微分方程组），$\sin (A)$，$\sqrt(A)$ 等。

    * 广义逆矩阵（Moore-Penrose伪逆）：对于非方阵或奇异矩阵，定义其“逆”，用于求解无解或有多解的矛盾线性方程组。是最小二乘法的理论基础。

    * 特殊矩阵类：深入研究具有特殊结构的矩阵，如对称矩阵、Hermite矩阵、正定矩阵、酉矩阵、Toeplitz矩阵等，它们具有更好的性质和更高效的计算方法。

    * 矩阵的扰动理论：研究当矩阵元素发生微小变化时，其特征值、特征向量等性质如何变化，对数值分析至关重要。

    * 非负矩阵：元素均为非负的矩阵，在经济学、概率论（马尔可夫链）和网络科学中有广泛应用。

    应用领域：

    * 数值计算：是各种计算算法的核心。

    * 优化理论：最小二乘、二次规划等。

    * 信号与图像处理：滤波、压缩、去噪。

    * 机器学习与数据科学：主成分分析（PCA）、线性判别分析（LDA）、推荐系统、神经网络（本质是层层的矩阵运算）。

    * 量子力学：算符可以用矩阵表示。

    * 网络科学：图的邻接矩阵、拉普拉斯矩阵。

    教材：

    * 《矩阵论》- 程云鹏，张凯院等：国内经典教材，内容全面，适合工科研究生。

    * 《Matrix Analysis》- Roger A. Horn, Charles R. Johnson：矩阵领域的“圣经”，理论深度很高，适合数学基础好的读者。

    * 《Linear Algebra and Its Applications》- Gilbert Strang：MIT经典教材，直观易懂，特别注重矩阵的应用和几何解释，非常适合入门和建立直觉。他的公开课在MIT OCW上可以找到。

    在线课程：

    * MIT OpenCourseWare - 18.065 Linear Algebra and Learning from Data：Gilbert Strang教授的新课，紧密结合了矩阵论和机器学习。

    * 3Blue1Brown的“线性代数的本质”系列视频：用极其出色的动画直观解释线性代数的核心概念，必看。

    Python中的相关库与函数:

    * NumPy

        * 创建矩阵

            ```py
            import numpy as np
            A = np.array([[1, 2], [3, 4]]) # 从列表创建
            B = np.mat('1 2; 3 4')         # 从字符串创建矩阵（更符合数学习惯）
            I = np.eye(3)                  # 3x3 单位矩阵
            Z = np.zeros((2, 2))           # 2x2 零矩阵
            ```

        * 基础运算

            ```py
            A + B    # 加法
            A - B    # 减法
            A * B    # **注意：这是逐元素相乘，不是矩阵乘法！**
            A @ B    # Python3.5+ 的矩阵乘法运算符
            np.dot(A, B) # 矩阵乘法
            np.linalg.inv(A) # 矩阵求逆（如果可逆）
            A.T      # 矩阵转置
            ```

        * 分解与高级运算

            ```py
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eig(A)

            # 奇异值分解（SVD）
            U, S, Vt = np.linalg.svd(A)

            # QR分解
            Q, R = np.linalg.qr(A)

            # 计算范数
            norm_l2 = np.linalg.norm(A, ord=2) # 2-范数
            norm_fro = np.linalg.norm(A, 'fro') # Frobenius范数

            # 求解线性方程组 Ax = b
            x = np.linalg.solve(A, b)

            # 计算伪逆（广义逆）
            A_pinv = np.linalg.pinv(A)
            ```

    * SciPy

        * 更特殊的分解

            ```py
            import scipy.linalg

            # Schur分解
            T, Z = scipy.linalg.schur(A)

            # LU分解（带 pivoting）
            P, L, U = scipy.linalg.lu(A)

            # 矩阵函数
            exp_A = scipy.linalg.expm(A)   # 矩阵指数 e^A
            sin_A = scipy.linalg.sinm(A)   # 矩阵正弦
            ```

    * 其他相关库

        `Scikit-learn`：机器学习库，内部大量使用SVD、QR等矩阵分解进行建模（如decomposition.PCA, decomposition.TruncatedSVD）。

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