# 贝叶斯统计 Note

Resource:

1. 《贝叶斯统计》，韦来生，2016第1版

## 第一章 绪论

**完备事件群（分划）**

设$B_1$，$B_2$，$\dots$，$B_n$（$n$为有限或无穷）满足下列条件：

1. 两两互不相交，即$B_i B_j = \varnothing$，$i \neq j$

1. 它们之并（和）正好是样本空间，即$\sum\limits_{i=1}^n B_i = \Omega$

则称$B_1$，$B_2$，$\dots$，$B_n$是样本空间$\Omega$的一个完备事件群，又称为$\Omega$的一个分划。

**全概率公式**

设$B_1$，$B_2$，$\dots$，$B_n$（$n$为有限或无穷）是样本空间$\Omega$中的一个分划，$A$为$\Omega$中的一个事件，则全概率公式为：

$$P(A) = P(\sum\limits_{i=1}^n A B_i) = \sum\limits_{i=1}^n P(A \mid B_i) P(B_i)$$

注：

1. 为什么$n$可以是无穷？有限和无穷有什么区别？

1. 全概率公式将整个事件$A$分解成一些两两互不相交的事件之并（和），直接计算$P(A)$不容易，但分解后的那些事件的概率容易计算，从而使$P(A)$的计算变得容易了。

**贝叶斯公式**

对于样本空间$\Omega$中的一个完备事件群$\{ B_1, B_2, \dots, B_n \}$，设$A$为$\Omega$中的一个事件，且$P(B_i) \gt 0$，$i = 1, 2, \dots, n$，$P(A) \gt 0$，则按条件概率计算方式有

$$P(B_i \mid A) = \dfrac{P(A \mid B_i) P(B_i)}{P(A)} = \dfrac{P(A \mid B_i) P(B_i)}{\sum\limits_{j=1}^n P(A \mid B_j) P(B_j)}, \quad i = 1, 2, \dots, n$$

这个公式称为贝叶斯公式（Bayes formula）。

注：

1. 从形式上看这个公式不过是条件概率定义与全概率公式之间的简单推论。其之所以著名，是在这个公式的哲理意义上。

    如果把事件$A$看成结果，把事件$\{ B_i \}$看成导致这一结果的可能原因，则可以形象地认为全概率公式是由原因推结果，而贝叶斯公式是由结果推原因。

三种信息：

数理统计学的任务是要通过样本推断总体。

* 总体信息

    样本具有两重性，当把样本视为随机变量时，它有概率分布，称为总体分布。如果我们已经知道总体的分布形式，这就给了我们一种信息，称为总体信息。

    总体信息很重要，但获取总体信息需要做大量的实验，进而统计分析。

* 样本信息

    样本信息即从总体中抽取的样本所提供的信息。

    样本点越多，提供的信息越多。我们希望对样本的加工、整理，对总体的分布或对总体的某些数字特征作出统计推断。

    总体信息和样本信息放在一起，也称为抽样信息（sampling information）。

    基于总体信息和样本信息进行统计推断的理论和方法称为经典（古典）统计学（classical statistics）。它的基本观点时：把样本看成来自有一定概率分布的总体，所研究的对象是这个总体而不局限于数据本身。

* 先验信息

    先验信息（prior information）指的是在抽样，有关统计推断问题中未知参数的一些信息。

    基于上述三种信息进行统计推断的方法和理论称为贝叶斯统计学。

拓展阅读：

1. 韦来生和张伟平（2013）
1. S. Kotz 和吴喜之（2000）
1. 茆诗松和汤银才（2012年第二版，1999年第一版）

**先验分布**

参数空间$\varTheta$上的任一概率分布都称为先验分布（prior distribution）。

表示约定：用$\pi(\theta)$表示随机变量$\theta$的概率函数。当$\theta$为离散型随机变量时，$\pi(\theta_i)$（$i = 1, 2, \dots$）表示事件$\{ \theta = \theta_i \}$的概率分布，即概率$P(\theta = \theta_i)$。当$\theta$为连续型随机变量时，$\pi(\theta)$表示$\theta$的密度函数，$\theta$的分布函数用$F^\pi (\theta)$表示。

先验分布$\pi(\theta)$是在抽取样本$X$之前对参数$\theta$可能取值的认识，后验分布是在抽样之后对$\theta$的认知，记为$\pi(\theta \mid x)$，其分布函数用$F^\pi (\theta \mid x)$表示。

**后验分布**

在获得样本$X$后，$\theta$的后验分布（posterior distribution）就是给定$X = x$条件下$\theta$的条件分布，记为$\pi(\theta \mid x)$。在有密度的情形下，它的密度函数为

$$\pi(\theta \mid x) = \dfrac{h(x, \theta)}{m(x)} = \dfrac{f(x \mid \theta) \pi (\theta)}{\int_{\varTheta} f(x \mid \theta) \pi(\theta) \mathrm d \theta}$$

其中，$h(x, \theta) = f(x \mid \theta) \pi(\theta)$为$X$和$\theta$的联合密度，而

$$m(x) = \int_{\varTheta} h(x, \theta) \mathrm d \theta = \int_{\varTheta} f(x \mid \theta) \pi(\theta) \mathrm d \theta$$

为$X$的边缘分布。

当$\theta$是离散型随机变量时，先验分布可用先验分布列$\{ \pi(\theta_i), i = 1, 2, \dots \}$表示，这时后验分布是如下离散形式：

$$\pi(\theta_i \mid x) = \dfrac{f(x \mid \theta_i) \pi(\theta_i)}{\sum\limits_i f(x \mid \theta_i) \pi(\theta_i)}, \quad i = 1, 2, \dots$$

假如样本来自的总体$X$也是离散的，只要把$\pi(\theta \mid x)$和$\pi(\theta_i \mid x)$中的密度函数$f(x \mid \theta_i)$换成事件$\{ X = x \mid \theta = \theta_i \}$的概率$P(X = x \mid \theta = \theta_i)$且将$\pi(\theta_i)$换成$P(\theta = \theta_i)$即可。