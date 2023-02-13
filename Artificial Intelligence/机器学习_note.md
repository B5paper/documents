# 机器学习 笔记

周志华

机器学习定义（非正式）：在计算机上，从数据中产生模型（model）的算法称为学习算法（learning algorithm）。机器学习是研究关于“学习算法”的学问。

**机器学习定义（正式）**（[Mitchell, 1997]）

假设用$P$来评估计算机程序在某任务类$T$上的性能，若一个程序通过利用经验$E$在$T$中任务上获得了性能改善，则我们就说关于$T$和$P$，该程序对$E$进行了学习。

注：

1. 从非正式的定义中可以看到，机器学习有四大要素：数据，模型，学习算法，评价指标。

**数据集**（data set）

数据集由很多条记录组成。每条记录是关于一个事件或对象的描述，称为一个示例（instance）或样本（sample）。

一个样本又由许多个属性（attribute）或特征（feature）组成。这些属性或特征用于描述事件或物体的某个表现或性质。比如颜色，大小，高低等。

属性的具体取值称为属性值（attribute value）。属性张成的空间称为属性空间（attribute space），样本空间（sample space）或输入空间。由于空间中每个点对应一个坐标向量，所以一个示例也被称作一个特征向量（feature vector）。

一般地，令$D = \{ \boldsymbol x_1, \boldsymbol x_2, \dots, \boldsymbol x_m \}$表示包含了$m$个示例的数据集，每个示例由$d$个属性描述，则每个示例$\boldsymbol x_i = (x_{i1}; x_{i2}; \dots ; x_{id})$是$d$维样本空间$\mathcal X$中的一个向量，$\boldsymbol x_i \in \mathcal X$，其中$x_{ij}$是$\boldsymbol x_i$在第$j$个属性上的取值，$d$称为样本$\boldsymbol x_i$的维数（dimensionality）。

**训练**

从数据中学得模型的过程称为学习（learning）或训练（training），这个过程通过执行某个学习算法来完成。

**训练数据**

训练过程中使用的数据称为训练数据（training data），其中每个样本称为一个训练样本（training sample），训练样本组成的集合称为训练集（training set）。

注：

1. 训练数据和训练集有什么区别？

**假设与真相**

学习到的模型对应了关于数据的某种潜在规律，因此也把这个模型称作假设（hypothesis）。而潜在规律自身，则称为真实或真相（ground-truth）。

**假设空间**

学习过程可以看作是一个在所有假设（hypothesis）组成的空间中进行搜索的过程，搜索目标是找到与训练集匹配（fit）的假设，即能够将训练集中的瓜判断正确的假设。

注：

1. 从作者的意思来看，假设空间指的是属性的取值的所有组合。空集也算是一个假设，表示无论什么属性，都无法满足 ground-truth。

**版本空间**

可能有多个假设与训练集一致，即存在一个与训练集一致的“假设集合”，我们称之为版本空间（version space）。

注：

1. 假如一个训练集是这样的：

    ```
    id    y f1 f2 f3
    1     1  1  0  1
    2     1  0  0  1
    3     0  2  1  2
    4     0  0  2  0 
    ```

    对于$y = 1$，当$f_2 = 0$时，$f_1$和$f_3$无论取什么值都不影响。这样的一组假设集合$H_1$是满足条件的。而当$f_3 = 1$时，无论$f_1$和$f_2$取何值，都不影响分类结果，这样的一组假设集合$H_2$也是满足$y = 1$的条件的。自然，当$f_2 = 0$，$f_3 = 1$时，无论$f_1$取什么值，也都满足$y = 1$的条件。因此$H_1$，$H_2$和$H_3$组成的集合称为版本集合。

**归纳偏好**

机器学习算法在学习过程中对某种类型假设的偏好，称为归纳偏好（inductive bias），或简称为偏好。

任何一个有效的机器学习算法必有其归纳偏好。

注：

1. 作者认为如果没有归纳偏好，那么模型会每次随机抽取一个等效假设对结果进行预测，这样的学习结果没有意义。但是，实际上这正是不确定性的体现。即使是人类，也无法给出确定的输出。我认为并不是没有意义的。

## 模型评估与选择

**错误率**

通常把分类错误的样本数占样本总数的比例称为错误率（error rate）。即如果在$m$个样本中有$a$个样本分类错误，则错误率$E = a / m$。

**精度**

精度（accuracy） = 1 - 错误率。

**误差**

一般地，我们把学习器的实际预测输出与样本的真实输出之间的差异称为误差（error），学习器在训练集上的误差称为训练误差（training error）或经验误差（empirical error），在新样本上的误差称为泛化误差（generalization error）。

显然，我们希望得到泛化误差小的学习器。

**过拟合与欠拟合**

把训练样本自身的一些特点当作了所有潜在样本都会具有的一般性质，这样会导致活化性能下降，这种现象在机器学习中称为过拟合（overfitting）。

对训练样本的一般性质尚未学好，这种现象称为欠拟合（underfitting）。

**留出法**

留出法（hold-out）直接将数据集$D$划分为两个互斥的集合，其中一个集合作为训练集$S$，另一个作为测试集$T$，即$D = S \cup T$，$S \cap T = \varnothing$。在$S$上训练出模型后，用$T$来评估其测试误差，作为对泛化误差的估计。

在使用留出法时，要尽可能保持数据的一致性，例如在分类任务中要保持样本的类别比例相似，通常使用分层采样（stratified sampling）。

**交叉验证法**

**自助法**

**性能度量**

在预测任务中，给定样例集$D = \{ (\boldsymbol x_1, y_1), (\boldsymbol x_2, y_2), \dots, (\boldsymbol x_m, y_m) \}$，其中$y_i$是示例$\boldsymbol x_i$的真实标记。要评估学习器$f$的性能，就要把学习器预测结果$f(\boldsymbol x)$与真实标记$y$进行比较。这样衡量模型泛化能力的评价标准，称为性能度量（performance measure）。

* 回归任务

    回归任务最常用的性能度量是均方误差（mean squared error）：

    $$E(f; D) = \dfrac 1 m \sum\limits_{i=1}^m (f(\boldsymbol x_i) - y_i)^2$$

    更一般的，对于数据分布$\mathcal D$和概率密度函数$p(\cdot)$，均方误差可描述为

    $$E(f; \mathcal D) = \int_{\boldsymbol x \sim \mathcal D} (f(\boldsymbol x) - y)^2 p(\boldsymbol x) \mathrm d \boldsymbol x$$

    注：

    1. 概率密度函数$p(\boldsymbol x)$是什么意思？一条输入数据$\boldsymbol x$可以有概率密度函数吗？假如$p(\boldsymbol x)$是一个联合概率密度，它该怎么计算？

* 分类任务

    **错误率与精度**

    错误率是分类错误的样本数占样本总数的比例。对样例集$D$，分类错误率定义为

    $$E(f; D) = \dfrac 1 m \sum\limits_{i=1}^m \mathbb I (f(\boldsymbol x_i) \neq y_i)$$

    精度是分类正确的样本数占样本总数的比例，定义为

    $$\begin{aligned} \mathrm{acc}(f; D) &= \dfrac 1 m \sum\limits_{i=1}^m \mathbb I (f(\boldsymbol x_i) = y_i) \\ &= 1 - E(f; D)\end{aligned}$$

    更一般地，对于数据分布$\mathcal D$和概率密度函数$p(\cdot)$，错误率与精度可分别描述为

    $$E(f; \mathcal D) = \int_{\boldsymbol x \sim \mathcal D} \mathbb I (f(\boldsymbol x) \neq y) p(\boldsymbol x) \mathrm d \boldsymbol x$$

    $$\begin{aligned} \mathrm{acc} (f; \mathcal D) &= \int_{\boldsymbol x \sim \mathcal D} \mathbb I (f(\boldsymbol x) = y) p(\boldsymbol x) \mathrm d \boldsymbol x \\ &= 1 - E(f; \mathcal D) \end{aligned}$$

    **查准率，查全率与 F1**

    对于二分类问题，可将样例根据其真实类别与学习器预测类别的组合划分为真正例（true positive），假正例（false positive），真反例（true negative），假反例（false negative）四种。

    可分别用$TP$，$FP$，$TN$，$FN$表示四种情况对应的样例数。有$TP + FP + TN + FN = 样例总数$。

    这四个分类结果可组成一个混淆矩阵（confusion matrix）：

    | GT | 预测为正例 | 预测为反例 |
    | - | - | - |
    | 正例 | $TP$（真正例） | $FN$（假反例） |
    | 反例 | $FP$（假正例） | $TN$（真反例） |

    查准率（precision）$P$的定义：

    $$P = \dfrac{TP}{TP + FP}$$

    查全率（recall）$R$的定义：

    $$R = \dfrac{TP}{TP + FN}$$

    查准率和查全率是一对矛盾的度量。为了多维优化，可以画出查准率-查全率曲线，简称$P-R$曲线。绘制方法如下：

    假设学习器能输出每个测试样本被分类为正例的得分$s_i$，然后把测试样本按照得分进行降序排列。此时设定一个阈值$t$，凡是得分$s_i$大于阈值$t$的，预测结果都被认定为正例。随着阈值$t$从 0 开始增大，查全率$R$会随之单调下降，此时我们就可以以查全率$R$为横坐标，查准率$P$为纵坐标，画出$P-R$曲线。

    注：

    1. 为什么随着阈值$t$的增加，查全率$R$会减小？查准率$P$会不会单调变化？

        随着阈值$t$的增大，分类的预测结果可如下表所示。

        | GT | s | $\mathrm{pred}(t = 0)$ | $\mathrm{pred}(t = 0.2)$ | $\mathrm{pred}(t = 0.4)$ | $\mathrm{pred}(t = 0.8)$ |
        | - | - | - | - | - | - |
        | 0 | 0.9 | 1 | 1 | 1 | 1 |
        | 1 | 0.8 | 1 | 1 | 1 | 1 |
        | 1 | 0.75 | 1 | 1 | 1 | 0 |
        | 0 | 0.2 | 1 | 1 | 0 | 0 |
        | 1 | 0.15 | 1 | 0 | 0 | 0 |
        | 0 | 0.1 | 1 | 0 | 0 | 0 |

        从表中可以看到，$t$好像一条分界线，这条分界线上面的数值全都是 1，分界线下面的数值全都是 0。
    
        根据$R = \dfrac{TP}{TP + FN}$，随着$t$的增大，分界线上移，预测值为 1 的数量减小，$TP$要么不变，要么减小。而分母$TP + FN$代表 GT 中所有类别为 1 的数量，显然这个数字是不变的。综上，分子不变或减小，分母不变，所以$R$不变或减小。

        根据$P = \dfrac{TP}{TP + FP}$，显然$TP$和$FP$都是不变或减小。$TP + FP$为所有预测为 1 的样例的数量，也是减小的。这样我们无法推测$P$是增大还是减小。将其分离变量：$\dfrac 1 P = 1 + \dfrac{FP}{TP}$。若$\mathrm{pred}$从 1 变成 0，对应的$GT = 1$，那么$TP$减小，$FP$不变，$P$减小；若$\mathrm{pred}$从 1 变成 0，对应的$GT = 0$，那么$FP$减小，$TP$不变，$P$增大。因此无法判定$P$值的单调性。

    1. `scikit-learn`上的解释：<https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py>

    1. P-R 曲线一定会经过`(1, 0)`和`(0, 1)`这两个点吗？

    **ROC 与 AUC**

## 线性模型

