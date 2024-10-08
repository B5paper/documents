# Machine Learning Note

## cache

* 感知器的学习算法

    1. 将权重$\boldsymbol w$初始化为零或一个极小的随机数

    2. 迭代所有的训练样本$\boldsymbol x^{(i)}$，执行以下操作：

        1. 计算输出值$\hat y$

        2. 更新权重

            $$w_j \leftarrow w_j + \Delta w_j$$

            $$\Delta w_j = \eta (y^{(i)} - \hat y^{(i)}) x_j^{(i)}$$

            其中，$\eta$是学习率，$0.0 \lt \eta \lt 1.0$。

* a simple perceptron function implemented by numpy

    ```python
    import numpy as np

    def perceptron(x: np.ndarray, w: np.ndarray, theta: float) -> float:
        z = w.T @ x
        z = z.item()  # convert z to python 'float' type
        print('z = %f' % z)
        if z >= theta:
            y = 1
        else:
            y = -1
        return y

    def main():
        w = np.array([0.3, 0.7])
        w = w.reshape(2, 1)
        x = np.array([0.5, 0.6])
        w = w.reshape(2, 1)
        theta = 0.5
        y = perceptron(w, x, theta)
        print('y = %f' % y)

    if __name__ == '__main__':
        main()
    ```

    output:

    ```
    z = 0.570000
    y = 1.000000
    ```

* 感知器算法

    设输入$\boldsymbol x = \begin{bmatrix} x_1 \\ \vdots \\ x_m \end{bmatrix}$，权重向量$\boldsymbol w = \begin{bmatrix} w_1 \\ \vdots \\ w_m \end{bmatrix}$，净输入$\boldsymbol z = w_1 x_1 + w_2 x_ 2 + \cdots + w_m x_m$，其实就是$z = \boldsymbol w^\intercal \boldsymbol x$
    
    定义激励函数为

    $$\phi (z) = \left\{ \begin{aligned} 1,\ &\text{if } z \geq \theta \\ -1,\ &\text{otherwise}  \end{aligned} \right.$$

    说明：

    1. 这里呈现的是多个输入神经元的输出通过乘一个权重后，累加作用于后置神经元。

        为什么这里刚好选择了乘权重？乘权重实际止是一个点乘操作，点乘正好对应了线性空间中的模式匹配准则。

* 术语

    感知器 perceptron

    自适应线性单元 adaptive linear neuron

    激励函数 activation function

    向量点积 vector dot product

    净输入 net input，这个指的就是$z = \boldsymbol w^\intercal \boldsymbol x$

* 鸢尾花数据集是事先已经知道了不同的类别，然后再在不同的类别上挑选了一些**主观**上认为有用的几个特征，再根据这些特征去训练模型。

    目前 AI 的难题是：事先不知道是否为相同的类别，也不知道选哪些特征可能有效。只能不断尝试收集新特征，直到突然发现有一个特征可以将一批数据集清晰地分类，那么就认为发现了一条新的“规律”。

* 鸢尾花数据集（Iris dataset）

    它包含了 Setosa、Versicolor 和 Virginica 三个品种总共 150 种鸢尾花的测量数据。

    数据表头：

    萼片长度，萼片宽度，花瓣长度，花瓣宽度，类标

    我们记此数据集为

    $$
    \bold X \in \mathbb R^{150 \times 4} = 
    \begin{bmatrix}
    x^{(1)}_1 &x^{(2)}_2 &x^{(3)}_3 &x^{(4)}_4 \\
    x^{(2)}_1 &x^{(2)}_2 &x^{(3)}_3 &x^{(4)}_4 \\
    \vdots &\vdots &\vdots &\vdots \\
    x^{(150)}_1 &x^{(150)}_2 &x^{(150)}_3 &x^{(150)}_4
    \end{bmatrix}
    $$

    记目标变量为

    $$
    \boldsymbol y =
    \begin{bmatrix}
    y^{(1)} \\
    \vdots \\
    y^{(150)}
    \end{bmatrix}
    (y \in \{ \text{Setosa, Versicolor, Virginica} \})
    $$

* 术语

    监督学习 supervised learning

    无监督学习 unsupervised learning

    强化学习 reinforcement learning

    分类 classification

    回归 regression

    样本的组别信息 group membership  (不知道是啥意思)

    二分类 binary classification

    多分类 multi-class classification

    负类别 negative class

    正类别 positve class

    线性回归 linear regression

    聚类中的簇 cluster

    数据降维 dimensionality reduction

## note

(empty)