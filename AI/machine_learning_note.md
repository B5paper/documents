# Machine Learning Note

## cache

* 对于$z = w_1 x_1 + w_2  x_2$的情况，我们首先模仿只有一个$x$的写法，得到

    ```py
    def phi(z, theta):
        if z >= theta:
            return 1
        else:
            return -1
        
    def z(x1, x2, w1, w2):
        return w1 * x1 + w2 * x2

    def z_gt(x1, x2):
        return 0.3 * x1 + 0.5 * x2

    def main_3():
        theta = 0.5
        w1, w2 = 0.7, 0.2
        for i in range(10):
            x1 = random.random() * 2
            x2 = random.random() * 2
            val_z_gt = z_gt(x1, x2)
            val_z = z(x1, x2, w1, w2)
            val_phi_gt = phi(val_z_gt, theta)
            val_phi = phi(val_z, theta)
            if abs(val_phi_gt - val_phi) < 0.0001:
                delta_w1 = 0
                delta_w2 = 0
            elif val_phi_gt - val_phi > 0:
                delta_w1 = 0.1
                delta_w2 = 0.1
            else:
                delta_w1 = -0.1
                delta_w2 = -0.1
            print('round {}, w1: {:.2f}, w2: {:.2f}, delta_w1: {:.2f}, delta_w2: {:.2f}'.format(
                i, w1, w2, delta_w1, delta_w2))
            w1 += delta_w1
            w2 += delta_w2
        return

    if __name__ == '__main__':
        main_3()
    ```

    output:

    ```
    round 0, w1: 0.70, w2: 0.20, delta_w1: -0.10, delta_w2: -0.10
    round 1, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 2, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 3, w1: 0.60, w2: 0.10, delta_w1: 0.10, delta_w2: 0.10
    round 4, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 5, w1: 0.70, w2: 0.20, delta_w1: -0.10, delta_w2: -0.10
    round 6, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 7, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 8, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    round 9, w1: 0.60, w2: 0.10, delta_w1: 0.00, delta_w2: 0.00
    ```

    我们看到$w_1$与$w_2$绑定在了一起，要么都增大，要么都减小。还缺少一个关键因素，如果$w_1$偏大，$w_2$偏小，我们希望`delta_w1`为负数，`delta_w2`为正数。不清楚该如何做到。

    思路二，我们可以把$x$看作已知量，把$w$看作变量，此时只要求出$z$对$w$的偏导，即$x$，即可得到变化的方向。由此可写出如下代码：

    ```py
    def phi(z, theta):
        if z >= theta:
            return 1
        else:
            return -1
        
    def z(x1, x2, w1, w2):
        return w1 * x1 + w2 * x2

    def z_gt(x1, x2):
        return 0.3 * x1 + 0.5 * x2

    def main_3():
        theta = 0.5
        w1, w2 = 0.7, 0.2
        for i in range(10):
            x1 = random.random() * 2
            x2 = random.random() * 2
            val_z_gt = z_gt(x1, x2)
            val_z = z(x1, x2, w1, w2)
            val_phi_gt = phi(val_z_gt, theta)
            val_phi = phi(val_z, theta)
            if abs(val_phi_gt - val_phi) < 0.0001:
                delta_w1 = 0
                delta_w2 = 0
            elif val_phi_gt - val_phi > 0:
                delta_w1 = 0.1 * x1
                delta_w2 = 0.1 * x2
            else:
                delta_w1 = -0.1 * x1
                delta_w2 = -0.1 * x2
            print('round {}, w1: {:.2f}, w2: {:.2f}, delta_w1: {:.2f}, delta_w2: {:.2f}'.format(
                i, w1, w2, delta_w1, delta_w2))
            w1 += delta_w1
            w2 += delta_w2
        return

    if __name__ == '__main__':
        main_3()
    ```

    output:

    ```
    round 0, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 1, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 2, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 3, w1: 0.70, w2: 0.20, delta_w1: 0.00, delta_w2: 0.00
    round 4, w1: 0.70, w2: 0.20, delta_w1: -0.09, delta_w2: -0.02
    round 5, w1: 0.61, w2: 0.18, delta_w1: 0.00, delta_w2: 0.00
    round 6, w1: 0.61, w2: 0.18, delta_w1: 0.00, delta_w2: 0.18
    round 7, w1: 0.61, w2: 0.36, delta_w1: 0.00, delta_w2: 0.00
    round 8, w1: 0.61, w2: 0.36, delta_w1: 0.00, delta_w2: 0.00
    round 9, w1: 0.61, w2: 0.36, delta_w1: 0.00, delta_w2: 0.00
    ```

    这个方向明显是对的。

    这里借助了偏导。如果不知道偏导数这个概念，是否可以用其他方式想出来？

* 画$y = \begin{cases} 1,\ x \geq \theta \\ -1,\ x \lt \theta \end{cases}$的图像，当$\theta = 0.5$时，有

    <div>
    <img width=700 src='../../Reference_resources/ref_35/pic_1.png'>
    </div>

    这里想讨论的是，这个函数是否需要能够微分？

    我们需要的似乎只是$y_{gt} - y$的正负，不需要知道这个函数的导数。

* 乘法的大与小

    对于$z = xy$，如果想增大$z$，那么当$x \gt 0$时，可以增大$y$；当$x \lt 0$时，可以减小$y$。如果想减小$z$，那么当$x \gt 0$时，需要减小$y$，当$x \lt 0$时，需要增大$y$。

    另外一种想法是，我们把$x$与$0$的大小比值提前，当$x \gt 0$，如果想增大$z$，那么需要增大$y$；如果想减小$z$，那么需要减小$y$。当$x \gt 0$，如果想增大$z$，那么需要减小$y$，如果想减小$z$，则需要增大$y$。

    现在的问题是：这样的分类讨论是否有更简洁统一的表述？

* 离散情况下通过负反馈复合函数的参数

    我们考虑最简单的一种情况：$z = wx$，$\phi(z) = \begin{cases} 1, \ z >= \theta \\ -1, \ z < \theta \end{cases}$，为简单起见，我们令$\theta = 0.5$。现在假如$w_{\mathrm{gt}} = 0.7$，我们猜测$w = 0.5$，并且我们只能计算$\phi(x)$，那么是否存在一个算法，可以使我们的猜测值$w$不断逼近真实值$w_{gt}$？

    想法：我们对$x$随机采样，并计算$\phi_{\mathrm{gt}}(x)$和$\phi(x)$，若$\phi_{\mathrm{gt}} - \phi \gt 0$，则说明$w$偏小。如果我们能证明前面这句话，那么就可以写出下面的算法：

    ```python
    import random
    def main_3():
        w = 0.5
        w_gt = 0.7
        theta = 0.5
        for i in range(10):
            x = random.random() * 1.5  # [0, 1.5)
            z = w * x
            z_gt = w_gt * x
            phi = 1 if z > theta else -1
            phi_gt = 1 if z_gt > theta else -1
            w_delta = 0.1 if phi_gt - phi > 0 else -0.1
            if abs(phi_gt - phi) < 0.0001:
                w_delta = 0
            elif phi_gt - phi > 0:
                w_delta = 0.1
            else:
                w_delta = -0.1
            print('round {}, x: {:.4}, phi: {}, phi_gt: {}, w: {}, w_delta: {}'.format(
                i, x, phi, phi_gt, w, w_delta)
            )
            w += w_delta
        return

    if __name__ == '__main__':
        main_3()
    ```

    output:

    ```
    round 0, x: 0.9404, phi: -1, phi_gt: 1, w: 0.5, w_delta: 0.1
    round 1, x: 0.9234, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 2, x: 0.8939, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 3, x: 0.7078, phi: -1, phi_gt: -1, w: 0.6, w_delta: 0
    round 4, x: 0.019, phi: -1, phi_gt: -1, w: 0.6, w_delta: 0
    round 5, x: 0.8664, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 6, x: 0.9605, phi: 1, phi_gt: 1, w: 0.6, w_delta: 0
    round 7, x: 0.5712, phi: -1, phi_gt: -1, w: 0.6, w_delta: 0
    round 8, x: 0.8324, phi: -1, phi_gt: 1, w: 0.6, w_delta: 0.1
    round 9, x: 0.4142, phi: -1, phi_gt: -1, w: 0.7, w_delta: 0
    ```

    从结果看，我们的假设很可能是对的。

    尝试分析一下：

    我们画出$\phi(z)$和$\phi_{\mathrm{gt}}(z)$的图像：

    <div>
    <img alt='phi(z)' width=700 src=''>
    </div>

    再画出$z(x)$和$z_{\mathrm{gt}}(x)$的图像：

    <div>
    <img alt='z(x)' width=700 src=''>
    </div>

    可见在$(x_0, x_1)$范围内，$\phi_{\mathrm{gt}}$总是$1$，$\phi$总是$-1$，此时会不断增大$w$，并且在$w \lt w_{\mathrm{gt}}$时，$\phi_{\mathrm{gt}} \gt \phi$总是成立。

    说明：

    1. 这个过程没有计算导数，只是一个负反馈通路，但是需要全程白盒，知道当某种现象发生时，如何调整可调参数。$\phi(z)$就是为了防止计算导数。

    2. 现在考虑$z(x_1, x_2) = w_1 x_1 + w_2 x_2$，上述方案是否仍然可行？

    3. 这样分析似乎也可以：对于某个$x$，当$\phi_{gt} > \phi$时，我们需要增大$\phi$，因此需要增大$z$，因为$x$是正值，因此需要增大$w$。

* 小宽度 + 小深度时，l1 loss 下的拟合任务，sigmoid 的效果比 relu 更好

    当使用 sigmoid + 大深度时，拟合曲线会经常在局部有高频 sin 曲线的形状。

    如果每次随机 fix 3/4 的参数，只让 1/4 的参数被 optimizer 改变，在有限的 epoch 内是否效果会更好？或者每次只改变一个 weight。假设：当节点数量变多时，每个节点贡献的 effort 很小，但是也有一些是互相抵消的，是否可以让单个节点在限制条件下（其他节点参数不动），快速调整，找到自己的位置，从而使得单个节点的 contribute 最大？这样的话，就不存在互相抵消的影响了。如果不行，那么说明小 effort 比抵消作用更加重要，这样就要求我们去关注 init random seed，随机搜索的种子采样点。

* python 机器学习上感知器的例子

    ```python
    import numpy as np

    class Perceptron(object):
        def __init__(self, eta=0.01, n_iter=10):
            self.eta = eta
            self.n_iter = n_iter

        def fit(self, X, y):
            self.w_ = np.zeros(1 + X.shape[1])
            self.errors_ = []

            for _ in range(self.n_iter):
                errors = 0
                for xi, target in zip(X, y):
                    update = self.eta * (target - self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                self.errors_.append(errors)
            return self
        
        def net_input(self, X):
            return np.dot(X, self.w_[1:]) + self.w_[0]
        
        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0, 1, -1)
    ```

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