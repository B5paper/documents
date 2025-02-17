* control system note

    * 对冲激响应（impulse response）的理解

        $$x(t) = \sum_{i=0}^n u(i \Delta T) \cdot \Delta T \cdot h_{\Delta} (t - i \Delta T)$$

        如何理解这个公式？$x(t)$是输出，$u(t)$是输入，显然$u(i \Delta T)$是将所有时间按照$\Delta T$进行分隔，得到很多的等宽区间，而$i \Delta T$就是这些区间的左端点（因为$i$从 0 开始计数，所以不是右端点）。

        我们假设$u(t)$输入如下所示：

        <div style='text-align:center'>
        <img width=300 alt="u(t)" src='./pic_1.png'>
        </div>

        $h_{\Delta}(t)$是对$\Delta T$宽度的一个信号的响应，我们假设如下图所示：

        <div style='text-align:center'>
        <img width=300 alt="h_\Delta(t)" src='./pic_2.png'>
        </div>

        那么$x(t)$就是多个$h_\Delta (t)$乘$u(t)$对应位置的值再相加。如下图所示：

        <div style='text-align:center'>
        <img width=300 alt='multiply_and_add' src='pic_3.png'>
        </div>

        小蓝点组成多个纵列，第 1 个纵列有 1 个，第 2 条纵列有 2 个小蓝圈，第 3 条纵列有 3 个小蓝圈，后面一直维持 4 个，再往后走慢慢又变成 3 个，2 个，1 个。

        现在我们把每条纵列上的小蓝圆圈的值相加起来，即得到$x(t)$的输出。这个相乘 -> 时延 -> 叠加的过程，就是形成冲激响应的过程。

        将$\Delta T$取极限趋近于 0，即可得到连续情况下的冲激响应。