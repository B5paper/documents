# Matplotlib Note

## cache

* `plt.figure()`

    syntax:

    ```py
    plt.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, clear=False, **kwargs)
    ```

    * num: 图形标识符（数字或字符串）

    * figsize: 图形尺寸（宽度, 高度），单位为英寸

    * dpi: 分辨率，每英寸点数

    * facecolor: 图形背景颜色

    * edgecolor: 图形边框颜色

    * clear: 如果为 True 且图形已存在，则清除该图形

    example:

    * 基本用法

        ```py
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建数据
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # 创建图形
        plt.figure()
        plt.plot(x, y)
        plt.title('基础图形')
        plt.show()
        ```

    * 指定图形尺寸

        ```py
        # 创建指定大小的图形
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'r-', linewidth=2)
        plt.title('自定义尺寸图形')
        plt.grid(True)
        plt.show()
        ```

    * 多图形管理

        ```py
        # 创建第一个图形
        plt.figure(1, figsize=(6, 4))
        plt.plot(x, np.sin(x), 'b-')
        plt.title('图形 1: 正弦函数')

        # 创建第二个图形
        plt.figure(2, figsize=(6, 4))
        plt.plot(x, np.cos(x), 'g-')
        plt.title('图形 2: 余弦函数')

        # 切换回第一个图形并添加内容
        plt.figure(1)
        plt.plot(x, np.cos(x), 'r--', alpha=0.5)
        plt.legend(['sin', 'cos'])

        plt.show()
        ```

    * 自定义背景和分辨率

        ```py
        # 高分辨率、自定义背景
        plt.figure(figsize=(10, 6), dpi=100, facecolor='lightgray')
        plt.plot(x, np.sin(x), label='sin(x)')
        plt.plot(x, np.cos(x), label='cos(x)')
        plt.legend()
        plt.title('高分辨率自定义背景图形')
        plt.grid(True, alpha=0.3)
        plt.show()
        ```

    * 清除现有图形

        ```py
        # 先创建一个图形
        plt.figure(1)
        plt.plot(x, y)
        plt.title('原始图形')

        # 清除并重新绘制
        plt.figure(1, clear=True)
        plt.plot(x, np.tan(x))
        plt.title('清除后重新绘制的图形')
        plt.ylim(-5, 5)
        plt.show()
        ```

    * 使用子图

        ```py
        # 创建图形并添加子图
        fig = plt.figure(figsize=(12, 4))

        # 添加第一个子图
        ax1 = fig.add_subplot(131)
        ax1.plot(x, np.sin(x))
        ax1.set_title('正弦函数')

        # 添加第二个子图
        ax2 = fig.add_subplot(132)
        ax2.plot(x, np.cos(x), 'r-')
        ax2.set_title('余弦函数')

        # 添加第三个子图
        ax3 = fig.add_subplot(133)
        ax3.plot(x, np.exp(-x), 'g-')
        ax3.set_title('指数衰减')

        plt.tight_layout()
        plt.show()
        ```

    * 保存高质量图形

        ```py
        # 创建高分辨率图形用于保存
        plt.figure(figsize=(8, 6), dpi=150)
        x = np.linspace(0, 2*np.pi, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)

        plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
        plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('三角函数')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存为高质量图片
        plt.savefig('high_quality_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        ```

* matplotlib hello world example

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    def main():
        x = np.linspace(0, 2 * np.pi, 200)
        y = np.sin(x)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()
        return

    if __name__ == '__main__':
        main()
    ```

    画一条 sin 曲线。

    说明：

    1. `x`与`y`的 shape 都为`(200, )`

    2. `ax.plot()`只接收 shape 为`(N, )`或者`(N, 1)`的 array，不接收其他 shape 的数据，比如`(1, N)`。

## 3D plot

### draw a 3d figure

```python
import matplotlib.pyplot as plt
import numpy as np

def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    line = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    xs = line[:, 0]
    ys = line[:, 1]
    zs = line[:, 2]
    ax.plot(xs, ys, zs)
    plt.show()

if __name__ == '__main__':
    main()
```

效果：

<div style='text-align:center'>
<img width=700 src='./pics/matplotlib_note/pic_1.png' />
</div>

### draw a 3d triangle

画三维的三角形主要用到的函数是`ax.plot_trisurf()`。

```python
import matplotlib.pyplot as plt
import numpy as np

def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    triangle = np.array([
        [-0.5, 0, 0],
        [0, 1, 0],
        [0.5, 0, 0]    
    ])
    xs = triangle[:, 0]
    ys = triangle[:, 1]
    zs = triangle[:, 2]
    vertex_idxs = [
        [0, 1, 2]
    ]
    ax.plot_trisurf(xs, ys, zs, triangles=vertex_idxs)
    plt.show()

if __name__ == '__main__':
    main()
```

我们需要分别指定所有顶点的 x 坐标，y 坐标，z 坐标，然后用`triangles`参数指定顶点的索引，通过类似 opengl VBO 的方式，画出三角形。另外我们还可以用`color`参数指定三角形的颜色。

效果：

<div style='text-align:center'>
<img width=700 src='./pics/matplotlib_note/pic_2.png'>
</div>

说明：

1. 如果`plot_trisurf()`函数不指定`triangles`参数，那么函数的行为会发生变化，使用另外一种模式画图。有空了看下。

另外一种绘制方式：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

triangles =  [
    ((1,1,1),(2,2,2),(1,3,4)),
    ((2,3,4),(9,9,9),(3,4,5)),
]

ax = plt.gca(projection="3d")

ax.add_collection(Poly3DCollection(triangles))

ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])

plt.show()
```

不太懂，有时间了看看。

### draw an animation

Ref: <https://matplotlib.org/stable/gallery/animation/animation_demo.html#sphx-glr-gallery-animation-animation-demo-py>

我们可以使用`ax.clear()`和`plt.pause(duration_seconds)`的组合来绘制动画。

```python
import matplotlib.pyplot as plt
import numpy as np

def create_3d_ax():
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d', autoscale_on = False)
    return ax

def preset_ax_config(ax):
    ax.set_box_aspect([1, 1, 1])
    ax.set_autoscale_on(False)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

def clear_ax(ax):
    ax.clear()
    preset_ax_config(ax)  # 每次清空完 ax 后需要重新设置 ax 的属性，不然会恢复到默认设置

def main():
    triangle = np.array([
        [-0.5, 0, 0],
        [0, 1, 0],
        [0.5, 0, 0]
    ])
    xs = triangle[:, 0]
    ys = triangle[:, 1]
    zs = triangle[:, 2]
    vertex_idxs = [
        [0, 1, 2]
    ]
    
    ax = create_3d_ax()
    for frame_idx in range(15):
        clear_ax(ax)
        ax.plot_trisurf(xs, ys, zs, triangles=vertex_idxs)
        xs += 0.1
        plt.pause(0.1)

if __name__ == '__main__':
    main()
```

### axes scaling

如果使用默认的配置，画出来的坐标轴尺度并不是一致的，并且坐标轴会随着绘制数据的变化而动态变化。

比如在下面这种情况下：

```python
import matplotlib.pyplot as plt
import numpy as np

def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    triangle = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [1, 0, 0]
    ])
    xs = triangle[:, 0]
    ys = triangle[:, 1]
    zs = triangle[:, 2]
    vertex_idxs = [
        [0, 1, 2]
    ]
    ax.plot_trisurf(xs, ys, zs, triangles=vertex_idxs)
    plt.show()

if __name__ == '__main__':
    main()
```

<div style='text-align:center'>
<img width=700 src='./pics/matplotlib_note/pic_4.png'>
</div>

三角形沿 x 轴的边的长度为 2，高为 1，但是从效果图来看，底边和高的长度几乎相同，这样明显是不对的。

仔细看图，x 轴的刻度范围是 -1 到 1，而 y 轴的坐标范围是 0 到 1，坐标轴的刻度尺度不一致使得图形变形。

下面的配置可以使得绘制坐标轴尺度相同，并且视图静态，不随着数据的变化而变化。

```python
fig = plt.figure()
ax = fig.add_subplot(projection = '3d', autoscale_on = False)
ax.set_box_aspect([1, 1, 1])
ax.set_autoscale_on(False)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
```

我们使用这个配置画一个三角形：

```python
import matplotlib.pyplot as plt
import numpy as np

def create_3d_ax():
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d', autoscale_on = False)
    return ax

def preset_ax_config(ax):
    ax.set_box_aspect([1, 1, 1])
    ax.set_autoscale_on(False)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

def main():
    triangle = np.array([
        [-0.5, 0, 0],
        [0, 1, 0],
        [0.5, 0, 0]
    ])

    xs = triangle[:, 0]
    ys = triangle[:, 1]
    zs = triangle[:, 2]

    vertex_idxs = [
        [0, 1, 2]
    ]

    ax = create_3d_ax()
    preset_ax_config(ax)
    ax.plot_trisurf(xs, ys, zs, triangles=vertex_idxs)
    plt.show()

if __name__ == '__main__':
    main()
```

效果：

<div style='text-align:center'>
<img width=700 src='./pics/matplotlib_note/pic_3.png'>
</div>

### Backend

在 jupyter notebook 中画 3d 图时，目前使用`TkCairo`作为 backend 的效果比较好。

```python
import matplotlib
matplotlib.use('TkCairo')
import matplotlib.pyplot as plt

# plot something
```

### Draw a spot

```python
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
x = [4]
y = [3]
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.grid()
plt.plot(x, y, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
plt.show()
```

### 使用 matplotlib 画一个球体

Ref: <https://saturncloud.io/blog/rendering-a-3d-sphere-in-matplotlib-a-guide/>

这个里面用到了球的参数方程和`np.outer()`，目前对这两个都不太熟，有时间了再看吧。