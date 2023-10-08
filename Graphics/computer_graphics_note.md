# Computer Graphics Note

## Ray tracing

所谓的渲染即把三维物体映射成二维的像素。通常有两种渲染方式：

In *object-order rendering*, each object is considered in turn, and for each object all the pixels that it influences are found and updated. In *image-order rendering*, each pixel is considered in turn, and for each pixel all the objects that influence it are found and the pixel value is computed.

Broadly speaking, image-order rendering is simpler to get working and more flexible in the effects that can be produced, and usually (though not always) takes much more execution time to produce a comparable image.

通常基于像素的渲染更简单，但耗时更长。

假设摄像机的位置为$\boldsymbol e$，空间中某个点为$\boldsymbol s$，那么我们可以用下面的式子描述一条方向上每个位置的坐标：

$$\boldsymbol p (t) = \boldsymbol e + t(\boldsymbol s - \boldsymbol e)$$

<div style='text-align:center'>
<img width=700 src='./pics/computer_graphics_note_pics/pic_1.jpg'>
</div>

其中，$t$表示某个点与摄像机位置$\boldsymbol e$的距离。$t$的范围为$(-\infty, \infty)$。

特别地，当$t = 0$时，$\boldsymbol p(0) = \boldsymbol e$；当$t = 1$时，$\boldsymbol p(1) = \boldsymbol s$。

之所以将$\boldsymbol p$写成$t$的函数，是因为后面我们求交点时，只要求解出$t$，就可以确实光线与物体的交点。

为进一步简化表示，我们令$\boldsymbol d = \boldsymbol s - \boldsymbol e$，表示射线的方向，那么就可以将射线表示为

$$\boldsymbol p(t) = \boldsymbol e + t \boldsymbol d$$

<div style='text-align:center'>
<img width=700 src='./pics/computer_graphics_note_pics/pic_2.jpg'>
</div>

如上图所示，我们只需要求出来$t_0$，就可以拿到交点坐标。

注意，这里的$\overset{\rightarrow} d$不一定是单位向量。

### 摄像机的坐标系

假设摄像机位于原点，$\boldsymbol u$为摄像机的右方向，$\boldsymbol v$为摄像机的上方向，$\boldsymbol w$为$\boldsymbol u \times \boldsymbol v$得到的方向，摄像机面向$- \boldsymbol w$方向。如下图所示：

<div style='text-align:center'>
<img width=700 src='./pics/computer_graphics_note_pics/pic_3.jpg'>
</div>

### 投影与光线生成

所谓的渲染，即把三维空间中的点按照一定方法平移后，映射到一个二维平面，这个平面就是我们最终得到的二维图像。

有两种常用的平移方式，一种是平行投影，一种是透视投影。

#### 平行投影（Orthographic Views，正投影）

<div style='text-align:center'>
<img width=700 src='./pics/computer_graphics_note_pics/pic_4.jpg'>
</div>

如图所示，我们把摄像机放在平面的中间，然后按照这个平面的方向发射光线，所有的光线都是平行的，最终撞到的物体的颜色就是要渲染的颜色。

首先我们把二维平面的像素位置映射到空间坐标系中。假设二维图片的宽度为$n_x$个像素，高度为$n_y$个像素，将其映射到三维空间中，左边界为$l$，右边界为$r$，上边界为$t$，下边界为$b$。对于某个像素点$(i, j)$（使用 xy 坐标系），其在三维空间中的坐标为：

$$\begin{aligned}
&u = l + (r - l)(i + 0.5) / n_x \\
&v = b + (t - b)(j + 0.5) / n_y
\end{aligned}
$$

此时我们可以得到，每条光线的出发点为

$$\boldsymbol e + u \boldsymbol u + v \boldsymbol v$$

方向为

$$-\boldsymbol w$$

#### 透视投影

<div style='text-align:center'>
<img width=700 src='./pics/computer_graphics_note_pics/pic_5.jpg'>
</div>

如图所示，首先将二维平面放到摄像机前$d$距离的位置，然后将相机位置和所有像素相连，即可得到每条光线的方向。

我们可以得到每条光线的初始位置：

$$\boldsymbol e$$

和方向：

$$-d \boldsymbol w + u \boldsymbol u + v \boldsymbol v$$

其中$u$，$v$的计算方法和平行投影中$u$，$v$的计算方法相同。

### Intersection

为了计算射线与物体是否相交，以及相交的交点位置，我们希望能解一个方程，如果这个方程有解，那么就说明相交，如果无解，就不相交。

这个方程的形式为

$$f(\boldsymbol p(t)) = 0 \Leftrightarrow f(\boldsymbol e + t \boldsymbol d) = 0$$

因为$\boldsymbol e$，$\boldsymbol d$都是已知，所以方程转化为求实数$t$是否存在。

#### 与球相交 ray-sephere intersection

假设一个球体的球心是$\boldsymbol c = (x_c, y_c, z_c)$，半径为$R$，那么它的球面上的点满足

$$(x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 - R^2 = 0$$

设$\boldsymbol p = (x, y, z)$，我们可以将这个方程重写成

$$(\boldsymbol p - \boldsymbol c) \cdot (\boldsymbol p - \boldsymbol c) - R^2 = 0$$

又因为$\boldsymbol p$只依赖于$t$，即$\boldsymbol p(t) = \boldsymbol e + t \boldsymbol d$，代入上面的方程后，可以得到一个只与$t$相关的方程：

$$(\boldsymbol d \cdot \boldsymbol d) t^2 + 2 \boldsymbol d \cdot (\boldsymbol e − \boldsymbol c) t + (\boldsymbol e − \boldsymbol c) \cdot (\boldsymbol e − \boldsymbol c) − R^2 = 0$$

这是一个关于$t$的二次函数。我们根据求根公式就可以得到方程的解。根据较小的$t$值即可得到第一次相交的点。

#### 与三角形相交 ray-triangle intersection

为了求解射线和三角形的相交，我们实际上是在求解射线的参数方程和三角形的参数方程的联立（不懂这一步，有空了看看参数方程相关的数学知识）：

$$
\left\{
\begin{aligned}
&x_e + t x_d = f(u, v) \\
&y_e + t y_d = g(u, v) \\
&z_e + t z_d = h(u, v) \\
\end{aligned}
\right.
$$

or

$$
\boldsymbol e + t \boldsymbol d = \boldsymbol f(u, v)
$$

设三角形三个顶点的坐标为$\boldsymbol a$，$\boldsymbol b$，$\boldsymbol c$，射线被表示为$\boldsymbol e + t \boldsymbol d$，那么直线与三角形所在平面相交，就可以表示为方程：

$$\boldsymbol e + t \boldsymbol d = \boldsymbol a + \beta (\boldsymbol b - \boldsymbol a) + \gamma (\boldsymbol c - \boldsymbol a)$$

解出$t$后，我们便可以得到交点$\boldsymbol p$的位置：$\boldsymbol p = \boldsymbol e + t \boldsymbol d$。

设$\boldsymbol e = (x_e, y_e, z_e)$，$\boldsymbol d = (x_d, y_d, z_d)$，$\boldsymbol a = (x_a, y_a, z_a)$，$\boldsymbol b = (x_b, y_b, z_b)$，$\boldsymbol c = (x_c, y_c, z_c)$，上述线性方程组可展开为：

$$
\left\{
\begin{aligned}
&x_e + t x_d = x_a + \beta (x_b − x_a) + \gamma (x_c − x_a) \\
&y_e + t y_d = y_a + \beta (y_b − y_a) + \gamma (y_c − y_a) \\
&z_e + t z_d = z_a + \beta (z_b − z_a) + \gamma (z_c − z_a)
\end{aligned}
\right.
$$

我们可以将其重新写成线性方程组的形式：

$$
\begin{bmatrix}
x_a − x_b &x_a − x_c &x_d \\
y_a − y_b &y_a − y_c &y_d \\
z_a − z_b &z_a − z_c &z_d
\end{bmatrix}
\begin{bmatrix}
\beta \\
\gamma \\
t
\end{bmatrix}
=
\begin{bmatrix}
x_a − x_e \\
y_a − y_e \\
z_a − z_e
\end{bmatrix}
$$

我们可以通过克拉默法则（Cramer’s rule）得到线性方程组的解：

（未完待续）

#### 射线与多边形的交点

### Shading

有很多种着色模型，下面是最简单的几种

* Lambertian Shading

    只考虑光照角度的影响，不考虑观察者的位置。假设单通道颜色的强度为$L$，光强为$I$，那么$L$可以被计算为：

    $$L = k_d I \max(0, \boldsymbol n \cdot \boldsymbol l)$$

    其中，$\boldsymbol n$是光线与物体交点处的物体表面的法线的单位向量，$\boldsymbol l$是从交点指向光源的单位法线向量。

    $k_d$ is the diffuse coefficient, or the surface color。$k_d$其实可以理解为对不同颜色通道的光的反射能力，其实也就代表了物体的颜色。

    $\boldsymbol n \cdot \boldsymbol l = \cos \theta$，其实代表了光照与法线的夹角。

    Lambertian shading is view independent: the color of a surface does not depend on the direction from which you look.

* Blinn-Phong Shading

    在 Lambert 光照的基础上加上镜面反射，就构成了 Blinn-Phong 光照模型。

    假设


