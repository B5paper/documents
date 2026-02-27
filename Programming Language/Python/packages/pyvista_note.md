# Pyvista Note

## cache

* 简述 pyvista 画如何画二维折线图，并给出 example

    PyVista 本身主要专注于三维网格和表面可视化，但它可以非常方便地结合 Matplotlib 来绘制二维折线图，或者利用 PyVista 的 `pyvista.Chart2D` 功能（如果可用）。

    以下是两种常用的方法：

    方法一：使用 PyVista 内置的 Chart2D（推荐，无需额外库）

    从 PyVista 0.32.0 版本开始，引入了 `Chart2D` 功能，可以直接在 PyVista 的渲染窗口中绘制二维图表。

    ```python
    import pyvista as pv
    import numpy as np

    # 生成数据
    x = np.linspace(0, 10, 20)
    y = np.sin(x)

    # 创建一个二维图表
    chart = pv.Chart2D()
    # 添加折线图
    line_plot = chart.line(x, y, 'r', width=3, label='sin(x)')

    # 显示图表（这会打开一个 PyVista 渲染窗口）
    chart.show()

    # 如果你有一个现有的绘图场景，也可以将图表嵌入进去
    # 例如：plotter = pv.Plotter()
    #      plotter.add_chart(chart)
    #      plotter.show()
    ```

    方法二：使用 Matplotlib 嵌入（兼容性更好）

    如果你只需要一个独立的二维折线图，最通用的做法是使用 Matplotlib。PyVista 通常与 Matplotlib 配合使用来处理这类需求。

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 创建折线图
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.plot(x, np.cos(x), 'r--', label='cos(x)')

    # 添加标签和标题
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('2D Line Plot using Matplotlib')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 显示图形
    plt.show()
    ```

    完整的 PyVista + 三维场景示例

    以下是一个更典型的 PyVista 使用场景：在三维场景旁边显示一个二维折线图：

    ```python
    import pyvista as pv
    import numpy as np

    # 创建一些三维数据（例如：一个球体）
    sphere = pv.Sphere(radius=0.5, center=(0, 0, 0))

    # 创建二维数据
    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    # 创建绘图器
    plotter = pv.Plotter(shape=(1, 2))  # 一行两列

    # 左侧子图：显示三维球体
    plotter.subplot(0, 0)
    plotter.add_mesh(sphere, color='lightblue', show_edges=True)
    plotter.add_text("3D Sphere", font_size=10)

    # 右侧子图：显示二维折线图（通过创建单独的 Chart2D）
    plotter.subplot(0, 1)
    chart = pv.Chart2D()
    chart.line(x, y, 'g', width=2, label='sin(x)')
    plotter.add_chart(chart)  # 将图表添加到子图中

    # 显示
    plotter.show()
    ```

    总结

    - 如果**只是画二维折线图**：使用 Matplotlib 最简单（方法二）。
    - 如果**需要在 PyVista 的三维场景旁边画二维折线图**：使用 `pyvista.Chart2D`（方法一或第三个示例）。
    - 确保你的 PyVista 版本较新（>=0.32.0）以获得 `Chart2D` 支持。

* 简述 pyvista 如何画类似 matplotlib animation 那样的动态图

    PyVista 本身不直接提供类似 `matplotlib.animation` 的高级动画模块，但可以通过几种方式实现动态可视化效果。

    主要方法

    1. 使用 `update` 函数配合定时器（最常用）

        ```python
        import pyvista as pv
        import numpy as np

        # 1. 准备基础数据
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2)  # 预先计算距离，优化性能

        # 2. 初始化网格
        # 初始高度设为 0
        Z = np.zeros_like(X)
        grid = pv.StructuredGrid(X, Y, Z)
        grid["scalars"] = Z.flatten() # 绑定颜色数据

        # 3. 创建绘图窗口
        pl = pv.Plotter(notebook=False)
        # 添加网格，设置颜色映射(cmap)和固定的颜色范围(clim)
        pl.add_mesh(grid, scalars="scalars", clim=[-1, 1], cmap="viridis")

        # 关键步骤：开启交互式更新模式
        pl.show(interactive_update=True)

        # 4. 动画循环
        t = 0
        try:
            # 模拟 500 帧动画，或者使用 while True 持续运行
            for i in range(500):
                t += 0.1
                
                # 计算新的波浪高度: $z = \sin(\sqrt{x^2 + y^2} - t)$
                new_z = np.sin(dist - t)
                
                # 更新物理坐标（Z轴）
                grid.points[:, 2] = new_z.flatten()
                
                # 更新颜色标量（让颜色随高度实时变化）
                grid["scalars"] = new_z.flatten()
                
                # 刷新渲染窗口
                pl.update()
                
                # 如果需要限制帧率，可以取消下面一行的注释
                # import time; time.sleep(0.01)

        except Exception as e:
            # 当手动关闭窗口时，防止报错
            print("动画已停止")

        # 5. 释放资源
        pl.close()
        ```

    2. 使用 `add_timer_event` 方法

        这个似乎有 bug。

        ```python
        import pyvista as pv
        import numpy as np

        # 1. 创建初始数据
        mesh = pv.Sphere(theta_resolution=40, phi_resolution=40)
        # 备份原始坐标，防止变形累积（重要！）
        orig_points = mesh.points.copy()

        # 2. 设置绘图器
        pl = pv.Plotter()
        # 初始化标量，确保 add_mesh 时能识别颜色映射
        mesh["scalars"] = np.zeros(mesh.n_points)
        actor = pl.add_mesh(mesh, scalars="scalars", cmap="magma", clim=[-1, 1])

        # 时间变量
        t = 0

        # 3. 定时器回调函数 (必须接受 obj 和 event 两个参数)
        def timer_callback(obj, event):
            global t
            t += 0.1
            
            # 基于【原始坐标】进行变形，防止球体越变越大或飞走
            new_points = orig_points.copy()
            # 变形公式：$y' = y + 0.1 \cdot \sin(x \cdot 2 + t)$
            new_points[:, 1] += 0.1 * np.sin(orig_points[:, 0] * 5 + t)
            
            # 更新网格点和颜色
            mesh.points = new_points
            mesh["scalars"] = np.sin(orig_points[:, 0] * 5 + t)
            
            # 刷新渲染
            pl.render()

        # 4. 设置定时器
        # 每 16ms 触发一次回调
        pl.add_timer_event(16, timer_callback)

        # 5. 显示窗口
        pl.show()
        ```

    3. 录制视频或保存帧

        ```python
        import pyvista as pv
        import numpy as np

        # 创建数据
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)

        grid = pv.StructuredGrid(x, y, z)

        # 创建绘图器
        pl = pv.Plotter(off_screen=True)  # 离屏渲染
        pl.add_mesh(grid, scalars=z.flatten(), clim=[-2, 2])

        # 打开视频写入器
        pl.open_movie("animation.mp4", framerate=24)

        # 动画循环
        for t in np.linspace(0, 2*np.pi, 100):
            # 更新数据
            z = np.sin(np.sqrt(x**2 + y**2) - t * 5)
            grid.points[:, 2] = z.flatten()
            grid["scalars"] = z.flatten()
            
            # 写入帧
            pl.write_frame()

        # 关闭
        pl.close()
        ```

    4. 使用 `orbit_on_path` 实现摄像机动画

        ```python
        import pyvista as pv

        # 创建模型
        mesh = pv.Cone()

        # 创建绘图器
        pl = pv.Plotter()
        pl.add_mesh(mesh)

        # 设置摄像机动画路径
        path = pl.generate_orbital_path(n_points=100, viewup=[0, 1, 0], shift=5)

        # 播放动画
        pl.orbit_on_path(path, write_frames=False)
        pl.show()
        ```

    注意事项

    1. **性能优化**：对于复杂模型，尽量减少每帧更新的数据量
    2. **帧率控制**：可以使用 `time.sleep()` 或定时器控制动画速度
    3. **交互性**：PyVista 的动画通常保持交互性，可以在动画过程中旋转视角
    4. **Jupyter Notebook**：在 notebook 中使用时，需要设置 `notebook=True`

    这些方法各有适用场景，第一种方法最灵活，第二种适合周期性更新，第三种适合生成视频文件。

* 简述 PyVista 的安装和用法

    我来简要介绍 PyVista 的安装和基本用法。

    安装

    使用 pip 安装

    ```bash
    pip install pyvista
    ```

    使用 conda 安装

    ```bash
    conda install -c conda-forge pyvista
    ```

    安装所有依赖（推荐）

    ```bash
    pip install pyvista[all]  # 包括所有绘图依赖
    ```

    基本用法

    1. 基础设置

        ```python
        import pyvista as pv

        # 开启使用笔记本绘图（如果在Jupyter中）
        pv.set_jupyter_backend('panel')  # 或 'ipygany', 'trame'
        ```

    2. 创建基本几何体

        ```python
        # 创建球体
        sphere = pv.Sphere(radius=1, center=(0, 0, 0))
        sphere.plot()

        # 创建立方体
        cube = pv.Cube()
        cube.plot()

        # 创建平面
        plane = pv.Plane()
        plane.plot()
        ```

    3. 数据可视化

        ```python
        import numpy as np

        # 创建点云数据
        points = np.random.rand(100, 3)  # 100个随机点
        point_cloud = pv.PolyData(points)

        # 添加标量数据
        point_cloud['values'] = np.random.rand(100)

        # 绘制
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud, scalars='values', point_size=5)
        plotter.show()
        ```

    4. 网格操作

        ```python
        # 加载或创建网格
        mesh = pv.Sphere()

        # 获取网格信息
        print(f"点数: {mesh.n_points}")
        print(f"单元数: {mesh.n_cells}")

        # 访问点坐标
        points = mesh.points

        # 访问单元
        cells = mesh.cell
        ```

    5. 多子图绘制

        ```python
        plotter = pv.Plotter(shape=(1, 2))

        # 左子图
        plotter.subplot(0, 0)
        plotter.add_mesh(pv.Sphere(), color='red')

        # 右子图
        plotter.subplot(0, 1)
        plotter.add_mesh(pv.Cube(), color='blue')

        plotter.show()
        ```

    6. 高级可视化

        ```python
        # 创建标量场
        mesh = pv.Cube()
        mesh['distance'] = mesh.points[:, 0]  # 使用x坐标作为标量

        # 添加等高线
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='distance', cmap='coolwarm')
        plotter.add_mesh(mesh.contour(), color='black', line_width=2)
        plotter.show()
        ```

    常用技巧

    保存和导出

    ```python
    # 保存为图片
    plotter.screenshot('output.png')

    # 导出网格
    mesh.save('mesh.vtk')
    ```

    交互式操作

    ```python
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.add_axes()  # 添加坐标轴
    plotter.add_bounding_box()  # 添加边界框
    plotter.show_grid()  # 显示网格
    plotter.show()
    ```

    PyVista 功能非常强大，以上只是基础用法。它能处理复杂的3D数据可视化、网格操作和科学计算可视化需求。

* pyvista 如何画 wireframe？如何在画 mesh 时只画边线不画填充？

    在 PyVista 中画 wireframe（线框）有多种方法，我来详细介绍：

    方法1：使用 `style='wireframe'` 参数

    最简单的方法是在 `add_mesh` 时设置 `style='wireframe'`：

    ```python
    import pyvista as pv

    # 创建球体
    sphere = pv.Sphere()

    # 绘制线框模式
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, style='wireframe', color='black', line_width=2)
    plotter.show()
    ```

    方法2：使用 `show_edges=True` 保留填充

    如果你想保留填充色，同时显示边线：

    ```python
    import pyvista as pv

    # 创建球体
    sphere = pv.Sphere()

    # 显示填充和边线
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, color='lightblue', show_edges=True, edge_color='black', line_width=1)
    plotter.show()
    ```

    方法3：自定义线框样式

    更详细的控制：

    ```python
    import pyvista as pv

    # 创建多个几何体
    sphere = pv.Sphere(center=(0, 0, 0))
    cube = pv.Cube(center=(2, 0, 0))
    cone = pv.Cone(center=(4, 0, 0))

    plotter = pv.Plotter()

    # 球体：纯线框，红色粗线
    plotter.add_mesh(sphere, style='wireframe', color='red', line_width=3)

    # 立方体：纯线框，蓝色虚线（注意：PyVista可能不支持虚线，这是示意）
    plotter.add_mesh(cube, style='wireframe', color='blue', line_width=2)

    # 圆锥：填充+边线
    plotter.add_mesh(cone, color='lightgreen', show_edges=True,
                     edge_color='darkgreen', line_width=2)

    plotter.show()
    ```

    方法4：提取边线单独绘制

    你可以提取网格的边线作为单独的线框网格来绘制：

    ```python
    import pyvista as pv

    # 创建网格
    mesh = pv.Cube()

    # 提取边线
    edges = mesh.extract_feature_edges()

    plotter = pv.Plotter()

    # 绘制原始网格（半透明填充）
    plotter.add_mesh(mesh, color='lightblue', opacity=0.3)

    # 绘制边线（黑色粗线）
    plotter.add_mesh(edges, color='black', line_width=3, render_lines_as_tubes=True)

    plotter.show()
    ```

    方法5：使用不同的颜色方案

    ```python
    import pyvista as pv
    import numpy as np

    # 创建一个带有标量数据的网格
    mesh = pv.Sphere()
    mesh['data'] = np.random.random(mesh.n_points)

    plotter = pv.Plotter()

    # 线框模式，但使用标量数据着色
    plotter.add_mesh(mesh, style='wireframe', scalars='data',
                     cmap='coolwarm', line_width=2)

    plotter.show()
    ```

    方法6：多子图对比

    ```python
    import pyvista as pv

    mesh = pv.Sphere()

    plotter = pv.Plotter(shape=(1, 3))

    # 纯线框
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, style='wireframe', color='black')
    plotter.add_text("Wireframe Only")

    # 纯填充
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, color='lightblue')
    plotter.add_text("Fill Only")

    # 填充+边线
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='red')
    plotter.add_text("Fill + Edges")

    plotter.show()
    ```

    实用技巧

    设置默认线框样式

    ```python
    # 设置全局默认样式
    pv.global_theme.render_lines_as_tubes = True  # 线条渲染为管状
    pv.global_theme.line_width = 2  # 默认线宽
    ```

    交互式切换

    ```python
    plotter = pv.Plotter()
    mesh_actor = plotter.add_mesh(pv.Sphere(), color='lightblue', show_edges=True)

    # 添加滑块来控制是否显示边线
    def toggle_edges(value):
        mesh_actor.GetProperty().SetEdgeVisibility(bool(value))
        plotter.render()

    plotter.add_slider_widget(toggle_edges, [0, 1], value=1, title="Show Edges")
    plotter.show()
    ```

    注意事项

    - `style='wireframe'` 只显示线框，不显示填充
    - `show_edges=True` 在填充基础上显示边线
    - 可以通过 `line_width` 调整线宽
    - 可以通过 `edge_color` 设置边线颜色（当 `show_edges=True` 时）
    - `render_lines_as_tubes=True` 可以让线条看起来更立体

    选择哪种方法取决于你的具体需求：是想要纯线框效果，还是填充加边线的效果。

* pyvista 如何画自定义的 mesh 数据（用户提供 node 坐标和 face 信息）？

    在PyVista中，你可以使用`pyvista.PolyData`或`pyvista.UnstructuredGrid`来创建自定义的mesh数据，具体取决于你的face信息类型。以下是几种常见的方法：

    方法1：使用 PolyData（适用于三角形/四边形面片）

    ```python
    import pyvista as pv
    import numpy as np

    # 定义节点坐标
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # 定义face信息（使用vtk格式：每个面以顶点数量开头，然后是顶点索引）
    # 格式: [n_vertices, idx1, idx2, idx3, ...]
    faces = np.hstack([
        [4, 0, 1, 2, 3],  # 底面四边形
        [3, 0, 1, 4],      # 三角形面
        [3, 1, 2, 4],      # 三角形面
        [3, 2, 3, 4],      # 三角形面
        [3, 3, 0, 4]       # 三角形面
    ])

    # 创建PolyData对象
    mesh = pv.PolyData(nodes, faces)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightblue',
                     point_size=10, render_points_as_spheres=True)
    plotter.show()
    ```

    方法2：使用 UnstructuredGrid（适用于混合单元类型）

    ```python
    import pyvista as pv
    import numpy as np

    # 定义节点坐标
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # 定义单元类型
    # VTK_QUAD = 9, VTK_TRIANGLE = 5
    cell_types = np.array([9, 5, 5, 5, 5])  # 1个四边形 + 4个三角形

    # 定义face连接信息
    cells = np.array([
        4, 0, 1, 2, 3,    # 四边形单元
        3, 0, 1, 4,        # 三角形单元
        3, 1, 2, 4,        # 三角形单元
        3, 2, 3, 4,        # 三角形单元
        3, 3, 0, 4         # 三角形单元
    ])

    # 创建UnstructuredGrid
    mesh = pv.UnstructuredGrid(cells, cell_types, nodes)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightcoral')
    plotter.show()
    ```

    方法3：从面列表创建（更直观的方式）

    ```python
    import pyvista as pv
    import numpy as np

    # 定义节点坐标
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # 以更直观的方式定义面（每个面是一个列表）
    face_list = [
        [0, 1, 2, 3],  # 底面四边形
        [0, 1, 4],      # 三角形面
        [1, 2, 4],      # 三角形面
        [2, 3, 4],      # 三角形面
        [3, 0, 4]       # 三角形面
    ]

    # 转换为vtk格式
    cells = []
    for face in face_list:
        cells.append(len(face))  # 添加顶点数
        cells.extend(face)       # 添加顶点索引

    cells = np.array(cells)

    # 创建mesh
    mesh = pv.PolyData(nodes, cells)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightgreen',
                     opacity=0.7, edge_color='black', line_width=2)
    plotter.add_points(mesh, color='red', point_size=10,
                       render_points_as_spheres=True)
    plotter.show()
    ```

    方法4：添加点数据和单元数据

    ```python
    import pyvista as pv
    import numpy as np

    # 创建mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    faces = np.hstack([
        [4, 0, 1, 2, 3],
        [3, 0, 1, 4],
        [3, 1, 2, 4],
        [3, 2, 3, 4],
        [3, 3, 0, 4]
    ])

    mesh = pv.PolyData(nodes, faces)

    # 添加点数据（例如每个节点的高度）
    mesh.point_data['height'] = nodes[:, 2]  # z坐标作为高度

    # 添加单元数据（例如每个面的面积）
    cell_areas = mesh.compute_cell_sizes()['Area']
    mesh.cell_data['area'] = cell_areas

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='height', show_edges=True,
                     cmap='viridis', edge_color='black')
    plotter.show()
    ```

    关键点说明：

    1. **VTK格式要求**：面信息需要使用特殊的VTK格式：`[n_vertices, idx1, idx2, ..., n_vertices, idx1, idx2, ...]`

    2. **节点索引**：节点索引从0开始

    3. **单元类型**：

       - 三角形：VTK_TRIANGLE (5)
       - 四边形：VTK_QUAD (9)
       - 更多类型可以参考VTK文档

    4. **性能考虑**：对于大型mesh，建议使用numpy数组而不是Python列表

    选择哪种方法取决于你的具体需求和数据结构。`PolyData`适用于纯表面网格，而`UnstructuredGrid`可以处理更复杂的单元类型。

* pyvista 如何画自定义的 mesh 数据（用户提供 node 坐标和 face 信息）？

    在PyVista中，你可以使用`pyvista.PolyData`或`pyvista.UnstructuredGrid`来创建自定义的mesh数据，具体取决于你的face信息类型。以下是几种常见的方法：

    方法1：使用 PolyData（适用于三角形/四边形面片）

    ```python
    import pyvista as pv
    import numpy as np

    # 定义节点坐标
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # 定义face信息（使用vtk格式：每个面以顶点数量开头，然后是顶点索引）
    # 格式: [n_vertices, idx1, idx2, idx3, ...]
    faces = np.hstack([
        [4, 0, 1, 2, 3],  # 底面四边形
        [3, 0, 1, 4],      # 三角形面
        [3, 1, 2, 4],      # 三角形面
        [3, 2, 3, 4],      # 三角形面
        [3, 3, 0, 4]       # 三角形面
    ])

    # 创建PolyData对象
    mesh = pv.PolyData(nodes, faces)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightblue',
                     point_size=10, render_points_as_spheres=True)
    plotter.show()
    ```

    方法2：使用 UnstructuredGrid（适用于混合单元类型）

    ```python
    import pyvista as pv
    import numpy as np

    # 定义节点坐标
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # 定义单元类型
    # VTK_QUAD = 9, VTK_TRIANGLE = 5
    cell_types = np.array([9, 5, 5, 5, 5])  # 1个四边形 + 4个三角形

    # 定义face连接信息
    cells = np.array([
        4, 0, 1, 2, 3,    # 四边形单元
        3, 0, 1, 4,        # 三角形单元
        3, 1, 2, 4,        # 三角形单元
        3, 2, 3, 4,        # 三角形单元
        3, 3, 0, 4         # 三角形单元
    ])

    # 创建UnstructuredGrid
    mesh = pv.UnstructuredGrid(cells, cell_types, nodes)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightcoral')
    plotter.show()
    ```

    方法3：从面列表创建（更直观的方式）

    ```python
    import pyvista as pv
    import numpy as np

    # 定义节点坐标
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # 以更直观的方式定义面（每个面是一个列表）
    face_list = [
        [0, 1, 2, 3],  # 底面四边形
        [0, 1, 4],      # 三角形面
        [1, 2, 4],      # 三角形面
        [2, 3, 4],      # 三角形面
        [3, 0, 4]       # 三角形面
    ]

    # 转换为vtk格式
    cells = []
    for face in face_list:
        cells.append(len(face))  # 添加顶点数
        cells.extend(face)       # 添加顶点索引

    cells = np.array(cells)

    # 创建mesh
    mesh = pv.PolyData(nodes, cells)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightgreen',
                     opacity=0.7, edge_color='black', line_width=2)
    plotter.add_points(mesh, color='red', point_size=10,
                       render_points_as_spheres=True)
    plotter.show()
    ```

    方法4：添加点数据和单元数据

    ```python
    import pyvista as pv
    import numpy as np

    # 创建mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    faces = np.hstack([
        [4, 0, 1, 2, 3],
        [3, 0, 1, 4],
        [3, 1, 2, 4],
        [3, 2, 3, 4],
        [3, 3, 0, 4]
    ])

    mesh = pv.PolyData(nodes, faces)

    # 添加点数据（例如每个节点的高度）
    mesh.point_data['height'] = nodes[:, 2]  # z坐标作为高度

    # 添加单元数据（例如每个面的面积）
    cell_areas = mesh.compute_cell_sizes()['Area']
    mesh.cell_data['area'] = cell_areas

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='height', show_edges=True,
                     cmap='viridis', edge_color='black')
    plotter.show()
    ```

    关键点说明：

    1. **VTK格式要求**：面信息需要使用特殊的VTK格式：`[n_vertices, idx1, idx2, ..., n_vertices, idx1, idx2, ...]`

    2. **节点索引**：节点索引从0开始

    3. **单元类型**：
       - 三角形：VTK_TRIANGLE (5)
       - 四边形：VTK_QUAD (9)
       - 更多类型可以参考VTK文档

    4. **性能考虑**：对于大型mesh，建议使用numpy数组而不是Python列表

    选择哪种方法取决于你的具体需求和数据结构。`PolyData`适用于纯表面网格，而`UnstructuredGrid`可以处理更复杂的单元类型。

* vtk 和 pyvista 是什么关系？

    VTK 和 PyVista 的关系

    1. **基本关系**

        **VTK (Visualization Toolkit)** 是底层的基础库，而 **PyVista** 是建立在 VTK 之上的高层封装库。

        ```
        应用程序层：    你的代码
                            ↓
        PyVista层：     PyVista (简化接口)
                            ↓
        VTK层：         VTK (C++核心库)
                            ↓
        图形硬件：        OpenGL/Vulkan等
        ```

    2. **详细对比**

        | 特性 | VTK | PyVista |
        |------|-----|---------|
        | **语言** | C++ 核心，有Python封装 | 纯Python |
        | **定位** | 完整的科学可视化工具包 | VTK的Pythonic封装 |
        | **API风格** | 传统的面向对象，VTK风格 | Python风格，更符合直觉 |
        | **代码量** | 需要写较多代码 | 简洁，代码量少 |
        | **学习曲线** | 陡峭 | 平缓 |
        | **灵活性** | 极高，可定制性强 | 高，但受封装限制 |
        | **性能** | 原生C++性能 | 接近VTK（底层仍是VTK） |

    3. **代码对比示例**

        VTK 方式（传统）：

        ```python
        import vtk

        # 创建球体
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(30)
        sphere.SetPhiResolution(30)

        # 创建mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        # 创建actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 创建renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

        # 创建render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        # 创建interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        # 开始可视化
        interactor.Initialize()
        interactor.Start()
        ```

        PyVista 方式（简洁）：

        ```python
        import pyvista as pv

        # 创建并可视化球体
        sphere = pv.Sphere(radius=1.0)
        sphere.plot()
        ```

        或者更详细的控制：

        ```python
        import pyvista as pv

        sphere = pv.Sphere(radius=1.0)
        plotter = pv.Plotter()
        plotter.add_mesh(sphere)
        plotter.show()
        ```

    4. **PyVista 的主要优势**

        1. **简化代码**：将VTK的复杂管道简化为直观的Python方法
        2. **NumPy集成**：直接使用NumPy数组操作数据
        3. **Jupyter友好**：可以在Notebook中直接显示3D图形
        4. **网格操作**：提供了丰富的网格处理和过滤功能
        5. **可视化**：简化的可视化接口，快速创建专业图形

    5. **底层关系**

        PyVista 在底层仍然使用 VTK：

        ```python
        import pyvista as pv

        # 创建PyVista网格
        mesh = pv.Sphere()

        # 实际上可以访问底层的VTK对象
        vtk_polydata = mesh  # PyVista网格本身就是VTK数据对象的子类
        vtk_points = mesh.points  # 实际上是VTK点的封装
        vtk_cells = mesh.cells    # 实际上是VTK单元的封装

        # 可以直接使用VTK过滤器
        vtk_filter = vtk.vtkSmoothPolyDataFilter()
        # ... 等
        ```

    6. **何时使用哪个？**

        **使用 PyVista 当：**

        - 快速原型开发
        - 数据分析和可视化
        - 教学和学习
        - 日常科学计算可视化
        - 需要与NumPy/Pandas等科学计算库集成

        **使用 VTK 当：**

        - 需要底层控制
        - 开发完整的应用程序
        - 需要特定的VTK功能而PyVista未封装
        - 性能要求极高
        - 需要集成到现有的VTK项目中

    7. **PyVista 的核心抽象**

        PyVista 将 VTK 的核心概念进行了Python化封装：

        - `pv.PolyData` → VTK的 vtkPolyData
        - `pv.UnstructuredGrid` → VTK的 vtkUnstructuredGrid
        - `pv.Plotter` → 封装了 VTK 的 renderer/window/interactor
        - `pv.DataSet` → 所有网格类型的基类

    总结

    **PyVista 是 VTK 的 Pythonic 接口**，它保留了 VTK 的强大功能，同时提供了更简洁、更符合Python习惯的API。如果你刚开始学习3D数据可视化，推荐从PyVista开始；如果需要更底层的控制，可以直接使用VTK，或者通过PyVista访问底层的VTK对象。


## topics
