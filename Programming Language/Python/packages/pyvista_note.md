# Pyvista Note

## cache

* pyvista 中，小四面体组成的 mesh 密度太高，如何才能降采样

    在 PyVista（以及它背后的 VTK 引擎）中，针对由小四面体（Tetrahedrons）组成的**非结构化网格（Unstructured Grid）**进行降采样，通常有几种不同的思路，具体取决于你是想保持体积网格（Volumetric Mesh）还是仅仅处理表面。

    以下是几种主流的方法：

    ---

    1. 表面抽稀 (Decimation)
        
        如果你最关心的是网格的外观，或者只需处理外壳，可以先提取网格的表面，然后使用 `decimate` 或 `decimate_pro`。

        ```python
        import pyvista as pv

        # 假设 mesh 是你的四面体网格
        # 1. 提取表面网格 (PolyData)
        surf = mesh.extract_surface()

        # 2. 抽稀（减少 50% 的面数）
        simplified_surf = surf.decimate(0.5)

        simplified_surf.plot(show_edges=True)
        ```

    ---

    2. 体素重采样 (Resampling via Voxelization)
        
        如果你需要保留内部的体积数据，最稳健的方法是将复杂的非结构化网格重采样到规则的线性网格（Uniform Grid）上。这可以极大地降低数据复杂度，同时平滑噪声。

        ```python
        # 定义目标网格的密度（比如 50x50x50）
        grid = pv.create_grid(mesh, dimensions=(50, 50, 50))

        # 将原始四面体数据映射到新网格上
        resampled_mesh = grid.sample(mesh)

        # 过滤掉落在原始网格之外的空值
        resampled_mesh = resampled_mesh.threshold()
        ```

    ---

    3. 使用剪裁或切片 (Sub-sampling by Selection)
        
        如果只是为了可视化减负，而不是为了改变几何结构，可以使用 `cell_centers` 配合随机采样。

        ```python
        import numpy as np

        # 随机选择 20% 的单元 (Cells)
        n_cells = mesh.n_cells
        indices = np.random.choice(n_cells, size=int(n_cells * 0.2), replace=False)
        downsampled_mesh = mesh.extract_cells(indices)

        downsampled_mesh.plot(show_edges=True)
        ```

    ---

    4. 顶点聚类 (Vertex Clustering)
        
        这是处理超高密度网格最快的方法之一。它通过将空间划分为小格子，并把落在同一个格子内的所有顶点合并为一个。

        ```python
        # clean 滤波器可以合并靠近的顶点
        # tolerance 参数控制合并的力度
        downsampled_mesh = mesh.clean(tolerance=0.01, remove_unused_points=True)
        ```

    ---

    **总结建议**

    | 需求场景 | 推荐方法 | 优点 |
    | :--- | :--- | :--- |
    | **仅为了渲染流畅** | `extract_surface().decimate()` | 速度最快，视觉效果好 |
    | **需要保留体积/内部属性** | `sample()` (Voxel Resampling) | 数据分布更均匀，适合后续计算 |
    | **快速减少数据量** | `extract_cells()` (随机抽样) | 保留了原始单元的形态 |

    **你想处理的是单纯的几何显示压力，还是需要把降采样后的网格用于后续的有限元计算（FEA）？** 如果是后者，我们可能需要讨论更高级的四面体网格重构工具（如 `tetgen`）。

* pyvista 如何显示帮助，查看当前的相机 position 和 lookat ?

    在 PyVista 中，查看帮助信息以及获取相机参数（位置、焦点、视角）非常直观。PyVista 的 `Plotter` 对象实际上封装了底层的 VTK 相机操作。

    以下是实现你需求的具体方法：

    1. 如何显示帮助信息
        
        PyVista 提供了内置的交互式帮助界面。

        * **在交互窗口中：** 当渲染窗口弹出时，按下键盘上的 **`h`** 键。屏幕上会直接列出所有快捷键指南（包括如何控制相机、切换平滑着色等）。
        * **在 Python 代码中：**
            ```python
            import pyvista as pv
            # 查看类或方法的文档
            help(pv.Plotter) 
            # 或者查看当前绘图器的相机属性说明
            print(pv.Plotter.camera.__doc__)
            ```

    ---

    2. 查看当前相机位置 (Position) 和 焦点 (Lookat)
        
        在 PyVista 中，相机的状态统称为 `camera_position`。它包含三个核心向量：**相机位置**、**焦点 (Lookat)** 以及 **上方向 (View up)**。

        方法 A：直接从 Plotter 对象获取
        
        这是最常用的方法，返回一个包含三个列表的元组：
        `[(x, y, z)_position, (x, y, z)_focal_point, (x, y, z)_view_up]`

        ```python
        import pyvista as pv

        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())
        pl.show() # 注意：在交互式窗口调整好角度并关闭后，下方的代码会打印最终位置

        # 获取相机完整参数
        print("Camera Position, Focal Point, and View Up:")
        print(pl.camera_position)

        # 如果你只需要特定的部分：
        print(f"Position: {pl.camera.position}")
        print(f"Look At (Focal Point): {pl.camera.focal_point}")
        ```

        方法 B：实时监控（在交互时打印）
        
        如果你想在拖动鼠标旋转模型时**实时**看到坐标变化，可以使用观察者模式：

        ```python
        import pyvista as pv

        def callback():
            # 实时打印位置和焦点
            pos = pl.camera.position
            focal = pl.camera.focal_point
            print(f"Pos: {pos} | LookAt: {focal}")

        pl = pv.Plotter()
        pl.add_mesh(pv.Cube())
        # 添加观察者，每当相机改变时触发回调
        pl.add_observer(pv.vtk.vtkCommand.ModifiedEvent, lambda obj, event: callback())
        pl.show()
        ```

    ---

    3. 常用相机属性总结

        | 属性 | 说明 |
        | :--- | :--- |
        | `pl.camera.position` | 相机在空间中的坐标 $(x, y, z)$ |
        | `pl.camera.focal_point` | 相机正对着的点（即 Lookat 点） |
        | `pl.camera.view_up` | 相机的“头顶”指向哪个方向 |
        | `pl.camera.zoom(2.0)` | 缩放倍率 |
        | `pl.camera_position` | 以上三者的集合（推荐用于保存和恢复视角） |

    > **提示：** 如果你想把当前调整好的视角应用到下一个脚本中，可以直接复制 `pl.camera_position` 打印出来的那个元组，然后在下次绘图时设置 `pl.camera_position = [(...)]` 即可完美复刻视角。

* 按 h 没反应

    如果在按下 **`h`** 键时没有反应，通常是因为当前的 **交互后端（Interactive Backend）** 不支持键盘事件，或者窗口没有获得 **焦点**。

    这种情况最常出现在 **Jupyter Notebook / Lab** 环境中，或者是使用了非交互式的静态渲染器。

    你可以根据你的运行环境尝试以下对策：

    1. 确保窗口处于“交互模式”
        
        如果你是在本地 Python 脚本中运行，请检查 `show()` 的调用方式：
        * **确保窗口被激活：** 点击一下弹出的 3D 渲染窗口，确保它是当前操作的活动窗口，然后再按 `h`。
        * **不要在非交互环境运行：** 如果你在某些 IDE（如 PyCharm 的 SciView 或某些远程服务器）中，它们可能会拦截键盘输入。

    2. Jupyter 环境下的特殊处理

        在 Jupyter 中，默认渲染可能被设置为静态图片。请尝试在代码最上方强制开启交互窗口：

        ```python
        import pyvista as pv

        # 强制使用传统的 VTK 交互窗口（会弹出独立窗口）
        pv.set_jupyter_backend('static') # 或者不设置，直接运行以下代码
        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())
        pl.show(interactive_update=True) 
        ```
        *注意：在 JupyterLab 里，建议使用 `pl.show(jupyter_backend='trame')`，这通常能更好地支持交互。*

    3. 代码替代方案：直接打印快捷键列表
        
        既然按下 `h` 没反应，我们可以直接用 Python 代码把这些“隐藏”的帮助信息打印出来。PyVista 的交互帮助本质上是 `RenderWindowInteractor` 的功能。

        你可以运行以下代码查看所有内置快捷键：

        ```python
        import pyvista as pv
        print(pv.Plotter.KEY_HELP_MSG)
        ```

    ---

    4. 无法查看相机位置？直接用“实时同步”
    
        如果你的目标是查看相机位置，但交互式操作受限，可以使用 **`camera_set`** 功能，或者在代码中强制打印。

        **最硬核的调试方法（无论按键是否失效）：**
        使用 `pl.enable_trackball_style()` 确保处于轨迹球缩放模式，并手动获取位置：

        ```python
        import pyvista as pv

        pl = pv.Plotter()
        pl.add_mesh(pv.Cube(), color='cyan')

        # 即使按键没反应，你也可以在程序关闭时获取最后停留的位置
        pl.show() 

        # 窗口关闭后，这里会自动打印你刚才手动调整好的位置
        print("最后调整的相机参数为：")
        print(pl.camera_position)
        ```

    **总结：**
    如果 `h` 没反应，说明你的环境限制了键盘映射。请确认你是在**本地电脑**运行 `python script.py` 还是在 **Jupyter/远程** 环境？如果是后者，键盘交互通常会被屏蔽。

* plotter.render() 用的这个

    明白了，问题出在 `plotter.render()` 的工作方式上。

    当你手动调用 `pl.render()` 而不是 `pl.show()` 时，你通常是在执行**非交互式渲染**（即“离线”渲染），或者是在一个你自己控制的循环中更新画面。在这种模式下，PyVista 的**交互器（Interactor）并没有启动**，所以它不会监听键盘事件，按 `h` 自然没反应。

    如果你想在使用 `render()` 的过程中查看相机参数，可以通过以下两种方式解决：

    1. 强制打印（最直接的方法）
        
        既然你在代码中控制渲染流程，你可以直接在 `render()` 后面打印当前的相机状态：

        ```python
        import pyvista as pv

        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())

        # 你的渲染逻辑
        pl.render()

        # 直接输出当前相机的三要素：[坐标, 焦点, 上方向]
        print("--- 当前相机快照 ---")
        print(f"Position:  {pl.camera.position}")
        print(f"Focal Pt:  {pl.camera.focal_point}")
        print(f"Camera Pos: {pl.camera_position}") # 这是一个包含上述信息的完整 list
        ```

    ---

    2. 在屏幕上直接显示坐标 (实时悬浮文字)
        
        既然按键没用，我们可以直接把坐标“刻”在渲染窗口里，这样每次 `render()` 时都能看到变化：

        ```python
        import pyvista as pv

        pl = pv.Plotter()
        pl.add_mesh(pv.Sphere())

        # 在屏幕角落添加一个动态更新的文字标签
        text_actor = pl.add_text(f"Pos: {pl.camera.position}", position='lower_left', font_size=10)

        def update_and_render():
            # 每次更新时同步文字
            text_actor.SetInput(f"Pos: {[round(x, 2) for x in pl.camera.position]}")
            pl.render()

        # 模拟你的渲染循环
        for _ in range(100):
            # 假设你在这里改变了相机或模型
            # pl.camera.azimuth += 1 
            update_and_render()

        pl.show()
        ```

    ---

    3. 为什么按 `h` 没反应？
        
        PyVista 的交互快捷键（如 `h`, `r`, `s` 等）是由 `pl.show()` 内部启动的 `interactor.Start()` 驱动的。
        * 如果你只用 `render()`，交互器处于静止状态。
        * **解决方法：** 如果你希望既能手动控制 `render()`，又想保留按键功能，你需要显式地调用交互器处理函数（这在复杂应用中较常见），或者改用 `pl.show(interactive_update=True)` 进入交互循环。

        **核心建议：**
        如果你只是想**调试**出合适的视角，建议先临时把 `pl.render()` 改成 `pl.show()`，在弹出的窗口里用鼠标转到满意的角度，**关闭窗口**，然后在控制台看最后打印出的 `pl.camera_position`。

    你是正在做一个动态模拟，还是在集成到某个 GUI 界面里？

* print(f"Position:  {pl.camera.position}") 这种方式不错，除了相机参数，帮我把 scale 参数，mesh 姿态等信息也打印出来吧

    在 PyVista 中，**Mesh 的姿态（Pose）** 通常由其 `user_matrix`（变换矩阵）决定，而 **Scale（缩放）** 可以存在于两个层面：一个是相机本身的缩放（Zoom），另一个是 Mesh 对象自身的缩放。

    既然你喜欢直接打印的方式，可以将以下代码片段整合到你的 `render()` 循环中。这会输出最核心的姿态和几何参数：

    核心参数打印代码

    ```python
    import pyvista as pv
    import numpy as np

    # 假设 pl 是你的 Plotter，mesh 是你添加的模型
    # mesh = pl.add_mesh(...)

    print("--- 实时状态快照 ---")

    # 1. 相机参数 (Camera)
    print(f"[Camera Position]: {pl.camera.position}")
    print(f"[Camera LookAt]:   {pl.camera.focal_point}")
    print(f"[Camera Zoom]:     {pl.camera.zoom(1.0):.2f}") # 获取当前缩放倍率

    # 2. Mesh 姿态 (Orientation & Position)
    # PyVista 的 mesh.orientation 返回的是绕 X, Y, Z 轴的角度 (Euler angles)
    print(f"[Mesh Orientation]: {mesh.orientation}") 
    print(f"[Mesh Center]:      {mesh.center}")      # Mesh 几何中心在世界坐标系的位置

    # 3. 缩放 (Scale)
    # 如果你通过 mesh.scale() 进行了缩放，其比例会体现在变换矩阵中
    # 或者查看 bounding box 的尺寸来直观感受大小
    print(f"[Mesh Bounds]:      {mesh.bounds}")      # [xmin, xmax, ymin, ymax, zmin, zmax]

    # 4. 变换矩阵 (Transformation Matrix)
    # 包含位移、旋转、缩放的 4x4 矩阵，是描述姿态最专业的方式
    # 如果你使用了平移、旋转命令，信息都在这里
    matrix = mesh.user_matrix
    print("[Mesh User Matrix]:\n", matrix)

    print("-" * 20)
    ```

    ---

    **关键概念解析**

    1. 关于 "Mesh 姿态" 的陷阱
        
        在 PyVista 中，有两种方式改变 Mesh 的位置：
        * **直接修改点坐标：** `mesh.rotate_x(45)`。这种方式下，`user_matrix` 依然是单位阵，因为你直接把模型顶点的坐标改了。
        * **使用用户矩阵：** `pl.add_mesh(mesh, user_matrix=my_matrix)`。这种方式下，Mesh 本身的顶点坐标没变，但在渲染时会被矩阵变换。
            * *建议：* 如果你发现 `mesh.orientation` 总是 `(0,0,0)`，那说明你是直接修改了顶点坐标。此时查看姿态的方法是观察 `mesh.center` 和 `mesh.bounds`。

    2. 什么是 LookAt (焦点)？
        
        打印出的 `pl.camera.focal_point` 就是相机注视的那个点。如果你发现模型移动了但相机没跟上，通常需要手动设置：
        ```python
        pl.camera.focal_point = mesh.center
        ```

    3. 推荐小技巧：使用 `add_axes()`
        
        既然 `h` 帮助文档按不出来，建议在初始化时加上坐标轴，配合打印信息会非常直观：
        ```python
        pl.add_axes() # 在左下角显示 XYZ 轴方向
        # 或者给特定的 mesh 绑定坐标轴
        pl.add_axes_at_origin(labels_off=False) 
        ```

    你现在是在做一个自动化生成视频的过程，还是在调试模型对齐的算法？如果是对齐算法，打印 `mesh.user_matrix` 会对你调试旋转矩阵（Rotation Matrix）非常有帮助。

* pv.CellType.TETRA 和 VOXEL 有什么区别

    在 PyVista (VTK) 中，`TETRA` 和 `VOXEL` 是两种完全不同的单元类型，它们在**拓扑结构**、**顶点顺序**以及**几何限制**上有显著区别。

    简单来说：**TETRA 是“全能型”的四面体，VOXEL 是“规则型”的正方体单元。**

    ---

    1. 结构与顶点数量

        | 特性 | **TETRA (四面体)** | **VOXEL (体素)** |
        | --- | --- | --- |
        | **顶点数** | 4 个顶点 | 8 个顶点 |
        | **面数量** | 4 个三角形面 | 6 个矩形面 |
        | **形状** | 任意形状的四面体（锥体） | 必须是各边平行于坐标轴的直角六面体 |
        | **VTK 枚举值** | `10` | `11` |

    ---

    2. 几何限制 (最核心区别)

        * **TETRA (Tetrahedron):**
        * 非常灵活。四个顶点可以在空间中任意摆放（只要不共面）。
        * 适用于**非结构化网格**，能完美拟合复杂的曲面边界。
        * 通常用于有限元分析 (FEA) 或物理仿真。


        * **VOXEL:**
        * **极其严苛**。在 VTK 定义中，VOXEL 的 12 条边必须分别平行于 $x, y, z$ 轴。
        * 它实际上是一个“轴对齐”的正方体或长方体。
        * 如果你有一个倾斜的六面体，你不能使用 `VOXEL`，而必须使用 `HEXAHEDRON` (六面体, Type 12)。

    ---

    3. 顶点排列顺序 (Indexing)

        VTK 对两者的顶点索引顺序要求不同，如果弄反了，渲染会出错（法线反转或畸变）。

        **TETRA (4 nodes)**

        通常遵循右手定则：底面三个点 $(0, 1, 2)$ 逆时针排列，第四个点 $(3)$ 在上方。

        **VOXEL (8 nodes)**

        顺序非常特殊，它是按照二进制递增坐标排序的：

        1. $(x_{min}, y_{min}, z_{min})$
        2. $(x_{max}, y_{min}, z_{min})$
        3. $(x_{min}, y_{max}, z_{min})$
        4. $(x_{max}, y_{max}, z_{min})$
        5. ...以此类推（先变 $x$，再变 $y$，最后变 $z$）。

    ---

    4. 什么时候用哪个？

        * **用 `TETRA` 的场景：**
        * 你正在从 `MeshLab`、`Gmsh` 或 `TetGen` 导入数据。
        * 你需要模拟复杂的几何体（如人体器官、机械零件）。
        * **你的数据源 `elems` 是 $m \times 4$ 的数组。**


        * **用 `VOXEL` 的场景：**
        * 你正在处理医学影像（MRI/CT 扫描堆栈）。
        * 你正在处理类似 Minecraft 的方块世界。
        * 数据本身是在规整的 3D 网格（Grid）上采样得到的。

    ---

    **避坑指南**

    如果你手中的 `elems` 是 **$m \times 8$** 的数组，且形状是规则的长方体，你可能想用 `VOXEL`；但如果这些长方体发生了旋转或变形，**请务必使用 `pv.CellType.HEXAHEDRON` (12)**，否则 VTK 会尝试强行将它们对齐到轴向，导致模型崩坏。

    **你现在手头的数据源是四列（四面体）还是八列（六面体/体素）？**

* pv.voxelize

    看来你正在查阅 `pv.voxelize` 的文档。这是一个非常经典的**空间离散化**工具，也就是将任意形状的网格（无论是高模四面体还是表面网格）“方块化”。

    如果你想通过这个方法实现**降采样**，它的核心逻辑是：用一堆规则的正方体（Voxels）去填充你的原始模型。

    1. 如何使用 `voxelize` 进行降采样？

        如果你的原始四面体网格太密（几十万个小四面体），你可以通过设置较大的 `density` 来生成一个低模的、由方块组成的网格。

        ```python
        import pyvista as pv

        # 假设 mesh 是你之前创建的高模 UnstructuredGrid
        # density 决定了方块的大小，数值越大，模型越“糊”（模数越低）
        low_res_voxel_mesh = mesh.voxelize(density=0.5) 

        # 查看结果
        low_res_voxel_mesh.plot(show_edges=True)

        ```

    2. 参数关键点拨

        * **`density` (密度/尺寸)**:
        * 这是最重要的参数。虽然叫 density，但它实际上代表**每个方块的边长**。
        * 如果传一个数字（如 `0.5`），则 $x, y, z$ 方向都是 $0.5$ 的正方体。
        * 如果传列表 `[0.5, 0.5, 1.0]`，你可以得到长方体体素。


        * **`check_surface`**:
        * 默认 `True`。它要求你的输入网格必须是**封闭流形**（Closed Manifold，没有破洞）。如果你的四面体网格边界有漏洞，这步会报错。


        * **`enclosed`**:
        * 如果设为 `True`，体素网格会稍微包裹住原始模型的边界；`False`（默认）则尽量保持在原始模型内部。

    3. 注意：它返回的是什么？

        调用 `voxelize` 后，你得到的不再是四面体网格（TETRA），而是**体素网格（VOXEL）**。

        * **如果你需要继续做有限元分析 (FEA)**：VOXEL 网格通常是可以直接用的，但如果你后续的算法只认四面体，你需要加一步转换：
        ```python
        # 将方块(VOXEL) 自动切分成四面体(TETRA)
        low_res_tet_mesh = low_res_voxel_mesh.tetrahedralize()

        ```

    4. 为什么要用这个方法做降采样？

        * **优点**：非常稳定，能保证生成的低模网格绝对没有自交，且网格质量（单元形状）非常统一。
        * **缺点**：会丢失原始模型的精细特征（比如尖锐的边缘会变成锯齿状）。

    ---

    **你想把降采样后的结果导出为哪种格式？或者你需要保持原始模型的精确边界（不想变锯齿）吗？**

    * 我看它写了 deprecated 

        眼尖！这是一个非常典型的 **API 迁移警告**。

        在 PyVista `0.46.0` 及之后的版本中，官方正在整理代码架构，将原本直接挂在 `pyvista` 命名空间下的全局函数，统一转移到了**过滤器（Filters）**系统中。

        1. 为什么会 Deprecated？

            PyVista 希望所有的处理操作（如平滑、切片、体素化）都通过统一的入口调用。

            * **旧写法：** `pv.voxelize(mesh, ...)`
            * **新写法：** `mesh.voxelize(...)`

            这本质上是把函数变成了一个**对象方法**。这样做的好处是支持“链式调用”，比如 `mesh.voxelize(0.5).extract_surface()`。

        ---

        2. 你应该如何修改代码？

            如果你使用的是较新版本的 PyVista，请直接在你的 mesh 对象上调用它：

            ```python
            import pyvista as pv

            # 1. 创建或读取你的高模网格
            # high_res_mesh = pv.UnstructuredGrid(...) 

            # 2. 正确的“非弃用”调用方式
            # 这里的 density 是体素的边长，数值越大，生成的网格越稀疏（低模）
            low_poly_mesh = high_res_mesh.voxelize(density=0.1, check_surface=False)

            # 3. 如果你需要把它转回四面体（低模四面体网格）
            final_low_tet_mesh = low_poly_mesh.tetrahedralize()

            final_low_tet_mesh.plot(show_edges=True)

            ```

        ---

        3. 补充：文档中提到的另外两个“兄弟”

            文档的 **See Also** 部分提到了两个替代方案，根据你的需求可能有奇效：

            1. **`voxelize_rectilinear`**:
            * **返回：** `RectilinearGrid`。
            * **特点：** 如果你的降采样只是为了存储数据（比如 CT 断层数据），这个格式比 `UnstructuredGrid` 更省内存，查询速度更快。


            2. **`voxelize_binary_mask`**:
            * **返回：** `ImageData` (由 0 和 1 组成的 3D 矩阵)。
            * **场景：** 如果你想把 3D 模型变成类似 NumPy 数组的 $0/1$ 矩阵（做深度学习或者计算体积），用这个最快。

        ---

        **避坑指南**

        如果你调用 `mesh.voxelize()` 报错说 `surface is not closed`，那是因为你的原始四面体网格**外表面有破洞**。

        你可以尝试先修复表面：

        ```python
        # 提取表面 -> 填补漏洞 -> 再体素化
        fixed_mesh = mesh.extract_surface().fill_holes(100)
        low_poly = fixed_mesh.voxelize(density=0.5)

        ```

        **你的原始网格是封闭的几何体吗？如果不确定，我可以教你如何检测网格的开口（Holes）。**

* Surface is not closed.

    意料之中的报错。这个 `RuntimeError` 是因为 `voxelize` 算法底层需要一个**明确的“内部”和“外部”**空间定义。如果你的四面体网格表面有孔洞、重叠面或者非流形边缘（Non-manifold edges），它就不知道哪些区域该填入体素。

    你有两种主要的解决策略：

    **方案 A：强行跳过检查（最快，但不一定稳）**

    如果你确信网格基本完整，只是有一些微小的数值缝隙，可以直接禁用检查。但请注意，如果空洞太大，生成的低模网格可能会丢失大片区域。

    ```python
    # 修改这一行，添加 check_surface=False
    grid = mesh.voxelize(density=0.5, check_surface=False)

    ```

    ---

    **方案 B：先修复表面，再体素化（最推荐）**

    如果你的目标是降采样并保持形状，通常的做法是：**提取表面 -> 补洞 -> 体素化**。

    ```python
    import pyvista as pv

    # 1. 提取四面体网格的表面（转换成三角面网格）
    surface = mesh.extract_surface()

    # 2. 尝试自动补洞 (hole_size 是空洞的最大周长，可以调大一点)
    # 同时进行表面清理（移除重复点和退化面）
    clean_surface = surface.clean().fill_holes(1000)

    # 3. 再次尝试体素化
    try:
        grid = clean_surface.voxelize(density=0.5, check_surface=True)
    except RuntimeError:
        print("表面依然不封闭，尝试强行体素化...")
        grid = clean_surface.voxelize(density=0.5, check_surface=False)

    grid.plot(show_edges=True)

    ```

    ---

    **方案 C：如果你的原始数据确实“很烂” (不封闭)**

    如果你的模型根本不是一个封闭的实体（比如它只是一片曲面，或者内部有大量断裂），`voxelize` 可能不是最佳选择。

    你可以尝试 **`select_enclosed_points`** 配合一个背景网格，这是一种更鲁棒的采样方式：

    ```python
    # 1. 创建一个覆盖整个模型的背景网格 (Uniform Grid)
    bounds = mesh.bounds
    dims = (50, 50, 50) # 你想要的低模分辨率
    grid = pv.ImageData(
        dimensions=dims,
        spacing=[(bounds[1]-bounds[0])/dims[0], 
                (bounds[3]-bounds[2])/dims[1], 
                (bounds[5]-bounds[4])/dims[2]],
        origin=[bounds[0], bounds[2], bounds[4]]
    )

    # 2. 找出哪些点在原始 mesh 内部
    # 这种方法对“不封闭”网格的容忍度稍高
    selection = grid.select_enclosed_points(mesh, tolerance=0.001)
    low_res_mesh = selection.threshold(0.5, scalars="selected_points")

    ```

    ---

    **排查建议：**

    你可以运行下面这行代码来**视觉定位**到底是哪里没封口：

    ```python
    # 提取并显示边界边（即没有邻接面的边，通常就是漏洞所在）
    edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=True)
    p = pv.Plotter()
    p.add_mesh(mesh, color="white", opacity=0.5)
    p.add_mesh(edges, color="red", line_width=5) # 红色的地方就是破洞
    p.show()

    ```

    **如果看到红线很多，说明你的原始四面体数据本身存在严重的拓扑问题。你需要我帮你看看如何用 Python 批量修复这些破洞吗？**

* pyvista 可以设置在降采样时使用多线程吗

    直接的答案是：**PyVista（以及底层的 VTK）的大部分内置 Filter（如 `voxelize`）并不原生支持 Python 层面的简单 `multiprocessing` 开关。**

    这是因为 VTK 的核心逻辑是用 C++ 编写的。虽然 VTK 内部有些算法（如 SMP 框架）会自动使用多线程，但像 `voxelize` 这种依赖于空间映射和拓扑检查的算法，往往是**单线程顺序执行**的。

    不过，如果你有多个网格需要处理，或者想加速单个大网格的降采样，可以参考以下几种实战方案：

    ---

    1. 任务级并行 (多进程处理多个 Mesh)

        如果你有一堆 `.vtk` 或 `.msh` 文件需要批量降采样，这是提升效率最快的方法。使用 Python 的 `concurrent.futures`：

        ```python
        import pyvista as pv
        from concurrent.futures import ProcessPoolExecutor

        def process_mesh(file_path):
            # 每个进程加载自己的数据，避免 GIL 锁
            mesh = pv.read(file_path)
            # 执行耗时的降采样
            low_res = mesh.voxelize(density=0.5, check_surface=False)
            low_res.save(f"low_{file_path}")
            return True

        files = ["m1.vtk", "m2.vtk", "m3.vtk"]
        with ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(process_mesh, files)

        ```

    ---

    2. 空间分割并行 (针对超大单个 Mesh)

        如果你的单个网格极其巨大（比如几千万个四面体），`voxelize` 会非常慢。你可以手动将空间切分成几块（Chunks），并行处理后合并：

        1. 将大 Mesh 按坐标轴切分为 4 份（使用 `mesh.clip`）。
        2. 开启 4 个进程分别对这 4 个块进行 `voxelize`。
        3. 使用 `pv.merge()` 将结果重新拼回一个 `UnstructuredGrid`。

    ---

    3. 利用 VTK 的加速器 (显卡/多核)

        虽然 PyVista 的 `voxelize` 默认不快，但你可以尝试切换到基于 **ImageData** 的方案，这在 VTK 中是有并行优化的：

        ```python
        # 使用多线程重采样算法 (Resampling)
        # 这种方法比传统的 voxelize 快得多，因为它基于探测（Probing）
        grid = pv.ImageData(dimensions=(100, 100, 100), spacing=(0.1, 0.1, 0.1), origin=mesh.center)
        resampled = grid.sample(mesh) # 这里内部调用了 VTK 的并行探测算法
        low_res = resampled.threshold(0.5) 

        ```

    ---

    4. 终极性能方案：Open3D (CPU 多线程)

        如果 PyVista 的速度确实令你抓狂，可以考虑将数据转给 **Open3D**。Open3D 的 `VoxelGrid` 下采样算法原生支持多线程加速，且速度极快。

        ```python
        import open3d as o3d
        import numpy as np

        # 将 PyVista 的点转给 Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.points)

        # 执行体素下采样 (这一步非常快)
        voxel_size = 0.5
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # 拿到降采样后的点中心，再构建你的低模
        new_points = np.asarray(downsampled_pcd.points)

        ```

    ---

    **总结建议**

    * **如果是单网格耗时：** 检查 `check_surface=False` 是否能加速（通常能快 30% 以上）；或者改用 `sample` 探测法。
    * **如果是多网格耗时：** 使用 `ProcessPoolExecutor`。
    * **如果是性能瓶颈：** 考虑先将四面体转成点云，用 Open3D 快速下采样，最后再重新剖分。

    **你的高模网格大约有多少个单元（Cells）？如果超过 500 万个，我们可能需要考虑更底层的分块处理逻辑。**

* 'UnstructuredGrid' object has no attribute 'tetrahedralize' 

    这个错误是因为在 PyVista 的最新版本中，这个方法的名称是 **`delaunay_3d`** 或者 **`extract_geometry`** 的变体。虽然逻辑上是“四面体化”，但 VTK/PyVista 将其分类在过滤器（Filters）中。

    对于 `UnstructuredGrid`（尤其是由 `voxelize` 生成的六面体/体素网格），将其转换为四面体网格的正确方法是使用 **`.delaunay_3d()`** 或 **`.tessellate()`**。

    1. 修正后的代码

        请将报错的那行改为以下任意一种（推荐第一种）：

        ```python
        # 方法 A：最通用，将非三角形单元（如 Voxel）切分为四面体
        low_tet = grid.tessellate()

        # 方法 B：如果想要重新生成更致密的四面体结构（基于点云重新构建）
        # alpha 参数控制网格的紧致程度
        low_tet = grid.delaunay_3d(alpha=0.1)

        # 方法 C：如果你只是想提取几何体并确保它是 UnstructuredGrid 格式
        low_tet = grid.cast_to_unstructured_grid()

        ```

    ---

    2. 为什么会报错？

        * **API 差异**：在 PyVista 中，只有 `PolyData`（表面网格）有类似 `triangulate` 的直接方法。
        * **UnstructuredGrid 的特殊性**：`UnstructuredGrid` 已经是一个体网格了，如果你想改变它的单元类型（比如从 Voxel 变 Tetra），通常需要调用 `tessellate`（镶嵌/细分）过滤器，它会将每一个 8 节点的六面体体素切割成 5 个或 6 个四面体。

    ---

    3. 性能小贴士

        如果你刚才通过 `voxelize` 得到了几百万个方块，调用 `tessellate()` 会让单元数量再翻 **5-6 倍**（因为 1 个方块 = 6 个四面体）。

        **如果你的目标是减少数据量，请务必增大 `voxelize(density=...)` 中的 `density` 值。**

    ---

    **完整闭环示例**

    为了确保你现在能跑通，这里是一个完整的最小化脚本：

    ```python
    import pyvista as pv

    # 1. 你的高模
    # mesh = ...

    # 2. 降采样（增加 density 值来减少数量）
    grid = mesh.voxelize(density=1.0, check_surface=False)

    # 3. 转换为四面体网格
    low_tet = grid.tessellate()

    print(f"原始单元数: {mesh.n_cells}")
    print(f"降采样后单元数: {low_tet.n_cells}")

    low_tet.save("low_res_tetra.vtk")
    low_tet.plot(show_edges=True)

    ```

    **运行 `tessellate()` 后，你的单元数量（Cells）是否符合你的预期（变少了还是变多了）？**

* pyvista 手动设置快捷键触发对话框

    PyVista 原生并没有直接弹出“文件保存对话框”的内置函数，因为它是一个科学计算库而非 GUI 库。但你可以结合 tkinter 轻易实现：

    ```Python
    import pyvista as pv
    from tkinter import filedialog

    plotter = pv.Plotter()
    plotter.add_mesh(pv.Cube())

    def save_callback():
        # 弹出系统保存对话框
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            plotter.screenshot(path)
            print(f"已保存至: {path}")

    # 给渲染窗口添加一个自定义按键（比如按 'd' 弹出对话框）
    plotter.add_key_event('d', save_callback)
    plotter.show()
    ```

    自定义 s 键保存图片：

    ```
    import pyvista as pv

    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere())

    # 定义一个带反馈的保存函数
    def custom_save():
        filename = "manual_save.png"
        plotter.screenshot(filename)
        print(f"--- 触发成功！截图已保存至: {filename} ---")

    # 重新绑定 's' 键
    plotter.add_key_event('s', custom_save)

    plotter.show()
    ```
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
