# fnirs note

## cache

* toast++ github repo: <https://github.com/toastpp/toastpp>

* NIRS-Toolbox 是个基于 matlab 的工具箱

    相关网站：

    * <https://help.nirx.de/hc/en-us/articles/17400352078108-NIRS-Toolbox>

    * <https://github.com/huppertt/nirs-toolbox>

    * <https://www.artinis.com/blogpost-all/2021/fnirs-analysis-toolbox-series-brain-analyzir>

    * <https://nirx.net/fnirs-analysis>

* 我现在想加载 lumo 的数据后，使用 mne-nirs 进行预处理，然后使用 toast++ 进行 3D 重建，给我一份 example 代码，加上可视化功能 

    这是一个非常有挑战性且完整的 DOT（扩散光学断层扫描）流程。要实现从 Lumo 原始数据到 3D 重建的转换，我们需要将 Lumo 的几何结构映射到 MNI 空间，利用 `mne-nirs` 处理时间序列信号，最后调用 `toast++` 进行空间域的反演。

    由于 `toast++` 的 3D 重建通常针对静态或差分图像，下面的示例将演示如何将 **MNE 处理后的 OD（光密度）变化量** 映射到 **3D 吸收系数 $\Delta\mu_a$** 的重建。

    **核心流程设计**

    1. **数据转换**：将 Lumo 原始强度转换为 `mne.io.Raw` 对象。
    2. **预处理**：使用 `mne-nirs` 进行强度到光密度（OD）的转换及滤波。
    3. **几何对齐**：使用你提供的 `regi` 函数将探头坐标对齐到 MNI152 头部模型。
    4. **Toast++ 建模**：在 3D Mesh 上配置源 ($Q$) 和检测器 ($M$)。
    5. **3D 重建**：求解差分反问题。

    ---

    **Python 示例代码**

    ```python
    import numpy as np
    import mne
    import mne_nirs
    import toast
    import pyvista as pv
    from scipy.sparse import linalg
    import os

    # 假设你之前的工具代码保存在 lumo_utils.py 中
    # from lumo_utils import * ## 1. 加载 Lumo 数据并转为 MNE 对象
    def convert_lumo_to_mne(int_bin, rec_data, sd_layout):
        # Lumo 数据格式: [nframe, recordings_per_frame]
        data = int_bin.data.T  # 转置为 [channels, times]
        
        # 创建 MNE info
        ch_names = [f"CH{i+1}" for i in range(rec_data.n_chans)]
        # 这里简单处理，实际应根据 wavelength 分配类型
        ch_types = ['fnirs_cw_amplitude'] * rec_data.n_chans 
        info = mne.create_info(ch_names=ch_names, sfreq=rec_data.framerate, ch_types=ch_types)
        
        raw = mne.io.RawArray(data, info)
        return raw

    ## 2. 使用 MNE-NIRS 预处理
    def preprocess_nirs(raw):
        # 1. 强度转光密度 (Optical Density)
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        # 2. 带通滤波 (去除漂移和心跳噪声)
        raw_od.filter(0.01, 0.2, fir_design='firwin')
        # 3. 提取某一段的时间平均值作为重建目标 (差分重建)
        # 比如：delta_OD = mean(task) - mean(baseline)
        data_od = raw_od.get_data()
        delta_od = np.mean(data_od, axis=1) # 简化：取全过程均值作为演示
        return delta_od

    ## 3. Toast++ 3D 重建配置
    def setup_toast_3d(mshs, sd_layout, mni_landmarks):
        # a. 创建 Toast Mesh
        # 注意：toast 需要节点和单元。这里假设使用 headVolumeMesh 进行前向计算
        nodes = mshs.headVolumeMesh.node
        elems = mshs.headVolumeMesh.elem[:, :4] # 提取四面体索引
        
        # 写入临时文件或直接通过内存创建（取决于 toast 版本，通常建议写 .msh）
        # 这里演示逻辑：
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: elems}, nodes)
        # toast_mesh = toast.Mesh(nodes, elems) # 伪代码，根据实际 API 调整
        
        # b. 注册探头到 MNI 空间并投影到头皮
        src_mni = regi_coords_sd_to_mni(sd_layout.src_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        det_mni = regi_coords_sd_to_mni(sd_layout.det_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        
        # 投影到头皮表面最近节点
        src_on_scalp = regi_coords_to_nearest_neighbors(src_mni, mshs.scalpSurfaceMesh.node)
        det_on_scalp = regi_coords_to_nearest_neighbors(det_mni, mshs.scalpSurfaceMesh.node)
        
        return src_on_scalp, det_on_scalp

    ## 4. 主重建程序
    def run_3d_reconstruction():
        # --- 加载数据 ---
        # (此处省略文件路径定义，参考你之前的测试代码)
        int_bin = lumo_load_all_intensity_files(lumo_bin_fpaths)
        rec_data = lumo_load_recording_data(toml_path)
        sd_layout = lumo_load_sd_layout(layout_json)
        mshs = load_mni152_mshs(mshs_file)
        
        # --- 预处理 ---
        raw = convert_lumo_to_mne(int_bin, rec_data, sd_layout)
        delta_od = preprocess_nirs(raw) # 得到每个通道的测量值 y
        
        # --- Toast 准备 ---
        # 创建 Toast 3D 网格
        tm = toast.Mesh() 
        # 这里建议将 mshs.headVolumeMesh 保存为 .msh 后加载
        # tm.ReadModel('adult_mni.msh') 
        
        # 设置源探测器 (Qvec/Mvec)
        # 假设 Lumo 共有 Nsrc 个源和 Ndet 个检测器
        qvec = tm.Qvec(type='Neumann', shape='Gaussian', width=2, pos=src_on_scalp)
        mvec = tm.Mvec(shape='Gaussian', width=2, pos=det_on_scalp)
        
        # 初始化背景参数 (标准组织光学参数)
        nlen = tm.NodeCount()
        mua = np.ones(nlen) * 0.01  # 背景吸收
        mus = np.ones(nlen) * 1.0   # 背景散射
        ref = np.ones(nlen) * 1.4   # 折射率
        freq = 0 # Lumo 是连续波 (CW)
        
        # --- 计算雅可比矩阵 ---
        print("Computing Jacobian...")
        phi = tm.Fields(None, qvec, mua, mus, ref, freq)
        aphi = tm.Fields(None, mvec, mua, mus, ref, freq)
        # 对于 CW 数据，测量值是强度，这里计算灵敏度矩阵
        J = tm.Jacobian(None, phi, aphi, None) # 简化的 Jacobian 调用
        
        # --- 线性反演 (Tikhonov Regularization) ---
        # y = J * x  => x = (J'J + alpha*I)^-1 J'y
        # 其中 x 是 delta_mua
        alpha = 0.01 * np.max(np.diag(J.T @ J)) # 正则化参数
        delta_mua = linalg.lsqr(J, delta_od, damp=alpha)[0]
        
        # --- 可视化 ---
        visualize_result(mshs, delta_mua)

    def visualize_result(mshs, delta_mua):
        # 使用 PyVista 在脑模上显示吸收变化
        nodes = mshs.headVolumeMesh.node
        elems = mshs.headVolumeMesh.elem[:, :4] # 四面体单元
        
        # 创建网格对象
        cell_type = np.array([pv.CellType.TETRA] * elems.shape[0])
        cells = np.hstack(np.c_[np.full(elems.shape[0], 4), elems])
        grid = pv.UnstructuredGrid(cells, cell_type, nodes)
        
        # 将结果映射到网格节点
        grid.point_data["Delta_Mua"] = delta_mua
        
        # 绘图
        plotter = pv.Plotter()
        # 绘制半透明头皮
        plotter.add_mesh(pv.PolyData(mshs.scalpSurfaceMesh.node, 
                                    np.hstack(np.c_[np.full(mshs.scalpSurfaceMesh.face.shape[0], 3), 
                                                    mshs.scalpSurfaceMesh.face])), 
                        opacity=0.1, color='tan')
        
        # 绘制重建后的吸收变化切片
        slices = grid.slice_orthogonal()
        plotter.add_mesh(slices, cmap="jet", scalars="Delta_Mua")
        
        # 也可以绘制阈值以上的体积 (例如激活区)
        thres = grid.threshold(np.percentile(delta_mua, 95))
        plotter.add_mesh(thres, color="red")
        
        plotter.show()

    ```

    ---

    **关键点说明**

    1. CW（连续波） vs FD（频域）

        你提供的 `toast++` 例子中 `freq = 100`，那是针对频域设备的（有相位信息）。**Lumo 是连续波设备**，因此在 `tm.Fields` 和 `tm.Jacobian` 中，`freq` 必须设为 **0**。此时我们只重建吸收系数 $\mu_a$，因为 CW 很难区分散射和吸收。

    2. 坐标注册 (Registration)

        Lumo 的坐标是基于其 Dock 的局部坐标。

        * 使用你写的 `regi_coords_sd_to_mni` 非常关键。
        * **注意**：`mni_landmarks` 的 5 个点顺序（Nasion, Inion, LPA, RPA, Cz）必须与 Lumo 导出 JSON 中的 `landmark_coords` 顺序完全一致，否则旋转矩阵 $R$ 会出错。

    3. 雅可比矩阵与内存

        在 3D 重建中，`J` 矩阵可能会非常大（通道数 $\times$ 节点数）。

        * Lumo 的通道数通常很多（几百到上千）。
        * MNI152 节点数约为 21 万。
        * 如果内存不足，建议使用 `toast.Basis` 将重建空间降低到较低分辨率的网格（类似你例子中的 `basis_inv`）。

    4. MNE-NIRS 的作用

        * **清理数据**：Lumo 的原始 binary 包含噪声。通过 `mne-nirs` 转换成 OD 并滤波，可以显著提高重建质量。
        * **差分重建**：通常我们重建的是 $\Delta\mu_a$（即任务态 vs 基线的变化）。公式为：

        $$\Delta \text{OD} = J \cdot \Delta \mu_a$$

        这里的 `delta_od` 就是 `mne` 处理后的信号差值。

    **建议下一步**

    您可以先尝试运行 `setup_toast_3d` 部分，使用 `pyvista` 确认探头是否正确“贴”在了 MNI152 的头皮上。如果探头悬空或者掉进了脑子里，重建结果将完全不可信。

    **需要我针对某个特定部分（如如何把 Mshs 对象转换成 Toast 兼容格式）深入写一下代码吗？**

* 我可以确认探头经过 regi_xxx 函数配准后，一定贴在 mni scalp mesh 头皮上，我已经做了可视化驻

    太棒了！既然坐标配准（Registration）已经解决，且探头已精准“贴合”在头皮上，我们接下来的核心任务是**打通数据流**：将 MNE 处理后的物理信号（$\Delta OD$）映射到 Toast++ 的前向模型矩阵（Jacobian）中，完成 3D 空间的反演。

    由于 Lumo 是 **CW（连续波）** 设备，我们通常只重建吸收系数的变化 $\Delta\mu_a$。

    **核心实现逻辑**

    1. **物理量对齐**：MNE-NIRS 输出的是 $\Delta OD = -\ln(I/I_0)$。在 Toast++ 中，对于 CW 数据，前向模型对应的 Jacobian 元素 $J_{ij} = \frac{\partial \ln \Phi_i}{\partial \mu_{a,j}}$。
    2. **网格转换**：将 `mshs.headVolumeMesh` 的节点和四面体单元导入 Toast。
    3. **正则化反演**：使用 Tikhonov 正则化处理高度欠定（Underdetermined）的问题。

    ---

    **3D 重建完整示例代码**

    ```python
    import numpy as np
    import toast
    import mne
    import mne_nirs
    import pyvista as pv
    from scipy.sparse import linalg

    def run_lumo_toast_reconstruction(int_bin, rec_data, sd_layout, mshs, mni_landmarks):
        """
        基于已配准坐标的 3D 重建流程
        """
        
        # --- 1. MNE-NIRS 预处理获取 Delta OD ---
        # 构造 MNE 格式 (简化版，仅演示数据流)
        data = int_bin.data.T  # [channels, time]
        info = mne.create_info(ch_names=[f'CH{i}' for i in range(rec_data.n_chans)], 
                            sfreq=rec_data.framerate, ch_types='fnirs_cw_amplitude')
        raw = mne.io.RawArray(data, info)
        
        # 预处理：强度 -> 光密度 -> 滤波 -> 提取均值变化
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        raw_od.filter(0.01, 0.1)
        
        # 假设我们要重建全段相对于初始时刻的变化
        od_data = raw_od.get_data()
        delta_od = od_data[:, -1] - od_data[:, 0] # 最后一个时刻 vs 第一个时刻
        
        # --- 2. Toast++ 网格加载 ---
        # 将 MNI152 的头模型转为 Toast 可识别的 Mesh
        nodes = mshs.headVolumeMesh.node
        # 注意：matlab 加载的索引通常从1开始，需要确认是否需 -1
        # 且四面体在 toast 中通常是 [n, 4]
        elems = mshs.headVolumeMesh.elem[:, :4] 
        
        tm = toast.Mesh()
        tm.SetNodes(nodes)
        tm.SetElements(elems)
        
        # --- 3. 设置源探测器 (Qvec & Mvec) ---
        # 使用你已经验证过的配准坐标
        src_mni = regi_coords_sd_to_mni(sd_layout.src_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        det_mni = regi_coords_sd_to_mni(sd_layout.det_coords_3d, sd_layout.landmark_coords, mni_landmarks)
        
        # 将源和探测器放置在头皮表面（Neumann 边界条件）
        qvec = tm.Qvec(type='Neumann', shape='Gaussian', width=2, pos=src_mni)
        mvec = tm.Mvec(shape='Gaussian', width=2, pos=det_mni)
        
        # --- 4. 配置光学参数 ---
        nlen = tm.NodeCount()
        # 典型生物组织背景值 (单位: mm^-1)
        mua = np.ones(nlen) * 0.01 
        mus = np.ones(nlen) * 1.0  
        ref = np.ones(nlen) * 1.4
        freq = 0  # CW 模式
        
        # --- 5. 计算 Jacobian ---
        print("Building Forward Model...")
        phi = tm.Fields(None, qvec, mua, mus, ref, freq)
        aphi = tm.Fields(None, mvec, mua, mus, ref, freq)
        
        # 计算灵敏度矩阵 (Jacobian)
        # 对于 CW，我们需要的是对 log 强度的导数
        proj = tm.Projection(phi, mvec) 
        # 获取 Jacobian (映射到节点空间)
        J = tm.Jacobian(None, phi, aphi, proj)
        
        # --- 6. 线性线性反演 ---
        # 因为通道数可能远小于节点数，这是一个极度病态的问题
        # 使用 Tikhonov 正则化: x = J^T * (J*J^T + alpha*I)^-1 * y
        print("Inverting...")
        # 选取正则化参数，通常根据最大奇异值的一定比例
        alpha = 1e-3 * np.linalg.norm(J @ J.T)
        
        # 求解: (J*J.T + alpha*I) * beta = delta_od
        A = J @ J.T + alpha * np.eye(J.shape[0])
        beta = np.linalg.solve(A, delta_od)
        # 映射回 3D 空间
        delta_mua = J.T @ beta

        return tm, delta_mua

    # --- 7. 可视化函数 ---
    def plot_3d_result(tm, delta_mua, mshs):
        nodes = tm.Nodes()
        elems = tm.Elements()
        
        # 创建 PyVista 对象
        # cells 数组格式: [n_points, p1, p2, p3, p4, n_points, ...]
        cells = np.hstack(np.c_[np.full(elems.shape[0], 4), elems])
        grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA]*elems.shape[0], nodes)
        
        grid.point_data["delta_mua"] = delta_mua
        
        plotter = pv.Plotter()
        
        # 添加半透明头皮作为背景
        scalp_poly = pv.PolyData(mshs.scalpSurfaceMesh.node, 
                                np.hstack(np.c_[np.full(mshs.scalpSurfaceMesh.face.shape[0], 3), 
                                                mshs.scalpSurfaceMesh.face]))
        plotter.add_mesh(scalp_poly, color="tan", opacity=0.15)
        
        # 关键：使用切片可视化脑内吸收变化
        slices = grid.slice_orthogonal(x=0, y=0, z=0)
        plotter.add_mesh(slices, cmap="RdBu_r", scalars="delta_mua", 
                        stitle="Absorption Change", clim=[-max(abs(delta_mua)), max(abs(delta_mua))])
        
        # 或者展示高吸收区域的等值面
        # threshold = grid.threshold(np.max(delta_mua) * 0.5)
        # plotter.add_mesh(threshold, color="red")
        
        plotter.add_scalar_bar()
        plotter.show()

    ```

    ---

    **几个关键细节提醒：**

    1. **Lumo 数据索引映射**：
        
        Lumo 的 `chans_list` 定义了哪些源-探测器对组成了通道。在 `tm.Qvec` 和 `tm.Mvec` 之后产生的 `phi` 和 `aphi` 矩阵，计算出的 `J` 矩阵行数默认是 `N_src * N_det`。你可能需要根据 `rec_data.chans_list_act` 提取出对应的行，使得 `J` 的行数与 `delta_od` 的长度一致。

        ```python
        # 假设 J 原始大小是 (Nsrc*Ndet, Nnodes)
        # 需要根据实际通道配置过滤
        J = J[active_channel_indices, :]

        ```

    2. **MNI152 组织分层**：

        你加载的 `headVolumeMesh` 包含 `labels` (Scalp, Skull, CSF, GM, WM)。
        * **进阶技巧**：在反演时，可以对不同的组织层设置不同的先验权重。例如，让 GM（灰质）层的正则化系数更小，从而引导重建结果更多地出现在脑皮层上。


    3. **单位一致性**：

        Toast 内部默认通常使用 `mm`。请确保你的 `mshs.node` 和 `sd_layout` 的坐标单位均为 `mm`。如果 MNI 坐标单位是厘米，请务必统一乘以 10。

    **下一步建议**

    你可以先尝试运行 `tm.Fields` 这一步。如果能顺利跑通前向计算并生成 `phi`（光通量分布），说明网格和源探测器配置完全正确。

    **要不要我帮你写一段代码，根据 Lumo 的 `chans_list` 自动从 Toast 的全采样 Jacobian 中提取出对应通道的子集？**

* `.nim` 文件

    ```nim
    NIM
    Mesh = ellips_tri10.msh
    SolutionType = N/A
    ImageSize = 32971
    EndHeader
    Image 0
    0.025 0.025 0.025 0.025 ...
    ```

* `ReadNim()`

    这段 C++ 代码展示了 `TOAST++` 底层库（FELIB）中 `ReadNim` 函数的实现逻辑。

    我们可以清晰地看到 `.nim` (Node Image) 或 `.rim` (Region Image) 文件的**内部结构**以及该函数的核心功能。

    1. 函数功能概述

        这段代码的主要作用是：**从一个文本格式的图像文件（.nim/ .rim）中提取特定的图像数据（标量场）以及关联的网格名称。**

        它并不直接读取网格的几何结构（那是 `Mesh::Read` 的工作），而是读取附加在网格节点或区域上的**数值数据**（例如吸收系数分布 $\mu_a$ 或散射系数分布 $\mu_s$）。

    ---

    2. 代码逻辑逐段解析

        **A. 文件头校验 (Header Check)**

        ```cpp
        if (!ifs.getline (cbuf, 256)) return false;
        if (strcmp (cbuf, "NIM") && strcmp (cbuf, "RIM")) return false;

        ```

        * **作用**：检查文件的第一行。
        * **关键点**：文件必须以关键字 `NIM` 或 `RIM` 开头。如果不是，函数立即返回 `false`（失败）。

        **B. 解析文件头信息 (Metadata Parsing)**

        ```cpp
        do {
            ifs.getline (cbuf, 256);
            if (!strncasecmp (cbuf, "ImageSize", 9))
                sscanf (cbuf+11, "%d", &imgsize);
            else if (!strncasecmp (cbuf, "Mesh", 4) && meshname)
                sscanf (cbuf+6, "%s", meshname);
        } while (strcasecmp (cbuf, "EndHeader"));

        ```

        * **作用**：在 `EndHeader` 标记出现前，循环查找关键参数。
        * **ImageSize**：读取数据的长度（通常对应网格的节点数或单元数）。
        * **Mesh**：读取该数据所依赖的网格文件名（例如 `mesh.msh`）。这解释了为什么函数参数里有一个 `meshname` 指针。

        **C. 索引定位与数据读取 (Data Extraction)**

        ```cpp
        for (;;) {
            do {
                ifs.getline (cbuf, 256);
            } while (ifs.good() && strncasecmp (cbuf, "Image", 5));
            // ...
            for (i = 0; i < imgsize; i++)
                ifs >> img[i];
            if (++j == idx) break;
        }

        ```

        * **作用**：`.nim` 文件可以包含多个数据集（Frames）。
        * **逻辑**：函数会查找以 `Image` 开头的标记。
        * **参数 `idx` 的意义**：如果你传入 `idx = 2`，它会跳过第一个图像块，读取文件中的第二个图像块。
        * **img 容器**：读取到的浮点数会被存储在 `RVector &img`（一个实数向量）中。

    ---

    3. 核心结论：它到底读了什么？

        基于源码，我们可以推断出一个 `.nim` 文件的典型结构如下：

        | 行/部分 | 内容示例 | 说明 |
        | --- | --- | --- |
        | **Header** | `NIM` | 文件类型标识 |
        |  | `ImageSize 1024` | 该网格有 1024 个节点 |
        |  | `Mesh mymesh.msh` | 关联的网格文件 |
        |  | `EndHeader` | 头部结束 |
        | **Data Block** | `Image 1` | 第一个数据块开始 |
        |  | `0.01 0.02 ...` | 1024 个浮点数值 |
        |  | `Image 2` | 第二个数据块开始 |
        |  | ... | ... |

## topics

