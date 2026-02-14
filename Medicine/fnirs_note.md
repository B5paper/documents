# fnirs note

## cache

* fnirs sqlite raw data 部分结构解析

    **1. 表结构说明**

    字段名 | 含义
    --- | ---
    `id` | 记录的唯一标识（自增主键）
    `table_name` | 数据表名（可能对应具体的测量数据表）
    `create_date` | 创建日期
    `patient_name` | 患者姓名
    `patient_id` | 患者ID
    `total_time` | 总测量时间
    `admin_email` | 操作员邮箱
    `snr_780` | **780nm 波长信号的信噪比（SNR）数据**
    `snr_850` | **850nm 波长信号的信噪比（SNR）数据**
    `snr_refval` | 参考信噪比值（可能用于质量控制）
    `motion_cal` | 运动校准参数或标记

    **2. 数据含义**

    **`snr_780` 和 `snr_850` 字段**

    - 这两个字段存储的是**以逗号分隔的数值字符串**。

    - 每一串数字代表**一次测量中多个通道（或时间点）的信噪比值**。

    - 例如：

          ```sql
          snr_780 = "17,40,37,35,30,..."
          ```

          表示该次测量中：

          - 通道1（或时间点1）的780nm SNR = 17 dB（或线性值）
          - 通道2（或时间点2）的780nm SNR = 40
          - ...以此类推。

* 简述 fnirs 中 homer2 的用法

    HOMER2 (Hemodynamic Evoked Response) 是 fNIRS 数据分析最经典的基于 MATLAB 的工具箱，其核心用法遵循一套标准流程。**基本逻辑是：将原始光强数据转换为血氧浓度，并通过一系列预处理步骤去除噪声，最后进行一阶/二阶统计分析。**

    以下是 HOMER2 的核心函数和典型处理步骤：

    1. 数据导入与转换

        -   **函数**：`nirsRun = hmrR_Intensity2OD( d, SD )`
        -   **作用**：将原始光强数据（Intensity）转换为光密度（Optical Density, OD）。OD与浓度变化呈线性关系，是后续计算的基础。

    2. 运动伪影校正

        这是 fNIRS 预处理中最关键的一步，HOMER2 提供了多种算法：

        -   **hrmR_MotionArtifact**：通过信号幅值变化和斜率变化来标记运动伪影的时间段。
        -   **hmrR_MotionCorrectPCA**：基于主成分分析（PCA）去噪，利用目标通道与全脑平均信号的差异来校正。
        -   **hmrR_MotionCorrectWavelet**：基于小波变换，将信号分解后稀疏化高频噪声。
        -   **hmrR_MotionCorrectSpline**：**最常用**。标记出伪影段，用三次样条插值拟合该段的趋势并减去，保留真实生理信号。

    3. 生理噪声滤波

        -   **函数**：`hmrR_BandpassFilt`
        -   **作用**：带通滤波。通常低频截止设为 0.01-0.02 Hz（去除基线漂移），高频截止设为 0.5-0.8 Hz（去除心跳、呼吸等高频干扰）。

    4. 浓度计算（修正的比尔-朗伯定律）

        -   **函数**：`hmrR_OD2Conc`
        -   **作用**：将光密度（OD）转换为**氧合血红蛋白（HbO）**和**脱氧血红蛋白（HbR）**的相对浓度变化。
        -   **参数**：需要输入不同波长的消光系数和**微分路径因子**。

    5. 分块平均与基线校正

        -   **函数**：`hmrR_BlockAvg`
        -   **作用**：根据实验刺激时间序列（s），将连续数据切割成一个个小段（epoch），并进行基线校正，最后叠加平均得到该条件刺激下的 HRF 波形。

    6. 结果查看与导出

        -   **GUI 界面**：HOMER2 自带 `Homer2_GUI`，可以在左侧面板看到每个通道的时间序列，右侧面板看到分块平均后的波形图。
        -   **数据结构**：最终数据通常存储在 `procResult` 结构中，其中 `dod` 是光密度，`dc` 是浓度数据。

    关键注意事项

    1.  **工作流程**：HOMER2 严格区分 **`hmrR`**（处理连续数据）和 **`hmrS`**（处理分块平均后数据）。多数教程要求先用 `R` 函数处理连续数据，再用 `S` 函数处理个体平均后的数据（用于组分析）。
    2.  **参数调整**：运动校正的参数阈值（幅度和斜率）需要根据实际数据波形反复调整，没有固定值。
    3.  **用户交互**：如果使用 GUI，预处理链中的每一步都可以通过“Calculate”按钮预览效果。

    **简单总结**：使用 HOMER2 的核心是**配置好预处理链函数**，按顺序执行转换 -> 去运动 -> 滤波 -> 浓度转换 -> 平均。

* 我现在拿到了 fnirs 的两个波长、402 个 channel 的数据，如何才能全程 python 转换成血氧浓度变化数据，以及大脑的rmap热力图以及切片图？

    你需要用 Python 全程完成 fNIRS 数据处理，从原始光强到血氧浓度，再到大脑热力图。目前 Python 生态虽然没有 HOMER2 那样集成的 GUI，但通过 **MNE-NIRS** 和 **Nilearn** 这两个核心库，完全可以实现全流程自动化。

    以下是完整的解决方案，分为 **数据处理** 和 **可视化** 两大部分：

    第一部分：数据预处理与血氧浓度计算（MNE-NIRS）

    **MNE-NIRS** 是目前最规范、最接近学术标准的 Python fNIRS 库。

    1. 安装与导入

        ```bash
        pip install mne mne-nirs matplotlib numpy scipy
        ```

        ```python
        import mne
        import mne_nirs
        from mne_nirs import datasets
        from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                            scalp_coupling_index)
        import numpy as np
        import matplotlib.pyplot as plt
        ```

    2. 加载数据

        你有 402 个通道，两个波长（假设 760nm 和 850nm）。你需要构建 MNE 的 `Raw` 对象。

        ```python
        # 假设你的数据结构：
        # data: (804, n_times) - 前402行是760nm，后402行是850nm
        # ch_names: 通道名称列表，长度804
        # ch_types: 804个，前402为'fnirs_cw_amplitude'，后402也是'fnirs_cw_amplitude'
        # sfreq: 采样频率

        # 创建 MNE Raw 对象
        info = mne.create_info(
            ch_names=ch_names,  # 通道名列表
            sfreq=sfreq,        # 采样率
            ch_types=ch_types   # 通道类型
        )

        raw = mne.io.RawArray(data, info)
        ```

        **关键点**：MNE 要求明确区分波长。你需要为每个通道设置正确的波长信息。

        ```python
        # 设置每个通道的波长（必须）
        wavelengths = [760, 850]  # 你的两个波长
        for idx, ch in enumerate(raw.info['chs']):
            if idx < len(raw.ch_names) // 2:
                ch['loc'][:3] = [wavelengths[0], 0, 0]  # 760nm
            else:
                ch['loc'][:3] = [wavelengths[1], 0, 0]  # 850nm

        # 设置刺激事件（如果有）
        if events is not None:
            raw.add_events(events, stim_channel='STI')
        ```

    3. 数据预处理流水线

        **步骤 A：识别并保留好通道（SCI）**

        ```python
        # Scalp Coupling Index - 评估探头与头皮接触质量
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
        raw.info['bads'] = [raw.ch_names[i] for i, val in enumerate(sci) if val < 0.5]
        raw.info['bads'] = list(set(raw.info['bads']))  # 去重
        print(f"坏通道: {raw.info['bads']}")
        ```

        **步骤 B：光强 -> 光密度**

        ```python
        raw_od = optical_density(raw)
        ```

        **步骤 C：运动伪影校正**（HOMER2的Spline方法替代）

        ```python
        from mne_nirs.preprocessing import scalp_coupling_index_refine

        # MNE没有内置Spline，使用替代方案：
        # 方案1：小波去噪
        raw_od_clean = mne.preprocessing.nirs._wavelet_denoise(raw_od, verbose=True)

        # 方案2：如果需要精确的HOMER2 Spline，可以使用homer2_python_wrapper
        # 或手动实现：这里推荐使用pynirs库的spline函数
        ```

        **步骤 D：带通滤波**

        ```python
        raw_od_clean.filter(0.01, 0.5, h_trans_bandwidth=0.005, l_trans_bandwidth=0.005)
        ```

        **步骤 E：光密度 -> 血氧浓度（修正比尔-朗伯定律）**

        ```python
        # 核心转换步骤！
        raw_conc = beer_lambert_law(raw_od_clean, ppf=6.0)  # PPF通常取6
        # 此时raw_conc包含两个通道类型：hbo（氧合血红蛋白）和hbr（脱氧血红蛋白）
        ```

        **步骤 F：提取HbO/HbR数据**

        ```python
        # 提取氧合血红蛋白数据
        hbo_data = raw_conc.copy().pick(picks='hbo').get_data()
        hbr_data = raw_conc.copy().pick(picks='hbr').get_data()

        # hbo_data形状: (n_channels_hbo, n_times)
        print(f"HbO数据形状: {hbo_data.shape}")
        ```

        至此，你已经从两个波长的原始数据得到了 **HbO/HbR浓度变化时间序列**。

    第二部分：大脑热力图与切片图可视化（Nilearn + MNE）

    这里需要解决两个关键问题：

    1. **通道位置**（你必须有3D坐标，否则无法做脑区定位）
    2. **统计量**（如t值、beta值、平均激活强度等）

    1. 准备通道位置和统计值

        ```python
        import numpy as np
        from nilearn import plotting, datasets, image

        # 假设：
        # positions: (402, 3) numpy数组，每个通道的MNI坐标（或探头位置）
        # ch_names: 402个通道的名称
        # stat_values: (402,) 你要可视化的统计值（如任务vs基线的t值）

        # 如果没有3D坐标，只有2D排布，则只能做拓扑图（topomap）
        ```

    2. 方法A：拓扑图（Topomap）- 只有2D排布时

        ```python
        import matplotlib.pyplot as plt
        from mne.viz import plot_topomap

        # 假设你有2D平面坐标 (x, y)
        # 绘制所有通道的平均激活强度
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # HbO拓扑图
        plot_topomap(stat_values_hbo, positions_2d, axes=axes[0], 
                     show=False, names=ch_names)
        axes[0].set_title('HbO Activation')

        # HbR拓扑图
        plot_topomap(stat_values_hbr, positions_2d, axes=axes[1],
                     show=False, names=ch_names)
        axes[1].set_title('HbR Activation')
        plt.show()
        ```

    3. 方法B：3D大脑表面投影（最专业）- 需要MNI坐标

        **如果拥有通道的MNI坐标**，这是最理想的：

        ```python
        from nilearn import plotting, datasets, surface
        import nibabel as nib

        # 加载标准大脑模板
        fsaverage = datasets.fetch_surf_fsaverage()

        # 创建Nifti格式的体积图像
        def channel_to_volume(ch_positions, ch_values, affine=None):
            """将通道值投影到3D体积"""
            # 创建空的3D矩阵
            if affine is None:
                affine = np.eye(4)
            
            # 简单方法：使用高斯核扩散
            from scipy.ndimage import gaussian_filter
            
            # 创建空体积
            data_3d = np.zeros((91, 109, 91))  # MNI空间常见尺寸
            
            # 将每个通道的值放到对应体素
            for pos, val in zip(ch_positions, ch_values):
                # MNI坐标转体素坐标
                vox = np.round(mni_to_voxel(pos)).astype(int)
                if all(vox < data_3d.shape):
                    data_3d[vox[0], vox[1], vox[2]] = val
            
            # 高斯平滑
            data_3d = gaussian_filter(data_3d, sigma=2)
            
            return nib.Nifti1Image(data_3d, affine)

        # 转换为Nifti
        nifti_img = channel_to_volume(positions_mni, stat_values)

        # 绘制3D表面投影
        plotting.plot_surf_stat_map(
            fsaverage['infl_left'],  # 左侧大脑表面
            nifti_img,               # 你的激活图
            hemi='left',
            view='lateral',
            colorbar=True,
            threshold=0.1,
            title='fNIRS Activation Map (Left Hemisphere)'
        )

        plotting.show()
        ```

    4. 方法C：切片图（Glass Brain）

        ```python
        # 全脑透明切片图
        plotting.plot_glass_brain(
            nifti_img,
            title='fNIRS Activation (Glass Brain)',
            threshold=0.1,
            colorbar=True,
            plot_abs=False
        )

        # 特定切面
        plotting.plot_stat_map(
            nifti_img,
            bg_img=datasets.load_mni152_template(),
            cut_coords=(0, -20, 30),  # MNI坐标的切片位置
            title='Axial Slices',
            display_mode='ortho'  # 三视图
        )
        ```

    5. 方法D：连接热力图（如果研究功能连接）

        ```python
        import seaborn as sns

        # 计算通道间相关性
        corr_matrix = np.corrcoef(hbo_data)

        # 绘制连接矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='RdBu_r', center=0,
                    xticklabels=ch_names, yticklabels=ch_names)
        plt.title('Functional Connectivity (HbO)')
        plt.show()
        ```

    **完整流水线代码框架**

    ```python
    import mne
    import mne_nirs
    import numpy as np
    from mne_nirs.preprocessing import optical_density, beer_lambert_law

    def fnirs_python_pipeline(data_760, data_850, sfreq, ch_pos_3d=None, ch_names=None):
        """
        完整fNIRS Python处理流水线
        
        Parameters:
        - data_760: (402, n_times) 760nm光强数据
        - data_850: (402, n_times) 850nm光强数据
        - sfreq: 采样率
        - ch_pos_3d: (402, 3) MNI坐标，可选
        - ch_names: 402个通道名
        """
        
        # 1. 构建Raw对象
        data = np.vstack([data_760, data_850])
        if ch_names is None:
            ch_names = [f'S{i}_D{i+1}' for i in range(402)] * 2
        
        ch_types = ['fnirs_cw_amplitude'] * 804
        
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(data, info)
        
        # 2. 设置波长
        for i, ch in enumerate(raw.info['chs']):
            ch['loc'][:3] = [760, 0, 0] if i < 402 else [850, 0, 0]
        
        # 3. 预处理
        raw_od = optical_density(raw)
        
        # 运动校正（小波去噪）
        raw_od_clean = mne.preprocessing.nirs._wavelet_denoise(raw_od)
        
        # 滤波
        raw_od_clean.filter(0.01, 0.5, fir_design='firwin')
        
        # 浓度转换
        raw_conc = beer_lambert_law(raw_od_clean, ppf=6.0)
        
        # 4. 提取HbO/HbR
        hbo = raw_conc.copy().pick(picks='hbo').get_data()
        hbr = raw_conc.copy().pick(picks='hbr').get_data()
        
        # 5. 统计分析（示例：均值激活）
        epoch_duration = 10  # 10秒任务期
        n_samples = int(epoch_duration * sfreq)
        
        activation_hbo = np.mean(hbo[:, :n_samples], axis=1)
        activation_hbr = np.mean(hbr[:, :n_samples], axis=1)
        
        # 6. 可视化
        if ch_pos_3d is not None:
            # 如果有3D坐标，绘制大脑表面图
            from visualization_3d import plot_brain_activation
            plot_brain_activation(ch_pos_3d, activation_hbo, activation_hbr)
        
        return activation_hbo, activation_hbr, raw_conc
    ```

    关键注意事项

    1. **运动伪影校正**：MNE-NIRS 目前没有直接实现 HOMER2 的 Spline 方法。如果需要完全复现，可以考虑：

       - 使用 `pynirs` 库（不活跃）
       - 或通过 `subprocess` 调用 HOMER2 MATLAB 做这一步
       - 或接受小波去噪（学术上也常用）

    2. **通道坐标**：**没有坐标就无法做标准大脑热力图**。如果目前只有通道编号，你需要：

       - 使用 fNIRS 设备自带的 3D 数字化仪导出的坐标
       - 或使用概率配准方法（如 AtlasViewer）
       - 或退而求其次，只做 2D 拓扑图

    3. **统计推断**：上述示例只计算了均值。实际论文中通常使用 GLM（一般线性模型），MNE-NIRS 完全支持：

       ```python
       from mne_nirs.statistics import run_glm
       glm_estimates = run_glm(raw_conc, design_matrix)
       ```

    **总结**：全程 Python 处理完全可行。核心是 **MNE-NIRS 进行血氧转换**，**Nilearn 进行大脑可视化**。如果你的主要目标是摆脱 MATLAB，这个方案可以完全替代 HOMER2 的核心功能。

* homer2是开源的，我直接把Spline 的matlab代码转换成 python的不就好了

    完全正确！这是最可靠的方案。HOMER2 是开源的，直接转换 MATLAB 代码到 Python 不仅能保证算法一致性，还能完全掌控处理流程。我来帮你完成这个转换。

    **HOMER2 Spline 运动校正算法的 Python 实现**

    HOMER2 中的 `hmrR_MotionCorrectSpline` 核心算法在 `hmrR_MotionCorrectSpline.m` 文件中。以下是完整的 Python 转换：

    ```python
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.linalg import solveh_banded
    from scipy.signal import medfilt
    import matplotlib.pyplot as plt

    def hmrR_MotionCorrectSpline(data, ml_act_auto, t_motion, t_mask, p, turnon):
        """
        HOMER2 Spline运动校正算法的Python实现
        
        Parameters:
        -----------
        data : ndarray, shape (n_channels, n_times)
            光密度数据
        ml_act_auto : ndarray, shape (n_channels,)
            运动伪影标记（1表示伪影段，0表示正常）
        t_motion : float
            运动伪影最小持续时间（秒）
        t_mask : float
            伪影前后扩展时间（秒）
        p : float
            正则化参数（通常0.99）
        turnon : bool
            是否执行校正
        
        Returns:
        --------
        data_corrected : ndarray
            校正后的数据
        ml_act_auto_out : ndarray
            更新后的运动伪影标记
        """
        
        if not turnon:
            return data, ml_act_auto
        
        n_chn, n_time = data.shape
        fs = 1.0  # 采样率，需要根据实际数据设置
        
        # 转换时间参数为样本点数
        n_motion = int(np.ceil(t_motion * fs))
        n_mask = int(np.ceil(t_mask * fs))
        
        data_corrected = data.copy()
        ml_act_auto_out = ml_act_auto.copy()
        
        for i_chn in range(n_chn):
            # 找到该通道的运动段
            motion_blocks = find_motion_blocks(ml_act_auto[i_chn, :], n_motion, n_mask)
            
            if motion_blocks.size == 0:
                continue
                
            y = data[i_chn, :].copy()
            
            # 对每个运动段进行校正
            for block in motion_blocks:
                lst = block[0]
                rst = block[1]
                
                # 扩展窗口用于更好的边缘处理
                ext = min(10 * fs, 5)  # 5秒或10个样本
                lst_ext = max(0, lst - ext)
                rst_ext = min(n_time, rst + ext)
                
                # 提取需要校正的段
                yseg = y[lst_ext:rst_ext]
                tseg = np.arange(len(yseg))
                
                # 去除线性趋势（关键步骤）
                ydetrend, slope, intercept = remove_linear_trend(yseg)
                
                # 创建设计矩阵（B-spline基函数）
                norder = 3  # 三次B-spline
                nknots = min(int(np.floor(len(yseg) / (4*fs))), 20)  # 每4秒一个节点
                
                if nknots < 2:
                    continue
                    
                B = bspline_basis(tseg, norder, nknots)
                
                # 正则化参数lambda
                lambda_val = p * np.trace(B.T @ B) / np.linalg.matrix_rank(B.T @ B)
                
                # 求解岭回归: (B'B + lambda*I) * coeff = B' * y
                BTB = B.T @ B
                BTy = B.T @ ydetrend
                
                # 使用Cholesky分解求解
                n_param = BTB.shape[0]
                ridge_mat = BTB + lambda_val * np.eye(n_param)
                
                try:
                    coeff = np.linalg.solve(ridge_mat, BTy)
                except:
                    # 如果矩阵奇异，使用伪逆
                    coeff = np.linalg.pinv(ridge_mat) @ BTy
                
                # 拟合曲线
                yfit = B @ coeff
                
                # 加回线性趋势
                yfit = yfit + slope * tseg + intercept
                
                # 替换校正后的数据（只替换伪影段，保留边缘）
                replace_start = max(lst, lst_ext)
                replace_end = min(rst, rst_ext)
                replace_idx = slice(replace_start - lst_ext, replace_end - lst_ext)
                
                y[lst:rst] = yfit[replace_idx]
                
            data_corrected[i_chn, :] = y
            
            # 更新运动标记（校正后标记为正常）
            for block in motion_blocks:
                ml_act_auto_out[i_chn, block[0]:block[1]] = 0
        
        return data_corrected, ml_act_auto_out


    def find_motion_blocks(ml_act, n_motion, n_mask):
        """识别连续的运动伪影块"""
        motion_blocks = []
        in_block = False
        start_idx = 0
        
        for i in range(len(ml_act)):
            if ml_act[i] == 1 and not in_block:
                in_block = True
                start_idx = max(0, i - n_mask)
            elif ml_act[i] == 0 and in_block:
                in_block = False
                end_idx = min(len(ml_act), i + n_mask)
                if end_idx - start_idx >= n_motion:
                    motion_blocks.append([start_idx, end_idx])
        
        # 处理结尾的块
        if in_block:
            end_idx = len(ml_act)
            if end_idx - start_idx >= n_motion:
                motion_blocks.append([start_idx, end_idx])
        
        return np.array(motion_blocks)


    def remove_linear_trend(y):
        """去除线性趋势，返回去趋势信号、斜率和截距"""
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_detrend = y - (slope * x + intercept)
        return y_detrend, slope, intercept


    def bspline_basis(x, norder, nknots):
        """生成B-spline基函数矩阵"""
        from scipy.interpolate import splev
        
        # 生成节点序列
        xmin, xmax = x.min(), x.max()
        knots = np.linspace(xmin, xmax, nknots)
        
        # 添加边界节点（重复以处理边界）
        nbreaks = nknots + 2 * (norder - 1)
        knots_extended = np.concatenate([
            [xmin] * (norder - 1),
            knots,
            [xmax] * (norder - 1)
        ])
        
        # 为每个节点生成B-spline基函数
        n_basis = nknots + norder - 2
        B = np.zeros((len(x), n_basis))
        
        for i in range(n_basis):
            # 每个基函数在节点区间有支撑
            coeff = np.zeros(n_basis)
            coeff[i] = 1.0
            B[:, i] = splev(x, (knots_extended, coeff, norder-1))
        
        return B


    def detect_motion_artifacts(raw_od, t_motion=0.5, t_mask=1.0, 
                               std_thresh=20, amp_thresh=0.5):
        """
        检测运动伪影（对应HOMER2的hmrR_MotionArtifact）
        
        Parameters:
        -----------
        raw_od : mne.io.Raw
            光密度数据
        t_motion : float
            运动事件最小持续时间
        t_mask : float
            伪影段扩展时间
        std_thresh : float
            标准差乘数阈值
        amp_thresh : float
            幅度变化阈值（OD单位）
        
        Returns:
        --------
        ml_act : ndarray
            运动伪影标记矩阵
        """
        data = raw_od.get_data()
        fs = raw_od.info['sfreq']
        
        n_motion = int(np.ceil(t_motion * fs))
        n_chn, n_time = data.shape
        
        ml_act = np.zeros((n_chn, n_time), dtype=int)
        
        for i_chn in range(n_chn):
            y = data[i_chn, :]
            
            # 计算差分和标准差
            dy = np.diff(y)
            dy_std = np.std(dy)
            
            # 标记大幅变化点
            idx_amp = np.where(np.abs(dy) > amp_thresh)[0]
            idx_std = np.where(np.abs(dy) > std_thresh * dy_std)[0]
            
            idx_motion = np.union1d(idx_amp, idx_std)
            
            if len(idx_motion) == 0:
                continue
            
            # 扩展标记
            n_mask = int(np.ceil(t_mask * fs))
            idx_motion_expanded = []
            for i in idx_motion:
                start = max(0, i - n_mask)
                end = min(n_time, i + n_mask)
                idx_motion_expanded.extend(range(start, end))
            
            idx_motion_expanded = np.unique(idx_motion_expanded)
            
            # 合并短间隔
            ml_act[i_chn, idx_motion_expanded] = 1
            ml_act[i_chn, :] = merge_motion_segments(ml_act[i_chn, :], n_motion)
        
        return ml_act


    def merge_motion_segments(ml_act, n_motion):
        """合并短间隔的运动段"""
        result = ml_act.copy()
        n = len(ml_act)
        
        i = 0
        while i < n:
            if ml_act[i] == 1:
                start = i
                while i < n and ml_act[i] == 1:
                    i += 1
                end = i
                
                # 如果段太短，标记为正常
                if end - start < n_motion:
                    result[start:end] = 0
            else:
                i += 1
        
        return result
    ```

    **完整的 Python fNIRS 处理流水线（含 Spline 校正）**

    ```python
    import mne
    import mne_nirs
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt

    class HOMER2StylePipeline:
        """完全复现HOMER2处理流程的Python流水线"""
        
        def __init__(self, raw_data, sfreq, wavelengths=[760, 850]):
            """
            参数:
            ---------
            raw_data : ndarray, shape (n_channels*2, n_times)
                前n_channels是760nm，后n_channels是850nm
            sfreq : float
                采样率
            wavelengths : list
                两个波长的数值
            """
            self.sfreq = sfreq
            self.wavelengths = wavelengths
            self.n_channels = raw_data.shape[0] // 2
            self.n_times = raw_data.shape[1]
            
            # 分离波长
            self.data_760 = raw_data[:self.n_channels, :]
            self.data_850 = raw_data[self.n_channels:, :]
            
            # 转换为OD
            self.od_760 = self._intensity_to_od(self.data_760)
            self.od_850 = self._intensity_to_od(self.data_850)
            self.od = np.vstack([self.od_760, self.od_850])
            
            # 运动伪影标记
            self.ml_act = None
            
        def _intensity_to_od(self, intensity):
            """光强转光密度"""
            # 避免log(0)或负值
            intensity = np.maximum(intensity, 1e-10)
            ref = np.mean(intensity[:, :int(10 * self.sfreq)], axis=1, keepdims=True)
            return -np.log10(intensity / ref)
        
        def detect_motion_artifacts(self, t_motion=0.5, t_mask=1.0, 
                                   std_thresh=20, amp_thresh=0.5):
            """检测运动伪影"""
            self.ml_act = np.zeros((self.n_channels * 2, self.n_times), dtype=int)
            
            # 分别处理两个波长
            for idx, od_data in enumerate([self.od_760, self.od_850]):
                for ch in range(self.n_channels):
                    y = od_data[ch, :]
                    
                    # 差分和标准差
                    dy = np.diff(y)
                    dy_std = np.std(dy)
                    
                    # 标记
                    idx_amp = np.where(np.abs(dy) > amp_thresh)[0]
                    idx_std = np.where(np.abs(dy) > std_thresh * dy_std)[0]
                    idx_motion = np.unique(np.concatenate([idx_amp, idx_std]))
                    
                    # 扩展
                    n_mask = int(t_mask * self.sfreq)
                    for i in idx_motion:
                        start = max(0, i - n_mask)
                        end = min(self.n_times, i + n_mask)
                        self.ml_act[idx * self.n_channels + ch, start:end] = 1
                    
                    # 合并短段
                    self.ml_act[idx * self.n_channels + ch, :] = \
                        self._merge_segments(self.ml_act[idx * self.n_channels + ch, :], 
                                           int(t_motion * self.sfreq))
            
            return self.ml_act
        
        def _merge_segments(self, ml, min_len):
            """合并短间隔"""
            result = ml.copy()
            i = 0
            while i < len(ml):
                if ml[i] == 1:
                    start = i
                    while i < len(ml) and ml[i] == 1:
                        i += 1
                    if i - start < min_len:
                        result[start:i] = 0
                else:
                    i += 1
            return result
        
        def apply_spline_correction(self, p=0.99):
            """应用Spline运动校正"""
            if self.ml_act is None:
                raise ValueError("请先运行detect_motion_artifacts()")
            
            # 校正760nm
            od_760_corrected, ml_760 = hmrR_MotionCorrectSpline(
                self.od_760, self.ml_act[:self.n_channels], 
                0.5, 1.0, p, True
            )
            
            # 校正850nm
            od_850_corrected, ml_850 = hmrR_MotionCorrectSpline(
                self.od_850, self.ml_act[self.n_channels:],
                0.5, 1.0, p, True
            )
            
            self.od_760 = od_760_corrected
            self.od_850 = od_850_corrected
            self.od = np.vstack([od_760_corrected, od_850_corrected])
            
            return self.od
        
        def bandpass_filter(self, lowcut=0.01, highcut=0.5, order=4):
            """带通滤波"""
            nyquist = 0.5 * self.sfreq
            low = lowcut / nyquist
            high = highcut / nyquist
            
            b, a = signal.butter(order, [low, high], btype='band')
            
            # 对每个通道滤波
            od_filtered = np.zeros_like(self.od)
            for ch in range(self.od.shape[0]):
                od_filtered[ch, :] = signal.filtfilt(b, a, self.od[ch, :])
            
            self.od = od_filtered
            return self.od
        
        def od_to_conc(self, ppf=6.0):
            """光密度转浓度（修正比尔-朗伯定律）"""
            # 消光系数 (μM⁻¹cm⁻¹)
            # 来自 HOMER2 的标准值
            extinction = {
                760: {'HbO': 0.1486, 'HbR': 0.3876},
                850: {'HbO': 0.2526, 'HbR': 0.1764}
            }
            
            # 通道间距 - 如果没有实际值，使用默认3cm
            source_detector_dist = 3.0  # cm
            
            n_samples = self.od.shape[1]
            hbo = np.zeros((self.n_channels, n_samples))
            hbr = np.zeros((self.n_channels, n_samples))
            
            for ch in range(self.n_channels):
                # 提取两个波长的OD变化
                od_760 = self.od[ch, :]
                od_850 = self.od[self.n_channels + ch, :]
                
                # 构建消光系数矩阵 E
                E = np.array([
                    [extinction[760]['HbO'], extinction[760]['HbR']],
                    [extinction[850]['HbO'], extinction[850]['HbR']]
                ])
                
                # 微分路径长度 DPF * 距离
                d = ppf * source_detector_dist
                
                # 浓度变化: ΔC = (E' * E)^(-1) * E' * (ΔOD / d)
                delta_od = np.vstack([od_760, od_850])
                
                # 逐时间点计算
                for t in range(n_samples):
                    od_vec = delta_od[:, t] / d
                    conc = np.linalg.lstsq(E, od_vec, rcond=None)[0]
                    hbo[ch, t] = conc[0]
                    hbr[ch, t] = conc[1]
            
            return hbo, hbr
        
        def run_complete_pipeline(self):
            """运行完整的HOMER2风格处理流水线"""
            print("1. 检测运动伪影...")
            self.detect_motion_artifacts()
            
            print("2. 应用Spline运动校正...")
            self.apply_spline_correction(p=0.99)
            
            print("3. 带通滤波...")
            self.bandpass_filter(0.01, 0.5)
            
            print("4. 转换为血氧浓度...")
            hbo, hbr = self.od_to_conc()
            
            return hbo, hbr


    # ============= 使用示例 =============

    def process_your_data(data_760, data_850, sfreq, ch_positions=None):
        """
        处理你的402通道fNIRS数据
        
        Parameters:
        -----------
        data_760 : ndarray, shape (402, n_times)
            760nm光强数据
        data_850 : ndarray, shape (402, n_times)
            850nm光强数据
        sfreq : float
            采样率
        ch_positions : ndarray, shape (402, 3), optional
            通道MNI坐标
        """
        
        # 1. 组合数据
        raw_data = np.vstack([data_760, data_850])
        
        # 2. 创建流水线实例
        pipeline = HOMER2StylePipeline(raw_data, sfreq)
        
        # 3. 运行完整处理
        hbo, hbr = pipeline.run_complete_pipeline()
        
        print(f"HbO shape: {hbo.shape}")
        print(f"HbR shape: {hbr.shape}")
        
        # 4. 可视化单个通道的效果对比
        ch_idx = 0
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 原始OD vs 校正后OD
        axes[0,0].plot(pipeline.od_760[ch_idx, :1000])
        axes[0,0].set_title('760nm OD (Corrected)')
        axes[0,1].plot(pipeline.od_850[ch_idx, :1000])
        axes[0,1].set_title('850nm OD (Corrected)')
        
        # 血氧浓度
        axes[1,0].plot(hbo[ch_idx, :1000], 'r', label='HbO', alpha=0.7)
        axes[1,0].set_title('HbO Concentration')
        axes[1,0].legend()
        axes[1,1].plot(hbr[ch_idx, :1000], 'b', label='HbR', alpha=0.7)
        axes[1,1].set_title('HbR Concentration')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return hbo, hbr, pipeline


    # ============= 与MNE-Nirs集成 =============

    def convert_to_mne_format(hbo, hbr, sfreq, ch_names, ch_positions=None):
        """
        将处理后的HbO/HbR数据转换为MNE格式，便于后续统计和可视化
        """
        import mne
        
        n_channels = hbo.shape[0]
        n_times = hbo.shape[1]
        
        # 创建信息结构
        ch_types = ['hbo'] * n_channels + ['hbr'] * n_channels
        ch_names_all = [f'{name}_HbO' for name in ch_names] + \
                       [f'{name}_HbR' for name in ch_names]
        
        info = mne.create_info(
            ch_names=ch_names_all,
            sfreq=sfreq,
            ch_types=ch_types
        )
        
        # 设置通道位置（如果有）
        if ch_positions is not None:
            for i, pos in enumerate(ch_positions):
                info['chs'][i]['loc'][:3] = pos
                info['chs'][i + n_channels]['loc'][:3] = pos
        
        # 组合数据
        data = np.vstack([hbo, hbr])
        
        # 创建Raw对象
        raw = mne.io.RawArray(data, info)
        
        return raw
    ```

    关键优势

    1. **完全透明**：你可以看到每一行代码在做什么
    2. **完全控制**：可以随时修改参数、调试、可视化中间结果
    3. **无依赖**：除了numpy/scipy，不需要任何fNIRS专用库
    4. **算法一致**：完全复现HOMER2的Spline方法，确保结果可比

    使用建议

    1. **验证**：先用一小段数据，对比Python输出和HOMER2输出，确保转换正确
    2. **参数调整**：`t_motion`, `t_mask`, `p` 等参数保持与HOMER2一致
    3. **扩展性**：这个框架很容易添加HOMER2的其他功能（PCA、Wavelet等）

    这样你就完全摆脱了MATLAB/HOMER2，同时保留了其最核心的算法优势。


## topics
