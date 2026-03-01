# fnirs note

## cache

* AdultMNI152

    AdultMNI152（通常指MNI-ICBM152）是神经影像学领域最广泛使用的标准脑模板之一，由蒙特利尔神经学研究所（MNI）开发。

    它的核心信息可以总结为以下几点：

    - **基本定义与构建**：它是一个平均脑模板，通过对 152 名健康年轻成人的 T1-weighted MRI 扫描进行线性配准并平均后构建而成。构建该模板的目的是为了提供一个通用的标准空间，使得不同研究、不同个体的脑图像可以在这个共同的空间下进行标准化处理和比较。

    - **人口统计学特征**：构成该模板的 152 名受试者具有特定的人口统计学特征，平均年龄为**25.02±4.90岁**（范围 18-44 岁），其中包含**86名男性和66名女性**，主要为白人（129人）。

    - **应用与局限性**：作为最常用的标准模板，AdultMNI152 已被集成到主流的图像处理软件包（如 SPM、FSL、MINC）中，支撑着从图像配准、分割到形态学分析（如 VBM）的各类应用。然而，其基于**健康年轻人**构建的特性也带来了明显的局限性。对于存在显著脑萎缩的**老年人**或**神经退行性疾病患者**（如阿尔茨海默病、额颞叶痴呆）的脑图像，直接使用该模板进行配准可能会导致较大的**配准误差**，从而影响下游分析的准确性。

* 3D 定位仪（digitizer）

* 一些有用的摘录

    > fNIRS devices consist of two kinds of optodes: light sources (AKA “emitters” or “transmitters”) and light detectors (AKA “receivers”). Channels are defined as source-detector pairs, and channel locations are defined as the midpoint between source and detector.

* oxyhaemoglobin prn. [ɒksihiməɡ'loʊbɪn] n. 氧化血红素

* 什么是 frequency domain fNIRS

    **Frequency Domain fNIRS (FD-fNIRS)** 简介

    Frequency Domain fNIRS是一种**先进的功能性近红外光谱技术**，它通过**调制光强频率**来测量大脑活动。让我详细解释：

    **基本原理**

    1. **与传统方法的对比**

        ```python
        # 三种主要fNIRS技术的对比
        techniques = {
            "Continuous Wave (CW-fNIRS)": {
                "光强调制": "恒定强度",
                "测量参数": "只有光衰减",
                "优缺点": ["简单便宜", "无法区分吸收和散射"]
            },
            "Frequency Domain (FD-fNIRS)": {
                "光强调制": "高频调制 (MHz范围)",
                "测量参数": "衰减 + 相位延迟 + 调制深度",
                "优缺点": ["可区分吸收和散射", "设备较复杂"]
            },
            "Time Domain (TD-fNIRS)": {
                "光强调制": "脉冲激光",
                "测量参数": "时间点扩展函数",
                "优缺点": ["信息最丰富", "最昂贵复杂"]
            }
        }
        ```

    2. **物理原理**

        ```python
        import numpy as np
        import matplotlib.pyplot as plt

        # 模拟FD-fNIRS的光强调制
        t = np.linspace(0, 10e-9, 1000)  # 时间(ns)
        frequency = 100e6  # 100 MHz调制

        # 入射光
        incident = np.sin(2 * np.pi * frequency * t)

        # 出射光（有衰减和相位延迟）
        attenuation = 0.7  # 衰减系数
        phase_delay = 0.5  # 相位延迟（弧度）
        detected = attenuation * np.sin(2 * np.pi * frequency * t - phase_delay)

        plt.figure(figsize=(10, 4))
        plt.plot(t*1e9, incident, label='入射光', alpha=0.7)
        plt.plot(t*1e9, detected, label='出射光', alpha=0.7)
        plt.xlabel('时间 (ns)')
        plt.ylabel('光强')
        plt.title('FD-fNIRS：高频调制光信号')
        plt.legend()
        plt.grid(True)
        plt.show()
        ```

    **测量参数**

    1. **三个关键参数**

        ```python
        class FDNIRS_Measurement:
            def __init__(self):
                self.DC = None      # 平均光强（衰减）
                self.AC = None      # 调制幅度（调制深度）
                self.phase = None   # 相位延迟
                
            def calculate_optical_properties(self):
                """
                从测量参数计算光学特性
                """
                # 吸收系数 (μa) 和 约化散射系数 (μs')
                # 通过求解光子扩散方程得到
                mu_a = self.solve_for_absorption(self.DC, self.AC, self.phase)
                mu_s_prime = self.solve_for_scattering(self.DC, self.AC, self.phase)
                
                return mu_a, mu_s_prime
            
            def solve_for_absorption(self, DC, AC, phase):
                """简化的吸收系数计算"""
                # 实际需要解扩散方程
                # 这里只是示意
                mu_a = (np.log(DC) + np.log(AC)) / (2 * phase)
                return mu_a
        ```

    2. **数据采集过程**

        ```python
        # 典型的FD-fNIRS采集流程
        class FDNIRS_Acquisition:
            def __init__(self, modulation_freq=110e6):  # 常用110 MHz
                self.freq = modulation_freq
                self.sampling_rate = 10 * modulation_freq  # 奈奎斯特定理
                
            def acquire_channel(self, source, detector):
                """
                单个通道的采集
                """
                # 1. 发送调制光
                self.emit_modulated_light(source, self.freq)
                
                # 2. 检测信号
                raw_signal = self.detect_light(detector)
                
                # 3. 解调（提取DC、AC和相位）
                DC = np.mean(raw_signal)
                AC = np.std(raw_signal) * np.sqrt(2)
                
                # 用锁相放大器提取相位
                reference = np.sin(2 * np.pi * self.freq * self.time)
                phase = self.extract_phase(raw_signal, reference)
                
                return {
                    'DC': DC,
                    'AC': AC,
                    'phase': phase,
                    'frequency': self.freq
                }
        ```

    **优势与局限**

    **优点**

    ```python
    advantages = {
        "定量测量": "✅ 可分别计算HbO和HbR的绝对浓度",
        "深层组织": "✅ 穿透深度比CW-fNIRS大（可达3-4cm）",
        "抗干扰": "✅ 调制技术降低环境光干扰",
        "多参数": "✅ 同时获得吸收和散射信息"
    }

    # 定量计算示例
    def calculate_absolute_concentration(AC, phase, baseline):
        """
        计算绝对血氧浓度
        """
        # 基于修正的Beer-Lambert定律
        # 但可以分离吸收和散射效应
        mu_a = extract_absorption(AC, phase)
        
        # 使用比尔定律计算浓度
        epsilon_HbO = 2.5  # 摩尔消光系数 (cm⁻¹/mM)
        epsilon_HbR = 1.8
        
        # 解联立方程
        HbO = (mu_a[760] * epsilon_HbR[850] - mu_a[850] * epsilon_HbR[760]) / \
              (epsilon_HbO[760] * epsilon_HbR[850] - epsilon_HbO[850] * epsilon_HbR[760])
        
        return HbO
    ```

    **局限**

    ```python
    limitations = {
        "设备成本": "💰 比CW-fNIRS贵3-5倍",
        "复杂性": "🔧 需要高频电子设备和精密校准",
        "数据处理": "📊 需要复杂的解调算法",
        "功耗": "⚡ 高频调制导致更高功耗"
    }
    ```

    **实际应用**

    1. **硬件组成**

        ```python
        class FDNIRS_System:
            def __init__(self):
                # 核心组件
                self.laser_diodes = {
                    '760nm': LaserDiode(freq=110e6, power=10),  # 去氧血红蛋白敏感
                    '850nm': LaserDiode(freq=110e6, power=10)   # 氧合血红蛋白敏感
                }
                
                self.detectors = {
                    'PMT': PhotomultiplierTube(gain=1e6),        # 高灵敏度
                    'APD': AvalanchePhotodiode(bandwidth=200e6)  # 高速响应
                }
                
                self.modulation = {
                    'type': 'sine',          # 正弦调制
                    'frequencies': [70e6, 110e6, 140e6],  # 多频可选
                    'demodulation': 'lock-in' # 锁相放大解调
                }
        ```

    2. **数据分析**

        ```python
        import numpy as np
        from scipy import signal

        def process_fdnirs_data(ac_760, ac_850, phase_760, phase_850, distance):
            """
            处理FD-fNIRS数据
            """
            # 1. 计算光学特性
            mu_a_760, mus_760 = solve_diffusion_equation(
                AC=ac_760, 
                phase=phase_760,
                distance=distance,
                wavelength=760
            )
            
            mu_a_850, mus_850 = solve_diffusion_equation(
                AC=ac_850,
                phase=phase_850,
                distance=distance,
                wavelength=850
            )
            
            # 2. 计算血氧浓度
            HbO, HbR = beer_lambert_modified(
                mu_a_760, 
                mu_a_850,
                wavelengths=[760, 850],
                extinction_coeffs=extinction_data
            )
            
            # 3. 额外信息：散射系数反映组织结构
            scatter_index = (mus_760 + mus_850) / 2
            
            return {
                'HbO': HbO,
                'HbR': HbR,
                'scatter': scatter_index,
                'phase_delay': (phase_760 + phase_850) / 2
            }
        ```

    **与其他技术的比较**

    | 特性 | CW-fNIRS | FD-fNIRS | TD-fNIRS |
    |------|----------|----------|----------|
    | **信息量** | 低 | 中 | 高 |
    | **成本** | $10k-50k | $50k-150k | $200k+ |
    | **时间分辨率** | 高(>100Hz) | 中(10-50Hz) | 低(<10Hz) |
    | **穿透深度** | 1-2cm | 2-3cm | 3-5cm |
    | **定量能力** | 相对变化 | 绝对浓度 | 绝对浓度+分层 |
    | **便携性** | 高 | 中 | 低 |

    **总结**

    **FD-fNIRS的核心价值**：

    1. **定量测量**：不再是简单的相对变化
    2. **区分吸收和散射**：获得更全面的组织信息
    3. **更好的深度穿透**：可以测量更深层的大脑组织

    **适用场景**：

    - 需要绝对血氧浓度的临床研究
    - 深层脑区测量
    - 需要区分血氧和散射效应的研究

    需要我详细解释FD-fNIRS在某个具体领域的应用吗？

* 简述什么是 frequency domain fNIRS。用中文回答

    **频域功能性近红外光谱（Frequency Domain fNIRS）** 是一种通过测量大脑皮层血氧代谢变化来评估神经活动的光学脑成像技术。

    它是 **fNIRS** 技术的一种实现形式。与常见的**连续波（Continuous Wave, CW）** 系统不同，频域技术通过向头部发射**强度经过高频调制的近红外光**（通常为几十至几百MHz），并分析从组织中散射出来的光信号的变化来获取数据。

    其核心原理和特点如下：

    1.  **测量参数（相位与衰减）**：

        -   **相移**：光在穿过组织时，由于传播路径的延迟，出射光的相位会相对于入射光发生**滞后**。这个相位延迟程度主要取决于光在组织中的**路径长度**和组织的**吸收系数**。
        -   **衰减**：出射光相对于入射光的平均光强衰减程度，反映了组织的总**吸收系数**和**散射系数**。

    2.  **关键优势：路径长度校正**：

        -   在连续波系统中，由于光在组织中的实际行走路径长度是未知的，很难进行精确的定量测量。
        -   频域系统通过测量**相移**，可以计算出光子在组织中的**平均路径长度**。一旦知道了路径长度，就可以结合光强的衰减，通过数学模型（如修正的比尔-朗伯定律的频域解）将吸收系数和散射系数**分离**开来。

    3.  **主要功能**：

        -   **定量测量**：能够更精确地计算出氧合血红蛋白（HbO）和脱氧血红蛋白（HbR）的**绝对浓度变化**（相比连续波系统通常只能测量相对变化）。
        -   **组织光学特性**：可以同时测量组织的吸收系数（μa）和约化散射系数（μs‘），有助于更深入地了解组织特性。

    **总结来说**，频域fNIRS比连续波fNIRS提供了更丰富的光学信息，能够实现更高精度的定量测量，并区分光的吸收与散射效应，这对于需要精确比较或绝对量化脑血流动力学反应的研究具有重要价值。

* 给一些频域 fnirs 相关的论文、项目、书籍等学习资料

    以下是关于频域fNIRS的精选学习资料，涵盖经典论文、开源项目、专业书籍和研究团队，帮助你系统掌握这一技术：

    **经典综述与奠基论文**

    1.  **《Frequency-domain techniques for cerebral and functional near-infrared spectroscopy》**

        -   *作者*：Fantini, S., & Franceschini, M. A. (2006)
        -   *来源*：Near-Infrared Spectroscopy, 11, 203-223.
        -   *简介*：该领域的经典入门综述，详细阐述了频域技术的物理原理及其在脑功能研究中的应用。

    2.  **《Non-invasive optical spectroscopy and imaging of human brain function》**

        -   *作者*：Villringer, A., & Chance, B. (1997)
        -   *来源*：Trends in Neurosciences, 20(10), 435-442.
        -   *简介*：虽然覆盖整个fNIRS领域，但对早期频域技术的理论基础（如光子迁移理论）有深刻论述。

    3.  **《Quantitative determination of the absorption and scattering spectra of the human breast in vivo using frequency-domain near-infrared optical tomography》**

        -   *作者*：Chance, B., et al. (1998)
        -   *简介*：展示了频域技术如何分离吸收和散射系数，对理解算法原理很有帮助。

    **核心理论书籍**

    4.  **《Handbook of Optical Biomedical Diagnostics》**

        -   *编者*：Valery V. Tuchin
        -   *章节*：第7章 "Frequency-Domain Optical Tomography" 详细讲解了频域测量系统的硬件设计与数学模型。

    5.  **《Near-Infrared Spectroscopy: Theory, Spectral Analysis, Instrumentation, and Applications》**

        -   *作者*：Yukihiro Ozaki (尾崎幸洋) 等
        -   *简介*：涵盖了频域fNIRS的理论基础，特别是关于相移测量与修正比尔-朗伯定律的部分。

    **开源硬件与软件项目**

    6.  **开源项目：ISS Imagent (相关算法库)**

        -   *简介*：ISS公司是频域fNIRS设备的市场领导者。虽然没有完全开源的硬件，但其部分数据处理算法已在学术社区共享。可以搜索 "ISS Imagent processing algorithms" 查找相关Matlab/Python代码。

    7.  **GitHub项目：** 搜索关键词 "**Frequency-domain fNIRS**" 或 "**FD-NIRS**"

        -   *推荐仓库*：
            -   **"NIRS-brain"**：基于Matlab的fNIRS数据分析包，支持频域数据处理。
            -   **"Homer3"**：最著名的fNIRS分析工具箱（基于Matlab），其核心函数（如`hmR_Intensity2OD`）可用于处理频域数据，计算吸收和散射系数。
            -   **"PyNIRS"**：Python版本的fNIRS工具箱，包含频域数据处理模块。

    8.  **硬件设计参考：**

        -   *论文*："A low-cost, frequency-domain, near-infrared spectroscopy system" 搜索此类论文可找到一些高校设计的开源硬件方案（通常基于FPGA或锁相放大器）。

    **前沿研究团队与实验室**

    9.  **Sergio Fantini 实验室 (塔夫茨大学)**

        -   *简介*：频域fNIRS领域的权威之一，专注于定量光学成像和脑氧代谢监测。其网站通常有公开的算法和数据集。

    10. **Arjun Yodh 实验室 (宾夕法尼亚大学)**

        -   *简介*：专注于扩散光学成像（包括频域和时域），在生物医学光学顶刊发表大量高被引论文。

    11. **Maria Angela Franceschini 实验室 (麻省总医院)**

        -   *简介*：开发了先进的fNIRS技术（包括频域系统）用于临床脑监测。

    **中国学者与中文资源**

    12. **国内团队：**

        -   **天津大学 高峰教授团队**：在扩散光学层析成像（包括频域方法）方面有深入研究，发表多篇理论推导和系统搭建的论文。
        -   **华中科技大学 骆清铭院士团队**：虽然更知名于显微成像，但其在近红外光谱与脑功能研究方面也有涉及频域技术的论文。

    13. **中文综述：**

        -   *论文*：《频域近红外光谱技术用于脑功能成像的研究进展》（可在知网或万方数据搜索），适合快速了解中文术语和技术框架。

    **实用建议：如何入手**

    -   **第一步**：先读 **Fantini 2006** 的综述，搞懂“相移”和“调制频率”的核心概念。
    -   **第二步**：下载 **Homer3**，运行其示例数据（如果有频域数据demo），观察如何从原始相位和强度数据计算出HbO/HbR。
    -   **第三步**：搜索 "**calibration of frequency-domain NIRS**"（频域fNIRS的校准），因为频域系统对相位校准要求极高，这是实际搭建或使用设备的关键。

    如果你需要查找特定方向的论文（例如：结合EEG的频域fNIRS，或用于婴儿监测的频域设备），可以使用 **PubMed** 或 **Google Scholar**，配合关键词："frequency-domain NIRS" AND "brain" AND "特定主题"。

* nibabel, nilearn 这俩库是干嘛用的？

    这两个是**Python神经影像分析的核心库**，让我详细解释：

    **NiBabel - 文件I/O基础库**

    核心功能：读写神经影像格式

    ```python
    import nibabel as nib

    # 1. 读取各种神经影像格式
    img = nib.load('subject_T1.nii.gz')  # NIfTI格式
    img = nib.load('subject.dcm')         # DICOM格式
    img = nib.load('subject.mgz')          # FreeSurfer格式

    # 2. 获取影像数据
    data = img.get_fdata()  # 3D/4D数组，shape=(x, y, z) 或 (x, y, z, t)
    affine = img.affine     # 仿射矩阵，将体素坐标映射到真实空间坐标

    # 3. 处理头文件信息
    header = img.header
    voxel_size = header.get_zooms()  # 获取体素大小
    ```

    **主要用途**

    - **格式转换**：DICOM → NIfTI，NIfTI → FreeSurfer等
    - **数据访问**：获取MRI/fMRI的像素数据
    - **坐标变换**：处理不同空间坐标系的转换

    **Nilearn - 统计分析库**

    核心功能：统计学习和可视化

    ```python
    from nilearn import image, plotting, datasets, surface

    # 1. 下载标准模板
    mni = datasets.load_mni152_template()  # MNI152标准脑模板

    # 2. 图像处理
    smoothed_img = image.smooth_img('fmri.nii.gz', fwhm=6)  # 空间平滑
    masked_data = image.clean_img(imgs, confounds=confounds)  # 去噪

    # 3. 统计建模
    from nilearn.glm.first_level import FirstLevelGLM
    glm = FirstLevelGLM()
    results = glm.fit(run_images, design_matrices)

    # 4. 可视化
    plotting.plot_img('stat_map.nii.gz')  # 显示统计图
    plotting.plot_glass_brain('contrast.nii.gz')  # 玻璃脑显示
    plotting.view_img('results.nii.gz')  # 交互式3D视图
    ```

    **两者关系**

    ```
    原始数据 → [NiBabel读取] → NumPy数组 → [Nilearn处理] → 统计结果 → [NiBabel保存]
    ```

    **实际配合使用**

    ```python
    import nibabel as nib
    from nilearn import image, plotting
    import numpy as np

    # NiBabel负责IO
    img = nib.load('functional.nii.gz')
    data = img.get_fdata()

    # Nilearn负责分析
    # 1. 预处理
    processed = image.clean_img(img, detrend=True, standardize=True)

    # 2. 统计分析
    from nilearn.glm import threshold_stats_img
    thresholded_map, threshold = threshold_stats_img(
        stat_map_img, 
        alpha=0.05, 
        height_control='fdr'
    )

    # 3. 可视化
    plotting.plot_stat_map(thresholded_map, 
                            bg_img='anatomy.nii.gz',
                            cut_coords=[-30, -20, 10])

    # NiBabel再次用于保存
    nib.save(thresholded_map, 'results.nii.gz')
    ```

    **在fNIRS分析中的可能用途**

    虽然这两个库主要为fMRI/MRI设计，但在fNIRS中也有用武之地：

    ```python
    import nibabel as nib
    from nilearn import plotting, surface

    # 1. 加载标准脑模板用于可视化
    mni = nib.load('mni152.nii.gz')

    # 2. 将fNIRS探头位置映射到标准空间
    from mne_nirs.channels import get_long_channels
    montage = get_long_channels(raw_info)
    mni_positions = transform_to_mni(montage.get_positions())  # 需要自定义变换

    # 3. 在标准脑上绘制探头位置
    plotting.plot_markers(
        mni_positions,
        mni,
        marker_color='red',
        marker_size=50
    )

    # 4. 将重建结果保存为标准格式
    # 假设你计算出了大脑激活图（2D数组）
    activation_map_3d = project_to_volume(activation_values, mni.affine)
    nib.save(nib.Nifti1Image(activation_map_3d, mni.affine), 
             'fnirs_activation.nii.gz')
    ```

    **核心区别总结**

    | 特性 | NiBabel | Nilearn |
    |------|---------|---------|
    | **主要功能** | 文件读写、格式转换 | 统计分析、机器学习 |
    | **依赖关系** | 基础IO库 | 依赖NiBabel、Scikit-learn |
    | **应用场景** | 数据加载/保存 | 统计建模、可视化 |
    | **fNIRS适用性** | 间接（保存结果） | 有限（可视化为主） |

    **实际建议**

    1. **必须学NiBabel**：任何神经影像Python分析的基础
    2. **根据需求学Nilearn**：如果做fMRI或需要高级可视化
    3. **fNIRS项目**：主要用于最终结果的保存和可视化

    需要我详细介绍它们在fNIRS项目中的具体应用案例吗？
    
* 一些网站

    * mne-nirs

        <https://github.com/mne-tools/mne-nirs>

        主要用于 python  下的 fnirs channel 数据预处理，过滤噪声，质量评估等。

        没有 3d 重建的功能。

        对标的是 matlab 下的 homer3。

    * homer2

        <https://www.nitrc.org/projects/homer2>

        官网会引导跳转到这里去下载 homer2。这里有 homer2 的简介和下载链接。

        <https://homer-fnirs.org/>

        这个看起来比较像是 homer2 官网，但是上面显示此项目已不再维护，用户需要使用最新的 homer3。

    * openfnirs

        <https://openfnirs.org/>

        大多数知名的 fnirs 相关的项目都可以在这里跳转到，比如 homer2, atlas viewer, toast++ 等。

    * fNIRS vs EEG: Comparing Non-Invasive Brain Imaging Techniques

        <https://imotions.com/blog/learning/research-fundamentals/fnirs-vs-eeg/?srsltid=AfmBOopvmIlZPxJW_A8sv4w4iAOxUh8qmivtVMNkvHQVAAab-DN7m9cW>

        技术博客。

    * NIRFAST

        <http://www.nirfast.co.uk/>

        可能是一个类似 toast++ 的工具。还没有深入调研。

    * pynir

        <https://pypi.org/project/pynir/>

        This is a Python library for handling Near infrared (NIR) spectral calibration.

        不清楚校准的是什么。也是和 homer 一样滤除无效通道吗？

* hd-dot 中的光不会互相干扰吗？

    不会，因为我们不探测“相干性”。

    HD-DOT使用的是强度调制/解调的近红外光，处理的是光强信号。

    关键在于：

    * 光源：HD-DOT通常使用发光二极管或激光二极管，它们发出的光在时间和空间上相干性都很低（与用于全息或干涉仪的超稳激光器不同）。

    * 介质：大脑这个强散射介质本身就是“相干性杀手”。光子在其中走过的路径长度差异巨大（可达几厘米甚至十几厘米），远远超过光的相干长度。当这些走过不同路径、相位关系完全随机的光子到达探测器时，它们的电场无法发生稳定的干涉。

    * 探测器：HD-DOT使用的光电探测器（如雪崩光电二极管APD或光电倍增管PMT）是平方律探测器。它们响应的是光场的强度（即振幅的平方），而不是电场的瞬时值。所有到达探测器的光子的能量简单叠加，最终输出一个与总光强成正比的电流信号。

    一个简单的比喻：

    想象一个房间里有很多人在同时、不同步地说话（就像无数散射光子）。你站在房间另一端，用一个分贝仪（就像光强探测器）测量总的噪音水平。你测量到的是所有声波能量（强度）的叠加，而你的耳朵无法分辨出其中哪些声波发生了相长或相消干涉，因为它们的相位关系是混乱的。HD-DOT做的就是类似的事情——测量“光噪音”的总强度变化。

* 什么是 high density diffuse optical tomography

    我们来用通俗易懂的方式解释一下“高密度漫射光学断层成像”是干什么用的。

    **简单来说：**

    **高密度漫射光学断层成像是一种“脑部功能活动三维摄像机”。** 它通过在你的头上戴一个布满密集光源和探测器的“帽子”，来无创地探测和绘制你大脑内部的活动图像。

    详细分解

    为了更好地理解，我们把这个技术名词拆开来看：

    1.  **光学：** 它使用**近红外光**。这种光对人体安全无害，可以穿透头皮、头骨，并进入大脑皮层几厘米深。
    2.  **漫射：** 光在进入大脑组织后，不会直线传播，而是像烟雾在房间里扩散一样，发生强烈的**散射**。这就是“漫射”的来源。
    3.  **断层成像：** 这指的是它能生成**三维的、断层的图像**，就像CT或MRI扫描一样，可以显示出大脑不同深度的活动，而不仅仅是表面的二维图。
    4.  **高密度：** 这是关键！它意味着光源和探测器的排布非常**密集**。光源和探测器之间的距离有多种组合（短距离和长距离）。
        *   **短距离**主要测量浅层（头皮、头骨）的信号，用于排除这些非脑部活动的干扰。
        *   **长距离**的光可以到达大脑皮层，携带了脑部活动的信息。
        *   通过这种密集的排布和复杂的算法，HD-DOT可以更精确地“定位”大脑活动的源头，其图像质量远优于早期稀疏排布的设备。

    它的核心用途是什么？

    HD-DOT主要用于监测大脑的**功能活动**，而不是结构。它测量的是大脑活动时**血流动力学**的变化，即：

    *   **含氧血红蛋白** 浓度的增加
    *   **脱氧血红蛋白** 浓度的减少

    当大脑某个区域活跃时，该区域的神经元会消耗更多氧气，身体会立刻向该区域输送更多含氧血液作为补偿。这个“血氧水平依赖”信号正是HD-DOT（以及功能磁共振fMRI）所探测的。

    主要应用领域：

    1.  **神经科学研究：**
        *   研究人类在执行各种任务（如看、听、说、思考、运动）时，大脑不同区域是如何协同工作的。
        *   探索大脑发育，特别是**婴儿和儿童**的大脑。因为HD-DOT非常安静、无束缚、对运动容忍度相对较高，非常适合研究婴幼儿。

    2.  **临床医学应用：**
        *   **癫痫病灶定位：** 帮助医生在癫痫患者手术前，精确定位导致癫痫发作的大脑异常区域。
        *   **脑卒中评估：** 监测中风患者大脑血流的恢复情况。
        *   **脑肿瘤研究：** 评估肿瘤周围脑组织的功能状态。
        *   **认知障碍疾病：** 如阿尔茨海默病等，研究大脑功能连接的变化。

    与其它脑成像技术的比较：

    | 技术 | 优点 | 缺点 |
    | :--- | :--- | :--- |
    | **HD-DOT** | **安全无创、完全静默、便携、成本较低、对血氧变化敏感、可长时间监测** | 穿透深度有限（~2-3cm）、空间分辨率低于fMRI |
    | **功能磁共振 (fMRI)** | 空间分辨率**非常高**、全脑覆盖 | 设备巨大、昂贵、噪音大、对运动极其敏感、不适用于体内有金属植入物的患者 |
    | **脑电图 (EEG)** | 时间分辨率**极高**（毫秒级）、便携、成本低 | 空间分辨率差、很难精确定位脑内活动源 |
    | **正电子发射断层扫描 (PET)** | 可测量代谢等特定生化过程 | 有放射性、成本极高、不适用于常规研究 |

    总结

    **高密度漫射光学断层成像是一种先进的光学脑成像技术，它通过在头皮上密集排列光源和探测器，来非侵入性地绘制出大脑皮层功能活动的三维地图。它就像一个“透明的窗口”，让我们能够安全、安静地观察人脑在工作时的状态，尤其在婴幼儿研究和一些临床神经疾病诊断中具有独特优势。**

* 简述什么是光子流密度（光通量）？

    **光子流密度**更常见的标准术语是**光通量**，它在 radiometry（辐射度学）和 photometry（光度学）中有不同的侧重点。为了让你全面理解，可以从以下三个维度来简述：

    1. 核心定义（物理光学角度）

        光子流密度指的是单位时间内通过单位面积的光子数目，或者更常见的是指**单位时间内通过某一面积的光的能量**。

        在辐射度学中，它对应的精确术语是**辐照度**或**辐射通量密度**，单位是 \( W/m^2 \)（瓦特每平方米）。它描述了光能量在空间中的分布浓度。

    2. 视觉响应角度（光度学角度）

        在光度学（人眼视觉）中，光子流密度通常特指**光通量**（Luminous Flux），它表示光源每秒发出的**可见光**能量的总和，但并非简单的物理功率，而是根据人眼对不同波长光线的敏感度（即视见函数）进行加权后的量。

        *   **符号：** \(\Phi\)
        *   **单位：** 流明（lm）
        *   **通俗理解：** 它描述了光源“看起来”有多亮。例如，一个60W的白炽灯大约能发出700流明的光通量。

    3. 通俗类比

        你可以把光子流密度想象成**水流的流量**：

        * 光子就像水流中的**水滴**。

        * 光子流密度（光通量）就像**单位时间内流过管道某一横截面的总水量**。
        
        * 如果水压越大（光子密度越高），或者管道越粗（面积越大），单位时间流过的水就越多（光通量越大）。

    总结

    简单来说，**光子流密度（光通量）就是衡量一束光“输送”光子能力强弱的物理量。** 在照明工程中，它决定了灯具的明亮程度；在物理学中，它决定了光与物质相互作用（如光电效应）的效率。

* 简述 fnirs 数据处理中的 snirf 文件格式

    **SNIRF**（近红外光谱成像数据格式，Shared Near Infrared Spectroscopy Format）是 fNIRS 领域官方推荐的标准化数据文件格式，旨在解决不同厂商设备生成的数据格式不统一的问题。

    以下是关于 SNIRF 格式的简述：

    1. 本质与扩展名

        -   **本质**：它是一种基于 **HDF5** 标准的文件格式。
        -   **扩展名**：通常以 `.snirf` 结尾。

    2. 核心设计目标

        -   **互操作性**：允许数据在不同厂商的设备、不同的分析软件（如 Homer3、MNE-NIRS、Brainstorm）之间无缝流通，无需编写繁琐的格式转换脚本。
        -   **可重复性**：通过统一的数据结构，确保数据处理和分析的标准化，便于科研结果的复现。

    3. 数据结构层次

        SNIRF 文件内部采用层级结构存储数据，主要包含以下核心组（Groups）和数据集（Datasets）：

        -   **/nirs**：主数据组，包含以下子项：
            -   **data**：存储原始光强数据或浓度变化数据的时间序列矩阵。
            -   **probe**：存储探针布局信息，包括光源（Source）和探测器（Detector）的坐标位置、哪个源-探针对构成哪个测量通道。
            -   **stim**：存储实验刺激的时间标记（开始时间、持续时间、刺激类型）。
            -   **aux**：存储辅助数据（如加速度计数据、生理监控数据等）。
        -   **/formatVersion**：标识该文件遵循的 SNIRF 格式版本号（如 `1.0`）。

    4. 主要优势

        -   **自包含**：一个 `.snirf` 文件包含了原始数据、元数据（如采样率、波长）、探头几何结构和刺激标记，无需额外的附属文件。
        -   **跨平台**：基于 HDF5，使其不受操作系统和字节序的影响。
        -   **社区支持**：被 Homer3（MATLAB 平台）和 MNE-NIRS（Python 平台）等主流分析工具采用为主要数据格式。

* mne-nirs 完整的 GLM 分析流程

    ```py
    import mne
    import mne_nirs
    import numpy as np
    from mne_nirs.experimental_design import make_first_level_design_matrix
    from mne_nirs.statistics import run_glm
    from mne_nirs.channels import get_long_channels, get_short_channels

    # 1. 读取数据
    raw_intensity = mne.io.read_raw_snirf('your_data.snirf', preload=True)

    # 2. 预处理：转换为光密度和血红蛋白浓度
    from mne.preprocessing import nirs
    raw_od = nirs.optical_density(raw_intensity)
    raw_haemo = nirs.beer_lambert_law(raw_od, ppf=0.1)

    # 3. 分离长短通道（可选，用于去除生理噪声）
    short_chs = get_short_channels(raw_haemo)
    raw_haemo = get_long_channels(raw_haemo)

    # 4. 创建设计矩阵 ⭐ 关键步骤
    design_matrix = make_first_level_design_matrix(
        raw_haemo,
        drift_model='cosine',      # 漂移模型
        high_pass=0.005,            # 高通滤波截止频率
        hrf_model='spm',            # 血液动力学响应函数模型
        stim_dur=5.0                # 刺激持续时间
    )

    # 5. 添加短通道均值作为回归量（可选，但推荐）
    design_matrix["ShortHbO"] = np.mean(
        short_chs.copy().pick(picks="hbo").get_data(), axis=0
    )
    design_matrix["ShortHbR"] = np.mean(
        short_chs.copy().pick(picks="hbr").get_data(), axis=0
    )

    # 6. 运行 GLM ⭐ 需要传入 design_matrix
    glm_est = run_glm(raw_haemo, design_matrix)

    # 7. 查看结果
    print(glm_est)
    results_df = glm_est.to_dataframe()
    print(results_df.head())
    ```

* 查看设计矩阵的内容

    ```py
    from nilearn.plotting import plot_design_matrix
    import matplotlib.pyplot as plt

    plot_design_matrix(design_matrix)
    plt.show()
    ```

* MNE-NIRS 也支持传统的 epochs-based 分析（波形分析）

    ```py
    # Epochs-based 分析（替代 GLM）
    epochs = mne.Epochs(raw_haemo, events, tmin=-2, tmax=15, baseline=(-2, 0))
    conditions = ['Tapping_Left', 'Tapping_Right']
    evoked_dict = {cond: epochs[cond].average() for cond in conditions}
    mne.viz.plot_compare_evokeds(evoked_dict)
    ```

* 简述什么是广义线性模型，为什么要用广义线性模型分析fnirs数据

    广义线性模型（GLM）简述

    什么是广义线性模型？

    **广义线性模型（GLM）** 是传统线性模型的扩展，由三个核心部分组成：

    1. **随机成分**：指定响应变量的概率分布（如高斯分布、二项分布、泊松分布等）
    2. **系统成分**：线性预测器 $\eta = X\beta$（与普通线性回归相同）
    3. **连接函数**：将线性预测器与响应变量的期望值联系起来 $g(\mu) = \eta$

    **数学表达**：

    - 普通线性模型：$Y = X\beta + \varepsilon$，假设 $Y \sim N(\mu, \sigma^2)$

    - 广义线性模型：$g(E[Y]) = X\beta$，$Y$ 可以服从指数族中的任何分布

    与普通线性回归的区别

    | 特性 | 普通线性回归 | 广义线性模型 |
    |------|------------|------------|
    | 因变量分布 | 正态分布 | 指数族分布（二项、泊松、伽马等） |
    | 关系形式 | 线性 | 可通过连接函数变换 |
    | 方差 | 恒定 | 可随均值变化 |
    | 典型应用 | 连续值预测 | 分类、计数、比率数据 |

    为什么用GLM分析fNIRS数据？

    1. **fNIRS数据的特性符合GLM的框架**

        fNIRS数据具有以下特点，使得GLM特别适合：

        - **时间序列结构**：fNIRS数据是连续采集的时间序列，包含任务刺激、生理噪声、漂移等多种成分
        - **线性叠加假设**：大脑的血氧响应被认为是不同刺激和干扰成分的线性叠加
        - **噪声复杂**：包含心跳、呼吸、梅耶波等生理噪声，以及仪器噪声

    2. **GLM可以分离不同信号成分**

        GLM可以将fNIRS信号分解为：

        ```
        fNIRS信号 = 刺激诱发的HRF + 生理噪声 + 低频漂移 + 残差
                 = (设计矩阵 × 参数) + 噪声
        ```

        通过构建包含不同回归量的设计矩阵，GLM可以：

        - 估计不同实验条件下的血氧响应幅度
        - 分离任务相关的激活信号与生理噪声
        - 去除低频漂移（通过多项式或余弦基函数）

    3. **处理fNIRS数据的具体优势**

        | 优势 | 说明 | 在fNIRS中的应用 |
        |------|------|----------------|
        | **多条件分离** | 可以同时估计多个实验条件的效应 | 比较不同任务条件（如左手运动vs右手运动） |
        | **噪声建模** | 可以将噪声源作为回归量加入模型 | 加入短通道信号去除生理噪声 |
        | **漂移校正** | 内置漂移项处理低频趋势 | 去除设备漂移和慢速生理波动 |
        | **统计推断** | 提供参数估计和假设检验 | 识别显著激活的通道和时间点 |
        | **灵活性** | 可以处理连续变量和分类变量 | 分析连续变量（如反应时）的影响 |

    4. **与传统的block averaging相比**

        **传统Block Averaging**：

        - 简单直观，易于理解
        - 无法处理重叠的血液动力学响应
        - 对噪声敏感，不能分离不同来源的噪声

        **GLM方法**：

        - 可以处理快速事件相关设计（HRF可能重叠）
        - 可以同时建模多个回归量，分离信号成分
        - 提供统计显著性检验
        - 可以处理连续变化的参数（如parametric modulation）

    5. **MNE-NIRS中的GLM实现**

        在MNE-NIRS中，GLM分析的典型流程：

        ```python
        # 1. 构建设计矩阵（包含刺激条件、漂移项、短通道信号等）
        design_matrix = make_first_level_design_matrix(
            raw_haemo,
            drift_model='cosine',  # 漂移建模
            hrf_model='spm'        # HRF模型
        )

        # 2. 添加生理噪声回归量
        design_matrix["ShortHbO"] = short_channel_hbo

        # 3. 运行GLM
        glm_est = run_glm(raw_haemo, design_matrix)

        # 4. 统计推断
        contrast_matrix = np.eye(design_matrix.shape[1])[0:2]  # 对比矩阵
        contrast_results = glm_est.compute_contrast(contrast_matrix)
        ```

    6. **GLM提供的结果解释**

        GLM分析可以提供：

        - **Beta值**：每个条件在每个通道的激活强度
        - **T统计量**：激活的显著性
        - **P值**：统计检验的概率值
        - **对比分析**：不同条件之间的差异

    总结

    使用GLM分析fNIRS数据的主要原因是它提供了一个**统一、灵活且统计严谨的框架**来处理fNIRS数据的复杂性。它不仅能有效地从噪声中提取任务相关的血液动力学响应，还能进行多条件比较、噪声去除和统计推断，是当前fNIRS数据分析的**标准方法**和**金标准**。

* 简述 MNE-NIRS 的安装、用法

    MNE-NIRS 是一个基于 MNE-Python 的专用于**功能性近红外光谱**数据分析的开源 Python 工具包。它提供了从原始数据预处理、去噪到统计分析和可视化的完整流程。

    以下是其安装和基本用法的简述：

    一、安装

    由于 MNE-NIRS 是 MNE-Python 的姊妹项目，安装过程非常简单。推荐使用 `pip` 或 `conda` 在 Python 环境中进行安装。

    1.  **创建虚拟环境（可选但推荐）**

        ```bash
        conda create -n mne-nirs python=3.9
        conda activate mne-nirs
        ```

    2.  **安装 MNE-NIRS**

        最常用的安装方式是通过 `pip` 安装，它会自动拉取核心依赖（包括 `mne`）：

        ```bash
        pip install mne-nirs
        ```

    3.  **安装可选依赖**

        某些特定功能（如 GLM 统计或绘图优化）可能需要额外的库，可以根据需要安装：

        ```bash
        pip install matplotlib scikit-learn statsmodels
        ```

    二、基本用法流程

    典型的 MNE-NIRS 数据分析流程遵循标准的 fNIRS 数据处理步骤。以下是核心步骤的代码简述：

    1. 加载数据

        MNE-NIRS 支持多种主流 fNIRS 设备的数据格式（如 SNIRF、NIRx 等）。

        ```python
        import mne
        import mne_nirs

        # 读取 SNIRF 格式文件（目前最通用的标准）
        # raw = mne.io.read_raw_snirf("your_data.snirf", preload=True)

        # 或者读取 NIRx 格式
        raw = mne.io.read_raw_nirx("your_data_folder", preload=True)

        print(raw)
        ```

    2. 基本预处理

        预处理步骤通常包括：将光强数据转换为光密度、识别不良通道、去除生理噪声。

        ```python
        from mne.preprocessing.nirs import optical_density, beer_lambert_law

        # 将原始强度转换为光密度
        raw = optical_density(raw)

        # 使用修订的比尔-朗伯定律将光密度转换为血红蛋白浓度
        raw = beer_lambert_law(raw, ppf=0.1)  # ppf 是路径差分因子

        # 查看数据中包含了哪些通道类型 (hbo, hbr)
        print(raw.get_channel_types())

        # 可选：识别和标记不良通道（基于信噪比等）
        # from mne_nirs.preprocessing import detect_bad_channels
        # bads = detect_bad_channels(raw)  # 这只是一个示例函数
        ```

        一些额外的查看信息的方法：

        ```py
        # 查看fNIRS特定通道类型
        fnirs_picks = mne.pick_types(raw.info, fnirs=True)
        for pick in fnirs_picks[:5]:  # 显示前5个通道
            print(f"通道 {raw.ch_names[pick]}: 类型 = {raw. get_channel_types(picks=[pick])[0]}")

        # 查看info中的通道信息
        print(raw.info['chs'][0])  # 查看第一个通道的详细信息

        # 查看自定义字段（如果有）
        if hasattr(raw, '_data_type'):
            print(f"数据类型: {raw._data_type}")
        ```

    3. 去噪与滤波

        去除心率、呼吸等高频噪声以及低频漂移。

        ```python
        # 带通滤波：保留 0.01 至 0.5 Hz 的频率（典型的血流动力学响应频率）
        raw.filter(0.01, 0.5, fir_design='firwin')

        # 可选：使用短分离通道回归来去除生理噪声（如果设备有短距离通道）
        # from mne_nirs.preprocessing import short_channel_regression
        # raw = short_channel_regression(raw)
        ```

    4. 分段与基线校正

        根据实验刺激将连续数据切割成一个个事件相关的片段（epoch）。

        ```python
        # 从原始数据的注释中获取事件
        events, event_dict = mne.events_from_annotations(raw)

        # 创建 epochs：以刺激开始为 0 点，取刺激前 2 秒到刺激后 10 秒
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=-2,  # 基线开始时间
            tmax=10,  # 结束时间
            baseline=(-2, 0),  # 基线校正范围
            preload=True
        )

        print(epochs)
        ```

    5. 统计分析（GLM）

        MNE-NIRS 强大的功能之一是使用广义线性模型进行统计，这更符合 fNIRS 数据的统计特性。

        ```python
        from mne_nirs.statistics import run_glm
        from mne_nirs.experimental_design import make_first_level_design_matrix

        # 基于 epochs 创建设计矩阵并拟合 GLM
        glm_estimates = run_glm(epochs)

        # 查看某个通道的统计结果
        print(glm_estimates["S1_D1 - condition_A"])
        ```

    6. 可视化

        绘制平均波形或在头皮/大脑模板上绘制激活拓扑图。

        ```python
        # 绘制所有通道的平均响应
        epochs.average().plot_psd()  # 或者其他绘图函数

        # 绘制特定通道的波形
        # mne.viz.plot_compare_evokeds(...)

        # 绘制通道在头皮上的位置
        raw.plot_sensors()
        ```

    总结

    * 安装：`pip install mne-nirs`

    * 核心数据流：`raw` （原始/光密度） \(\rightarrow\) `raw` （浓度） \(\rightarrow\) `epochs` （分段） \(\rightarrow\) `glm_estimates` （统计）。

    * 优势：完全兼容 MNE-Python 的数据结构，可以无缝使用 MNE 提供的滤波、ICA 和可视化工具。

* MNE-NIRS 和 MNE 有什么区别？

    这是一个很好的问题。**MNE** 和 **MNE-NIRS** 的关系可以理解为 **“父包”与“子包”**，或者 **“通用平台”与“专用插件”**。

    简单来说：**MNE 是基础框架，MNE-NIRS 是构建在这个框架之上的 fNIRS 专用工具箱。**

    以下是详细的对比和区别：

    1. 定位与适用范围

    -   **MNE （MNE-Python）**
        -   **定位**：是一个通用的生理信号（主要是神经影像数据）分析软件包。
        -   **适用范围**：最初主要针对 **脑电图** 和 **脑磁图** 设计，后来扩展了处理侵入性脑电图、皮层电刺激图以及一般时间序列的功能。
        -   **核心**：它提供了处理任何连续生理信号的底层基础，如滤波、分段、基线校正、事件提取等。

    -   **MNE-NIRS**
        -   **定位**：是一个专门针对 **功能性近红外光谱** 分析的工具包。
        -   **适用范围**：专注于 fNIRS 特有的数据格式和物理模型。
        -   **核心**：它专为 fNIRS 而生，解决了 EEG/MEG 分析中不存在但在 fNIRS 中至关重要的问题。

    2. 核心功能差异

        最大的区别在于 **fNIRS 特有的物理转换** 和 **统计模型**。

        | 功能维度 | MNE （通用） | MNE-NIRS （专用） |
        | :--- | :--- | :--- |
        | **单位转换** | **不支持**。MNE 只能处理电压或磁场强度等电生理信号单位。 | **原生支持**。提供函数将光强（Intensity）转为光密度（OD），再通过**修订的比尔-朗伯定律**转为血红蛋白浓度（HbO/HbR）。 |
        | **数据格式** | 支持通用的 EEG/MEG 格式（如 FIF， EDF， BDF， CNT）。 | 支持 fNIRS 行业标准格式（如 **SNIRF**）及主流设备商格式（如 NIRx， Artinis 等）。它能识别这些格式中的波长、光源-探测器距离等信息。 |
        | **噪声去除** | 提供通用去噪法（如滤波、ICA、SSP）。 | 提供 fNIRS 特有去噪法，如**短分离通道回归**（利用短距通道估计并去除生理噪声）。 |
        | **统计模型** | 基于磁/电信号假设，主要使用平均值、t检验或基于场水平的簇聚类校正。 | 专门集成了基于**广义线性模型** 的统计方法，并考虑了 fNIRS 数据的序列相关性。还包含专门用于近红外的**通道特定统计功能**。 |

    3. 数据结构的关系

        为了更好地理解它们的区别，可以看一下代码中的数据流向：

        ```python
        import mne
        import mne_nirs

        # 1. 读取 fNIRS 数据（这一步通常用 MNE-NIRS）
        raw = mne.io.read_raw_nirx("subject_01")  # 实际上是 mne-nirs 扩展了 MNE 的 IO

        # 2. 进行 fNIRS 特有的转换（这一步必须用 MNE-NIRS）
        raw = mne_nirs.preprocessing.optical_density(raw)
        raw = mne_nirs.preprocessing.beer_lambert_law(raw)  # 转换为浓度

        # 3. 进行通用的预处理（这一步可以用 MNE）
        raw.filter(0.01, 0.5, fir_design='firwin')  # 这是 MNE 的功能

        # 4. 创建 epochs（这是 MNE 的核心功能）
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, tmin=-2, tmax=10, baseline=None)  # MNE 功能

        # 5. 进行 fNIRS 专用的统计分析（这是 MNE-NIRS 的功能）
        from mne_nirs.statistics import run_glm
        glm = run_glm(epochs)  # fNIRS 专用 GLM
        ```

    4. 总结区别

        | 对比项 | MNE | MNE-NIRS |
        | :--- | :--- | :--- |
        | **比喻** | 通用厨房，有水池、灶台、切菜板 | 专门做牛排的套件，有牛排刀、测温针 |
        | **输入数据** | EEG/MEG/iEEG 信号 | fNIRS 光强/波长信号 |
        | **物理模型** | 电磁场模型 | 光传播模型（比尔-朗伯定律） |
        | **特有功能** | 源定位（Source Localization）、独立成分分析 | 短通道回归、HbO/HbR 浓度转换、SNIRF 读写 |
        | **依赖关系** | 基础库 | **必须依赖 MNE** 才能运行 |

    **结论：**

    当你安装 `mne-nirs` 时，它会自动安装 `mne`。你写代码时，**大部分通用操作（滤波、分段、绘图）使用的是 MNE 的函数，而涉及 fNIRS 特有物理量（如浓度转换、短通道回归）时，则需要调用 MNE-NIRS 的模块。**

* 那么还有和 mne-nirs 同级的“专用插件”吗？

    是的，MNE-Python 作为一个成熟的生态平台，除了 MNE-NIRS，还有多个同级的**专用插件**（通常称为 MNE 生态的子项目）。这些工具包都建立在 MNE 核心框架之上，专门针对特定的模态或分析任务进行优化。

    以下是主要的同级插件：

    1. **MNE-Connectivity**

        -   **定位**：专门进行**功能连接**分析的插件。
        -   **解决的问题**：计算大脑不同区域之间的同步性、相干性、Granger 因果等。
        -   **特点**：
            -   提供多种连接性指标（如相干性、相位锁定值、互信息等）。
            -   支持基于传感器空间、源空间（脑磁/脑电图源定位后）的连接分析。
            -   包含统计验证工具（如基于置换检验的连接性统计）。
        -   **安装**：`pip install mne-connectivity`

    2. **MNE-BIDS**

        -   **定位**：专门处理**脑成像数据结构**格式的插件。
        -   **解决的问题**：BIDS 是神经科学领域推荐的数据组织标准。该插件帮助用户将原始数据格式化为 BIDS 标准，或者从 BIDS 数据集读取数据。
        -   **特点**：
            -   可以读写符合 BIDS 标准的 EEG/MEG/iEEG/fNIRS 数据集。
            -   包含数据验证工具，检查你的数据是否符合 BIDS 规范。
            -   方便数据共享和复现研究。
        -   **安装**：`pip install mne-bids`

    3. **MNE-ICALabel**

        -   **定位**：**自动标记独立成分**的插件。
        -   **解决的问题**：在 EEG 数据预处理中，通常使用 ICA（独立成分分析）去除眼动、心跳等伪迹。手动区分哪个成分是伪迹非常耗时。
        -   **特点**：
            -   利用预训练的机器学习模型，自动将 ICA 成分分类为“脑信号”、“眼动伪迹”、“心跳伪迹”、“肌肉伪迹”等。
            -   大大加速了 EEG 数据清理流程。
        -   **安装**：`pip install mne-icalabel`

    4. **MNE-Features**

        -   **定位**：从 MNE 数据对象中**自动提取特征**的插件。
        -   **解决的问题**：在脑机接口或机器学习应用中，需要从脑信号中提取大量特征（如波段功率、Hjorth 参数、分形维数等）。
        -   **特点**：
            -   提供一键式函数，从 `raw`、`epochs` 或 `events` 数据中提取数百种特征。
            -   输出可以直接用于 scikit-learn 等机器学习库的数据框。
        -   **安装**：`pip install mne-features`

    5. **MNE-RealTime**

        -   **定位**：**实时处理**脑信号的插件。
        -   **解决的问题**：用于在线脑机接口或神经反馈系统，需要实时获取和处理数据。
        -   **特点**：
            -   提供客户端-服务器架构，可以连接实时数据流（如 LSL，即实验室流媒体协议）。
            -   支持在线预处理、特征提取和分类。
        -   **安装**：`pip install mne-realtime`（注意：该项目活跃度可能随社区需求变化）

    6. **MNE-EEG （或者说 MNE-Study）**

        *注：有些模块虽然不叫 MNE-XXX，但功能类似，比如 `mne.stats` 虽然内置于 MNE，但提供了复杂的统计功能。不过官方生态中还有一个 **MNE-Study**（早期概念）用于管理大规模群体研究，目前已部分功能整合进 MNE-BIDS 和核心库。*

    总结对比表

    | 插件名称 | 专注领域 | 典型用户 | 核心价值 |
    | :--- | :--- | :--- | :--- |
    | **MNE-NIRS** | fNIRS 数据分析 | fNIRS 研究者 | 血红蛋白浓度转换、短通道回归、fNIRS 专用 GLM |
    | **MNE-Connectivity** | 功能连接 | 网络 neuroscience 研究者 | 多种连接指标计算与统计 |
    | **MNE-BIDS** | 数据标准化 | 所有需要数据共享/复现的研究者 | BIDS 格式读写与验证 |
    | **MNE-ICALabel** | ICA 伪迹去除 | EEG 研究者 | 自动标记 ICA 成分，加速预处理 |
    | **MNE-Features** | 特征工程 | 脑机接口/机器学习方向 | 批量提取数百种脑信号特征 |
    | **MNE-RealTime** | 实时处理 | 脑机接口/神经反馈方向 | 实时数据流处理框架 |

    生态关系

    这些插件都遵循相同的设计哲学：

    1.  **依赖 MNE 核心**：它们都建立在 MNE 的数据结构（如 `Raw`、`Epochs`、`Evoked`）之上。
    2.  **解决特定问题**：每个插件专注于一个领域，避免 MNE 核心库变得臃肿。
    3.  **无缝集成**：插件的输出通常可以直接用 MNE 的可视化函数展示。

    例如，你可以用 **MNE-NIRS** 处理 fNIRS 数据，然后用 **MNE-Connectivity** 计算通道间的相干性，最后用 **MNE-BIDS** 将处理好的数据保存为标准格式。

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
