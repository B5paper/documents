## cache

* 如何安装 wsl2

    在 2026 年，安装 WSL2（Windows Subsystem for Linux）已经变得极其简单。微软大幅简化了流程，现在只需要一个命令即可搞定。

    **必须以管理员身份打开 Windows PowerShell 或 CMD** 来执行安装命令：

    ---

    **第一步：一键安装 (最简单)**

    1.  右键点击 Windows 开始按钮，选择 **终端（管理员）** 或 **PowerShell（管理员）**。

    2.  输入以下命令：
        ```powershell
        wsl --install
        ```
        * **这个命令会干什么？** 它会自动启用所需的 Windows 可选组件（虚拟机平台和 Linux 子系统），下载最新的 Linux 内核，并默认下载安装 **Ubuntu** 发行版。

    3.  **重启电脑**（这是必须的，以完成组件启用）。

    注：

    1. 国内下载这个镜像很慢，需要提前开 system proxy 代理

    ---

    **第二步：设置 Linux 用户名和密码**

    1.  重启后，会自动弹出一个 Ubuntu 窗口完成最后的安装。
    2.  按照提示输入一个 **用户名** 和 **密码**（密码输入时屏幕不会显示字符，这是正常的）。
    3.  完成后，你就拥有了一个真正的 Linux 环境。

    ---

    **第三步：验证版本是否为 WSL2**

    在 PowerShell 中输入：
    ```powershell
    wsl -l -v
    ```
    确保 `VERSION` 列显示的是 `2`。如果是 `1`，可以通过以下命令转换：
    ```powershell
    wsl --set-version Ubuntu 2
    ```
    ---

    **进阶：如果你想要特定的 Linux 发行版**

    如果你不想用 Ubuntu，可以先查看有哪些可选：
    ```powershell
    wsl --list --online
    ```
    然后安装指定的版本，例如 Debian：
    ```powershell
    wsl --install -d Debian
    ```

    **安装过程中如果遇到 `0x800701bc` 之类的错误码，通常是因为内核补丁没更新，或者 BIOS 里的“虚拟化技术 (Virtualization Technology)”没打开。你需要进入 BIOS 开启它吗？**

* wsl 如何使用 cuda

    在 WSL2 中使用 CUDA 相比以前简单得多，因为现在的 WSL2 支持 **GPU 降临（GPU Paravirtualization）**。

    最重要的原则是：**千万不要在 WSL2 内部安装 NVIDIA 显卡驱动！**

    以下是标准安装步骤：

    ---

    1. 宿主机（Windows）准备
        
        * 确保你在 Windows 上已经安装了最新的 **NVIDIA Game Ready** 或 **Studio 驱动**。
        * 只要 Windows 能识别显卡，WSL2 就会通过内核驱动接口直接调用它。

    2. 在 WSL2 内部验证硬件
        
        打开你的 Ubuntu (WSL2) 终端，输入：
        ```bash
        nvidia-smi
        ```
        如果你能看到显卡信息表格（和 Windows 上看到的一样），说明硬件通道已经打通。

    ---

    3. 安装 CUDA Toolkit (针对 WSL2)
        
        虽然驱动是用 Windows 的，但开发环境（编译器 `nvcc` 等）需要在 WSL2 里安装。为了避免污染系统路径，建议直接通过 **Conda** 安装，这是最省心的方法。

        在 WSL2 的终端里：

        1.  **安装 Linux 版 Miniconda**（如果还没装）：
            ```bash
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
            bash Miniconda3-latest-Linux-x86_64.sh
            ```
        2.  **创建一个包含 CUDA 的环境**：
            ```bash
            # 创建环境并安装 pytorch 和对应的 cuda toolkit
            conda create -n gpu_env python=3.12 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
            ```
    ---

    **避坑指南**

    1.  **不要在 WSL 里手动下载 `.run` 格式的驱动包安装**：这会破坏 WSL 的驱动映射，导致 `nvidia-smi` 失效。
    2.  **显存共享**：WSL2 的显存是动态分配的，但在高负载下，建议关闭 Windows 上不必要的占用显存的程序。
    3.  **版本匹配**：确保你的 `pytorch-cuda` 版本（如 12.1）不要超过 Windows 驱动支持的最高版本（通过 `nvidia-smi` 右上角查看 `CUDA Version`）。

    你在 WSL2 中运行 `nvidia-smi` 有正常输出吗？如果输出了错误信息，可能需要在 Windows 功能里检查“虚拟机平台”是否完全开启。

    注：

    1. cuda 的版本不要使用 cuda 13，变化较大

    1. 不建议直接用 sudo apt install nvidia-cuda-toolkit。版本太落后

        驱动冲突：apt 可能会尝试顺便帮你装一个 Linux 版的 NVIDIA 驱动（内核模块），而在 WSL 中，驱动必须由 Windows 宿主机提供。如果 apt 覆盖了驱动链接，你的 nvidia-smi 可能会直接挂掉。

    1. cuda, cutensor, nccl 这三个是互相独立的 apt 包

* 卸载 cuda 13

    明智的选择。CUDA 13.x 确实太前沿了，很多深度学习库（如现有的 PyTorch 稳定版或某些算子库）可能还没来得及做完美的适配。退回 **CUDA 12.x**（特别是 12.1 或 12.4）是目前最稳妥的生产环境方案。

    在卸载之前，我先为你拆解那两个关键操作的含义，然后再告诉你如何干净地“撤销”它们。

    ---

    1. 这两个操作是什么含义？

        **操作 A：`cuda-repository-pin-600` (优先级控制)**

        * **含义**：这是一个 APT 策略文件。它的作用是告诉 Ubuntu 的包管理器：**“当多个仓库都有同一个包时，优先从 NVIDIA 官方仓库下载。”**
        * **如果不做会怎样**：Ubuntu 自带的源里可能也有老旧的 CUDA 相关组件。没有这个文件，`apt` 可能会因为版本冲突或者来源优先级问题拒绝安装官方包，或者安装了错误的系统版本。

        **操作 B：`cuda-*-keyring.gpg` (数字签名/信任证明)**

        * **含义**：这是 NVIDIA 仓库的“数字身份证”。它把公钥拷贝到系统信任的钥匙链里。
        * **如果不做会怎样**：当你执行 `apt update` 时，系统会报错说“由于没有签名，该仓库无法验证”，并出于安全考虑拒绝从该源下载任何东西。

    ---

    2. 卸载与逆操作步骤

        要彻底切换版本，请按顺序执行以下操作：

        **Step 1: 卸载 CUDA 13 软件本身**

        先清理掉已经安装的工具包和不再需要的依赖：
        ```bash
        sudo apt-get purge -y cuda-toolkit-13-2 cuda-toolkit-13-0
        sudo apt-get autoremove -y
        sudo apt-get autoclean
        ```

        **Step 2: 移除本地仓库记录 (重要)**

        由于你之前是用 `dpkg -i` 安装的 `.deb` 本地包，它在系统中注册了一个仓库。
        ```bash
        # 查找本地仓库包的名称（通常以 cuda-repo 开头）
        dpkg -l | grep cuda-repo

        # 卸载这个本地仓库包（根据你的版本号，应该是下面这个）
        sudo dpkg -P cuda-repo-wsl-ubuntu-13-2-local
        ```

        **Step 3: 逆操作——删除配置文件和密钥**

        现在执行你关心的“逆操作”：
        ```bash
        # 1. 移除优先级配置文件
        sudo rm /etc/apt/preferences.d/cuda-repository-pin-600

        # 2. 移除密钥（可以使用通配符，或者指定具体名称）
        sudo rm /usr/share/keyrings/cuda-*-keyring.gpg

        # 3. 移除残留的本地元数据缓存目录（如果有）
        sudo rm -rf /var/cuda-repo-wsl-ubuntu-13-2-local
        ```

        **Step 4: 刷新 APT 状态**

        ```bash
        sudo apt-get update
        ```

    ---

    3. 如何正确安装 CUDA 12？

        为了避免再次陷入环境变量和依赖的泥潭，我强烈建议你这次换个思路：**不要全局安装，改用 Conda 管理。**

        如果你在 WSL2 里已经装了 Conda：
        ```bash
        # 1. 创建新环境并直接指定 CUDA 12.1 和常用库
        conda create -n toast_env python=3.12
        conda activate toast_env

        # 2. 一次性安装 toolkit、nccl 和 cutensor（Conda 会处理好所有依赖）
        conda install -c nvidia cuda-toolkit=12.1 nccl cutensor
        ```

        **如果你坚持要用 `apt` 全局安装 CUDA 12：**
        请去官网选择 `CUDA 12.4`（目前的黄金稳定版），步骤和你之前一样，但记得把路径里的 `13-2` 全部改为 `12-4`。

    **避坑提醒**：安装完 CUDA 12 后，别忘了把 `.bashrc` 里的环境变量从 `cuda-13.0` 改成 `cuda-12.4`！

    你需要我帮你找 CUDA 12.4 在 WSL2 下的具体官方安装命令吗？

* 即使是 wsl 下，vim-gtk3 也可以使用系统粘贴板

